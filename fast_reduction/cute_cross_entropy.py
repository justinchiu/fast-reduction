"""
CuTe DSL cross-entropy and entropy kernels for Hopper (sm90).

Based on quack's CrossEntropy (Guo, Zadouri, Dao 2025).

Three kernel variants:
  1. CrossEntropyOnly   — CE loss only (2 reductions: max, sum_exp)
  2. EntropyOnly        — entropy only (3 reductions: max, sum_exp, sum_x_exp)
  3. CrossEntropyEntropy — both in one pass (3 reductions)

All share the same tiling/copy/cluster infrastructure via _KernelBase.
"""

import math
from functools import partial
from typing import Optional, Type, Tuple

import torch
from torch import Tensor

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync

from fast_reduction import cute_utils
from fast_reduction.reduce import row_reduce, online_softmax_reduce


# ---- helpers ----

_torch2cute = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _make_fake_tensor(dtype, shape, divisibility=1, leading_dim=-1):
    if leading_dim < 0:
        leading_dim = len(shape) + leading_dim
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride,
        assumed_align=divisibility * dtype.width // 8,
    )


def _tiled_copy_2d(dtype, threads_per_row, num_threads, num_copy_elems=1):
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(), dtype,
        num_bits_per_copy=num_copy_elems * dtype.width,
    )
    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row), order=(1, 0),
    )
    val_layout = cute.make_layout((1, num_copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


@cute.jit
def _predicate_k(tAcA, limit):
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (cute.size(tAcA.shape, mode=[0, 1]),
             cute.size(tAcA.shape, mode=[1]),
             cute.size(tAcA.shape, mode=[2])),
            stride=(cute.size(tAcA.shape, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


@cute.jit
def _copy(src, dst, *, pred=None, is_async=False):
    num_copy_elems = src.shape[0][0]
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(
        copy_op, src.element_type,
        num_bits_per_copy=min(128, num_copy_elems * src.element_type.width),
    )
    cute.copy(copy_atom, src, dst, pred=pred)


# ---- shared kernel infrastructure ----

class _KernelBase:
    """Shared tiling, copy, cluster, and reduction buffer infrastructure."""

    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        self.dtype = dtype
        self.N = N
        self.stage = 2
        self.reduction_dtype = Float32
        self.reload_from = None if N <= 16384 else "smem"

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _num_threads(self):
        return 128 if self.N <= 16384 else 256

    def _set_cluster_n(self):
        N = self.N
        if const_expr(self.dtype.width == 16):
            thresholds = [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]
        else:
            thresholds = [(16 * 1024, 1), (64 * 1024, 2), (128 * 1024, 4), (256 * 1024, 8)]
        for limit, cluster in thresholds:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    def _get_tiled_copy(self, vecsize):
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()
        num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n)
        tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = _tiled_copy_2d(self.dtype, threads_per_row, num_threads, vecsize)
        return tiled_copy, tiler_mn, threads_per_row

    def _get_reduction_buffer_layout(self, tv_layout):
        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        warps_per_row = (
            num_warps
            if cute.rank(tv_layout.shape[0]) == 1
            else max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
        )
        return cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, self.cluster_n), self.stage),
            order=(1, 0, 2),
        )

    def _allocate_reduction_buffer_and_mbar(self, smem, tv_layout):
        reduction_buffer = smem.allocate_tensor(
            self.reduction_dtype,
            self._get_reduction_buffer_layout(tv_layout),
            byte_alignment=8,
        )
        if const_expr(self.cluster_n > 1):
            mbar_ptr = smem.allocate_array(Int64, num_elems=self.stage)
        else:
            mbar_ptr = None
        return reduction_buffer, mbar_ptr

    @cute.jit
    def _initialize_cluster(self, tidx, mbar_ptr, num_warps):
        if const_expr(self.cluster_n > 1):
            if tidx < self.stage:
                cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()


# ---- kernel 1: cross-entropy only ----

class CrossEntropyOnly(_KernelBase):
    """CuTe DSL kernel: cross-entropy loss only.

    2 reductions: max, sum_exp.  Writes loss[M].
    """

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,       # (M, N)
        mTarget: cute.Tensor,  # (M,)
        mLoss: cute.Tensor,    # (M,)
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        vecsize = math.gcd(self.N, 128 // const_expr(mX.element_type.width))
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        self.kernel(
            mX, mTarget, mLoss,
            tiler_mn, tiled_copy, threads_per_row,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mTarget: cute.Tensor,
        mLoss: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, idX)]

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        thr_copy = tiled_copy.get_slice(tidx)
        tXgX = thr_copy.partition_S(gX)
        tXsX = thr_copy.partition_D(sX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tXrX = cute.make_fragment_like(tXgX)

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = None if is_even_N else _predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        copy = partial(_copy, pred=tXpX)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        row = tXcX[0][0]
        target = Int32.zero
        if row < shape[0]:
            target = Int32(mTarget[row])

        # ---- load: GMEM -> SMEM -> registers ----
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        if const_expr(not is_even_N):
            cute_utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        # ---- pass 1: max ----
        max_x = row_reduce(
            x, cute.ReductionOp.MAX, threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
            init_val=-Float32.inf,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )

        if const_expr(self.reload_from == "smem"):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        # ---- exp(x - max) ----
        log2_e = math.log2(math.e)
        exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=False)

        # ---- pass 2: sum_exp ----
        sum_exp = row_reduce(
            exp_x, cute.ReductionOp.ADD, threads_per_row,
            reduction_buffer[None, None, 1],
            mbar_ptr + 1 if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
        )

        # ---- extract target logit ----
        target_logit = Float32.zero
        should_ignore = Boolean(target < 0)
        if row < shape[0] and tXcX[0][1] == 0 and not should_ignore:
            target_logit = Float32(mX[row, target])

        # ---- write loss ----
        if (
            tXcX[0][1] == 0
            and row < shape[0]
            and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
        ):
            lse = max_x + cute.math.log(sum_exp, fastmath=True)
            ce_loss = (lse - target_logit) if not should_ignore else Float32.zero
            mLoss[row] = mLoss.element_type(ce_loss)


# ---- kernel 2: entropy only ----

class EntropyOnly(_KernelBase):
    """CuTe DSL kernel: entropy only.

    3 reductions: max, sum_exp, sum(x * exp(x - max)).  Writes entropy[M].
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        super().__init__(dtype, N)
        self.stage = 3  # 3 passes need 3 mbarrier stages to avoid phase reuse

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,         # (M, N)
        mEntropy: cute.Tensor,   # (M,)
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        vecsize = math.gcd(self.N, 128 // const_expr(mX.element_type.width))
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        self.kernel(
            mX, mEntropy,
            tiler_mn, tiled_copy, threads_per_row,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mEntropy: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, idX)]

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        thr_copy = tiled_copy.get_slice(tidx)
        tXgX = thr_copy.partition_S(gX)
        tXsX = thr_copy.partition_D(sX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tXrX = cute.make_fragment_like(tXgX)

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = None if is_even_N else _predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        copy = partial(_copy, pred=tXpX)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        row = tXcX[0][0]

        # ---- load: GMEM -> SMEM -> registers ----
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        if const_expr(not is_even_N):
            # Use finite sentinel (not -inf) so x*exp(x-max) = -1e4*0 = -0
            # instead of -inf*0 = NaN.
            cute_utils.fill_oob(tXsX, tXpX, tXsX.element_type(-1e4))
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        # ---- pass 1: max ----
        max_x = row_reduce(
            x, cute.ReductionOp.MAX, threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
            init_val=-Float32.inf,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )

        if const_expr(self.reload_from == "smem"):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        # ---- exp(x - max) ----
        log2_e = math.log2(math.e)
        exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=False)

        # ---- pass 2: sum_exp ----
        sum_exp = row_reduce(
            exp_x, cute.ReductionOp.ADD, threads_per_row,
            reduction_buffer[None, None, 1],
            mbar_ptr + 1 if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
        )

        # ---- pass 3: sum(x * exp(x - max)) ----
        x_times_exp = x * exp_x
        sum_x_exp = row_reduce(
            x_times_exp, cute.ReductionOp.ADD, threads_per_row,
            reduction_buffer[None, None, 2],
            mbar_ptr + 2 if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
        )

        # ---- write entropy ----
        if (
            tXcX[0][1] == 0
            and row < shape[0]
            and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
        ):
            lse = max_x + cute.math.log(sum_exp, fastmath=True)
            ent = lse - sum_x_exp / sum_exp
            mEntropy[row] = mEntropy.element_type(ent)


# ---- kernel 3: cross-entropy + entropy (joint) ----

class CrossEntropyEntropy(_KernelBase):
    """CuTe DSL kernel: cross-entropy + entropy from logits in one pass.

    3 reductions: max, sum_exp, sum(x * exp(x - max)).
    Writes both loss[M] and entropy[M].
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        super().__init__(dtype, N)
        self.stage = 3  # 3 passes need 3 mbarrier stages to avoid phase reuse

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,       # (M, N)
        mTarget: cute.Tensor,  # (M,)
        mLoss: cute.Tensor,    # (M,)
        mEntropy: cute.Tensor, # (M,)
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        vecsize = math.gcd(self.N, 128 // const_expr(mX.element_type.width))
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        self.kernel(
            mX, mTarget, mLoss, mEntropy,
            tiler_mn, tiled_copy, threads_per_row,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mTarget: cute.Tensor,
        mLoss: cute.Tensor,
        mEntropy: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, idX)]

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        thr_copy = tiled_copy.get_slice(tidx)
        tXgX = thr_copy.partition_S(gX)
        tXsX = thr_copy.partition_D(sX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tXrX = cute.make_fragment_like(tXgX)

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = None if is_even_N else _predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        copy = partial(_copy, pred=tXpX)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        row = tXcX[0][0]
        target = Int32.zero
        if row < shape[0]:
            target = Int32(mTarget[row])

        # ---- load: GMEM -> SMEM -> registers ----
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        if const_expr(not is_even_N):
            # Use finite sentinel (not -inf) so x*exp(x-max) = -1e4*0 = -0
            # instead of -inf*0 = NaN.
            cute_utils.fill_oob(tXsX, tXpX, tXsX.element_type(-1e4))
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        # ---- pass 1: max ----
        max_x = row_reduce(
            x, cute.ReductionOp.MAX, threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
            init_val=-Float32.inf,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )

        if const_expr(self.reload_from == "smem"):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        # ---- exp(x - max) ----
        log2_e = math.log2(math.e)
        exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=False)

        # ---- pass 2: sum_exp ----
        sum_exp = row_reduce(
            exp_x, cute.ReductionOp.ADD, threads_per_row,
            reduction_buffer[None, None, 1],
            mbar_ptr + 1 if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
        )

        # ---- pass 3: sum(x * exp(x - max)) for entropy ----
        x_times_exp = x * exp_x
        sum_x_exp = row_reduce(
            x_times_exp, cute.ReductionOp.ADD, threads_per_row,
            reduction_buffer[None, None, 2],
            mbar_ptr + 2 if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
        )

        # ---- extract target logit ----
        target_logit = Float32.zero
        should_ignore = Boolean(target < 0)
        if row < shape[0] and tXcX[0][1] == 0 and not should_ignore:
            target_logit = Float32(mX[row, target])

        # ---- write outputs ----
        if (
            tXcX[0][1] == 0
            and row < shape[0]
            and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
        ):
            lse = max_x + cute.math.log(sum_exp, fastmath=True)
            ce_loss = (lse - target_logit) if not should_ignore else Float32.zero
            mLoss[row] = mLoss.element_type(ce_loss)
            ent = lse - sum_x_exp / sum_exp
            mEntropy[row] = mEntropy.element_type(ent)


# ---- torch wrappers ----

_compile_cache_ce = {}
_compile_cache_ent = {}
_compile_cache_joint = {}


@torch.library.custom_op("fast_reduction::ce_fwd", mutates_args={"loss"})
def ce_fwd_out(
    x: Tensor,
    target: Tensor,
    loss: Tensor,
) -> None:
    """Cross-entropy forward, writing into pre-allocated output."""
    assert x.dim() == 2 and target.dim() == 1
    assert x.is_cuda and target.is_cuda
    N = x.size(1)
    dtype = _torch2cute[x.dtype]
    target_dtype = {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64}[target.dtype]
    key = (dtype, target_dtype, N)
    if key not in _compile_cache_ce:
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        x_cute = _make_fake_tensor(dtype, (batch_sym, N), div)
        target_cute = _make_fake_tensor(target_dtype, (batch_sym,))
        loss_cute = _make_fake_tensor(Float32, (batch_sym,))
        op = CrossEntropyOnly(dtype, N)
        _compile_cache_ce[key] = cute.compile(
            op, x_cute, target_cute, loss_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _compile_cache_ce[key](x, target, loss)


def ce_fwd(x: Tensor, target: Tensor) -> Tensor:
    """Cross-entropy loss from logits. Returns loss [M] fp32."""
    M = x.size(0)
    loss = torch.empty(M, device=x.device, dtype=torch.float32)
    ce_fwd_out(x, target, loss)
    return loss


@torch.library.custom_op("fast_reduction::entropy_fwd", mutates_args={"entropy"})
def entropy_fwd_out(
    x: Tensor,
    entropy: Tensor,
) -> None:
    """Entropy forward, writing into pre-allocated output."""
    assert x.dim() == 2
    assert x.is_cuda
    N = x.size(1)
    dtype = _torch2cute[x.dtype]
    key = (dtype, N)
    if key not in _compile_cache_ent:
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        x_cute = _make_fake_tensor(dtype, (batch_sym, N), div)
        entropy_cute = _make_fake_tensor(Float32, (batch_sym,))
        op = EntropyOnly(dtype, N)
        _compile_cache_ent[key] = cute.compile(
            op, x_cute, entropy_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _compile_cache_ent[key](x, entropy)


def entropy_fwd(x: Tensor) -> Tensor:
    """Entropy from logits. Returns entropy [M] fp32."""
    M = x.size(0)
    entropy = torch.empty(M, device=x.device, dtype=torch.float32)
    entropy_fwd_out(x, entropy)
    return entropy


@torch.library.custom_op("fast_reduction::ce_entropy_fwd", mutates_args={"loss", "entropy"})
def ce_entropy_fwd_out(
    x: Tensor,
    target: Tensor,
    loss: Tensor,
    entropy: Tensor,
) -> None:
    """Cross-entropy + entropy forward, writing into pre-allocated outputs."""
    assert x.dim() == 2 and target.dim() == 1
    assert x.is_cuda and target.is_cuda
    N = x.size(1)
    dtype = _torch2cute[x.dtype]
    target_dtype = {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64}[target.dtype]
    key = (dtype, target_dtype, N)
    if key not in _compile_cache_joint:
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        x_cute = _make_fake_tensor(dtype, (batch_sym, N), div)
        target_cute = _make_fake_tensor(target_dtype, (batch_sym,))
        loss_cute = _make_fake_tensor(Float32, (batch_sym,))
        entropy_cute = _make_fake_tensor(Float32, (batch_sym,))
        op = CrossEntropyEntropy(dtype, N)
        _compile_cache_joint[key] = cute.compile(
            op, x_cute, target_cute, loss_cute, entropy_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _compile_cache_joint[key](x, target, loss, entropy)


def ce_entropy_fwd(x: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Cross-entropy loss and entropy from logits in one pass.

    Returns (loss [M], entropy [M]) both fp32.
    """
    M = x.size(0)
    loss = torch.empty(M, device=x.device, dtype=torch.float32)
    entropy = torch.empty(M, device=x.device, dtype=torch.float32)
    ce_entropy_fwd_out(x, target, loss, entropy)
    return loss, entropy
