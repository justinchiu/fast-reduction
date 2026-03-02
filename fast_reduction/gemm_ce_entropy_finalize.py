"""
Finalization kernel for Level 5 GEMM epilogue fused CE+entropy.

After the GEMM epilogue produces partial reductions [N_tiles, M, 4] (where the 4
values per row are: max_x, sum_exp, sum_x_exp, target_logit), this kernel merges
partials across N_tiles using online softmax and writes:
  - loss[M]    = lse - target_logit
  - entropy[M] = lse - sum_x_exp / sum_exp

This is a simple row-reduction kernel — each thread handles one row.  The partial
buffer is ~32 MB for V=128256, M=4096, CTA_N=256 (N_tiles ≈ 502).
"""

import math
from typing import Tuple

import torch
from torch import Tensor

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from typing import Optional

from cutlass import Int32, Float32, const_expr


class Finalize:
    """CuTe DSL kernel: merge partial reductions from GEMM epilogue tiles.

    Each block handles BLOCK_M rows.  Within each row, sequentially reads
    N_tiles partials and merges with online softmax.
    """

    BLOCK_M = 128  # rows per block

    @cute.jit
    def __call__(
        self,
        mPartials: cute.Tensor,   # (N_tiles, M, 4) fp32
        mLoss: cute.Tensor,       # (M,) fp32
        mEntropy: cute.Tensor,    # (M,) fp32
        mLSE: Optional[cute.Tensor],  # (M,) fp32, optional LSE output
        n_tiles: Int32,           # number of N tiles to merge
        stream: cuda.CUstream,
    ):
        M = mPartials.shape[1]
        self.kernel(mPartials, mLoss, mEntropy, mLSE, n_tiles).launch(
            grid=[cute.ceil_div(M, self.BLOCK_M), 1, 1],
            block=[self.BLOCK_M, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mPartials: cute.Tensor,   # (N_tiles, M, 4)
        mLoss: cute.Tensor,       # (M,)
        mEntropy: cute.Tensor,    # (M,)
        mLSE: Optional[cute.Tensor],  # (M,)
        n_tiles: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        M = mPartials.shape[1]
        row = bidx * self.BLOCK_M + tidx

        if row < M:
            # Online softmax merge across N_tiles
            global_max = -Float32.inf
            global_sum_exp = Float32(0.0)
            global_sum_x_exp = Float32(0.0)
            global_target_logit = Float32(0.0)

            log2_e = Float32(math.log2(math.e))

            for t in cutlass.range(n_tiles, unroll=1):
                p_max = Float32(mPartials[t, row, 0])
                p_sum_exp = Float32(mPartials[t, row, 1])
                p_sum_x_exp = Float32(mPartials[t, row, 2])
                p_target = Float32(mPartials[t, row, 3])

                # Online softmax merge
                new_max = cute.arch.fmax(global_max, p_max)
                adj_old = cute.math.exp2((global_max - new_max) * log2_e, fastmath=True)
                adj_new = cute.math.exp2((p_max - new_max) * log2_e, fastmath=True)

                global_sum_exp = global_sum_exp * adj_old + p_sum_exp * adj_new
                global_sum_x_exp = global_sum_x_exp * adj_old + p_sum_x_exp * adj_new
                global_max = new_max
                global_target_logit += p_target  # only one tile has nonzero target_logit

            # Compute final outputs
            lse = global_max + cute.math.log(global_sum_exp, fastmath=True)
            ce_loss = lse - global_target_logit
            entropy = lse - global_sum_x_exp * cute.arch.rcp_approx(global_sum_exp)

            mLoss[row] = ce_loss
            mEntropy[row] = entropy
            if const_expr(mLSE is not None):
                mLSE[row] = lse


# ---- compile cache and torch wrapper ----

_compile_cache_finalize = {}


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


@torch.library.custom_op(
    "fast_reduction::finalize_ce_entropy", mutates_args={"loss", "entropy"}
)
def finalize_ce_entropy_out(
    partials: Tensor,   # (N_tiles, M, 4) fp32
    loss: Tensor,       # (M,) fp32
    entropy: Tensor,    # (M,) fp32
    n_tiles: int,
) -> None:
    """Merge partial reductions -> loss + entropy."""
    _finalize_ce_entropy_impl(partials, loss, entropy, n_tiles, lse=None)


@torch.library.custom_op(
    "fast_reduction::finalize_ce_entropy_lse",
    mutates_args={"loss", "entropy", "lse"},
)
def finalize_ce_entropy_lse_out(
    partials: Tensor,   # (N_tiles, M, 4) fp32
    loss: Tensor,       # (M,) fp32
    entropy: Tensor,    # (M,) fp32
    lse: Tensor,        # (M,) fp32
    n_tiles: int,
) -> None:
    """Merge partial reductions -> loss + entropy + lse."""
    _finalize_ce_entropy_impl(partials, loss, entropy, n_tiles, lse=lse)


def _finalize_ce_entropy_impl(partials, loss, entropy, n_tiles, lse=None):
    """Shared implementation for finalize with/without LSE output."""
    assert partials.dim() == 3 and partials.size(2) == 4
    assert partials.is_cuda
    has_lse = lse is not None
    key = ("finalize", has_lse)
    if key not in _compile_cache_finalize:
        M_sym = cute.sym_int()
        N_tiles_sym = cute.sym_int()
        partials_fake = _make_fake_tensor(Float32, (N_tiles_sym, M_sym, 4))
        loss_fake = _make_fake_tensor(Float32, (M_sym,))
        entropy_fake = _make_fake_tensor(Float32, (M_sym,))
        lse_fake = _make_fake_tensor(Float32, (M_sym,)) if has_lse else None
        n_tiles_val = Int32(0)
        op = Finalize()
        _compile_cache_finalize[key] = cute.compile(
            op,
            partials_fake,
            loss_fake,
            entropy_fake,
            lse_fake,
            n_tiles_val,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    if has_lse:
        _compile_cache_finalize[key](partials, loss, entropy, lse, n_tiles)
    else:
        _compile_cache_finalize[key](partials, loss, entropy, None, n_tiles)


def finalize_ce_entropy(
    partials: Tensor,
    n_tiles: int,
) -> Tuple[Tensor, Tensor]:
    """Merge partial reductions -> (loss, entropy).

    partials: (N_tiles, M, 4) fp32
    Returns: (loss [M], entropy [M]) both fp32
    """
    M = partials.size(1)
    loss = torch.empty(M, device=partials.device, dtype=torch.float32)
    entropy = torch.empty(M, device=partials.device, dtype=torch.float32)
    finalize_ce_entropy_out(partials, loss, entropy, n_tiles)
    return loss, entropy


def finalize_ce_entropy_with_lse(
    partials: Tensor,
    n_tiles: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Merge partial reductions -> (loss, entropy, lse).

    partials: (N_tiles, M, 4) fp32
    Returns: (loss [M], entropy [M], lse [M]) all fp32
    """
    M = partials.size(1)
    loss = torch.empty(M, device=partials.device, dtype=torch.float32)
    entropy = torch.empty(M, device=partials.device, dtype=torch.float32)
    lse = torch.empty(M, device=partials.device, dtype=torch.float32)
    finalize_ce_entropy_lse_out(partials, loss, entropy, lse, n_tiles)
    return loss, entropy, lse
