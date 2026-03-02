"""
Fused CuTe DSL dlogits kernel: computes dlogits in one HBM pass.

Given fp32 logits [M, V], target [M], LSE [M], entropy [M], and per-row
gradient scalars g_ce [M] and g_ent [M], computes:

    dz_j = p_j * (g_ce - g_ent * (log_p_j + H)) - g_ce * 1{j == target}

where p_j = exp(z_j - lse), log_p_j = z_j - lse, H = entropy.

Replaces 4-5 unfused PyTorch element-wise ops (~14ms/chunk at V=128256)
with a single kernel (~1ms at 3.35 TB/s).

Design: 2D grid (M rows, ceil(V / BLOCK_N) column tiles). Each block has
BLOCK_N threads; each thread computes one dlogits element per column tile.
No shared memory or inter-thread communication needed (purely pointwise).
"""

import math
from typing import Tuple

import torch
from torch import Tensor

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr


class ComputeDlogits:
    """CuTe DSL kernel: fused dlogits computation.

    2D grid: grid[0] = M rows, grid[1] = ceil(V / BLOCK_N) column tiles.
    Each thread handles one element: (row, col).
    """

    BLOCK_N = 256  # threads per block, one thread per column element

    @cute.jit
    def __call__(
        self,
        mLogits: cute.Tensor,    # (M, V) fp32 — input logits
        mTarget: cute.Tensor,    # (M,) int32/int64 — target indices
        mLSE: cute.Tensor,       # (M,) fp32 — log-sum-exp
        mEntropy: cute.Tensor,   # (M,) fp32 — entropy
        mGCE: cute.Tensor,       # (M,) fp32 — per-row CE gradient
        mGEnt: cute.Tensor,      # (M,) fp32 — per-row entropy gradient
        mDLogits: cute.Tensor,   # (M, V) fp32 — output dlogits
        vocab_size: Int32,
        stream: cuda.CUstream,
    ):
        M = mLogits.shape[0]
        self.kernel(
            mLogits, mTarget, mLSE, mEntropy, mGCE, mGEnt, mDLogits, vocab_size,
        ).launch(
            grid=[M, cute.ceil_div(vocab_size, self.BLOCK_N), 1],
            block=[self.BLOCK_N, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mLogits: cute.Tensor,    # (M, V) fp32
        mTarget: cute.Tensor,    # (M,)
        mLSE: cute.Tensor,       # (M,) fp32
        mEntropy: cute.Tensor,   # (M,) fp32
        mGCE: cute.Tensor,       # (M,) fp32
        mGEnt: cute.Tensor,      # (M,) fp32
        mDLogits: cute.Tensor,   # (M, V) fp32
        vocab_size: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        row = cute.arch.block_idx()[0]
        col_tile = cute.arch.block_idx()[1]
        col = col_tile * self.BLOCK_N + tidx

        if col < vocab_size:
            lse = Float32(mLSE[row])
            ent = Float32(mEntropy[row])
            g_ce = Float32(mGCE[row])
            g_ent = Float32(mGEnt[row])
            target = Int32(mTarget[row])

            # dz_j = p_j * (g_ce - g_ent*(log_p_j + ent)) - g_ce*1{j==t}
            # = p_j * (A + B * log_p_j) - g_ce*1{j==t}
            row_a = g_ce - g_ent * ent
            row_b = -g_ent

            log2_e = Float32(math.log2(math.e))

            z_j = Float32(mLogits[row, col])
            log_p = z_j - lse
            p_j = cute.math.exp2(log_p * log2_e, fastmath=True)
            dz = p_j * (row_a + row_b * log_p)
            if col == target:
                dz = dz - g_ce
            mDLogits[row, col] = dz


# ---- compile cache and torch wrapper ----

_compile_cache_dlogits = {}


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
    "fast_reduction::compute_dlogits", mutates_args={"dlogits"}
)
def compute_dlogits_out(
    logits: Tensor,     # (M, V) fp32
    target: Tensor,     # (M,) int32 or int64
    lse: Tensor,        # (M,) fp32
    entropy: Tensor,    # (M,) fp32
    g_ce: Tensor,       # (M,) fp32
    g_ent: Tensor,      # (M,) fp32
    dlogits: Tensor,    # (M, V) fp32
) -> None:
    """Fused dlogits computation, writing into pre-allocated output."""
    assert logits.dim() == 2
    assert logits.is_cuda
    V = logits.size(1)
    target_dtype = {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64}[target.dtype]
    key = ("dlogits", target_dtype, V)
    if key not in _compile_cache_dlogits:
        M_sym = cute.sym_int()
        logits_fake = _make_fake_tensor(Float32, (M_sym, V), divisibility=1)
        target_fake = _make_fake_tensor(target_dtype, (M_sym,))
        lse_fake = _make_fake_tensor(Float32, (M_sym,))
        ent_fake = _make_fake_tensor(Float32, (M_sym,))
        gce_fake = _make_fake_tensor(Float32, (M_sym,))
        gent_fake = _make_fake_tensor(Float32, (M_sym,))
        dlogits_fake = _make_fake_tensor(Float32, (M_sym, V), divisibility=1)
        vocab_val = Int32(0)
        op = ComputeDlogits()
        _compile_cache_dlogits[key] = cute.compile(
            op,
            logits_fake, target_fake, lse_fake, ent_fake,
            gce_fake, gent_fake, dlogits_fake, vocab_val,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _compile_cache_dlogits[key](
        logits, target, lse, entropy, g_ce, g_ent, dlogits, V,
    )


def compute_dlogits(
    logits: Tensor,
    target: Tensor,
    lse: Tensor,
    entropy: Tensor,
    g_ce: Tensor,
    g_ent: Tensor,
) -> Tensor:
    """Fused dlogits from logits + LSE + entropy + gradient scalars.

    logits:  (M, V) fp32
    target:  (M,) int32/int64
    lse:     (M,) fp32
    entropy: (M,) fp32
    g_ce:    (M,) fp32 — per-row CE gradient scalar
    g_ent:   (M,) fp32 — per-row entropy gradient scalar

    Returns: dlogits (M, V) fp32
    """
    dlogits = torch.empty_like(logits)
    compute_dlogits_out(logits, target, lse, entropy, g_ce, g_ent, dlogits)
    return dlogits
