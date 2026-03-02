"""
Level 5: GEMM with fused CE+entropy epilogue.

For each chunk of the batch dimension:
  1. Launch GEMM with custom epilogue → produces partial reductions
     [N_tiles, chunk_size, 4] instead of full [chunk_size, V] logits.
  2. Launch finalization kernel → merges partials across N_tiles
     to produce loss[chunk] + entropy[chunk].

The logits never hit HBM — they are reduced inside the GEMM epilogue
directly from the accumulator registers via scratch SMEM.

Backward pass: recomputes logits via cuBLAS matmul, computes dlogits
element-wise, then matmul for d_hidden/d_weight. The backward does
NOT use the GEMM epilogue — it's matmul-bound, not memory-bound.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass import Int32, Float32
from cutlass.cute.runtime import from_dlpack

from quack.cute_dsl_utils import (
    get_device_capacity,
    get_max_active_clusters,
    torch2cute_dtype_map,
)
from quack.gemm_wrapper_utils import GemmWrapperBase, GemmTensorInfo

from fast_reduction.gemm_ce_entropy_epilogue import GemmCEEntropySm90
from fast_reduction.gemm_ce_entropy_epilogue_v2 import GemmCEEntropyV2Sm90
from fast_reduction.gemm_ce_entropy_finalize import finalize_ce_entropy
from fast_reduction.kernel import (
    _mm_setup,
    _compute_dlogits_chunk,
    fused_linear_xent_entropy_backward,
)


# ---- compile caches ----
_compile_cache_gemm_ce = {}
_compile_cache_gemm_ce_v2 = {}


def _gemm_ce_entropy_kernel(
    A: Tensor,       # (1, M, K) or (M, K) — hidden states for one chunk
    B: Tensor,       # (1, N, K) or (N, K) — weight.mT (already transposed)
    target: Tensor,  # (M,) int64
    partials: Tensor, # (N_tiles, M, 4) fp32
    V: int,          # vocab size (for N-boundary check)
    tile_M: int = 128,
    tile_N: int = 256,
    cluster_M: int = 2,
    cluster_N: int = 1,
) -> None:
    """Launch the GEMM with CE+entropy epilogue."""
    # Prepare tensors: ensure 3D (M, K, L) / (N, K, L) with L=1
    if A.ndim == 2:
        A = A.unsqueeze(0)   # (1, M, K)
    if B.ndim == 2:
        B = B.unsqueeze(0)   # (1, N, K)

    M = A.shape[1]
    N = B.shape[1]
    K = A.shape[2]

    # Standard quack tensor prep: permute (L,M,K) → (M,K,L)
    A_perm = A.permute(1, 2, 0).contiguous()  # (M, K, 1)
    B_perm = B.permute(1, 2, 0).contiguous()  # (N, K, 1)

    # Determine layout majors and dtypes
    A_info = GemmTensorInfo(A_perm)
    B_info = GemmTensorInfo(B_perm)
    A_info.dtype = torch2cute_dtype_map[A_perm.dtype]
    B_info.dtype = torch2cute_dtype_map[B_perm.dtype]

    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
    }
    A_info.major = GemmWrapperBase.get_major_order(A_perm, major_configs["A"])
    B_info.major = GemmWrapperBase.get_major_order(B_perm, major_configs["B"])

    A_cute = GemmWrapperBase.create_cute_tensor(A_perm, A_info.major, major_configs["A"])
    B_cute = GemmWrapperBase.create_cute_tensor(B_perm, B_info.major, major_configs["B"])

    # Epilogue arguments
    mTarget = from_dlpack(target.detach(), assumed_align=8).mark_layout_dynamic(leading_dim=0)
    mPartials = from_dlpack(partials.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=2)

    epi_args = GemmCEEntropySm90.EpilogueArguments(
        mTarget=mTarget,
        mPartials=mPartials,
        vocab_size=Int32(V),
        total_M=Int32(M),
    )

    tile_shape_mn = (tile_M, tile_N)
    cluster_shape_mnk = (cluster_M, cluster_N, 1)
    acc_dtype = Float32

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N)

    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore=None,
        batch_idx_permute=None,
        max_swizzle_size=8,
    )

    current_stream = cutlass_torch.current_stream()

    # Compile key
    compile_key = (
        A_info.dtype, B_info.dtype,
        A_info.major, B_info.major,
        tile_shape_mn, cluster_shape_mnk,
        "ce_entropy_epilogue",
    )

    if compile_key not in _compile_cache_gemm_ce:
        gemm_obj = GemmCEEntropySm90(
            acc_dtype,
            A_info.dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            pingpong=False,
            is_persistent=True,
        )

        _compile_cache_gemm_ce[compile_key] = cute.compile(
            gemm_obj,
            A_cute,
            B_cute,
            None,   # mD — no logits output
            None,   # mC — no C input
            epi_args,
            scheduler_args,
            None,   # varlen_args
            current_stream,
        )

    _compile_cache_gemm_ce[compile_key](
        A_cute,
        B_cute,
        None,   # mD
        None,   # mC
        epi_args,
        scheduler_args,
        None,   # varlen_args
        current_stream,
    )


def gemm_fused_ce_entropy(
    hidden_states: Tensor,
    weight: Tensor,
    target: Tensor,
    bias: Optional[Tensor] = None,
    chunk_size: int = 4096,
    tile_M: int = 128,
    tile_N: int = 256,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Level 5: GEMM with fused CE+entropy epilogue.

    Returns (ce_loss, entropy, log_probs), each shaped like ``target``.

    For each chunk:
      1. GEMM epilogue reduces logits → partial stats [N_tiles, chunk, 4]
      2. Finalization kernel merges partials → loss[chunk] + entropy[chunk]

    The logits tensor [chunk, V] is NEVER materialized in HBM.
    """
    batch_shape = hidden_states.shape[:-1]
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    target_1d = target.reshape(-1)
    B = hidden_2d.shape[0]
    H = hidden_2d.shape[1]
    V = weight.shape[0]

    ce_loss = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)
    entropy = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)

    actual_chunk = min(chunk_size, B)
    N_tiles = (V + tile_N - 1) // tile_N

    # Pre-allocate partial buffer (reused every chunk)
    partials_buf = torch.empty(
        N_tiles, actual_chunk, 4,
        device=hidden_2d.device, dtype=torch.float32,
    )

    # Weight transposed for GEMM: (V, H) → B tensor is (N, K) = (V, H)
    # quack expects B as (N, K) and internally transposes to (K, N)
    # The GEMM computes A @ B^T = (M, K) @ (K, N) = (M, N)
    # So we pass weight directly as B in shape (1, V, H)
    weight_3d = weight.unsqueeze(0)  # (1, V, H)
    # After permute in _gemm_ce_entropy_kernel: (V, H, 1)
    # B_perm.mT gives (H, V, 1) which is (K, N, L) — correct for GEMM

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        h_chunk = hidden_2d[start:end]          # (chunk_len, H)
        t_chunk = target_1d[start:end]           # (chunk_len,)
        chunk_len = end - start

        # Ensure alignment for TMA (multiple of 8)
        assert chunk_len % 8 == 0 or chunk_len == B, (
            f"chunk_len={chunk_len} must be multiple of 8"
        )

        partials_chunk = partials_buf[:, :chunk_len, :]  # (N_tiles, chunk_len, 4)

        # Convert hidden to the right format
        # A = (1, chunk_len, H), already in (M, K) = (chunk_len, H) layout
        h_3d = h_chunk.unsqueeze(0)  # (1, chunk_len, H)

        # Launch GEMM with CE+entropy epilogue
        _gemm_ce_entropy_kernel(
            h_3d, weight_3d, t_chunk, partials_chunk, V,
            tile_M=tile_M, tile_N=tile_N,
        )

        # Launch finalization kernel: merge partials → loss + entropy
        loss_chunk, ent_chunk = finalize_ce_entropy(partials_chunk, N_tiles)
        ce_loss[start:end] = loss_chunk
        entropy[start:end] = ent_chunk

    log_probs = -ce_loss

    return (
        ce_loss.view(batch_shape),
        entropy.view(batch_shape),
        log_probs.view(batch_shape),
    )


# ========================================================================
# Level 5.2: Two-pass epilogue for improved numerical precision
# ========================================================================

def _gemm_ce_entropy_v2_kernel(
    A: Tensor,       # (1, M, K) or (M, K)
    B: Tensor,       # (1, N, K) or (N, K)
    target: Tensor,  # (M,) int64
    partials: Tensor, # (N_tiles, M, 4) fp32
    V: int,
    tile_M: int = 128,
    tile_N: int = 256,
    cluster_M: int = 2,
    cluster_N: int = 1,
) -> None:
    """Launch the GEMM with two-pass CE+entropy epilogue (v2)."""
    if A.ndim == 2:
        A = A.unsqueeze(0)
    if B.ndim == 2:
        B = B.unsqueeze(0)

    M = A.shape[1]
    N = B.shape[1]
    K = A.shape[2]

    A_perm = A.permute(1, 2, 0).contiguous()
    B_perm = B.permute(1, 2, 0).contiguous()

    A_info = GemmTensorInfo(A_perm)
    B_info = GemmTensorInfo(B_perm)
    A_info.dtype = torch2cute_dtype_map[A_perm.dtype]
    B_info.dtype = torch2cute_dtype_map[B_perm.dtype]

    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
    }
    A_info.major = GemmWrapperBase.get_major_order(A_perm, major_configs["A"])
    B_info.major = GemmWrapperBase.get_major_order(B_perm, major_configs["B"])

    A_cute = GemmWrapperBase.create_cute_tensor(A_perm, A_info.major, major_configs["A"])
    B_cute = GemmWrapperBase.create_cute_tensor(B_perm, B_info.major, major_configs["B"])

    mTarget = from_dlpack(target.detach(), assumed_align=8).mark_layout_dynamic(leading_dim=0)
    mPartials = from_dlpack(partials.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=2)

    epi_args = GemmCEEntropyV2Sm90.EpilogueArguments(
        mTarget=mTarget,
        mPartials=mPartials,
        vocab_size=Int32(V),
        total_M=Int32(M),
    )

    tile_shape_mn = (tile_M, tile_N)
    cluster_shape_mnk = (cluster_M, cluster_N, 1)
    acc_dtype = Float32

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N)

    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore=None,
        batch_idx_permute=None,
        max_swizzle_size=8,
    )

    current_stream = cutlass_torch.current_stream()

    compile_key = (
        A_info.dtype, B_info.dtype,
        A_info.major, B_info.major,
        tile_shape_mn, cluster_shape_mnk,
        "ce_entropy_epilogue_v2",
    )

    if compile_key not in _compile_cache_gemm_ce_v2:
        gemm_obj = GemmCEEntropyV2Sm90(
            acc_dtype,
            A_info.dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            pingpong=False,
            is_persistent=True,
        )

        _compile_cache_gemm_ce_v2[compile_key] = cute.compile(
            gemm_obj,
            A_cute,
            B_cute,
            None,
            None,
            epi_args,
            scheduler_args,
            None,
            current_stream,
        )

    _compile_cache_gemm_ce_v2[compile_key](
        A_cute,
        B_cute,
        None,
        None,
        epi_args,
        scheduler_args,
        None,
        current_stream,
    )


def gemm_fused_ce_entropy_v2(
    hidden_states: Tensor,
    weight: Tensor,
    target: Tensor,
    bias: Optional[Tensor] = None,
    chunk_size: int = 4096,
    tile_M: int = 128,
    tile_N: int = 256,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Level 5.2: GEMM with two-pass CE+entropy epilogue (improved precision).

    Same interface as gemm_fused_ce_entropy (v5.1). The two-pass epilogue
    eliminates online softmax merge and fastmath, reducing epilogue error
    to near-zero. Remaining error is only from WGMMA bf16 matmul precision.

    Returns (ce_loss, entropy, log_probs), each shaped like ``target``.
    """
    batch_shape = hidden_states.shape[:-1]
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    target_1d = target.reshape(-1)
    B = hidden_2d.shape[0]
    H = hidden_2d.shape[1]
    V = weight.shape[0]

    ce_loss = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)
    entropy = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)

    actual_chunk = min(chunk_size, B)
    N_tiles = (V + tile_N - 1) // tile_N

    partials_buf = torch.empty(
        N_tiles, actual_chunk, 4,
        device=hidden_2d.device, dtype=torch.float32,
    )

    weight_3d = weight.unsqueeze(0)

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        h_chunk = hidden_2d[start:end]
        t_chunk = target_1d[start:end]
        chunk_len = end - start

        assert chunk_len % 8 == 0 or chunk_len == B, (
            f"chunk_len={chunk_len} must be multiple of 8"
        )

        partials_chunk = partials_buf[:, :chunk_len, :]

        h_3d = h_chunk.unsqueeze(0)

        _gemm_ce_entropy_v2_kernel(
            h_3d, weight_3d, t_chunk, partials_chunk, V,
            tile_M=tile_M, tile_N=tile_N,
        )

        loss_chunk, ent_chunk = finalize_ce_entropy(partials_chunk, N_tiles)
        ce_loss[start:end] = loss_chunk
        entropy[start:end] = ent_chunk

    log_probs = -ce_loss

    return (
        ce_loss.view(batch_shape),
        entropy.view(batch_shape),
        log_probs.view(batch_shape),
    )


# ========================================================================
# Level 5 backward (shared by v5.1 and v5.2)
# ========================================================================

class GemmFusedCEEntropy(torch.autograd.Function):
    """Differentiable Level 5 GEMM with fused CE+entropy epilogue.

    Forward uses GEMM epilogue (logits never hit HBM).
    Backward recomputes logits via cuBLAS and uses PyTorch ops for dlogits.
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, target, bias, chunk_size):
        ce_loss, entropy, log_probs = gemm_fused_ce_entropy(
            hidden_states, weight, target, bias=bias, chunk_size=chunk_size,
        )
        ctx.save_for_backward(hidden_states, weight, target, entropy)
        ctx.bias = bias
        ctx.chunk_size = chunk_size
        return ce_loss, entropy, log_probs

    @staticmethod
    def backward(ctx, g_ce, g_ent, g_lp):
        hidden_states, weight, target, entropy = ctx.saved_tensors
        if g_lp is not None:
            g_ce = g_ce - g_lp
        # Reuse the same chunked backward as levels 3-4
        d_hidden, d_weight, d_bias = fused_linear_xent_entropy_backward(
            hidden_states, weight, target, g_ce, g_ent, entropy,
            bias=ctx.bias, chunk_size=ctx.chunk_size,
        )
        return d_hidden, d_weight, None, d_bias, None


def gemm_fused_ce_entropy_differentiable(
    hidden_states: Tensor,
    weight: Tensor,
    target: Tensor,
    bias: Optional[Tensor] = None,
    chunk_size: int = 4096,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Differentiable version of gemm_fused_ce_entropy (Level 5).

    Same interface, but supports .backward() on the returned tensors.
    """
    return GemmFusedCEEntropy.apply(hidden_states, weight, target, bias, chunk_size)
