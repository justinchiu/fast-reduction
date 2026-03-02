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
from fast_reduction.gemm_ce_entropy_bwd_epilogue import GemmCEEntropyBwdSm90
from fast_reduction.gemm_ce_entropy_finalize import (
    finalize_ce_entropy,
    finalize_ce_entropy_with_lse,
)
from fast_reduction.cute_dlogits import compute_dlogits
from fast_reduction.kernel import (
    _mm_setup,
    _compute_dlogits_chunk,
    _compute_dlogits_from_lse,
    fused_linear_xent_entropy_backward,
)


# ---- compile caches ----
_compile_cache_gemm_ce = {}
_compile_cache_gemm_ce_v2 = {}
_compile_cache_gemm_ce_bwd = {}


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


def _gemm_ce_entropy_bwd_kernel(
    A: Tensor,        # (1, M, K) or (M, K)
    B: Tensor,        # (1, N, K) or (N, K)
    target: Tensor,   # (M,) int64
    partials: Tensor, # (N_tiles, M, 4) fp32
    logits: Tensor,   # (M, V) fp32 — logits output
    V: int,
    tile_M: int = 128,
    tile_N: int = 256,
    cluster_M: int = 2,
    cluster_N: int = 1,
) -> None:
    """Launch the GEMM with CE+entropy epilogue that also writes logits."""
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
    mLogits = from_dlpack(logits.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=1)

    epi_args = GemmCEEntropyBwdSm90.EpilogueArguments(
        mTarget=mTarget,
        mPartials=mPartials,
        mLogits=mLogits,
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
        "ce_entropy_bwd_epilogue",
    )

    if compile_key not in _compile_cache_gemm_ce_bwd:
        gemm_obj = GemmCEEntropyBwdSm90(
            acc_dtype,
            A_info.dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            pingpong=False,
            is_persistent=True,
        )

        _compile_cache_gemm_ce_bwd[compile_key] = cute.compile(
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

    _compile_cache_gemm_ce_bwd[compile_key](
        A_cute,
        B_cute,
        None,
        None,
        epi_args,
        scheduler_args,
        None,
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


# ========================================================================
# Level 5 fast backward: GEMM forward + interleaved backward
# ========================================================================

def _gemm_fused_fwd_bwd(
    hidden_states: Tensor,
    weight: Tensor,
    target: Tensor,
    bias: Optional[Tensor] = None,
    chunk_size: int = 4096,
    ce_weight: float = 1.0,
    ent_weight: float = -1.0,
    tile_M: int = 128,
    tile_N: int = 256,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Level 5 forward + backward in one pass.

    Forward: GEMM epilogue → partials → finalize (with LSE).
    Backward: recompute logits via cuBLAS → dlogits using saved LSE →
              d_hidden, d_weight matmuls.

    Returns: (ce_loss, entropy, d_hidden, d_weight, d_bias)
    """
    batch_shape = hidden_states.shape[:-1]
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    target_1d = target.reshape(-1)
    B, H = hidden_2d.shape
    V = weight.shape[0]

    ce_loss = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)
    entropy = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)
    d_hidden = torch.empty_like(hidden_2d)
    d_weight = torch.zeros(V, H, device=hidden_2d.device, dtype=torch.float32)
    d_bias = None
    if bias is not None:
        d_bias = torch.zeros(V, device=hidden_2d.device, dtype=torch.float32)

    use_native, mm_dtype, weight_t, bias_for_mm = _mm_setup(hidden_2d, weight, bias)

    actual_chunk = min(chunk_size, B)
    N_tiles = (V + tile_N - 1) // tile_N

    # GEMM epilogue partials buffer (reused per chunk)
    partials_buf = torch.empty(
        N_tiles, actual_chunk, 4,
        device=hidden_2d.device, dtype=torch.float32,
    )
    # Logits buffer for backward recomputation
    logits_buf = torch.empty(
        actual_chunk, V, device=hidden_2d.device, dtype=mm_dtype,
    )
    logits_fp32_buf = (
        logits_buf
        if mm_dtype == torch.float32
        else torch.empty(actual_chunk, V, device=hidden_2d.device, dtype=torch.float32)
    )

    weight_3d = weight.unsqueeze(0)

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk_len = end - start
        h_chunk = hidden_2d[start:end]
        t_chunk = target_1d[start:end]

        assert chunk_len % 8 == 0 or chunk_len == B

        # 1. GEMM epilogue forward: logits never hit HBM
        partials_chunk = partials_buf[:, :chunk_len, :]
        h_3d = h_chunk.unsqueeze(0)
        _gemm_ce_entropy_kernel(
            h_3d, weight_3d, t_chunk, partials_chunk, V,
            tile_M=tile_M, tile_N=tile_N,
        )

        # 2. Finalize with LSE output
        loss_chunk, ent_chunk, lse_chunk = finalize_ce_entropy_with_lse(
            partials_chunk, N_tiles,
        )
        ce_loss[start:end] = loss_chunk
        entropy[start:end] = ent_chunk

        # 3. Recompute logits via cuBLAS for backward
        logits_mm = logits_buf[:chunk_len]
        h_mm = h_chunk if use_native else h_chunk.float()
        torch.mm(h_mm, weight_t, out=logits_mm)
        if bias_for_mm is not None:
            logits_mm.add_(bias_for_mm)

        logits_fp32 = logits_mm
        if mm_dtype != torch.float32:
            logits_fp32 = logits_fp32_buf[:chunk_len]
            logits_fp32.copy_(logits_mm)

        # 4. Compute dlogits from cuBLAS logits
        #    NOTE: We use cuBLAS logits + their own logsumexp (not WGMMA LSE)
        #    because WGMMA and cuBLAS produce slightly different logits,
        #    and mixing LSE from one with logits from the other is inconsistent.
        g_ce_chunk = torch.full((chunk_len,), ce_weight, device=logits_fp32.device)
        g_ent_chunk = torch.full((chunk_len,), ent_weight, device=logits_fp32.device)
        dlogits = _compute_dlogits_chunk(
            logits_fp32, t_chunk, g_ce_chunk, g_ent_chunk, ent_chunk,
        )

        # 5. d_hidden = dlogits @ weight
        dlogits_mm = dlogits.to(mm_dtype)
        weight_for_mm = weight if use_native else weight.float()
        d_hidden[start:end] = torch.mm(dlogits_mm, weight_for_mm).to(hidden_2d.dtype)

        # 6. d_weight += dlogits.T @ hidden
        d_weight.add_(torch.mm(dlogits_mm.t(), h_chunk.to(mm_dtype)).float())

        # 7. d_bias
        if d_bias is not None:
            d_bias.add_(dlogits.sum(dim=0))

    d_weight_out = d_weight.to(weight.dtype)
    d_bias_out = d_bias.to(bias.dtype) if d_bias is not None else None

    return (
        ce_loss.view(batch_shape),
        entropy.view(batch_shape),
        d_hidden.view_as(hidden_states),
        d_weight_out,
        d_bias_out,
    )


class GemmFusedCEEntropyFast(torch.autograd.Function):
    """Fast Level 5 GEMM with interleaved forward/backward.

    Forward: GEMM epilogue for loss/ent, recompute logits for backward,
    pre-compute d_hidden/d_weight. Backward just scales by dloss.
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, target, bias, chunk_size,
                ce_weight, ent_weight):
        ce_loss, entropy, d_hidden, d_weight, d_bias = _gemm_fused_fwd_bwd(
            hidden_states, weight, target, bias=bias, chunk_size=chunk_size,
            ce_weight=ce_weight, ent_weight=ent_weight,
        )
        loss = (ce_weight * ce_loss.sum() + ent_weight * entropy.sum())

        ctx.save_for_backward(d_hidden, d_weight)
        ctx.d_bias = d_bias
        ctx.has_bias = bias is not None
        ctx.mark_non_differentiable(ce_loss, entropy)
        return loss, ce_loss, entropy

    @staticmethod
    def backward(ctx, dloss, _g_ce, _g_ent):
        d_hidden, d_weight = ctx.saved_tensors
        d_hidden = d_hidden * dloss
        d_weight = d_weight * dloss
        d_bias = None
        if ctx.has_bias:
            d_bias = ctx.d_bias * dloss
        return d_hidden, d_weight, None, d_bias, None, None, None


def gemm_fused_ce_entropy_fast(
    hidden_states: Tensor,
    weight: Tensor,
    target: Tensor,
    bias: Optional[Tensor] = None,
    chunk_size: int = 4096,
    ce_weight: float = 1.0,
    ent_weight: float = -1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Fast Level 5 GEMM with pre-computed gradients.

    Returns (loss, ce_loss, entropy):
      - loss: scalar, requires grad
      - ce_loss: per-element CE, detached
      - entropy: per-element entropy, detached
    """
    return GemmFusedCEEntropyFast.apply(
        hidden_states, weight, target, bias, chunk_size, ce_weight, ent_weight,
    )


# ========================================================================
# Level 5M: Megakernel — GEMM epilogue writes logits + fused CuTe dlogits
# ========================================================================

def _gemm_megakernel_fwd_bwd(
    hidden_states: Tensor,
    weight: Tensor,
    target: Tensor,
    bias: Optional[Tensor] = None,
    chunk_size: int = 4096,
    ce_weight: float = 1.0,
    ent_weight: float = -1.0,
    tile_M: int = 128,
    tile_N: int = 256,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Level 5 megakernel forward + backward in one pass.

    Forward: GEMM epilogue → partials + logits → finalize (with LSE).
    Backward: fused CuTe dlogits kernel → d_hidden, d_weight matmuls.

    Key difference from _gemm_fused_fwd_bwd: no cuBLAS logits recompute,
    no logsumexp recompute, fused dlogits kernel instead of 4-5 PyTorch ops.

    Returns: (ce_loss, entropy, d_hidden, d_weight, d_bias)
    """
    batch_shape = hidden_states.shape[:-1]
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    target_1d = target.reshape(-1)
    B, H = hidden_2d.shape
    V = weight.shape[0]

    ce_loss = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)
    entropy = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)
    d_hidden = torch.empty_like(hidden_2d)
    d_weight = torch.zeros(V, H, device=hidden_2d.device, dtype=torch.float32)
    d_bias = None
    if bias is not None:
        d_bias = torch.zeros(V, device=hidden_2d.device, dtype=torch.float32)

    use_native, mm_dtype, weight_t, bias_for_mm = _mm_setup(hidden_2d, weight, bias)

    actual_chunk = min(chunk_size, B)
    N_tiles = (V + tile_N - 1) // tile_N

    # GEMM epilogue partials buffer (reused per chunk)
    partials_buf = torch.empty(
        N_tiles, actual_chunk, 4,
        device=hidden_2d.device, dtype=torch.float32,
    )
    # Logits buffer from GEMM epilogue (fp32, reused per chunk)
    logits_buf = torch.empty(
        actual_chunk, V, device=hidden_2d.device, dtype=torch.float32,
    )

    weight_3d = weight.unsqueeze(0)

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk_len = end - start
        h_chunk = hidden_2d[start:end]
        t_chunk = target_1d[start:end]

        assert chunk_len % 8 == 0 or chunk_len == B

        # 1. GEMM epilogue forward: partials + logits
        partials_chunk = partials_buf[:, :chunk_len, :]
        logits_chunk = logits_buf[:chunk_len]
        h_3d = h_chunk.unsqueeze(0)
        _gemm_ce_entropy_bwd_kernel(
            h_3d, weight_3d, t_chunk, partials_chunk, logits_chunk, V,
            tile_M=tile_M, tile_N=tile_N,
        )

        # 2. Finalize with LSE output
        loss_chunk, ent_chunk, lse_chunk = finalize_ce_entropy_with_lse(
            partials_chunk, N_tiles,
        )
        ce_loss[start:end] = loss_chunk
        entropy[start:end] = ent_chunk

        # 3. Fused CuTe dlogits kernel (one kernel, replaces 4-5 PyTorch ops)
        g_ce_chunk = torch.full(
            (chunk_len,), ce_weight, device=hidden_2d.device, dtype=torch.float32,
        )
        g_ent_chunk = torch.full(
            (chunk_len,), ent_weight, device=hidden_2d.device, dtype=torch.float32,
        )
        dlogits = compute_dlogits(
            logits_chunk, t_chunk, lse_chunk, ent_chunk,
            g_ce_chunk, g_ent_chunk,
        )

        # 4. d_hidden = dlogits @ weight
        dlogits_mm = dlogits.to(mm_dtype)
        weight_for_mm = weight if use_native else weight.float()
        d_hidden[start:end] = torch.mm(dlogits_mm, weight_for_mm).to(hidden_2d.dtype)

        # 5. d_weight += dlogits.T @ hidden
        d_weight.add_(torch.mm(dlogits_mm.t(), h_chunk.to(mm_dtype)).float())

        # 6. d_bias
        if d_bias is not None:
            d_bias.add_(dlogits.sum(dim=0))

    d_weight_out = d_weight.to(weight.dtype)
    d_bias_out = d_bias.to(bias.dtype) if d_bias is not None else None

    return (
        ce_loss.view(batch_shape),
        entropy.view(batch_shape),
        d_hidden.view_as(hidden_states),
        d_weight_out,
        d_bias_out,
    )


class GemmMegakernel(torch.autograd.Function):
    """Level 5 megakernel: GEMM epilogue writes logits + fused CuTe dlogits.

    Forward: GEMM epilogue for loss/ent/logits, fused dlogits kernel,
    pre-compute d_hidden/d_weight. Backward just scales by dloss.
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, target, bias, chunk_size,
                ce_weight, ent_weight):
        ce_loss, entropy, d_hidden, d_weight, d_bias = _gemm_megakernel_fwd_bwd(
            hidden_states, weight, target, bias=bias, chunk_size=chunk_size,
            ce_weight=ce_weight, ent_weight=ent_weight,
        )
        loss = (ce_weight * ce_loss.sum() + ent_weight * entropy.sum())

        ctx.save_for_backward(d_hidden, d_weight)
        ctx.d_bias = d_bias
        ctx.has_bias = bias is not None
        ctx.mark_non_differentiable(ce_loss, entropy)
        return loss, ce_loss, entropy

    @staticmethod
    def backward(ctx, dloss, _g_ce, _g_ent):
        d_hidden, d_weight = ctx.saved_tensors
        d_hidden = d_hidden * dloss
        d_weight = d_weight * dloss
        d_bias = None
        if ctx.has_bias:
            d_bias = ctx.d_bias * dloss
        return d_hidden, d_weight, None, d_bias, None, None, None


def gemm_megakernel_fast(
    hidden_states: Tensor,
    weight: Tensor,
    target: Tensor,
    bias: Optional[Tensor] = None,
    chunk_size: int = 4096,
    ce_weight: float = 1.0,
    ent_weight: float = -1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Level 5 megakernel with pre-computed gradients.

    Returns (loss, ce_loss, entropy):
      - loss: scalar, requires grad
      - ce_loss: per-element CE, detached
      - entropy: per-element entropy, detached
    """
    return GemmMegakernel.apply(
        hidden_states, weight, target, bias, chunk_size, ce_weight, ent_weight,
    )
