"""
Fused linear + cross-entropy + entropy for Hopper (H100, sm90).

Strategy (following quack's chunked linear CE pattern):
  - Chunk the batch dimension into tiles of `chunk_size`
  - For each chunk:
      1. matmul:  logits_chunk = hidden_chunk @ W.T       (cuBLAS / torch.mm)
      2. CE+ent:  (loss, entropy) from logits_chunk        (CuTe DSL kernel)
  - Logits tensor is [chunk_size, V] and reused each iteration,
    so peak memory is O(chunk_size * V) instead of O(B * V).

Backward pass:
  - Same chunking: recompute logits via matmul, compute dlogits element-wise,
    then d_hidden = dlogits @ W and accumulate d_weight += dlogits.T @ hidden_chunk.

Target throughput: >= 90 % of H100 HBM3 peak (3.35 TB/s model bandwidth).
"""

from typing import Optional, Tuple

import torch

from fast_reduction.cute_cross_entropy import ce_fwd, entropy_fwd, ce_entropy_fwd


# ===========================================================================
#  Matmul dtype helpers (shared by forward and backward)
# ===========================================================================

def _mm_setup(hidden_2d, weight, bias):
    """Prepare matmul dtype and transposed weight for chunked forward/backward."""
    use_native = (
        hidden_2d.dtype == weight.dtype
        and hidden_2d.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    mm_dtype = hidden_2d.dtype if use_native else torch.float32
    weight_t = (
        weight.t().contiguous()
        if use_native
        else weight.float().t().contiguous()
    )
    bias_for_mm = None if bias is None else bias.to(dtype=mm_dtype)
    return use_native, mm_dtype, weight_t, bias_for_mm


# ===========================================================================
#  Forward (levels 3 & 4)
# ===========================================================================

def fused_linear_xent_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused linear + cross-entropy + entropy.

    Returns (ce_loss, entropy, log_probs), each shaped like `target`.

    Chunks the batch dimension so only [chunk_size, V] logits are
    materialized at a time, then runs the CuTe DSL CE+entropy kernel
    on each chunk.
    """
    batch_shape = hidden_states.shape[:-1]
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    target_1d = target.reshape(-1)
    B = hidden_2d.shape[0]
    V = weight.shape[0]

    ce_loss = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)
    entropy = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)

    use_native, mm_dtype, weight_t, bias_for_mm = _mm_setup(hidden_2d, weight, bias)

    # Pre-allocate logits buffer (reused every chunk)
    actual_chunk = min(chunk_size, B)
    logits_mm_buf = torch.empty(
        actual_chunk, V, device=hidden_2d.device, dtype=mm_dtype
    )
    # CuTe reduction needs fp32 logits. If matmul is bf16, upcast once.
    logits_reduce_buf = (
        logits_mm_buf
        if mm_dtype == torch.float32
        else torch.empty(actual_chunk, V, device=hidden_2d.device, dtype=torch.float32)
    )

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        h_chunk = hidden_2d[start:end]
        t_chunk = target_1d[start:end]
        chunk_len = end - start

        logits_chunk_mm = logits_mm_buf[:chunk_len]
        h_chunk_mm = h_chunk if use_native else h_chunk.float()
        torch.mm(
            h_chunk_mm,
            weight_t,
            out=logits_chunk_mm,
        )

        if bias_for_mm is not None:
            logits_chunk_mm.add_(bias_for_mm)

        logits_chunk = logits_chunk_mm
        if mm_dtype != torch.float32:
            logits_chunk = logits_reduce_buf[:chunk_len]
            logits_chunk.copy_(logits_chunk_mm)

        # CuTe DSL kernel: one pass over [chunk_len, V] -> (loss, entropy)
        loss_chunk, ent_chunk = ce_entropy_fwd(logits_chunk, t_chunk)
        ce_loss[start:end] = loss_chunk
        entropy[start:end] = ent_chunk

    log_probs = -ce_loss

    return (
        ce_loss.view(batch_shape),
        entropy.view(batch_shape),
        log_probs.view(batch_shape),
    )


def separate_linear_xent_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked linear + separate CuTe CE kernel + CuTe entropy kernel.

    Same chunked matmul, but calls two separate kernels per chunk
    (each reads the logits independently). This is level 3 — shows
    the cost of reading logits twice vs the joint kernel (level 4).
    """
    batch_shape = hidden_states.shape[:-1]
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    target_1d = target.reshape(-1)
    B = hidden_2d.shape[0]
    V = weight.shape[0]

    ce_loss = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)
    entropy = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)

    use_native, mm_dtype, weight_t, bias_for_mm = _mm_setup(hidden_2d, weight, bias)

    actual_chunk = min(chunk_size, B)
    logits_mm_buf = torch.empty(
        actual_chunk, V, device=hidden_2d.device, dtype=mm_dtype
    )
    logits_reduce_buf = (
        logits_mm_buf
        if mm_dtype == torch.float32
        else torch.empty(actual_chunk, V, device=hidden_2d.device, dtype=torch.float32)
    )

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        h_chunk = hidden_2d[start:end]
        t_chunk = target_1d[start:end]
        chunk_len = end - start

        logits_chunk_mm = logits_mm_buf[:chunk_len]
        h_chunk_mm = h_chunk if use_native else h_chunk.float()
        torch.mm(
            h_chunk_mm,
            weight_t,
            out=logits_chunk_mm,
        )

        if bias_for_mm is not None:
            logits_chunk_mm.add_(bias_for_mm)

        logits_chunk = logits_chunk_mm
        if mm_dtype != torch.float32:
            logits_chunk = logits_reduce_buf[:chunk_len]
            logits_chunk.copy_(logits_chunk_mm)

        # Two separate kernel launches — each reads logits independently
        ce_loss[start:end] = ce_fwd(logits_chunk, t_chunk)
        entropy[start:end] = entropy_fwd(logits_chunk)

    log_probs = -ce_loss

    return (
        ce_loss.view(batch_shape),
        entropy.view(batch_shape),
        log_probs.view(batch_shape),
    )


# ===========================================================================
#  Backward (levels 3 & 4)
# ===========================================================================

def _compute_dlogits_chunk(logits_fp32, target, g_ce, g_ent, entropy):
    """Compute dlogits for one chunk, all in fp32.  Modifies logits_fp32 in-place.

    dz_j = p_j * (g_ce - g_ent * (log_p_j + ent)) - g_ce * 1_{j=target}

    where p_j = exp(z_j - lse), log_p_j = z_j - lse.

    Factored as:  dz_j = p_j * (A + B * log_p_j) - g_ce * 1_{j=target}
    where A = g_ce - g_ent * ent (per row), B = -g_ent (per row).

    Memory: uses 2 [M,V] fp32 buffers (logits_fp32 reused in-place + dlogits).
    """
    M = logits_fp32.shape[0]
    lse = torch.logsumexp(logits_fp32, dim=-1, keepdim=True)  # [M, 1]
    logits_fp32.sub_(lse)  # log_p in-place over logits_fp32

    # Per-row coefficients
    row_a = (g_ce - g_ent * entropy).unsqueeze(1)  # [M, 1]
    row_b = (-g_ent).unsqueeze(1)  # [M, 1]

    # dlogits = row_b * log_p + row_a  (new [M,V] alloc via mul, in-place add)
    dlogits = logits_fp32.mul(row_b).add_(row_a)

    # p = exp(log_p), in-place over logits_fp32
    logits_fp32.exp_()

    # dlogits = p * factor, in-place
    dlogits.mul_(logits_fp32)

    # Subtract g_ce at target positions
    dlogits[torch.arange(M, device=target.device), target] -= g_ce

    return dlogits


def fused_linear_xent_entropy_backward(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    g_ce: torch.Tensor,
    g_ent: torch.Tensor,
    entropy: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Chunked backward for fused linear + CE + entropy.

    Recomputes logits per chunk, computes dlogits element-wise,
    then accumulates d_hidden, d_weight, d_bias via matmul.

    Args:
        hidden_states: [B, H] or [*, H]
        weight: [V, H]
        target: [B] or [*]
        g_ce: [B] or [*] — upstream gradient for CE loss
        g_ent: [B] or [*] — upstream gradient for entropy
        entropy: [B] or [*] — entropy values from forward pass
        bias: [V] optional
        chunk_size: chunk size for matmul tiling

    Returns: (d_hidden, d_weight, d_bias)
    """
    batch_shape = hidden_states.shape[:-1]
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    target_1d = target.reshape(-1)
    g_ce_1d = g_ce.reshape(-1).float()
    g_ent_1d = g_ent.reshape(-1).float()
    ent_1d = entropy.reshape(-1).float()
    B, H = hidden_2d.shape
    V = weight.shape[0]

    d_hidden = torch.empty_like(hidden_2d)
    d_weight = torch.zeros(V, H, device=hidden_2d.device, dtype=torch.float32)
    d_bias = None
    if bias is not None:
        d_bias = torch.zeros(V, device=hidden_2d.device, dtype=torch.float32)

    use_native, mm_dtype, weight_t, bias_for_mm = _mm_setup(hidden_2d, weight, bias)

    actual_chunk = min(chunk_size, B)
    logits_buf = torch.empty(
        actual_chunk, V, device=hidden_2d.device, dtype=mm_dtype
    )

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk_len = end - start
        h_chunk = hidden_2d[start:end]

        # 1. Recompute logits
        logits_mm = logits_buf[:chunk_len]
        h_mm = h_chunk if use_native else h_chunk.float()
        torch.mm(h_mm, weight_t, out=logits_mm)
        if bias_for_mm is not None:
            logits_mm.add_(bias_for_mm)

        logits_fp32 = logits_mm.float()

        # 2. Compute dlogits element-wise in fp32
        dlogits = _compute_dlogits_chunk(
            logits_fp32,
            target_1d[start:end],
            g_ce_1d[start:end],
            g_ent_1d[start:end],
            ent_1d[start:end],
        )

        # 3. d_hidden = dlogits @ weight  [chunk, V] @ [V, H] -> [chunk, H]
        dlogits_mm = dlogits.to(mm_dtype)
        weight_for_mm = weight if use_native else weight.float()
        d_hidden[start:end] = torch.mm(dlogits_mm, weight_for_mm).to(hidden_2d.dtype)

        # 4. d_weight += dlogits.T @ hidden  [V, chunk] @ [chunk, H] -> [V, H]
        #    Use bf16 matmul + fp32 accumulation (standard mixed-precision pattern)
        d_weight.add_(torch.mm(dlogits_mm.t(), h_chunk.to(mm_dtype)).float())

        # 5. d_bias += dlogits.sum(dim=0)
        if d_bias is not None:
            d_bias.add_(dlogits.sum(dim=0))

    d_weight_out = d_weight.to(weight.dtype)
    d_bias_out = d_bias.to(bias.dtype) if d_bias is not None else None

    return d_hidden.view_as(hidden_states), d_weight_out, d_bias_out


# ===========================================================================
#  autograd.Function wrapper (levels 3 & 4)
# ===========================================================================

class FusedLinearXentEntropy(torch.autograd.Function):
    """Differentiable fused linear + cross-entropy + entropy.

    Forward uses CuTe DSL kernels (levels 3/4).
    Backward recomputes logits per chunk and uses PyTorch ops for dlogits.
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, target, bias, chunk_size):
        ce_loss, entropy, log_probs = fused_linear_xent_entropy(
            hidden_states, weight, target, bias=bias, chunk_size=chunk_size,
        )
        ctx.save_for_backward(hidden_states, weight, target, entropy)
        ctx.bias = bias
        ctx.chunk_size = chunk_size
        return ce_loss, entropy, log_probs

    @staticmethod
    def backward(ctx, g_ce, g_ent, g_lp):
        hidden_states, weight, target, entropy = ctx.saved_tensors
        # log_probs = -ce_loss, so g_lp contributes -g_lp to the ce gradient
        if g_lp is not None:
            g_ce = g_ce - g_lp
        d_hidden, d_weight, d_bias = fused_linear_xent_entropy_backward(
            hidden_states, weight, target, g_ce, g_ent, entropy,
            bias=ctx.bias, chunk_size=ctx.chunk_size,
        )
        return d_hidden, d_weight, None, d_bias, None


def fused_linear_xent_entropy_differentiable(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable version of fused_linear_xent_entropy.

    Same interface, but supports .backward() on the returned tensors.
    """
    return FusedLinearXentEntropy.apply(hidden_states, weight, target, bias, chunk_size)
