"""
Fused linear + cross-entropy + entropy for Hopper (H100, sm90).

Strategy (following quack's chunked linear CE pattern):
  - Chunk the batch dimension into tiles of `chunk_size`
  - For each chunk:
      1. matmul:  logits_chunk = hidden_chunk @ W.T       (cuBLAS / torch.mm)
      2. CE+ent:  (loss, entropy) from logits_chunk        (CuTe DSL kernel)
  - Logits tensor is [chunk_size, V] and reused each iteration,
    so peak memory is O(chunk_size * V) instead of O(B * V).

For the backward pass (future work):
  - Same chunking, but also compute dlogits (softmax - one_hot) in the CE kernel,
    then dx_chunk = dlogits @ W  and accumulate dW += dlogits.T @ hidden_chunk.

Target throughput: >= 90 % of H100 HBM3 peak (3.35 TB/s model bandwidth).
"""

from typing import Optional, Tuple

import torch

from fast_reduction.cute_cross_entropy import ce_fwd, entropy_fwd, ce_entropy_fwd


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

    # Apples-to-apples matmul path: keep GEMM in input dtype (e.g., bf16)
    # when hidden/weight dtypes match and are GEMM-friendly. Fallback to fp32
    # only for mixed/unsupported dtype combinations.
    use_native_mm_dtype = (
        hidden_2d.dtype == weight.dtype
        and hidden_2d.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    mm_dtype = hidden_2d.dtype if use_native_mm_dtype else torch.float32
    weight_t = (
        weight.t().contiguous()
        if use_native_mm_dtype
        else weight.float().t().contiguous()
    )
    bias_for_mm = None if bias is None else bias.to(dtype=mm_dtype)

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
        h_chunk_mm = h_chunk if use_native_mm_dtype else h_chunk.float()
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

    use_native_mm_dtype = (
        hidden_2d.dtype == weight.dtype
        and hidden_2d.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    mm_dtype = hidden_2d.dtype if use_native_mm_dtype else torch.float32
    weight_t = (
        weight.t().contiguous()
        if use_native_mm_dtype
        else weight.float().t().contiguous()
    )
    bias_for_mm = None if bias is None else bias.to(dtype=mm_dtype)

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
        h_chunk_mm = h_chunk if use_native_mm_dtype else h_chunk.float()
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
