"""
Pure-PyTorch references for benchmarking.

Three variants with increasing fusion:
  1. linear_xent        — linear + cross-entropy only (quack already has this)
  2. linear_entropy     — linear + entropy only
  3. linear_xent_entropy — linear + cross-entropy + entropy (the target)

Each returns different subsets of (ce_loss, entropy, log_probs).
All intentionally simple and correct, not fast.

Memory-efficient variants (chunked) are also provided — these are the
real baselines since they don't materialise the full [B, V] logits.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ===========================================================================
#  Unfused baselines (full logits materialisation)
# ===========================================================================

def baseline_linear_xent(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear + cross-entropy.  Returns (ce_loss, log_probs)."""
    batch_shape = hidden_states.shape[:-1]
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    target_1d = target.reshape(-1)

    logits = F.linear(hidden_2d, weight, bias)
    log_softmax = F.log_softmax(logits.float(), dim=-1)
    ce_loss = F.nll_loss(log_softmax, target_1d, reduction="none")
    log_probs = log_softmax[torch.arange(len(target_1d), device=target_1d.device), target_1d]

    return ce_loss.view(batch_shape), log_probs.view(batch_shape)


def baseline_linear_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Linear + entropy.  Returns entropy."""
    batch_shape = hidden_states.shape[:-1]
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])

    logits = F.linear(hidden_2d, weight, bias)
    log_softmax = F.log_softmax(logits.float(), dim=-1)
    softmax = log_softmax.exp()
    entropy = -(softmax * log_softmax).sum(dim=-1)

    return entropy.view(batch_shape)


def baseline_linear_xent_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear + cross-entropy + entropy.  Returns (ce_loss, entropy, log_probs).

    Matmul in input dtype (bf16 or fp32), reduction in fp32.
    """
    batch_shape = hidden_states.shape[:-1]
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    target_1d = target.reshape(-1)

    logits = F.linear(hidden_2d, weight, bias)
    log_softmax = F.log_softmax(logits.float(), dim=-1)
    softmax = log_softmax.exp()

    ce_loss = F.nll_loss(log_softmax, target_1d, reduction="none")
    entropy = -(softmax * log_softmax).sum(dim=-1)
    log_probs = log_softmax[torch.arange(len(target_1d), device=target_1d.device), target_1d]

    return (
        ce_loss.view(batch_shape),
        entropy.view(batch_shape),
        log_probs.view(batch_shape),
    )


def baseline_linear_xent_entropy_backward(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    g_ce: Optional[torch.Tensor] = None,
    g_ent: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Reference backward for linear + CE + entropy using PyTorch autograd.

    Computes gradients for loss = sum(g_ce * ce) + sum(g_ent * entropy).
    Default g_ce=1, g_ent=-1 matches the benchmark loss ce.sum() - ent.sum().

    Returns (ce_loss, entropy, d_hidden, d_weight, d_bias).
    All computation in fp32 for ground truth.
    """
    hidden_2d = hidden_states.detach().float().reshape(-1, hidden_states.shape[-1])
    weight_fp32 = weight.detach().float()
    target_1d = target.reshape(-1)
    B = hidden_2d.shape[0]

    hidden_2d.requires_grad_(True)
    weight_fp32.requires_grad_(True)

    bias_fp32 = None
    if bias is not None:
        bias_fp32 = bias.detach().float()
        bias_fp32.requires_grad_(True)

    logits = F.linear(hidden_2d, weight_fp32, bias_fp32)
    log_softmax = F.log_softmax(logits, dim=-1)
    softmax = log_softmax.exp()

    ce_loss = F.nll_loss(log_softmax, target_1d, reduction="none")
    entropy = -(softmax * log_softmax).sum(dim=-1)

    if g_ce is None:
        g_ce = torch.ones_like(ce_loss)
    else:
        g_ce = g_ce.reshape(-1).float()
    if g_ent is None:
        g_ent = -torch.ones_like(entropy)
    else:
        g_ent = g_ent.reshape(-1).float()

    loss = (g_ce * ce_loss + g_ent * entropy).sum()
    loss.backward()

    batch_shape = hidden_states.shape[:-1]
    d_hidden = hidden_2d.grad.to(hidden_states.dtype).view_as(hidden_states)
    d_weight = weight_fp32.grad.to(weight.dtype)
    d_bias = bias_fp32.grad.to(bias.dtype) if bias_fp32 is not None else None

    return (
        ce_loss.detach().view(batch_shape),
        entropy.detach().view(batch_shape),
        d_hidden,
        d_weight,
        d_bias,
    )


# ===========================================================================
#  Chunked baselines (memory-efficient — no full [B, V] materialisation)
# ===========================================================================

def chunked_linear_xent(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked linear + cross-entropy.  Returns (ce_loss, log_probs)."""
    batch_shape = hidden_states.shape[:-1]
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    target_1d = target.reshape(-1)
    B, V = hidden_2d.shape[0], weight.shape[0]

    ce_loss = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)
    log_probs = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        logits = F.linear(hidden_2d[start:end], weight, bias)
        log_sm = F.log_softmax(logits.float(), dim=-1)
        t = target_1d[start:end]
        ce_loss[start:end] = F.nll_loss(log_sm, t, reduction="none")
        log_probs[start:end] = log_sm[torch.arange(end - start, device=t.device), t]

    return ce_loss.view(batch_shape), log_probs.view(batch_shape)


def chunked_linear_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Chunked linear + entropy.  Returns entropy."""
    batch_shape = hidden_states.shape[:-1]
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    B = hidden_2d.shape[0]

    entropy = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        logits = F.linear(hidden_2d[start:end], weight, bias)
        log_sm = F.log_softmax(logits.float(), dim=-1)
        sm = log_sm.exp()
        entropy[start:end] = -(sm * log_sm).sum(dim=-1)

    return entropy.view(batch_shape)


def chunked_linear_xent_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked linear + cross-entropy + entropy.  Returns (ce_loss, entropy, log_probs).

    Memory: O(chunk_size * V) instead of O(B * V).
    """
    batch_shape = hidden_states.shape[:-1]
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    target_1d = target.reshape(-1)
    B = hidden_2d.shape[0]

    ce_loss = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)
    entropy = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)
    log_probs = torch.empty(B, device=hidden_2d.device, dtype=torch.float32)

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        logits = F.linear(hidden_2d[start:end], weight, bias)
        log_sm = F.log_softmax(logits.float(), dim=-1)
        sm = log_sm.exp()
        t = target_1d[start:end]
        ce_loss[start:end] = F.nll_loss(log_sm, t, reduction="none")
        entropy[start:end] = -(sm * log_sm).sum(dim=-1)
        log_probs[start:end] = log_sm[torch.arange(end - start, device=t.device), t]

    return (
        ce_loss.view(batch_shape),
        entropy.view(batch_shape),
        log_probs.view(batch_shape),
    )
