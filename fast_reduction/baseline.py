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

    logits = F.linear(hidden_2d.float(), weight.float(), None if bias is None else bias.float())
    log_softmax = F.log_softmax(logits, dim=-1)
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

    logits = F.linear(hidden_2d.float(), weight.float(), None if bias is None else bias.float())
    log_softmax = F.log_softmax(logits, dim=-1)
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

    This is the numerical ground truth that all kernel implementations are
    tested against.
    """
    batch_shape = hidden_states.shape[:-1]
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    target_1d = target.reshape(-1)

    logits = F.linear(hidden_2d.float(), weight.float(), None if bias is None else bias.float())
    log_softmax = F.log_softmax(logits, dim=-1)
    softmax = log_softmax.exp()

    ce_loss = F.nll_loss(log_softmax, target_1d, reduction="none")
    entropy = -(softmax * log_softmax).sum(dim=-1)
    log_probs = log_softmax[torch.arange(len(target_1d), device=target_1d.device), target_1d]

    return (
        ce_loss.view(batch_shape),
        entropy.view(batch_shape),
        log_probs.view(batch_shape),
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
        logits = F.linear(
            hidden_2d[start:end].float(), weight.float(),
            None if bias is None else bias.float(),
        )
        log_sm = F.log_softmax(logits, dim=-1)
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
        logits = F.linear(
            hidden_2d[start:end].float(), weight.float(),
            None if bias is None else bias.float(),
        )
        log_sm = F.log_softmax(logits, dim=-1)
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
        logits = F.linear(
            hidden_2d[start:end].float(), weight.float(),
            None if bias is None else bias.float(),
        )
        log_sm = F.log_softmax(logits, dim=-1)
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
