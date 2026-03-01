"""
Correctness tests for baseline and fused linear + xent + entropy.

Baseline tests run on CPU (no GPU required).
Fused kernel tests require CUDA (CuTe DSL).
"""

import pytest
import torch
import torch.nn.functional as F

from fast_reduction.baseline import baseline_linear_xent_entropy


def _reference_ce_entropy(logits, target):
    """Torch-native ground truth: returns (ce_loss, entropy, log_probs)."""
    log_sm = F.log_softmax(logits.float(), dim=-1)
    sm = log_sm.exp()
    ce = F.nll_loss(log_sm, target, reduction="none")
    ent = -(sm * log_sm).sum(dim=-1)
    lp = log_sm[torch.arange(len(target)), target]
    return ce, ent, lp


def test_baseline_matches_torch():
    torch.manual_seed(42)
    B, H, V = 32, 64, 128
    hidden = torch.randn(B, H)
    weight = torch.randn(V, H)
    target = torch.randint(0, V, (B,))

    ce, ent, lp = baseline_linear_xent_entropy(hidden, weight, target)

    logits = F.linear(hidden.float(), weight.float())
    ce_ref, ent_ref, lp_ref = _reference_ce_entropy(logits, target)

    torch.testing.assert_close(ce, ce_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ent, ent_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(lp, lp_ref, atol=1e-5, rtol=1e-5)


def test_baseline_log_probs_equals_neg_ce():
    torch.manual_seed(7)
    B, H, V = 16, 32, 64
    hidden = torch.randn(B, H)
    weight = torch.randn(V, H)
    target = torch.randint(0, V, (B,))

    ce, _, lp = baseline_linear_xent_entropy(hidden, weight, target)
    torch.testing.assert_close(lp, -ce, atol=1e-6, rtol=1e-6)


def test_baseline_with_bias():
    torch.manual_seed(11)
    B, H, V = 24, 48, 97
    hidden = torch.randn(B, H)
    weight = torch.randn(V, H)
    bias = torch.randn(V)
    target = torch.randint(0, V, (B,))

    ce, ent, lp = baseline_linear_xent_entropy(hidden, weight, target, bias=bias)

    logits = F.linear(hidden.float(), weight.float(), bias.float())
    ce_ref, ent_ref, lp_ref = _reference_ce_entropy(logits, target)

    torch.testing.assert_close(ce, ce_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ent, ent_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_matches_baseline_cuda():
    """CuTe DSL kernel matches PyTorch baseline on GPU."""
    from fast_reduction.kernel import fused_linear_xent_entropy

    torch.manual_seed(99)
    B, H, V = 256, 128, 1024
    hidden = torch.randn(B, H, device="cuda")
    weight = torch.randn(V, H, device="cuda")
    target = torch.randint(0, V, (B,), device="cuda")

    ce_fused, ent_fused, lp_fused = fused_linear_xent_entropy(hidden, weight, target)
    ce_base, ent_base, lp_base = baseline_linear_xent_entropy(
        hidden.cpu(), weight.cpu(), target.cpu(),
    )

    torch.testing.assert_close(ce_fused.cpu(), ce_base, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(ent_fused.cpu(), ent_base, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lp_fused.cpu(), lp_base, atol=1e-3, rtol=1e-3)


def test_entropy_nonnegative():
    """Entropy of any distribution is >= 0."""
    torch.manual_seed(13)
    B, H, V = 64, 32, 50
    hidden = torch.randn(B, H)
    weight = torch.randn(V, H)
    target = torch.randint(0, V, (B,))

    _, ent, _ = baseline_linear_xent_entropy(hidden, weight, target)
    assert (ent >= -1e-6).all(), f"entropy has negative values: {ent.min()}"


def test_batch_shapes():
    """Supports 3D hidden_states [batch, seq, hidden]."""
    torch.manual_seed(17)
    batch, seq, H, V = 4, 8, 32, 64
    hidden = torch.randn(batch, seq, H)
    weight = torch.randn(V, H)
    target = torch.randint(0, V, (batch, seq))

    ce, ent, lp = baseline_linear_xent_entropy(hidden, weight, target)
    assert ce.shape == (batch, seq)
    assert ent.shape == (batch, seq)
    assert lp.shape == (batch, seq)
