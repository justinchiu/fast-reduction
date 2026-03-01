"""
Test entropy accuracy at large V where cluster reduction (cluster_n > 1) is active.

This exercises the mbarrier phase reuse bug that caused MAE 2690 (EntropyOnly)
and MAE 3.6 (CrossEntropyEntropy) at V=128256.
"""

import pytest
import torch
import torch.nn.functional as F


def _entropy_ref(logits_fp32):
    """fp32 reference entropy: -sum(softmax * log_softmax)."""
    log_sm = F.log_softmax(logits_fp32, dim=-1)
    sm = log_sm.exp()
    return -(sm * log_sm).sum(dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("V", [32000, 65536, 128256])
def test_entropy_only_large_v(V):
    """EntropyOnly kernel matches fp32 reference at large V."""
    from fast_reduction.cute_cross_entropy import entropy_fwd

    torch.manual_seed(42)
    B = 512
    logits = torch.randn(B, V, device="cuda", dtype=torch.float32)

    ent = entropy_fwd(logits)
    ent_ref = _entropy_ref(logits)

    mae = (ent - ent_ref).abs().mean().item()
    print(f"EntropyOnly V={V}: MAE={mae:.6f}")
    # Should be < 0.01 (was 2690 before fix)
    torch.testing.assert_close(ent, ent_ref, atol=0.01, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("V", [32000, 65536, 128256])
def test_ce_entropy_joint_large_v(V):
    """CrossEntropyEntropy kernel matches fp32 reference at large V."""
    from fast_reduction.cute_cross_entropy import ce_entropy_fwd

    torch.manual_seed(42)
    B = 512
    logits = torch.randn(B, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (B,), device="cuda")

    loss, ent = ce_entropy_fwd(logits, target)

    log_sm = F.log_softmax(logits, dim=-1)
    ce_ref = F.nll_loss(log_sm, target, reduction="none")
    ent_ref = _entropy_ref(logits)

    ce_mae = (loss - ce_ref).abs().mean().item()
    ent_mae = (ent - ent_ref).abs().mean().item()
    print(f"CrossEntropyEntropy V={V}: CE MAE={ce_mae:.6f}, Ent MAE={ent_mae:.6f}")
    torch.testing.assert_close(loss, ce_ref, atol=0.01, rtol=1e-3)
    # Should be < 0.01 (was 3.6 before fix)
    torch.testing.assert_close(ent, ent_ref, atol=0.01, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ce_only_large_v_unaffected():
    """CrossEntropyOnly (2 passes) should be unaffected — no mbarrier reuse."""
    from fast_reduction.cute_cross_entropy import ce_fwd

    torch.manual_seed(42)
    B, V = 512, 128256
    logits = torch.randn(B, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (B,), device="cuda")

    loss = ce_fwd(logits, target)
    log_sm = F.log_softmax(logits, dim=-1)
    ce_ref = F.nll_loss(log_sm, target, reduction="none")

    mae = (loss - ce_ref).abs().mean().item()
    print(f"CrossEntropyOnly V={V}: CE MAE={mae:.6f}")
    torch.testing.assert_close(loss, ce_ref, atol=0.01, rtol=1e-3)
