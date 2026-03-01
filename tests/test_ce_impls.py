"""
Correctness tests for baseline and fused linear + xent + entropy.

Baseline tests run on CPU (no GPU required).
Fused kernel tests require CUDA (CuTe DSL).

Correctness tests compare against fp32 ground truth (fp32 matmul + fp32 reduction).
"""

import pytest
import torch
import torch.nn.functional as F

from fast_reduction.baseline import baseline_linear_xent_entropy


def _fp32_reference(hidden, weight, target):
    """fp32 matmul + fp32 reduction ground truth."""
    logits = F.linear(hidden.float(), weight.float())
    log_sm = F.log_softmax(logits, dim=-1)
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
    ce_ref, ent_ref, lp_ref = _fp32_reference(hidden, weight, target)

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_matches_baseline_cuda():
    """CuTe DSL kernel matches fp32 ground truth."""
    from fast_reduction.kernel import fused_linear_xent_entropy

    torch.manual_seed(99)
    B, H, V = 256, 128, 1024
    hidden = torch.randn(B, H, device="cuda")
    weight = torch.randn(V, H, device="cuda")
    target = torch.randint(0, V, (B,), device="cuda")

    ce_fused, ent_fused, lp_fused = fused_linear_xent_entropy(hidden, weight, target)
    ce_ref, ent_ref, lp_ref = _fp32_reference(
        hidden.cpu(), weight.cpu(), target.cpu(),
    )

    torch.testing.assert_close(ce_fused.cpu(), ce_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(ent_fused.cpu(), ent_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lp_fused.cpu(), lp_ref, atol=1e-3, rtol=1e-3)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_finalize_kernel():
    """Standalone finalization kernel: merge hand-computed partials."""
    from fast_reduction.gemm_ce_entropy_finalize import finalize_ce_entropy

    torch.manual_seed(42)
    M, V = 128, 1024
    N_tiles = 4

    logits = torch.randn(M, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (M,), device="cuda")

    log_sm = F.log_softmax(logits, dim=-1)
    sm = log_sm.exp()
    ce_ref = F.nll_loss(log_sm, target, reduction="none")
    ent_ref = -(sm * log_sm).sum(dim=-1)

    chunk_n = (V + N_tiles - 1) // N_tiles
    partials = torch.zeros(N_tiles, M, 4, device="cuda", dtype=torch.float32)

    for t in range(N_tiles):
        n_start = t * chunk_n
        n_end = min(n_start + chunk_n, V)
        chunk = logits[:, n_start:n_end]

        partials[t, :, 0] = chunk.max(dim=-1).values
        partials[t, :, 1] = torch.exp(chunk - partials[t, :, 0:1]).sum(dim=-1)
        partials[t, :, 2] = (chunk * torch.exp(chunk - partials[t, :, 0:1])).sum(dim=-1)

        mask = (target >= n_start) & (target < n_end)
        local_idx = (target - n_start).clamp(0, n_end - n_start - 1)
        target_vals = chunk[torch.arange(M, device="cuda"), local_idx]
        partials[t, :, 3] = torch.where(mask, target_vals, torch.zeros_like(target_vals))

    loss, entropy = finalize_ce_entropy(partials, N_tiles)

    torch.testing.assert_close(loss, ce_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(entropy, ent_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gemm_fused_matches_fp32():
    """Level 5 GEMM epilogue vs fp32 ground truth.

    Error is dominated by WGMMA bf16 matmul precision (fp32 acc).
    Epilogue+finalization contribute ~0.000003 CE MAE.
    """
    from fast_reduction.gemm_kernel import gemm_fused_ce_entropy

    torch.manual_seed(123)
    B, H, V = 256, 128, 1024
    hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, V, (B,), device="cuda")

    ce_fused, ent_fused, lp_fused = gemm_fused_ce_entropy(
        hidden, weight, target, chunk_size=B,
    )
    ce_ref, ent_ref, lp_ref = _fp32_reference(
        hidden.cpu(), weight.cpu(), target.cpu(),
    )

    torch.testing.assert_close(ce_fused.cpu(), ce_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ent_fused.cpu(), ent_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lp_fused.cpu(), lp_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gemm_fused_v2_matches_fp32():
    """Level 5.2 two-pass epilogue vs fp32 ground truth."""
    from fast_reduction.gemm_kernel import gemm_fused_ce_entropy_v2

    torch.manual_seed(123)
    B, H, V = 256, 128, 1024
    hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, V, (B,), device="cuda")

    ce_v2, ent_v2, lp_v2 = gemm_fused_ce_entropy_v2(
        hidden, weight, target, chunk_size=B,
    )
    ce_ref, ent_ref, lp_ref = _fp32_reference(
        hidden.cpu(), weight.cpu(), target.cpu(),
    )

    torch.testing.assert_close(ce_v2.cpu(), ce_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ent_v2.cpu(), ent_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lp_v2.cpu(), lp_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gemm_v1_v2_identical():
    """v5.1 and v5.2 produce near-identical results.

    The two-pass epilogue (v5.2) eliminates online merge and fastmath,
    but the single-pass merge in v5.1 contributes negligible error
    (CE MAE ~0.000003), so CE is bitwise identical.
    Entropy may differ by ~1e-5 due to accumulation order.
    """
    from fast_reduction.gemm_kernel import gemm_fused_ce_entropy, gemm_fused_ce_entropy_v2

    torch.manual_seed(42)
    B, H, V = 128, 256, 4096
    hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, V, (B,), device="cuda")

    ce_v1, ent_v1, _ = gemm_fused_ce_entropy(hidden, weight, target, chunk_size=B)
    ce_v2, ent_v2, _ = gemm_fused_ce_entropy_v2(hidden, weight, target, chunk_size=B)

    torch.testing.assert_close(ce_v1, ce_v2, atol=0, rtol=0)
    torch.testing.assert_close(ent_v1, ent_v2, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_levels3_4_entropy_no_nan_non_aligned_vocab():
    """Level 3/4 entropy remains finite for non-aligned vocab sizes."""
    from fast_reduction.kernel import fused_linear_xent_entropy, separate_linear_xent_entropy

    torch.manual_seed(777)
    B, H, V = 512, 128, 1000  # non-aligned V exercises OOB tail path
    hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, V, (B,), device="cuda")

    ce4, ent4, lp4 = fused_linear_xent_entropy(hidden, weight, target, chunk_size=B)
    ce3, ent3, lp3 = separate_linear_xent_entropy(hidden, weight, target, chunk_size=B)

    assert not torch.isnan(ent4).any(), "Level 4 entropy contains NaN for non-aligned V."
    assert not torch.isnan(ent3).any(), "Level 3 entropy contains NaN for non-aligned V."

    # Reference with bf16 matmul (matches level 3/4 GEMM dtype path).
    logits = F.linear(hidden, weight)
    log_sm = F.log_softmax(logits.float(), dim=-1)
    sm = log_sm.exp()
    ce_ref = F.nll_loss(log_sm, target, reduction="none")
    ent_ref = -(sm * log_sm).sum(dim=-1)
    lp_ref = log_sm[torch.arange(B, device="cuda"), target]

    torch.testing.assert_close(ce4, ce_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(ent4, ent_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lp4, lp_ref, atol=1e-3, rtol=1e-3)

    torch.testing.assert_close(ce3, ce_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(ent3, ent_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lp3, lp_ref, atol=1e-3, rtol=1e-3)
