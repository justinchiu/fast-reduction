"""
Backward pass correctness tests.

Tests gradient correctness of the fused linear + CE + entropy backward
against the fp32 reference baseline (PyTorch autograd).

Loss function: L = sum(g_ce * ce_loss) + sum(g_ent * entropy)
Default: g_ce=1, g_ent=-1 → L = ce_loss.sum() - entropy.sum()
"""

import pytest
import torch
import torch.nn.functional as F

from fast_reduction.baseline import baseline_linear_xent_entropy_backward


def _fp32_reference_grads(hidden, weight, target, bias=None, g_ce=None, g_ent=None):
    """Compute fp32 ground truth gradients via PyTorch autograd."""
    return baseline_linear_xent_entropy_backward(
        hidden, weight, target, bias=bias, g_ce=g_ce, g_ent=g_ent,
    )


# ===========================================================================
#  Reference backward tests (CPU, no CUDA required)
# ===========================================================================

class TestReferenceBackward:
    def test_basic_gradients(self):
        """Reference backward produces non-zero gradients with correct shapes."""
        torch.manual_seed(42)
        B, H, V = 32, 64, 128
        hidden = torch.randn(B, H)
        weight = torch.randn(V, H)
        target = torch.randint(0, V, (B,))

        ce, ent, d_hidden, d_weight, d_bias = _fp32_reference_grads(
            hidden, weight, target,
        )

        assert d_hidden.shape == hidden.shape
        assert d_weight.shape == weight.shape
        assert d_bias is None
        assert d_hidden.abs().sum() > 0
        assert d_weight.abs().sum() > 0

    def test_with_bias(self):
        """Reference backward computes d_bias when bias is provided."""
        torch.manual_seed(42)
        B, H, V = 32, 64, 128
        hidden = torch.randn(B, H)
        weight = torch.randn(V, H)
        target = torch.randint(0, V, (B,))
        bias = torch.randn(V)

        ce, ent, d_hidden, d_weight, d_bias = _fp32_reference_grads(
            hidden, weight, target, bias=bias,
        )

        assert d_bias is not None
        assert d_bias.shape == (V,)
        assert d_bias.abs().sum() > 0

    def test_gradcheck_small(self):
        """Numerical gradient check on small sizes."""
        torch.manual_seed(7)
        B, H, V = 4, 8, 16
        hidden = torch.randn(B, H, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(V, H, dtype=torch.float64, requires_grad=True)
        target = torch.randint(0, V, (B,))

        def fn(h, w):
            logits = F.linear(h, w)
            log_sm = F.log_softmax(logits, dim=-1)
            sm = log_sm.exp()
            ce = F.nll_loss(log_sm, target, reduction="none")
            ent = -(sm * log_sm).sum(dim=-1)
            return ce.sum() - ent.sum()

        torch.autograd.gradcheck(fn, (hidden, weight), eps=1e-6, atol=1e-4)


# ===========================================================================
#  Level 4 backward tests (CUDA required)
# ===========================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestLevel4Backward:
    def test_d_hidden_matches_reference(self):
        """Level 4 d_hidden matches fp32 reference."""
        from fast_reduction.kernel import fused_linear_xent_entropy_backward, fused_linear_xent_entropy

        torch.manual_seed(99)
        B, H, V = 256, 128, 1024
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16)
        target = torch.randint(0, V, (B,), device="cuda")

        # Forward
        ce, ent, _ = fused_linear_xent_entropy(hidden, weight, target, chunk_size=B)

        # Backward: loss = ce.sum() - ent.sum()
        g_ce = torch.ones(B, device="cuda", dtype=torch.float32)
        g_ent = -torch.ones(B, device="cuda", dtype=torch.float32)
        d_hidden, d_weight, _ = fused_linear_xent_entropy_backward(
            hidden, weight, target, g_ce, g_ent, ent, chunk_size=B,
        )

        # Reference
        _, _, d_hidden_ref, d_weight_ref, _ = _fp32_reference_grads(
            hidden.cpu(), weight.cpu(), target.cpu(),
        )

        d_hidden_mae = (d_hidden.float().cpu() - d_hidden_ref.float()).abs().mean()
        d_weight_mae = (d_weight.float().cpu() - d_weight_ref.float()).abs().mean()

        # bf16 matmul introduces error; tolerances match forward accuracy
        assert d_hidden_mae < 0.05, f"d_hidden MAE={d_hidden_mae:.6f}"
        assert d_weight_mae < 0.05, f"d_weight MAE={d_weight_mae:.6f}"

    def test_autograd_function(self):
        """FusedLinearXentEntropy.apply supports backward()."""
        from fast_reduction.kernel import fused_linear_xent_entropy_differentiable

        torch.manual_seed(42)
        B, H, V = 128, 64, 512
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, V, (B,), device="cuda")

        ce, ent, lp = fused_linear_xent_entropy_differentiable(
            hidden, weight, target, chunk_size=B,
        )
        loss = ce.sum() - ent.sum()
        loss.backward()

        assert hidden.grad is not None
        assert weight.grad is not None
        assert hidden.grad.shape == hidden.shape
        assert weight.grad.shape == weight.shape

        # Compare with reference
        _, _, d_hidden_ref, d_weight_ref, _ = _fp32_reference_grads(
            hidden.detach().cpu(), weight.detach().cpu(), target.cpu(),
        )

        d_hidden_mae = (hidden.grad.float().cpu() - d_hidden_ref.float()).abs().mean()
        d_weight_mae = (weight.grad.float().cpu() - d_weight_ref.float()).abs().mean()

        assert d_hidden_mae < 0.05, f"d_hidden MAE={d_hidden_mae:.6f}"
        assert d_weight_mae < 0.05, f"d_weight MAE={d_weight_mae:.6f}"

    def test_with_bias(self):
        """Level 4 backward computes d_bias correctly."""
        from fast_reduction.kernel import fused_linear_xent_entropy_differentiable

        torch.manual_seed(42)
        B, H, V = 128, 64, 512
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        bias = torch.randn(V, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, V, (B,), device="cuda")

        ce, ent, lp = fused_linear_xent_entropy_differentiable(
            hidden, weight, target, bias=bias, chunk_size=B,
        )
        loss = ce.sum() - ent.sum()
        loss.backward()

        assert bias.grad is not None
        assert bias.grad.shape == (V,)

        _, _, _, _, d_bias_ref = _fp32_reference_grads(
            hidden.detach().cpu(), weight.detach().cpu(), target.cpu(),
            bias=bias.detach().cpu(),
        )
        d_bias_mae = (bias.grad.float().cpu() - d_bias_ref.float()).abs().mean()
        assert d_bias_mae < 0.05, f"d_bias MAE={d_bias_mae:.6f}"

    def test_non_aligned_vocab(self):
        """Backward works with non-power-of-2 vocab sizes."""
        from fast_reduction.kernel import fused_linear_xent_entropy_differentiable

        torch.manual_seed(777)
        B, H, V = 256, 64, 1000
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, V, (B,), device="cuda")

        ce, ent, lp = fused_linear_xent_entropy_differentiable(
            hidden, weight, target, chunk_size=B,
        )
        loss = ce.sum() - ent.sum()
        loss.backward()

        assert not torch.isnan(hidden.grad).any(), "d_hidden contains NaN"
        assert not torch.isnan(weight.grad).any(), "d_weight contains NaN"

        _, _, d_hidden_ref, d_weight_ref, _ = _fp32_reference_grads(
            hidden.detach().cpu(), weight.detach().cpu(), target.cpu(),
        )
        d_hidden_mae = (hidden.grad.float().cpu() - d_hidden_ref.float()).abs().mean()
        assert d_hidden_mae < 0.05, f"d_hidden MAE={d_hidden_mae:.6f}"

    def test_3d_hidden(self):
        """Backward supports 3D hidden_states [batch, seq, hidden]."""
        from fast_reduction.kernel import fused_linear_xent_entropy_differentiable

        torch.manual_seed(17)
        batch, seq, H, V = 4, 32, 64, 512
        hidden = torch.randn(batch, seq, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, V, (batch, seq), device="cuda")

        ce, ent, lp = fused_linear_xent_entropy_differentiable(
            hidden, weight, target, chunk_size=128,
        )
        assert ce.shape == (batch, seq)
        loss = ce.sum() - ent.sum()
        loss.backward()

        assert hidden.grad.shape == (batch, seq, H)
        assert weight.grad.shape == (V, H)

    def test_chunked_matches_unchunked(self):
        """Backward with chunk_size < B gives same result as chunk_size = B."""
        from fast_reduction.kernel import fused_linear_xent_entropy_backward, fused_linear_xent_entropy

        torch.manual_seed(42)
        B, H, V = 256, 128, 1024
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16)
        target = torch.randint(0, V, (B,), device="cuda")

        # Forward (same for both)
        ce, ent, _ = fused_linear_xent_entropy(hidden, weight, target, chunk_size=B)

        g_ce = torch.ones(B, device="cuda", dtype=torch.float32)
        g_ent = -torch.ones(B, device="cuda", dtype=torch.float32)

        # Unchunked (chunk_size = B)
        d_h1, d_w1, _ = fused_linear_xent_entropy_backward(
            hidden, weight, target, g_ce, g_ent, ent, chunk_size=B,
        )
        # Chunked (chunk_size = 64)
        d_h2, d_w2, _ = fused_linear_xent_entropy_backward(
            hidden, weight, target, g_ce, g_ent, ent, chunk_size=64,
        )

        # d_hidden should be identical (no accumulation across chunks)
        torch.testing.assert_close(d_h1, d_h2, atol=0, rtol=0)
        # d_weight uses bf16 matmul + fp32 accumulation across chunks;
        # different chunking changes bf16 rounding, so tolerance matches bf16 precision
        torch.testing.assert_close(d_w1.float(), d_w2.float(), atol=0.05, rtol=0.05)


# ===========================================================================
#  Level 5 backward tests (CUDA required)
# ===========================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestLevel5Backward:
    def test_autograd_function(self):
        """GemmFusedCEEntropy.apply supports backward()."""
        from fast_reduction.gemm_kernel import gemm_fused_ce_entropy_differentiable

        torch.manual_seed(123)
        B, H, V = 256, 128, 1024
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, V, (B,), device="cuda")

        ce, ent, lp = gemm_fused_ce_entropy_differentiable(
            hidden, weight, target, chunk_size=B,
        )
        loss = ce.sum() - ent.sum()
        loss.backward()

        assert hidden.grad is not None
        assert weight.grad is not None

        _, _, d_hidden_ref, d_weight_ref, _ = _fp32_reference_grads(
            hidden.detach().cpu(), weight.detach().cpu(), target.cpu(),
        )

        d_hidden_mae = (hidden.grad.float().cpu() - d_hidden_ref.float()).abs().mean()
        d_weight_mae = (weight.grad.float().cpu() - d_weight_ref.float()).abs().mean()

        assert d_hidden_mae < 0.05, f"d_hidden MAE={d_hidden_mae:.6f}"
        assert d_weight_mae < 0.05, f"d_weight MAE={d_weight_mae:.6f}"

    def test_cross_level_consistency(self):
        """Level 4 and Level 5 backward produce similar gradients."""
        from fast_reduction.kernel import fused_linear_xent_entropy_differentiable
        from fast_reduction.gemm_kernel import gemm_fused_ce_entropy_differentiable

        torch.manual_seed(42)
        B, H, V = 256, 128, 1024
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16)
        target = torch.randint(0, V, (B,), device="cuda")

        # Level 4
        h4 = hidden.clone().requires_grad_(True)
        w4 = weight.clone().requires_grad_(True)
        ce4, ent4, _ = fused_linear_xent_entropy_differentiable(
            h4, w4, target, chunk_size=B,
        )
        (ce4.sum() - ent4.sum()).backward()

        # Level 5
        h5 = hidden.clone().requires_grad_(True)
        w5 = weight.clone().requires_grad_(True)
        ce5, ent5, _ = gemm_fused_ce_entropy_differentiable(
            h5, w5, target, chunk_size=B,
        )
        (ce5.sum() - ent5.sum()).backward()

        # Forward values differ slightly (WGMMA vs cuBLAS), so backward will too.
        # But they should be close.
        d_h_mae = (h4.grad.float() - h5.grad.float()).abs().mean()
        d_w_mae = (w4.grad.float() - w5.grad.float()).abs().mean()

        assert d_h_mae < 0.1, f"d_hidden cross-level MAE={d_h_mae:.6f}"
        assert d_w_mae < 0.1, f"d_weight cross-level MAE={d_w_mae:.6f}"


# ===========================================================================
#  Fast backward tests (CUDA required)
# ===========================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFastBackward:
    def test_level4_fast_matches_reference(self):
        """Level 4 fast backward matches fp32 reference."""
        from fast_reduction.kernel import fused_linear_xent_entropy_fast

        torch.manual_seed(42)
        B, H, V = 256, 128, 1024
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, V, (B,), device="cuda")

        loss, ce, ent = fused_linear_xent_entropy_fast(
            hidden, weight, target, chunk_size=B,
        )
        loss.backward()

        assert hidden.grad is not None
        assert weight.grad is not None

        _, _, d_hidden_ref, d_weight_ref, _ = _fp32_reference_grads(
            hidden.detach().cpu(), weight.detach().cpu(), target.cpu(),
        )

        d_hidden_mae = (hidden.grad.float().cpu() - d_hidden_ref.float()).abs().mean()
        d_weight_mae = (weight.grad.float().cpu() - d_weight_ref.float()).abs().mean()

        assert d_hidden_mae < 0.05, f"d_hidden MAE={d_hidden_mae:.6f}"
        assert d_weight_mae < 0.05, f"d_weight MAE={d_weight_mae:.6f}"

    def test_level4_fast_matches_slow(self):
        """Level 4 fast and slow backward produce identical gradients."""
        from fast_reduction.kernel import (
            fused_linear_xent_entropy_differentiable,
            fused_linear_xent_entropy_fast,
        )

        torch.manual_seed(99)
        B, H, V = 256, 128, 1024
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16)
        target = torch.randint(0, V, (B,), device="cuda")

        # Slow path
        h_slow = hidden.clone().requires_grad_(True)
        w_slow = weight.clone().requires_grad_(True)
        ce, ent, _ = fused_linear_xent_entropy_differentiable(
            h_slow, w_slow, target, chunk_size=B,
        )
        (ce.sum() - ent.sum()).backward()

        # Fast path
        h_fast = hidden.clone().requires_grad_(True)
        w_fast = weight.clone().requires_grad_(True)
        loss, _, _ = fused_linear_xent_entropy_fast(
            h_fast, w_fast, target, chunk_size=B,
        )
        loss.backward()

        # Should be very close (same code path for dlogits, same matmul)
        dh_diff = (h_slow.grad.float() - h_fast.grad.float()).abs().max()
        dw_diff = (w_slow.grad.float() - w_fast.grad.float()).abs().max()

        assert dh_diff < 0.01, f"d_hidden max diff={dh_diff:.6f}"
        assert dw_diff < 0.01, f"d_weight max diff={dw_diff:.6f}"

    def test_level5_fast_matches_reference(self):
        """Level 5 fast backward matches fp32 reference."""
        from fast_reduction.gemm_kernel import gemm_fused_ce_entropy_fast

        torch.manual_seed(123)
        B, H, V = 256, 128, 1024
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, V, (B,), device="cuda")

        loss, ce, ent = gemm_fused_ce_entropy_fast(
            hidden, weight, target, chunk_size=B,
        )
        loss.backward()

        _, _, d_hidden_ref, d_weight_ref, _ = _fp32_reference_grads(
            hidden.detach().cpu(), weight.detach().cpu(), target.cpu(),
        )

        d_hidden_mae = (hidden.grad.float().cpu() - d_hidden_ref.float()).abs().mean()
        d_weight_mae = (weight.grad.float().cpu() - d_weight_ref.float()).abs().mean()

        assert d_hidden_mae < 0.05, f"d_hidden MAE={d_hidden_mae:.6f}"
        assert d_weight_mae < 0.05, f"d_weight MAE={d_weight_mae:.6f}"

    def test_fast_non_aligned_vocab(self):
        """Fast backward works with non-power-of-2 vocab sizes."""
        from fast_reduction.kernel import fused_linear_xent_entropy_fast

        torch.manual_seed(777)
        B, H, V = 256, 64, 1000
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, V, (B,), device="cuda")

        loss, ce, ent = fused_linear_xent_entropy_fast(
            hidden, weight, target, chunk_size=B,
        )
        loss.backward()

        assert not torch.isnan(hidden.grad).any(), "d_hidden contains NaN"
        assert not torch.isnan(weight.grad).any(), "d_weight contains NaN"

        _, _, d_hidden_ref, d_weight_ref, _ = _fp32_reference_grads(
            hidden.detach().cpu(), weight.detach().cpu(), target.cpu(),
        )
        d_hidden_mae = (hidden.grad.float().cpu() - d_hidden_ref.float()).abs().mean()
        assert d_hidden_mae < 0.05, f"d_hidden MAE={d_hidden_mae:.6f}"

    def test_fast_detached_outputs(self):
        """Fast backward returns detached ce and ent (no grad)."""
        from fast_reduction.kernel import fused_linear_xent_entropy_fast

        torch.manual_seed(42)
        B, H, V = 128, 64, 512
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, V, (B,), device="cuda")

        loss, ce, ent = fused_linear_xent_entropy_fast(
            hidden, weight, target, chunk_size=B,
        )

        # loss should require grad, ce and ent should not
        assert loss.requires_grad
        assert not ce.requires_grad
        assert not ent.requires_grad

        # loss should be scalar
        assert loss.dim() == 0

        # ce and ent should be per-element
        assert ce.shape == (B,)
        assert ent.shape == (B,)

    def test_level5_megakernel_matches_reference(self):
        """Level 5 megakernel backward matches fp32 reference."""
        from fast_reduction.gemm_kernel import gemm_megakernel_fast

        torch.manual_seed(123)
        B, H, V = 256, 128, 1024
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, V, (B,), device="cuda")

        loss, ce, ent = gemm_megakernel_fast(
            hidden, weight, target, chunk_size=B,
        )
        loss.backward()

        _, _, d_hidden_ref, d_weight_ref, _ = _fp32_reference_grads(
            hidden.detach().cpu(), weight.detach().cpu(), target.cpu(),
        )

        d_hidden_mae = (hidden.grad.float().cpu() - d_hidden_ref.float()).abs().mean()
        d_weight_mae = (weight.grad.float().cpu() - d_weight_ref.float()).abs().mean()

        assert d_hidden_mae < 0.05, f"d_hidden MAE={d_hidden_mae:.6f}"
        assert d_weight_mae < 0.05, f"d_weight MAE={d_weight_mae:.6f}"

    def test_level5_megakernel_matches_slow(self):
        """Level 5 megakernel and standard L5 backward produce similar gradients."""
        from fast_reduction.gemm_kernel import (
            gemm_fused_ce_entropy_fast,
            gemm_megakernel_fast,
        )

        torch.manual_seed(99)
        B, H, V = 256, 128, 1024
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16)
        target = torch.randint(0, V, (B,), device="cuda")

        # Standard L5 fast
        h_slow = hidden.clone().requires_grad_(True)
        w_slow = weight.clone().requires_grad_(True)
        loss1, _, _ = gemm_fused_ce_entropy_fast(h_slow, w_slow, target, chunk_size=B)
        loss1.backward()

        # Megakernel
        h_mega = hidden.clone().requires_grad_(True)
        w_mega = weight.clone().requires_grad_(True)
        loss2, _, _ = gemm_megakernel_fast(h_mega, w_mega, target, chunk_size=B)
        loss2.backward()

        # Both use WGMMA logits, so gradients should be very close
        dh_mae = (h_slow.grad.float() - h_mega.grad.float()).abs().mean()
        dw_mae = (w_slow.grad.float() - w_mega.grad.float()).abs().mean()

        assert dh_mae < 0.1, f"d_hidden cross-method MAE={dh_mae:.6f}"
        assert dw_mae < 0.1, f"d_weight cross-method MAE={dw_mae:.6f}"

    def test_megakernel_non_aligned_vocab(self):
        """Megakernel backward works with non-power-of-2 vocab sizes."""
        from fast_reduction.gemm_kernel import gemm_megakernel_fast

        torch.manual_seed(777)
        B, H, V = 256, 64, 1000
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, V, (B,), device="cuda")

        loss, ce, ent = gemm_megakernel_fast(
            hidden, weight, target, chunk_size=B,
        )
        loss.backward()

        assert not torch.isnan(hidden.grad).any(), "d_hidden contains NaN"
        assert not torch.isnan(weight.grad).any(), "d_weight contains NaN"

        _, _, d_hidden_ref, d_weight_ref, _ = _fp32_reference_grads(
            hidden.detach().cpu(), weight.detach().cpu(), target.cpu(),
        )
        d_hidden_mae = (hidden.grad.float().cpu() - d_hidden_ref.float()).abs().mean()
        assert d_hidden_mae < 0.05, f"d_hidden MAE={d_hidden_mae:.6f}"

    def test_megakernel_detached_outputs(self):
        """Megakernel returns detached ce and ent (no grad)."""
        from fast_reduction.gemm_kernel import gemm_megakernel_fast

        torch.manual_seed(42)
        B, H, V = 128, 64, 512
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, V, (B,), device="cuda")

        loss, ce, ent = gemm_megakernel_fast(
            hidden, weight, target, chunk_size=B,
        )

        assert loss.requires_grad
        assert not ce.requires_grad
        assert not ent.requires_grad
        assert loss.dim() == 0
        assert ce.shape == (B,)
        assert ent.shape == (B,)

    def test_fast_chunked(self):
        """Fast backward works with chunk_size < B."""
        from fast_reduction.kernel import fused_linear_xent_entropy_fast

        torch.manual_seed(42)
        B, H, V = 256, 128, 1024
        hidden = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16)
        target = torch.randint(0, V, (B,), device="cuda")

        # Unchunked
        h1 = hidden.clone().requires_grad_(True)
        w1 = weight.clone().requires_grad_(True)
        loss1, _, _ = fused_linear_xent_entropy_fast(h1, w1, target, chunk_size=B)
        loss1.backward()

        # Chunked
        h2 = hidden.clone().requires_grad_(True)
        w2 = weight.clone().requires_grad_(True)
        loss2, _, _ = fused_linear_xent_entropy_fast(h2, w2, target, chunk_size=64)
        loss2.backward()

        # d_hidden should be identical (no accumulation across chunks)
        torch.testing.assert_close(h1.grad, h2.grad, atol=0, rtol=0)
        # d_weight may differ slightly due to bf16 chunked accumulation
        torch.testing.assert_close(
            w1.grad.float(), w2.grad.float(), atol=0.05, rtol=0.05,
        )
