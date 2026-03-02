# fast-reduction

CuTe DSL + PyTorch implementations for `linear + cross-entropy + entropy` on Hopper (H100, `sm90`), with a benchmark ladder from unfused to increasingly fused variants.

## Forward Performance

B=32768, H=4096, V=128256, bf16, chunk=4096, on a single H100 80GB:

| Level | Variant | Time | Peak mem | CE MAE | Ent MAE |
|-------|---------|------|----------|--------|---------|
| 1 | torch unfused | 113 ms | 56 GB | 0.481 | 0.029 |
| 2 | torch chunked | 112 ms | 10 GB | 0.481 | 0.029 |
| 3 | CuTe separate | 76 ms | 5.2 GB | 0.481 | 0.029 |
| 4 | CuTe joint | 71 ms | 5.2 GB | 0.481 | 0.029 |
| 5 | GEMM epilogue | 67 ms | 1.3 GB | 0.0012 | 0.00002 |
| 5.2 | GEMM epilogue v2 | 70 ms | 1.3 GB | 0.0012 | 0.00002 |

Error is MAE vs fp32 ground truth (fp32 matmul + fp32 reduction).

## Forward + Backward Performance

B=32768, H=4096, V=128256, bf16, chunk=4096:

| Level | Variant | Fwd+Bwd | Peak mem | CE MAE | dH MAE |
|-------|---------|---------|----------|--------|--------|
| 1 | torch unfused | OOM | — | — | — |
| 2 | torch chunked | 308 ms | 38.5 GB | 0.481 | 0.034 |
| 4 | CuTe joint (autograd) | 350 ms | 14.5 GB | 0.481 | 0.034 |
| 5 | GEMM epilogue (autograd) | 346 ms | 14.5 GB | 0.0012 | 0.044 |
| 4F | CuTe joint fast | 262 ms | 13.5 GB | 0.481 | 0.034 |
| 5F | GEMM epilogue fast | 348 ms | 14.5 GB | 0.0012 | 0.044 |
| **5M** | **GEMM megakernel** | **243 ms** | **13.6 GB** | **0.0012** | **0.0009** |

The "fast" variants (4F, 5F, 5M) pre-compute gradients during the forward pass
(liger/quack pattern), so backward just scales by the upstream scalar.

**5M (megakernel)** is the fastest and most accurate backward. It extends the
GEMM epilogue to also write fp32 logits to GMEM, then uses a fused CuTe dlogits
kernel instead of 4-5 unfused PyTorch ops. This eliminates the cuBLAS logits
recompute and logsumexp recompute from 5F, and the fp32 WGMMA logits give 38x
better gradient accuracy than the bf16 cuBLAS logits used by levels 2-4.

## Why Level 5 Is 400x More Precise

Levels 1-4 compute logits via `torch.mm(bf16, bf16)` which outputs **bf16** —
the fp32 accumulator gets truncated to bf16 on write to HBM. Softmax/CE then
operates on these truncated logits, giving CE MAE ≈ 0.48.

Level 5 fuses the reduction into the GEMM epilogue and reads the **fp32
accumulator directly** from registers/SMEM — logits never touch HBM. This
avoids bf16 truncation entirely, giving CE MAE ≈ 0.0012.

| | Logit precision | CE MAE | Ent MAE |
|---|---|---|---|
| Levels 1-4 (bf16 logits from HBM) | bf16 (8-bit mantissa) | 0.481 | 0.029 |
| Level 5 (fp32 acc from registers) | fp32 (23-bit mantissa) | 0.0012 | 0.00002 |

## Level 5 Error Decomposition

Using the WGMMA GEMM with default epilogue (fp32 output) as reference to
isolate epilogue error from matmul error:

| Source | CE MAE | CE max |
|--------|--------|--------|
| WGMMA bf16 matmul (fp32 acc) vs fp32 matmul | 0.001163 | — |
| v5.1 epilogue + finalization | 0.000003 | 0.000031 |
| v5.2 epilogue + finalization | 0.000003 | 0.000031 |

The epilogue contributes 0.3% of the total error. The remaining 99.7% is from
WGMMA K-reduction order differing from sequential fp32.

v5.1 and v5.2 produce identical CE and near-identical entropy (max diff ~1e-5).
The two-pass epilogue in v5.2 eliminates online softmax merge and `fastmath`,
but the single-pass merge in v5.1 is already precise because it operates over
only 8 subtiles within a CTA of 256 vocab columns, where adjustment factors
are ≈ 1.0.

## Repository Layout

```text
fast_reduction/
  __init__.py
  baseline.py                       # PyTorch references (unfused + chunked)
  kernel.py                         # chunked matmul + CuTe reductions (levels 3 & 4)
  cute_cross_entropy.py             # CuTe kernels: CE-only, entropy-only, joint
  cute_dlogits.py                   # Fused CuTe dlogits kernel (level 5M backward)
  reduce.py                         # reduction primitives
  cute_utils.py                     # low-level PTX helpers
  gemm_ce_entropy_epilogue.py       # Level 5 GEMM epilogue mixin (single-pass)
  gemm_ce_entropy_epilogue_v2.py    # Level 5.2 GEMM epilogue mixin (two-pass)
  gemm_ce_entropy_bwd_epilogue.py   # Level 5M GEMM epilogue (partials + logits output)
  gemm_ce_entropy_finalize.py       # Finalization kernel (shared by v5.1, v5.2, 5M)
  gemm_kernel.py                    # Level 5 driver + megakernel (v5.1, v5.2, 5M)

benchmarks/
  bench_linear_xent_entropy.py      # forward-only benchmark
  bench_fwd_bwd.py                  # forward + backward benchmark

tests/
  test_ce_impls.py
  test_backward.py                  # backward gradient correctness tests
  test_entropy_large_v.py           # large-V entropy accuracy tests

docs/
  memory_bound_kernels.md
```

## Install

```bash
uv sync --extra dev
```

## Run Tests

```bash
uv run pytest tests/ -v
```

## Run Benchmarks

```bash
# Forward-only
uv run python benchmarks/bench_linear_xent_entropy.py

# Forward + backward
uv run python benchmarks/bench_fwd_bwd.py
```

## API

```python
# Forward-only (inference)
from fast_reduction.gemm_kernel import gemm_fused_ce_entropy

ce_loss, entropy, log_probs = gemm_fused_ce_entropy(
    hidden_states,  # [B, H] or [..., H]  bf16
    weight,         # [V, H]
    target,         # [B] or [...]
    chunk_size=4096,
)

# Training (forward + backward, pre-computed gradients)
from fast_reduction.gemm_kernel import gemm_megakernel_fast

loss, ce_loss, entropy = gemm_megakernel_fast(
    hidden_states,  # [B, H]  bf16, requires_grad=True
    weight,         # [V, H]  bf16, requires_grad=True
    target,         # [B]
    chunk_size=4096,
    ce_weight=1.0,
    ent_weight=-1.0,
)
loss.backward()  # hidden_states.grad, weight.grad populated
```
