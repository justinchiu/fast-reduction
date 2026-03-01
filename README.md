# fast-reduction

CuTe DSL + PyTorch implementations for `linear + cross-entropy + entropy` on Hopper (H100, `sm90`), with a benchmark ladder from unfused to increasingly fused variants.

## Performance Ladder

B=32768, H=4096, V=128256, bf16, chunk=4096, on a single H100 80GB:

| Level | Variant | Time | Peak mem | CE MAE | Ent MAE |
|-------|---------|------|----------|--------|---------|
| 1 | torch unfused | 118 ms | 57 GB | 0.482 | 0.028 |
| 2 | torch chunked | 114 ms | 10 GB | 0.482 | 0.028 |
| 3 | CuTe separate | 74 ms | 5.3 GB | 0.482 | NaN* |
| 4 | CuTe joint | 69 ms | 5.3 GB | 0.482 | NaN* |
| 5 | GEMM epilogue | 66 ms | 1.3 GB | 0.0012 | 0.00002 |
| 5.2 | GEMM epilogue v2 | 68 ms | 1.3 GB | 0.0012 | 0.00002 |

Error is MAE vs fp32 ground truth (fp32 matmul + fp32 reduction).

\* Levels 3-4 produce NaN entropy at V=128256 with bf16 logits (pre-existing
CuTe reduction kernel issue).

## Why Level 5 Is 400x More Precise

Levels 1-4 compute logits via `torch.mm(bf16, bf16)` which outputs **bf16** —
the fp32 accumulator gets truncated to bf16 on write to HBM. Softmax/CE then
operates on these truncated logits, giving CE MAE ≈ 0.48.

Level 5 fuses the reduction into the GEMM epilogue and reads the **fp32
accumulator directly** from registers/SMEM — logits never touch HBM. This
avoids bf16 truncation entirely, giving CE MAE ≈ 0.0012.

| | Logit precision | CE MAE | Ent MAE |
|---|---|---|---|
| Levels 1-4 (bf16 logits from HBM) | bf16 (8-bit mantissa) | 0.482 | 0.028 |
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
  reduce.py                         # reduction primitives
  cute_utils.py                     # low-level PTX helpers
  gemm_ce_entropy_epilogue.py       # Level 5 GEMM epilogue mixin (single-pass)
  gemm_ce_entropy_epilogue_v2.py    # Level 5.2 GEMM epilogue mixin (two-pass)
  gemm_ce_entropy_finalize.py       # Finalization kernel (shared by v5.1 and v5.2)
  gemm_kernel.py                    # Level 5 driver (v5.1 and v5.2)

benchmarks/
  bench_linear_xent_entropy.py

tests/
  test_ce_impls.py

docs/
  memory_bound_kernels.md
```

## Install

```bash
uv sync --extra dev
```

## Run Tests

```bash
uv run pytest tests/test_ce_impls.py -v
```

## Run Benchmarks

```bash
uv run python benchmarks/bench_linear_xent_entropy.py
uv run python benchmarks/bench_linear_xent_entropy.py --B 32768 --V 128256 --H 4096
```

## API

```python
from fast_reduction.gemm_kernel import gemm_fused_ce_entropy, gemm_fused_ce_entropy_v2

# Level 5: GEMM epilogue (recommended — fastest, most precise)
ce_loss, entropy, log_probs = gemm_fused_ce_entropy(
    hidden_states,  # [B, H] or [..., H]  bf16
    weight,         # [V, H]
    target,         # [B] or [...]
    chunk_size=4096,
)

# Level 5.2: two-pass epilogue (identical precision, ~5% slower)
ce_loss, entropy, log_probs = gemm_fused_ce_entropy_v2(
    hidden_states, weight, target, chunk_size=4096,
)
```
