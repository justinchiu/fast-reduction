# OOB Fill Value Issue in CuTe Entropy Kernels

## Problem

The CuTe reduction kernels (`EntropyOnly`, `CrossEntropyEntropy`) tile the vocab
dimension N into blocks. When V is not evenly divisible by the tile size
(`is_even_N = False`), out-of-bounds (OOB) lanes in the last tile must be filled
with a sentinel value.

The original code fills OOB lanes with `-inf`:

```python
cute_utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
```

This is correct for:
- **max reduction**: `max(real, -inf) = real` — OOB lanes ignored
- **sum_exp reduction**: `exp(-inf - max) = 0` — OOB lanes contribute nothing

But it causes **NaN in entropy** via the `x * exp(x - max)` term:
- OOB lane: `x = -inf`, `exp_x = exp(-inf - max) = 0`
- Product: `-inf * 0 = NaN` (IEEE 754)
- NaN propagates through `sum_x_exp` → `entropy = NaN`

This affects any V that doesn't align with the kernel's tile size (e.g., V=128256).

## Fix

Replace `-inf` with a finite sentinel `-1e4` in the two entropy-computing kernels:

```python
cute_utils.fill_oob(tXsX, tXpX, tXsX.element_type(-1e4))
```

Why `-1e4` works:
- **max**: `-1e4` is far below any real logit (typically [-10, 10]), so max is unaffected
- **sum_exp**: `exp(-1e4 - max)` underflows to **exactly 0.0** in fp32 (below denorm range)
- **x * exp_x**: `(-1e4) * 0.0 = -0.0` — finite, contributes 0 to sum

The `CrossEntropyOnly` kernel does not compute `x * exp_x`, so it can keep `-inf`.

Zero extra cost: no additional ops, no extra SMEM round-trips, compiled away for
aligned V via `const_expr`.

## Remaining Issue

The sentinel fix eliminates NaN but does **not** fix a separate accuracy problem
in the CuTe entropy kernels at large V (128256):

| Kernel | Ent MAE (vs fp32 ref) |
|--------|-----------------------|
| EntropyOnly (Level 3) | 2690 |
| CrossEntropyEntropy (Level 4) | 3.6 |
| torch log_softmax (Levels 1-2) | 0.029 |
| GEMM epilogue (Level 5) | 0.00002 |

The entropy error at large V is a separate numerical issue (likely in the
cross-cluster reduction or `rcp_approx` precision), not caused by the OOB fill.
