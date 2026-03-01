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

## Remaining Issue — FIXED

The sentinel fix eliminated NaN, but a separate accuracy problem remained in the
CuTe entropy kernels at large V (128256):

| Kernel | Ent MAE (before) | Ent MAE (after) |
|--------|-----------------|-----------------|
| EntropyOnly (Level 3) | 2690 | 0.000002 |
| CrossEntropyEntropy (Level 4) | 3.6 | 0.000002 |
| torch log_softmax (Levels 1-2) | 0.029 | 0.029 |
| GEMM epilogue (Level 5) | 0.00002 | 0.00002 |

### Root cause: mbarrier phase reuse in 3-pass cluster reduction

Both entropy kernels perform 3 reduction passes (max, sum_exp, sum_x_exp) but
only allocated 2 mbarrier stages (`self.stage = 2`). Pass 3 reused `mbar_ptr + 0`
(same as pass 1) with `phase=0`, but that phase had already been consumed by
pass 1. After pass 1, the mbarrier phase advances to 1, so `mbarrier_wait(phase=0)`
in pass 3 **returned immediately with stale data** — reading max values from pass 1
instead of the sum_x_exp values from pass 3.

This only manifests when `cluster_n > 1` (V > 16384), which is why small-V tests
passed. The block-level reduction path uses `__syncthreads()` which has no phase
semantics, so cluster_n=1 was unaffected.

### Fix

1. Increased `self.stage` from 2 to 3 in `EntropyOnly` and `CrossEntropyEntropy`,
   giving each reduction pass its own buffer slot and mbarrier.
2. Changed pass 3 to use `reduction_buffer[..., 2]` and `mbar_ptr + 2`.
3. Replaced `rcp_approx(sum_exp)` with proper division (`/ sum_exp`) in the
   entropy formula for maximum precision.
