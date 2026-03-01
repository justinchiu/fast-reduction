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
only allocated 2 mbarrier stages (`self.stage = 2`). The passes were assigned:

```python
# Pass 1 (max):      buffer slot 0, mbar_ptr + 0, waits phase 0  ✓
max_x = row_reduce(x, ..., reduction_buffer[None, None, 0],
                   mbar_ptr + 0, ...)

# Pass 2 (sum_exp):  buffer slot 1, mbar_ptr + 1, waits phase 0  ✓
sum_exp = row_reduce(exp_x, ..., reduction_buffer[None, None, 1],
                     mbar_ptr + 1, ...)

# Pass 3 (sum_x_exp): buffer slot 0, mbar_ptr + 0, waits phase 0  ✗ BUG
sum_x_exp = row_reduce(x_times_exp, ..., reduction_buffer[None, None, 0],
                       mbar_ptr + 0, ...)
```

Inside `cluster_reduce`, each pass ends with `mbarrier_wait(mbar_ptr, phase=0)`.
After pass 1 completes, `mbar_ptr + 0`'s phase advances to 1. When pass 3 reuses
`mbar_ptr + 0` and waits on phase 0, that phase is **already completed** — the
wait returns immediately before pass 3's async DSMEM stores have landed. The
thread reads **stale max values from pass 1** instead of sum_x_exp, so entropy
computes `lse - max_x / sum_exp` instead of `lse - sum_x_exp / sum_exp`.

This only manifests when `cluster_n > 1` (V > 16384), which is why small-V tests
passed. The block-level reduction path uses `__syncthreads()` which has no phase
semantics, so cluster_n=1 was unaffected.

### Fix

**1. Allocate 3 stages instead of 2** so each pass gets its own mbarrier and
buffer slot:

```python
class EntropyOnly(_KernelBase):
    def __init__(self, dtype, N):
        super().__init__(dtype, N)
        self.stage = 3  # was 2

class CrossEntropyEntropy(_KernelBase):
    def __init__(self, dtype, N):
        super().__init__(dtype, N)
        self.stage = 3
```

**2. Point pass 3 at the new slot 2** (instead of reusing slot 0):

```python
# Before:
sum_x_exp = row_reduce(..., reduction_buffer[None, None, 0],
                       mbar_ptr + 0, ...)
# After:
sum_x_exp = row_reduce(..., reduction_buffer[None, None, 2],
                       mbar_ptr + 2, ...)
```

**3. Replace approximate reciprocal with exact division:**

```python
# Before:
ent = lse - sum_x_exp * cute.arch.rcp_approx(sum_exp)
# After:
ent = lse - sum_x_exp / sum_exp
```
