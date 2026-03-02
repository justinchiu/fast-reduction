# Backward Pass for Fused Linear + CE + Entropy

## Current Status

Phases 1-3 complete. Phase 3 implemented the quack/liger interleaved pattern for Level 4,
achieving a 25% speedup. Level 5 fast backward provides no speedup because WGMMA and cuBLAS
produce different logits (LSE mismatch prevents reuse).

Next: CuTe backward kernel for dlogits (fuse 4 element-wise passes → 1 kernel).

## Performance (B=32768, H=4096, V=128256, bf16, chunk=4096)

| Level | Variant | Fwd (ms) | Fwd+Bwd (ms) | Peak mem | CE MAE | dH MAE |
|-------|---------|----------|---------------|----------|--------|--------|
| 1 | torch unfused | OOM | OOM | — | — | — |
| 2 | torch chunked | 112 | 308 | 38.5 GB | 0.4807 | 0.0335 |
| 4 | CuTe joint (slow) | 70 | 351 | 14.5 GB | 0.4807 | 0.0335 |
| **4F** | **CuTe joint (fast)** | **—** | **262** | **13.5 GB** | **0.4807** | **0.0335** |
| 5 | GEMM epilogue (slow) | 67 | 346 | 14.5 GB | 0.0012 | 0.0438 |
| 5F | GEMM epilogue (fast) | — | 348 | 14.5 GB | 0.0012 | 0.0438 |

**Level 4 fast is the best fwd+bwd path** at 262ms — 25% faster than Level 4 slow,
15% faster than torch chunked. Level 5 fast provides no speedup because backward
must recompute logits via cuBLAS regardless.

## Per-Chunk Backward Profiling (chunk=4096, V=128256)

| Step | Time (ms) | Notes |
|------|-----------|-------|
| Recompute logits (torch.mm) | 6.0 | Extra matmul, avoidable |
| logsumexp | 4.4 | Already computed in forward |
| dlogits element-wise (5 ops) | 14.3 | 5 unfused HBM passes over 2GB |
| d_hidden matmul | 4.8 | Necessary |
| d_weight matmul | 4.3 | Necessary (bf16 + fp32 accum) |
| **Total** | **~34** | **~15ms achievable (just 3 matmuls)** |

## Math

Per-token logits `z`, target `y`, softmax `p = exp(z - lse)`, log-probs `log_p = z - lse`.

CE gradient:      `dL_ce/dz_j = p_j - 1{j=y}`
Entropy gradient: `dH/dz_j = -p_j * (log_p_j + H)`

Combined with upstream `g_ce`, `g_ent` per token:

```
dlogits_j = g_ce * (p_j - 1{j=y}) + g_ent * (-p_j * (log_p_j + H))
          = p_j * (g_ce - g_ent*(log_p_j + H)) - g_ce * 1{j=y}
```

Then per chunk:
- `d_hidden_chunk = dlogits_chunk @ weight`       [chunk,V] @ [V,H] -> [chunk,H]
- `d_weight += dlogits_chunk.T @ hidden_chunk`     [V,chunk] @ [chunk,H] -> [V,H]
- `d_bias += dlogits_chunk.sum(dim=0)`             if bias

## Reference Implementations Analysis

### Quack (`~/quack/quack/linear_cross_entropy.py`)

**Key insight: ALL gradient computation happens during the forward pass.**

```
Forward (per chunk):
  1. logits = x_chunk @ W.T                          # cuBLAS matmul
  2. loss, dlogits = ce_fwd(logits, target, dx=buf)  # CuTe kernel writes dlogits IN-PLACE
  3. d_hidden_chunk = dlogits @ W                     # matmul (logits still warm in cache)
  4. d_weight += dlogits.T @ x_chunk                  # accumulate (defers last chunk)

Backward:
  d_hidden *= dloss_scalar                            # element_mul, O(B*H)
  d_weight += last_chunk_dw * dloss                   # one small matmul
```

The CuTe forward kernel (`CrossEntropy`) computes softmax probabilities during the
reduction, then writes `p_j - 1{j=target}` directly to the logits buffer (in-place
overwrite). This IS the dlogits for CE. No separate backward kernel needed for the
reduction.

The `ChunkedLinearCrossEntropyFunction` defers the last chunk's d_weight GEMM to
the backward pass so it can be scaled by `dloss`. All other chunks' d_weight
contributions are pre-accumulated during forward.

### Liger-Kernel (`~/Liger-Kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py`)

Same strategy, Triton instead of CuTe:

```
Forward (per chunk):
  1. logits = x_chunk @ W.T
  2. loss = triton_ce_fwd(logits, target)             # 1st pass: compute loss
     dlogits = triton_ce_bwd(logits, target)          # 2nd pass: overwrite logits with dlogits
  3. d_hidden_chunk = dlogits @ W
  4. d_weight += dlogits.T @ x_chunk

Backward:
  element_mul(d_hidden, dloss)                        # Triton kernel, O(B*H)
  element_mul(d_weight, dloss)                        # Triton kernel, O(V*H)
```

### Pattern Summary

Both implementations share the same core pattern:

1. **Compute dlogits DURING the forward pass** — logits are already in registers/L2
   cache from the reduction kernel, so writing dlogits costs almost nothing extra.
2. **Immediately compute d_hidden and d_weight via matmul** — while logits/dlogits
   are still cache-warm, no need to re-read them from HBM later.
3. **Save pre-computed gradients** in autograd ctx — backward just scales by the
   upstream scalar gradient (one element-wise multiply).

This eliminates:
- Logits recomputation matmul (~6ms/chunk)
- logsumexp recomputation (~4.4ms/chunk)
- Multiple unfused HBM round-trips for element-wise dlogits (~14ms/chunk → 0)

## Phase 3 Implementation: Interleaved Forward+Backward

### What Was Implemented

Following the quack/liger pattern, Phase 3 interleaves forward and backward in a
single chunked loop. For Level 4, this avoids logits recomputation and uses pre-computed
LSE from the CuTe kernel.

**Level 4 fast path** (`fused_linear_xent_entropy_fast`):
```python
for chunk:
    logits = h_chunk @ W.T                                # forward matmul
    loss, ent, lse = ce_entropy_fwd_with_lse(logits, t)   # CuTe kernel (with LSE output)
    dlogits = _compute_dlogits_from_lse(logits, t, lse, ent)  # no logsumexp recompute
    d_hidden[chunk] = dlogits @ W                          # backward matmul
    d_weight += dlogits.T @ h_chunk                        # accumulate
```

The autograd.Function (`FusedLinearXentEntropyFast`) returns `(scalar_loss, ce_detached, ent_detached)`.
Backward just scales pre-computed d_hidden/d_weight by the upstream dloss scalar.

### Why Level 5 Fast Doesn't Help

Level 5 forward uses WGMMA (custom matmul in GEMM epilogue) which produces slightly
different logits than cuBLAS. At V=128256, the LSE difference is ~0.48 MAE. Using
WGMMA LSE with cuBLAS-recomputed logits produces inconsistent softmax (doesn't sum to 1),
causing d_hidden MAE of ~0.89 — unacceptable.

The Level 5 fast path must use cuBLAS logsumexp for consistency, eliminating the LSE
reuse benefit. The remaining benefit (single-loop structure) provides negligible speedup.

### What Saves Time in Level 4

Per chunk at V=128256:
- **Logits recomputation avoided**: 6ms saved (shared with forward in same loop)
- **logsumexp avoided**: 4.4ms saved (uses CuTe kernel LSE output)
- Element-wise dlogits: ~10ms (still 4 PyTorch ops, reduced from 5)
- d_hidden matmul: 4.8ms (unchanged)
- d_weight matmul: 4.3ms (unchanged)

Total savings: ~10.4ms/chunk × 8 chunks = ~83ms → matches observed 89ms improvement.

### Future Optimization: CuTe Backward Kernel

The remaining bottleneck is 4 unfused element-wise passes for dlogits (~10ms/chunk).
A CuTe kernel fusing `sub(lse) → exp → mul(factor) → scatter(target)` into one pass
would reduce this to ~1ms/chunk, saving ~72ms total.

## Implementation Files

| File | Changes |
|------|---------|
| `fast_reduction/cute_cross_entropy.py` | Added LSE output to `CrossEntropyEntropy`, `ce_entropy_fwd_with_lse()` |
| `fast_reduction/kernel.py` | Added `_compute_dlogits_from_lse()`, `_fused_fwd_bwd()`, `FusedLinearXentEntropyFast`, `fused_linear_xent_entropy_fast()` |
| `fast_reduction/gemm_kernel.py` | Added `_gemm_fused_fwd_bwd()`, `GemmFusedCEEntropyFast`, `gemm_fused_ce_entropy_fast()` |
| `fast_reduction/gemm_ce_entropy_finalize.py` | Added LSE output to `Finalize`, `finalize_ce_entropy_with_lse()` |
| `fast_reduction/baseline.py` | `baseline_linear_xent_entropy_backward()` (Phase 2) |
| `fast_reduction/__init__.py` | Updated exports |
| `tests/test_backward.py` | 17 tests (3 reference + 6 level4 + 2 level5 + 6 fast) |
| `benchmarks/bench_fwd_bwd.py` | Forward+backward benchmark with fast variants |

### Running Tests

```bash
CUDA_VISIBLE_DEVICES=0 uv run pytest tests/test_backward.py -v
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/bench_fwd_bwd.py
```
