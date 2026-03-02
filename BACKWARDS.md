# Backward Pass for Fused Linear + CE + Entropy

## Current Status

Phase 1 (reference backward) and Phase 2 (chunked PyTorch-ops backward) are complete.
The backward is **correct** but **slow** — same speed for Level 4 and Level 5 because
both recompute logits via `torch.mm` and use unfused PyTorch element-wise ops for dlogits.

Next: Phase 3 — fuse dlogits into the forward CuTe kernel (quack/liger pattern).

## Current Backward Performance (B=32768, H=4096, V=128256, bf16, chunk=4096)

| Level | Variant | Fwd (ms) | Fwd+Bwd (ms) | Peak mem | CE MAE | dH MAE |
|-------|---------|----------|---------------|----------|--------|--------|
| 1 | torch unfused | OOM | OOM | — | — | — |
| 2 | torch chunked | 722 | 1437 | 11.0 GB | 0.0000 | 0.0000 |
| 4 | CuTe joint | 675 | 1101 | 13.9 GB | 0.0001 | 0.0007 |
| 5 | GEMM epilogue | 68 | 495 | 13.9 GB | 0.0012 | 0.0018 |

L4 and L5 backward are identical (same code path). L5's forward advantage doesn't help.

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

## Plan: Phase 3 — Fused Backward

### Approach

Modify the forward to compute dlogits during the CuTe kernel, then compute
d_hidden/d_weight immediately in the same forward loop.

### Step 1: CuTe kernel variant that outputs dlogits

Modify `CrossEntropyEntropy` in `cute_cross_entropy.py` (or add a new class) to
output dlogits in-place over the logits buffer.

The kernel already computes `p_j` (softmax) and `log_p_j` during the online reduction.
At the end, after writing loss and entropy, also write:

```python
# For default g_ce=1, g_ent=-1:
dlogits_j = p_j * (1 + log_p_j + H) - 1{j=target}
```

For general upstream gradients, write the "unscaled" form:
```python
dlogits_j = p_j          # at non-target positions
dlogits_j = p_j - 1.0    # at target position
```
This is the CE-only gradient. The entropy contribution (`-p_j * (log_p_j + H)`)
can be written to a second output or combined. However, since both quack and liger
only handle CE (not CE+entropy), we need to extend their pattern.

**Option A: Write CE-dlogits + save log_p and H for entropy dlogits.**
Forward kernel outputs `p_j - 1{j=y}` in-place. Backward (if needed for non-trivial
upstream) reads this plus saved `log_p_j` and `H` to reconstruct full dlogits.
Problem: log_p is [M,V], too large to save.

**Option B: Write combined dlogits for fixed g_ce=1, g_ent=-1.**
Forward kernel outputs `p_j * (1 + log_p_j + H) - 1{j=y}` in-place. Backward
just scales d_hidden and d_weight by the upstream scalar. This only works for the
standard loss = ce.sum() - ent.sum(). For other upstream shapes, fall back to
the current recompute path.

**Option C: Write softmax `p_j` to dlogits buffer, save entropy `H`.**
Forward kernel overwrites logits with `p_j`. Backward reads `p_j`, computes
`log_p_j = log(p_j)`, then assembles full dlogits:
`dlogits_j = g_ce*(p_j - 1{j=y}) + g_ent*(-p_j*(log_p_j + H))`
This is element-wise but needs only 1 HBM read of [M,V] (softmax), not
recomputing logits+logsumexp. Saves ~10ms/chunk vs current, costs ~4ms/chunk
for the element-wise pass.

**Recommended: Option B** for the common case, with **Option C as fallback**.

### Step 2: Restructure forward loop

```python
def fused_linear_xent_entropy_fwd_bwd(hidden, weight, target, chunk_size):
    d_hidden = empty_like(hidden)
    d_weight = zeros(V, H, fp32)

    for chunk:
        logits = h_chunk @ W.T                              # matmul
        loss, ent = ce_entropy_fwd_bwd(logits, target)      # CuTe kernel, writes dlogits in-place
        d_hidden[chunk] = logits @ W                        # logits buffer now contains dlogits
        d_weight += logits.T @ h_chunk                      # accumulate

    return ce, entropy, d_hidden, d_weight
```

### Step 3: Simplify autograd.Function backward

```python
class FusedLinearXentEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, target, bias, chunk_size):
        ce, ent, log_probs, d_hidden, d_weight = fused_fwd_bwd(...)
        ctx.save_for_backward(d_hidden, d_weight)
        return ce, ent, log_probs

    @staticmethod
    def backward(ctx, g_ce, g_ent, g_lp):
        d_hidden, d_weight = ctx.saved_tensors
        # For loss = ce.sum() - ent.sum(), dloss = 1.0
        # d_hidden and d_weight already contain the correct gradients
        # If upstream isn't uniform 1.0, need to scale or fall back
        return d_hidden, d_weight, None, d_bias, None
```

### Step 4: Level 5 extension

For Level 5 (GEMM epilogue), the epilogue kernel already has `p_j` and `log_p_j`
available in registers. Extend `epi_visit_subtile()` to also write dlogits to
the output buffer. Then the driver loop immediately does the d_hidden and d_weight
matmuls.

### Memory Trade-off

Pre-computing gradients means saving during forward:
- `d_hidden [B,H]`: 256 MB at B=32768, H=4096, bf16
- `d_weight [V,H]`: 1 GB at V=128256, H=4096, bf16

Total: ~1.3 GB extra vs current recompute approach (which saves only entropy [B] = 128KB).

This is acceptable — both quack and liger make this trade-off because the backward
speedup (3-4x) far outweighs the memory cost. And peak memory is still much less
than the unfused baseline (65 GB).

### Expected Performance

Per chunk (V=128256, chunk=4096):
- Forward CuTe kernel + dlogits: ~0.8ms (marginal cost over forward-only)
- d_hidden matmul: ~4.8ms
- d_weight matmul: ~4.3ms
- **Total per chunk: ~10ms** (vs ~34ms currently)

Full B=32768, 8 chunks:
- **Fwd+Bwd: ~80ms** for the reduction part
- Plus matmul forward time: ~675ms (L4) or ~68ms (L5)
- **Expected L4 total: ~755ms** (vs 1101ms currently, 1.5x speedup)
- **Expected L5 total: ~148ms** (vs 495ms currently, 3.3x speedup)

## Implementation Files

| File | Changes |
|------|---------|
| `fast_reduction/cute_cross_entropy.py` | Add `ce_entropy_fwd_bwd` variant that outputs dlogits in-place |
| `fast_reduction/kernel.py` | Restructure forward loop to compute d_hidden/d_weight during forward |
| `fast_reduction/gemm_kernel.py` | Extend Level 5 epilogue to output dlogits |
| `fast_reduction/gemm_ce_entropy_epilogue.py` | Add dlogits output to epilogue mixin |
| `tests/test_backward.py` | Update tests for new path |
| `benchmarks/bench_fwd_bwd.py` | Re-benchmark |

## Current Implementation (Phase 2)

### Files Modified

- `fast_reduction/baseline.py` — Added `baseline_linear_xent_entropy_backward()`
- `fast_reduction/kernel.py` — Added `_compute_dlogits_chunk()`, `fused_linear_xent_entropy_backward()`, `FusedLinearXentEntropy`, `fused_linear_xent_entropy_differentiable()`
- `fast_reduction/gemm_kernel.py` — Added `GemmFusedCEEntropy`, `gemm_fused_ce_entropy_differentiable()`
- `fast_reduction/__init__.py` — Updated exports
- `tests/test_backward.py` — 11 tests across 3 classes
- `benchmarks/bench_fwd_bwd.py` — Forward+backward benchmark

### Running Tests

```bash
CUDA_VISIBLE_DEVICES=0 uv run pytest tests/test_backward.py -v
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/bench_fwd_bwd.py
```
