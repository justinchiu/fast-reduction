# Progress

## Current State (2026-03-01)

All 5 levels are implemented and tests pass (8/8).

### Benchmark Results (B=32768, H=4096, V=128256, bf16, chunk=4096)

| Level | Variant | Time | Peak mem | Matmul precision |
|-------|---------|------|----------|-----------------|
| 1 | torch unfused | 711 ms | 65 GB | fp32 |
| 2 | torch chunked | 723 ms | 11 GB | fp32 |
| 3 | CuTe separate | 680 ms | 5.4 GB | fp32 |
| 4 | CuTe joint | 675 ms | 5.4 GB | fp32 |
| **5** | **GEMM epilogue fused** | **68 ms** | **1.3 GB** | **bf16 (WGMMA)** |

Level 5 is ~10x faster and uses ~4x less memory.

### Accuracy Analysis

Level 5's apparent "error" (0.003 CE max error vs fp32 reference) is NOT a bug.
It's the inherent precision difference between bf16 and fp32 matmul:

| Comparison | CE max error | Explanation |
|-----------|-------------|-------------|
| Level 4 vs fp32 ref | 0.00006 | Level 4 does **fp32 matmul** (casts bf16→fp32 before torch.mm) |
| Level 5 vs fp32 ref | 0.003 | WGMMA does **bf16×bf16→fp32 accum** — less precise inputs |
| cuBLAS bf16 vs fp32 ref | 1.22 | bf16 output truncation loses much more |
| Level 5 epilogue partials vs fp32 matmul partials | 0.00002 | The **reduction logic is correct** |

Key findings:
- Level 5's GEMM epilogue reduction is numerically correct (partials match fp32 matmul to 5 decimal places)
- The 0.003 error comes entirely from bf16 input multiplication precision (mantissa truncation before multiply), not from the reduction
- Level 5 is actually **more precise** than standard bf16 matmul (1.22 error) because the fp32 accumulator is never truncated to bf16
- Level 4 uses fp32 matmul (`h_chunk.float() @ weight.float().t()`), which is why it appears more accurate but is slower

### Known Issues

1. **Levels 1-4 use fp32 matmul**: `kernel.py` casts bf16 to fp32 before `torch.mm`.
   This makes them slower than necessary. Should be changed to bf16 matmul for
   a fair comparison, but note this will increase their CE error from 0.00006 to ~1.2.

2. **Level 4 entropy comparison shows NaN**: Pre-existing issue in the entropy
   output of `fused_linear_xent_entropy` at large V. Not related to Level 5.

3. **Benchmark fairness**: Level 5 (bf16 WGMMA) vs Levels 1-4 (fp32 cuBLAS) is
   not apples-to-apples. The 10x speedup is partly from bf16 vs fp32 matmul,
   not just epilogue fusion. Need to benchmark with bf16 matmul for Levels 1-4.

### Architecture

```
Level 5 data flow:
  hidden [B,H] bf16 ──┐
                       ├─ WGMMA (bf16×bf16→fp32 accum) ─→ epilogue reduces
  weight [V,H] bf16 ──┘     in registers, never hits HBM    accumulator to
                                                              4 partials/row
                                                                    │
                                   finalize kernel ←────────────────┘
                                   merges N_tiles partials via online softmax
                                        │
                               loss[B], entropy[B]
```

### Files

| File | Purpose | Status |
|------|---------|--------|
| `fast_reduction/gemm_ce_entropy_epilogue.py` | GEMM epilogue mixin | Done |
| `fast_reduction/gemm_ce_entropy_finalize.py` | Finalization kernel | Done |
| `fast_reduction/gemm_kernel.py` | Level 5 driver | Done |
| `fast_reduction/kernel.py` | Levels 3-4 driver | Done |
| `fast_reduction/cute_cross_entropy.py` | CuTe CE/entropy kernels | Done |
| `fast_reduction/reduce.py` | Reduction primitives | Done |
| `fast_reduction/cute_utils.py` | PTX utilities | Done |
| `tests/test_ce_impls.py` | Correctness tests (8/8 pass) | Done |
| `benchmarks/bench_linear_xent_entropy.py` | All 5 levels benchmark | Done |

### Next Steps

- [ ] Fix levels 1-4 to use bf16 matmul for fair comparison
- [ ] Re-benchmark with bf16 matmul across all levels
- [ ] Investigate Level 4 entropy NaN at large V
- [ ] Nsight Compute profiling of Level 5
- [ ] Backward pass implementation
