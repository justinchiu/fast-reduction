# fast-reduction

## Goal

Implement a **fused linear + cross-entropy + entropy** kernel for H100 (Hopper, sm90) that hits
>= 90 % of H100 HBM3 peak throughput (3.35 TB/s).

## Hardware

8 x H100 80 GB (Hopper, sm90, HBM3).

## Performance ladder

Five levels of fusion, measured at B=32768, H=4096, V=128256, bf16, chunk=4096:

| Level | Variant | Time | Peak mem | Approach |
|-------|---------|------|----------|----------|
| 1 | torch unfused | 711 ms | 65 GB | Full [B,V] logits, separate F.cross_entropy + entropy |
| 2 | torch chunked | 723 ms | 11 GB | Chunked [chunk,V], same PyTorch ops |
| 3 | CuTe separate | 680 ms | 5.4 GB | Chunked matmul + 2 CuTe kernels (CE, entropy) |
| 4 | CuTe joint | 675 ms | 5.4 GB | Chunked matmul + 1 CuTe kernel (CE+entropy) |
| 5 | GEMM epilogue | — | — | Reduction fused into GEMM epilogue (logits never hit HBM) |

**CuTe reduction kernels alone** (no matmul, B=4096, V=128256, fp32):

| Kernel | Time | BW | % of H100 peak |
|--------|------|----|-----------------|
| CE only | 0.70 ms | 2987 GB/s | 89% |
| entropy only | 0.77 ms | 2740 GB/s | 82% |
| joint CE+entropy | 0.76 ms | 2770 GB/s | 83% |

The reduction kernels are near speed of light. The bottleneck in levels 1-4 is the
`torch.mm` matmul (~34 TFLOP per call at these sizes). Level 5 eliminates the logits
HBM round-trip entirely by computing reductions in the GEMM epilogue.

## Architecture: what's fused and what's not

Levels 1-4 all do **chunked matmul** (`torch.mm` via cuBLAS) followed by a
separate **CuTe DSL reduction kernel**. The logits `[chunk, V]` buffer is
reused each iteration, so peak memory is O(chunk_size * V) instead of O(B * V).
But the logits still go through HBM between the matmul and the reduction.

Level 5 would fuse the reduction into the GEMM epilogue so logits stay in
registers/SMEM and never touch HBM. Quack has the infrastructure for this
(`gemm_sm90.py` extensible epilogue with `epi_visit_subtile()` hooks) but
has not implemented a cross-entropy or entropy epilogue.

## Key files

```
fast_reduction/
  __init__.py
  baseline.py              pure-PyTorch reference (unfused + chunked variants)
  kernel.py                chunked matmul + CuTe reduction (levels 3 & 4)
  cute_cross_entropy.py    CuTe DSL kernels: CE-only, entropy-only, joint CE+entropy
  reduce.py                reduction primitives (thread → warp → block → cluster)
  dsl_utils.py             low-level PTX: DSMEM, f32↔i64 packing, pointer arithmetic

benchmarks/
  bench_linear_xent_entropy.py   wall-clock, peak mem, model BW (all 4 levels)

tests/
  test_ce_impls.py         correctness tests vs PyTorch reference

docs/
  memory_bound_kernels.md  reference blogpost (Guo, Zadouri, Dao)
```

## Testing

```bash
uv run pytest
uv run pytest tests/test_ce_impls.py -v
```

## Benchmarking

```bash
uv run python benchmarks/bench_linear_xent_entropy.py
uv run python benchmarks/bench_linear_xent_entropy.py --B 65536 --V 128256 --H 4096
```

## API

```python
from fast_reduction.kernel import fused_linear_xent_entropy, separate_linear_xent_entropy

# Level 4: joint CE+entropy (one CuTe kernel per chunk)
ce_loss, entropy, log_probs = fused_linear_xent_entropy(
    hidden_states,  # [B, H]  bf16 or fp32
    weight,         # [V, H]
    target,         # [B]     torch.long
    bias=None,      # [V]     optional
    chunk_size=4096,
)

# Level 3: separate CE + entropy (two CuTe kernels per chunk)
ce_loss, entropy, log_probs = separate_linear_xent_entropy(
    hidden_states, weight, target, chunk_size=4096,
)
```

Ground truth: `fast_reduction.baseline_linear_xent_entropy` (same signature minus chunk_size).

## Dependencies

```toml
[project]
dependencies = [
    "nvidia-cutlass-dsl>=4.4.1",
    "apache-tvm-ffi",
    "torch>=2.5",
]
```

## Reference

- Quack repo (Dao-AILab/quack) — CuTe DSL cross-entropy, softmax, and GEMM kernels. Clone at ~/quack.
- Quack GEMM epilogue framework: `quack/gemm_sm90.py` (extensible mixin with `epi_visit_subtile()`)
- Quack epilogue examples: `gemm_act.py` (activation fusion), `gemm_dact.py` (backward activation)
