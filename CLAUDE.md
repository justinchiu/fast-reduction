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
| 5 | GEMM epilogue | 68 ms | 1.3 GB | Reduction fused into GEMM epilogue (logits never hit HBM) |

**CuTe reduction kernels alone** (no matmul, B=4096, V=128256, fp32):

| Kernel | Time | BW | % of H100 peak |
|--------|------|----|-----------------|
| CE only | 0.70 ms | 2987 GB/s | 89% |
| entropy only | 0.77 ms | 2740 GB/s | 82% |
| joint CE+entropy | 0.76 ms | 2770 GB/s | 83% |

The reduction kernels are near speed of light. The bottleneck in levels 1-4 is the
`torch.mm` matmul (~34 TFLOP per call at these sizes). Level 5 eliminates the logits
HBM round-trip entirely by computing reductions in the GEMM epilogue.

**Forward + backward** (B=32768, H=4096, V=128256, bf16, chunk=4096):

| Level | Variant | Fwd+Bwd | Peak mem | Notes |
|-------|---------|---------|----------|-------|
| 1 | torch unfused | OOM | — | Full [B,V] logits |
| 2 | torch chunked | 309 ms | 38.5 GB | Autograd saves logits per chunk |
| 4 | CuTe joint | 351 ms | 13.3 GB | Recomputes logits in backward |
| 5 | GEMM epilogue | 345 ms | 13.3 GB | Fwd: GEMM epilogue, Bwd: recompute |

The backward recomputes logits via cuBLAS (1 extra matmul per chunk),
trading ~80ms of compute for **25 GB less peak memory** (13.3 vs 38.5 GB).
At B=4096 (single chunk), Level 5 fwd+bwd = 49ms vs torch chunked = 38ms.

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
  baseline.py              pure-PyTorch reference (unfused + chunked + backward)
  kernel.py                chunked matmul + CuTe reduction (levels 3 & 4) + backward + autograd
  cute_cross_entropy.py    CuTe DSL kernels: CE-only, entropy-only, joint CE+entropy
  reduce.py                reduction primitives (thread → warp → block → cluster)
  cute_utils.py             low-level PTX: DSMEM, f32↔i64 packing, pointer arithmetic
  gemm_ce_entropy_epilogue.py   Level 5 GEMM epilogue mixin
  gemm_ce_entropy_finalize.py   Level 5 finalization kernel
  gemm_kernel.py                Level 5 driver + backward + autograd

benchmarks/
  bench_linear_xent_entropy.py   forward-only: wall-clock, peak mem, model BW, accuracy
  bench_fwd_bwd.py               forward + backward: time, peak mem, gradient accuracy
  profile.sh                     Nsight Compute profiling script

tests/
  test_ce_impls.py         forward correctness tests vs PyTorch reference
  test_backward.py         backward gradient correctness tests vs fp32 reference
  test_entropy_large_v.py  large-V entropy accuracy tests (cluster reduction)

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
# Forward-only benchmark
uv run python benchmarks/bench_linear_xent_entropy.py
uv run python benchmarks/bench_linear_xent_entropy.py --B 32768 --V 128256 --H 4096
uv run python benchmarks/bench_linear_xent_entropy.py --no-accuracy   # speed/memory only

# Forward + backward benchmark
uv run python benchmarks/bench_fwd_bwd.py
uv run python benchmarks/bench_fwd_bwd.py --B 4096 --V 128256 --H 4096
```

## API

```python
from fast_reduction.kernel import fused_linear_xent_entropy, separate_linear_xent_entropy

# Level 4: joint CE+entropy (one CuTe kernel per chunk) — forward only
ce_loss, entropy, log_probs = fused_linear_xent_entropy(
    hidden_states,  # [B, H]  bf16 or fp32
    weight,         # [V, H]
    target,         # [B]     torch.long
    bias=None,      # [V]     optional
    chunk_size=4096,
)

# Level 3: separate CE + entropy (two CuTe kernels per chunk) — forward only
ce_loss, entropy, log_probs = separate_linear_xent_entropy(
    hidden_states, weight, target, chunk_size=4096,
)
```

### Differentiable API (supports .backward())

```python
from fast_reduction.kernel import fused_linear_xent_entropy_differentiable
from fast_reduction.gemm_kernel import gemm_fused_ce_entropy_differentiable

# Level 4: differentiable (autograd.Function)
hidden.requires_grad_(True)
weight.requires_grad_(True)
ce_loss, entropy, log_probs = fused_linear_xent_entropy_differentiable(
    hidden, weight, target, bias=None, chunk_size=4096,
)
loss = ce_loss.sum() - entropy.sum()
loss.backward()  # hidden.grad, weight.grad populated

# Level 5: differentiable (autograd.Function)
ce_loss, entropy, log_probs = gemm_fused_ce_entropy_differentiable(
    hidden, weight, target, bias=None, chunk_size=4096,
)
```

The backward recomputes logits per chunk (no logits stored in memory),
computes dlogits element-wise in fp32, then uses bf16 matmul for
d_hidden and d_weight accumulation.

Ground truth: `fast_reduction.baseline_linear_xent_entropy` (same signature minus chunk_size).
Ground truth backward: `fast_reduction.baseline_linear_xent_entropy_backward`.

## Dependencies

```toml
[project]
dependencies = [
    "nvidia-cutlass-dsl>=4.4.1",
    "apache-tvm-ffi",
    "torch>=2.5",
]
```

## GPU Usage

This machine has 8 x H100 GPUs. Always use an idle GPU for experiments to avoid
interfering with other work.

### Check GPU utilization

```bash
# Quick check: shows index, utilization %, memory used/total
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

# Full dashboard
nvidia-smi
```

### Run on a specific unused GPU

```bash
# Run on GPU 1
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/bench_linear_xent_entropy.py

# Run on GPU 3
CUDA_VISIBLE_DEVICES=3 uv run pytest tests/test_ce_impls.py -v
```

Always check `nvidia-smi` first, pick a GPU with 0% utilization and 0 MB used,
then prefix your command with `CUDA_VISIBLE_DEVICES=<gpu_id>`.

## Reference

- Quack repo (Dao-AILab/quack) — CuTe DSL cross-entropy, softmax, and GEMM kernels. Clone at ~/quack.
- Quack GEMM epilogue framework: `quack/gemm_sm90.py` (extensible mixin with `epi_visit_subtile()`)
- Quack epilogue examples: `gemm_act.py` (activation fusion), `gemm_dact.py` (backward activation)
