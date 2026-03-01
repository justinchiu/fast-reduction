# fast-reduction

CuTe DSL + PyTorch implementations for `linear + cross-entropy + entropy` on Hopper (H100, `sm90`), with a benchmark ladder from unfused to increasingly fused variants.

## Status

The performance ladder has 5 levels:

1. torch unfused
2. torch chunked
3. CuTe separate (`xent` kernel + `entropy` kernel)
4. CuTe joint (`xent+entropy` in one kernel)
5. GEMM epilogue fusion

Implemented in this repo: levels `1-4`.
Level `5` is planned (not implemented yet).

## Repository Layout

```text
fast_reduction/
  __init__.py
  baseline.py              # PyTorch references (unfused + chunked)
  kernel.py                # chunked matmul + CuTe reductions (levels 3 & 4)
  cute_cross_entropy.py    # CuTe kernels: CE-only, entropy-only, joint
  reduce.py                # reduction primitives
  dsl_utils.py             # low-level DSL/PTX helpers

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

`pyproject.toml` depends on:
- `torch>=2.5`
- `nvidia-cutlass-dsl>=4.4.1`
- `apache-tvm-ffi>=0.1.9`

## Run Tests

```bash
uv run pytest
uv run pytest tests/test_ce_impls.py -v
```

## Run Benchmarks

```bash
uv run python benchmarks/bench_linear_xent_entropy.py
uv run python benchmarks/bench_linear_xent_entropy.py --B 65536 --V 128256 --H 4096
```

## API

```python
from fast_reduction.baseline import baseline_linear_xent_entropy
from fast_reduction.kernel import fused_linear_xent_entropy, separate_linear_xent_entropy

# Level 4: one CuTe kernel per chunk (joint CE + entropy)
ce_loss, entropy, log_probs = fused_linear_xent_entropy(
    hidden_states,  # [B, H] or [..., H]
    weight,         # [V, H]
    target,         # [B] or [...]
    bias=None,      # optional [V]
    chunk_size=4096,
)

# Level 3: two CuTe kernels per chunk (separate CE, entropy)
ce_loss, entropy, log_probs = separate_linear_xent_entropy(
    hidden_states, weight, target, chunk_size=4096,
)

# Numerical reference
ce_ref, ent_ref, lp_ref = baseline_linear_xent_entropy(hidden_states, weight, target)
```

## Notes

- Levels 1-4 still use `torch.mm` for chunked matmul and run reductions afterward.
- Level 5 would fuse reduction into GEMM epilogue so logits do not round-trip through HBM.
