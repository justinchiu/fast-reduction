#!/usr/bin/env bash
# Nsight Compute profiling for the fused linear + CE + entropy benchmark.
# Usage: bash benchmarks/profile.sh
set -euo pipefail

ncu --set full \
    --kernel-name "regex:kernel" \
    -o profile \
    uv run python benchmarks/bench_linear_xent_entropy.py --iters 1 --warmup 1
