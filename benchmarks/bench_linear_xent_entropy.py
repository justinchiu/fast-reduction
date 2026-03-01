"""
Benchmark: linear + cross-entropy + entropy

Compares 4 levels of fusion:
  1. torch unfused        — full [B,V] logits, separate ops
  2. torch chunked        — chunked [chunk,V], separate ops
  3. quack separate       — chunked matmul + 2 CuTe kernels (xent, entropy)
  4. quack joint          — chunked matmul + 1 CuTe kernel (xent+entropy)

Reports wall-clock time, peak memory, and model memory throughput.

Usage:
    uv run python benchmarks/bench_linear_xent_entropy.py
    uv run python benchmarks/bench_linear_xent_entropy.py --B 65536 --V 128256 --H 4096
"""

import argparse
import gc
import time

import torch
import torch.nn.functional as F

from fast_reduction.baseline import (
    baseline_linear_xent_entropy,
    chunked_linear_xent_entropy,
)
from fast_reduction.kernel import fused_linear_xent_entropy, separate_linear_xent_entropy


def bytes_to_gb(b: int) -> float:
    return b / (1 << 30)


def bytes_to_mb(b: int) -> float:
    return b / (1 << 20)


def model_memory_bytes(B: int, V: int, H: int, dtype_bytes: int) -> int:
    """Minimum bytes through HBM for fused linear+xent+entropy.

    Read:  hidden [B,H] + weight [V,H] + target [B]
    Write: ce_loss [B] + entropy [B] + log_probs [B]
    """
    read_bytes = B * H * dtype_bytes + V * H * dtype_bytes + B * 8
    write_bytes = B * 4 * 3
    return read_bytes + write_bytes


def benchmark_fn(fn, warmup=5, iters=20):
    """Returns (avg_ms, peak_mem_bytes)."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / iters
    peak = torch.cuda.max_memory_allocated()
    return elapsed_ms, peak


def report(name, ms, peak, min_bytes):
    bw = min_bytes / (ms / 1000) / 1e9 if ms > 0 else 0
    print(f"  {name:40s}  {ms:8.2f} ms  {bytes_to_mb(peak):8.0f} MB  {bw:6.0f} GB/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=32768, help="batch (num tokens)")
    parser.add_argument("--H", type=int, default=4096, help="hidden dim")
    parser.add_argument("--V", type=int, default=128256, help="vocab size")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--chunk", type=int, default=4096, help="chunk size for chunked variants")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    B, H, V = args.B, args.H, args.V
    chunk = args.chunk
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    dtype_bytes = 4 if dtype == torch.float32 else 2

    device = "cuda"
    hidden = torch.randn(B, H, device=device, dtype=dtype)
    weight = torch.randn(V, H, device=device, dtype=dtype)
    target = torch.randint(0, V, (B,), device=device, dtype=torch.long)

    min_bytes = model_memory_bytes(B, V, H, dtype_bytes)
    logits_size = B * V * dtype_bytes

    print(f"B={B}  H={H}  V={V}  dtype={args.dtype}  chunk={chunk}")
    print(f"Full logits [B,V] = {bytes_to_gb(logits_size):.2f} GB")
    print(f"Chunk logits [chunk,V] = {bytes_to_mb(chunk * V * dtype_bytes):.0f} MB")
    print(f"Minimum model bytes = {bytes_to_mb(min_bytes):.0f} MB")
    print(f"H100 HBM3 peak = 3350 GB/s")
    print()
    print(f"  {'variant':40s}  {'time':>8s}  {'peak mem':>8s}  {'model BW':>8s}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*8}")

    # 1. torch unfused
    def run_unfused():
        return baseline_linear_xent_entropy(hidden, weight, target)
    ms, peak = benchmark_fn(run_unfused, args.warmup, args.iters)
    report("1. torch unfused", ms, peak, min_bytes)

    # 2. torch chunked
    def run_chunked():
        return chunked_linear_xent_entropy(hidden, weight, target, chunk_size=chunk)
    ms, peak = benchmark_fn(run_chunked, args.warmup, args.iters)
    report("2. torch chunked", ms, peak, min_bytes)

    # 3. CuTe separate (xent kernel + entropy kernel, two reads of logits)
    def run_separate():
        return separate_linear_xent_entropy(hidden, weight, target, chunk_size=chunk)
    ms, peak = benchmark_fn(run_separate, args.warmup, args.iters)
    report("3. CuTe separate xent + entropy", ms, peak, min_bytes)

    # 4. CuTe joint xent+entropy (one read of logits)
    def run_joint():
        return fused_linear_xent_entropy(hidden, weight, target, chunk_size=chunk)
    ms, peak = benchmark_fn(run_joint, args.warmup, args.iters)
    report("4. CuTe joint xent+entropy", ms, peak, min_bytes)

    print()


if __name__ == "__main__":
    main()
