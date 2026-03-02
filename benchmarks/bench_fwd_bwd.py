"""
Benchmark: forward + backward for linear + cross-entropy + entropy

Compares levels with backward pass support:
  1. torch unfused (autograd)
  2. torch chunked (autograd)
  4. CuTe joint (autograd.Function)
  5. GEMM epilogue (autograd.Function)

Reports:
  - Forward-only time (ms)
  - Forward + backward time (ms)
  - Peak memory (GB)
  - Forward CE MAE, Ent MAE (vs fp32 reference)
  - Gradient d_hidden MAE (vs fp32 reference)

Default: B=32768, H=4096, V=128256, bf16, chunk=4096

Usage:
    uv run python benchmarks/bench_fwd_bwd.py
    uv run python benchmarks/bench_fwd_bwd.py --B 4096 --V 128256 --H 4096
    uv run python benchmarks/bench_fwd_bwd.py --no-accuracy
"""

import argparse
import gc
import time

import torch
import torch.nn.functional as F

from fast_reduction.baseline import (
    baseline_linear_xent_entropy,
    baseline_linear_xent_entropy_backward,
)
from fast_reduction.kernel import (
    fused_linear_xent_entropy_differentiable,
)
from fast_reduction.gemm_kernel import (
    gemm_fused_ce_entropy_differentiable,
)


def bytes_to_gb(b: int) -> float:
    return b / (1 << 30)


def benchmark_fn(fn, warmup=3, iters=10):
    """Returns (avg_ms, peak_mem_bytes)."""
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


def fmt_err(val):
    if val is None:
        return "—"
    if val < 0.0001:
        return f"{val:.1e}"
    return f"{val:.4f}"


def report(name, fwd_ms, fwd_bwd_ms, peak, ce_mae=None, ent_mae=None, dh_mae=None):
    print(
        f"  {name:40s}  {fwd_ms:8.1f} ms  {fwd_bwd_ms:8.1f} ms  "
        f"{bytes_to_gb(peak):6.1f} GB  {fmt_err(ce_mae):>10s}  "
        f"{fmt_err(ent_mae):>10s}  {fmt_err(dh_mae):>10s}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=32768, help="batch (num tokens)")
    parser.add_argument("--H", type=int, default=4096, help="hidden dim")
    parser.add_argument("--V", type=int, default=128256, help="vocab size")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--chunk", type=int, default=4096, help="chunk size")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--no-accuracy", action="store_true")
    args = parser.parse_args()

    B, H, V = args.B, args.H, args.V
    chunk = args.chunk
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    device = "cuda"
    torch.manual_seed(42)
    hidden = torch.randn(B, H, device=device, dtype=dtype)
    weight = torch.randn(V, H, device=device, dtype=dtype)
    target = torch.randint(0, V, (B,), device=device, dtype=torch.long)

    # Compute fp32 reference for accuracy
    ce_ref = ent_ref = d_hidden_ref = None
    if not args.no_accuracy:
        print("Computing fp32 ground truth (chunked)...")
        # Chunked to limit peak memory
        ref_chunk = min(2048, B)
        ce_parts, ent_parts, dh_parts = [], [], []
        for start in range(0, B, ref_chunk):
            end = min(start + ref_chunk, B)
            h_cpu = hidden[start:end].cpu()
            w_cpu = weight.cpu()
            t_cpu = target[start:end].cpu()
            ce_r, ent_r, dh_r, _, _ = baseline_linear_xent_entropy_backward(
                h_cpu, w_cpu, t_cpu,
            )
            ce_parts.append(ce_r)
            ent_parts.append(ent_r)
            dh_parts.append(dh_r)
        ce_ref = torch.cat(ce_parts)
        ent_ref = torch.cat(ent_parts)
        d_hidden_ref = torch.cat(dh_parts)
        del ce_parts, ent_parts, dh_parts
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nB={B}  H={H}  V={V}  dtype={args.dtype}  chunk={chunk}")
    print()
    print(
        f"  {'variant':40s}  {'fwd':>8s}  {'fwd+bwd':>8s}  "
        f"{'peak':>6s}  {'CE MAE':>10s}  {'Ent MAE':>10s}  {'dH MAE':>10s}"
    )
    print(
        f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}"
    )

    # ---- 1. torch unfused (autograd) ----
    def run_unfused_fwd():
        h = hidden.detach().requires_grad_(True)
        w = weight.detach().requires_grad_(True)
        logits = F.linear(h, w)
        log_sm = F.log_softmax(logits.float(), dim=-1)
        sm = log_sm.exp()
        ce = F.nll_loss(log_sm, target, reduction="none")
        ent = -(sm * log_sm).sum(dim=-1)
        return ce, ent, h, w

    def run_unfused_fwd_bwd():
        ce, ent, h, w = run_unfused_fwd()
        loss = ce.sum() - ent.sum()
        loss.backward()
        return h, w

    try:
        fwd_ms, _ = benchmark_fn(run_unfused_fwd, args.warmup, args.iters)
        fwd_bwd_ms, peak = benchmark_fn(run_unfused_fwd_bwd, args.warmup, args.iters)
        ce_mae = ent_mae = dh_mae = None
        if ce_ref is not None:
            ce, ent, h, w = run_unfused_fwd()
            (ce.sum() - ent.sum()).backward()
            ce_mae = (ce.float().cpu() - ce_ref).abs().mean().item()
            ent_mae = (ent.float().cpu() - ent_ref).abs().mean().item()
            dh_mae = (h.grad.float().cpu() - d_hidden_ref.float()).abs().mean().item()
        report("1. torch unfused", fwd_ms, fwd_bwd_ms, peak, ce_mae, ent_mae, dh_mae)
    except torch.cuda.OutOfMemoryError:
        print(f"  {'1. torch unfused':40s}  OOM")
        gc.collect()
        torch.cuda.empty_cache()

    # ---- 2. torch chunked (autograd) ----
    def run_chunked_fwd():
        h = hidden.detach().requires_grad_(True)
        w = weight.detach().requires_grad_(True)
        ce_all = []
        ent_all = []
        for start in range(0, B, chunk):
            end = min(start + chunk, B)
            logits = F.linear(h[start:end], w)
            log_sm = F.log_softmax(logits.float(), dim=-1)
            sm = log_sm.exp()
            ce_all.append(F.nll_loss(log_sm, target[start:end], reduction="none"))
            ent_all.append(-(sm * log_sm).sum(dim=-1))
        return torch.cat(ce_all), torch.cat(ent_all), h, w

    def run_chunked_fwd_bwd():
        ce, ent, h, w = run_chunked_fwd()
        loss = ce.sum() - ent.sum()
        loss.backward()
        return h, w

    fwd_ms, _ = benchmark_fn(run_chunked_fwd, args.warmup, args.iters)
    fwd_bwd_ms, peak = benchmark_fn(run_chunked_fwd_bwd, args.warmup, args.iters)
    ce_mae = ent_mae = dh_mae = None
    if ce_ref is not None:
        ce, ent, h, w = run_chunked_fwd()
        (ce.sum() - ent.sum()).backward()
        ce_mae = (ce.float().cpu() - ce_ref).abs().mean().item()
        ent_mae = (ent.float().cpu() - ent_ref).abs().mean().item()
        dh_mae = (h.grad.float().cpu() - d_hidden_ref.float()).abs().mean().item()
    report("2. torch chunked", fwd_ms, fwd_bwd_ms, peak, ce_mae, ent_mae, dh_mae)

    # ---- 4. CuTe joint (autograd.Function) ----
    def run_fused_fwd():
        h = hidden.detach().requires_grad_(True)
        w = weight.detach().requires_grad_(True)
        ce, ent, lp = fused_linear_xent_entropy_differentiable(
            h, w, target, chunk_size=chunk,
        )
        return ce, ent, h, w

    def run_fused_fwd_bwd():
        ce, ent, h, w = run_fused_fwd()
        loss = ce.sum() - ent.sum()
        loss.backward()
        return h, w

    fwd_ms, _ = benchmark_fn(run_fused_fwd, args.warmup, args.iters)
    fwd_bwd_ms, peak = benchmark_fn(run_fused_fwd_bwd, args.warmup, args.iters)
    ce_mae = ent_mae = dh_mae = None
    if ce_ref is not None:
        ce, ent, h, w = run_fused_fwd()
        (ce.sum() - ent.sum()).backward()
        ce_mae = (ce.float().cpu() - ce_ref).abs().mean().item()
        ent_mae = (ent.float().cpu() - ent_ref).abs().mean().item()
        dh_mae = (h.grad.float().cpu() - d_hidden_ref.float()).abs().mean().item()
    report("4. CuTe joint (autograd)", fwd_ms, fwd_bwd_ms, peak, ce_mae, ent_mae, dh_mae)

    # ---- 5. GEMM epilogue (autograd.Function) ----
    def run_gemm_fwd():
        h = hidden.detach().requires_grad_(True)
        w = weight.detach().requires_grad_(True)
        ce, ent, lp = gemm_fused_ce_entropy_differentiable(
            h, w, target, chunk_size=chunk,
        )
        return ce, ent, h, w

    def run_gemm_fwd_bwd():
        ce, ent, h, w = run_gemm_fwd()
        loss = ce.sum() - ent.sum()
        loss.backward()
        return h, w

    fwd_ms, _ = benchmark_fn(run_gemm_fwd, args.warmup, args.iters)
    fwd_bwd_ms, peak = benchmark_fn(run_gemm_fwd_bwd, args.warmup, args.iters)
    ce_mae = ent_mae = dh_mae = None
    if ce_ref is not None:
        ce, ent, h, w = run_gemm_fwd()
        (ce.sum() - ent.sum()).backward()
        ce_mae = (ce.float().cpu() - ce_ref).abs().mean().item()
        ent_mae = (ent.float().cpu() - ent_ref).abs().mean().item()
        dh_mae = (h.grad.float().cpu() - d_hidden_ref.float()).abs().mean().item()
    report("5. GEMM epilogue (autograd)", fwd_ms, fwd_bwd_ms, peak, ce_mae, ent_mae, dh_mae)

    print()


if __name__ == "__main__":
    main()
