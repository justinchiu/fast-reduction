# Backward Plan (After Forward Completion)

Current direction: finish forward implementations first, then implement backward in phases.

## Goal

Add correct and efficient backward for `linear + cross-entropy + entropy` with the same chunked memory strategy used in forward.

Outputs to support:
- `d_hidden` (same shape as `hidden_states`)
- `d_weight` (same shape as `weight`)
- `d_bias` when bias is present

## Constraints

- Keep memory as `O(chunk_size * V)` (no full `[B, V]` materialization).
- Match PyTorch autograd numerics for bf16/fp32 forward usage.
- Preserve current forward APIs.

## Math to Implement

Per-token logits: `z`, target: `y`, softmax: `p`.

- CE gradient wrt logits:
  `dL_ce/dz = p - one_hot(y)`
- Entropy:
  `H = -sum_j p_j log p_j`
- Entropy gradient wrt logits:
  `dH/dz_j = -p_j * (log p_j + H)`

With upstream grads `g_ce`, `g_ent` (shape `[B]`):

`dlogits = g_ce * (p - one_hot(y)) + g_ent * (-p * (log p + H))`

Then per chunk:
- `d_hidden_chunk = dlogits_chunk @ weight`
- `d_weight += dlogits_chunk.T @ hidden_chunk`
- `d_bias += row_sum(dlogits_chunk)` (if bias exists)

## Phased Execution

### Phase 1: Reference Backward Path (Correctness First)

1. Add a reference backward implementation using torch ops only.
2. Wrap in a custom `torch.autograd.Function` that reuses current forward outputs and saved tensors.
3. Verify gradients against a full torch baseline on small/medium shapes.

Exit criteria:
- Max error thresholds met for `d_hidden`, `d_weight`, `d_bias` across fp32/bf16.
- Works for 2D and 3D hidden shapes.

### Phase 2: CuTe Backward Reduction Kernel

1. Add a kernel that computes `dlogits` from logits/target/upstream grads (CE+entropy joint).
2. Reuse chunked loop from `kernel.py`; keep a reusable `dlogits` buffer.
3. Use GEMM calls for `d_hidden` and `d_weight` accumulation per chunk.

Exit criteria:
- Numerically matches Phase 1.
- No extra full-logits allocations.

### Phase 3: Performance Tuning

1. Tune chunk size and accumulation dtype (fp32 accumulation; cast outputs as needed).
2. Reduce launch overhead (preallocated outputs, minimized temporary tensors).
3. Add benchmark rows for backward-only and forward+backward.

Exit criteria:
- Clear speedup over pure torch backward baseline.
- Stable memory footprint under large `B, V`.

## Tests to Add

- Gradient parity test vs torch baseline (`d_hidden`, `d_weight`, `d_bias`).
- Bias/no-bias coverage.
- Non-contiguous input handling expectations (or explicit `.contiguous()` policy).
- Mixed precision coverage (bf16 inputs with fp32 accumulation).

## Risks

- Entropy gradient numerical sensitivity for large logits.
- Backward kernel compile/cache behavior across multiple vocab sizes.
- Accumulation precision and determinism tradeoffs for `d_weight`.

