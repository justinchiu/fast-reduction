# Level 5 Plan: GEMM Epilogue Fusion

Objective: implement level 5 where CE + entropy reductions happen in GEMM epilogue, so logits do not round-trip through HBM.

## Target Outcome

- Replace level 4 pipeline:
  - `torch.mm` writes `[chunk, V]` logits
  - separate reduction kernel reads logits
- With level 5 pipeline:
  - GEMM accumulator fragments feed epilogue visitor
  - epilogue writes only `{ce_loss, entropy, log_probs}` per token

## Key Idea

Compute row-wise online softmax statistics inside epilogue:

- running max: `m`
- running sum exp: `s = sum(exp(z - m))`
- running weighted sum: `sx = sum(z * exp(z - m))`
- tracked target logit: `z_y`

After full row reduction:
- `lse = m + log(s)`
- `ce = lse - z_y`
- `ent = lse - sx / s`
- `log_prob = -ce`

This preserves level 4 math while removing logits HBM traffic.

## Implementation Plan

### Phase 0: Integration Spike

1. Inspect/confirm quack epilogue extension points (`gemm_sm90.py`, `epi_visit_subtile()`).
2. Implement a minimal epilogue visitor that captures per-row `max` only.
3. Validate execution/compilation stability for one fixed shape.

Exit criteria:
- Visitor hook is proven and testable in this repo.

### Phase 1: CE-Only Epilogue

1. Add row-state structs for `m`, `s`, and `z_y`.
2. Process each output subtile with numerically stable online updates.
3. Emit CE output only (`loss[B]`).
4. Compare against current CE forward on representative shapes.

Exit criteria:
- CE parity within expected tolerance for fp32/bf16 modes.

### Phase 2: Joint CE + Entropy Epilogue

1. Extend row-state with `sx`.
2. Emit both CE and entropy outputs from final row state.
3. Keep optional `log_probs` output as `-ce`.

Exit criteria:
- Parity with level 4 (`ce_entropy_fwd`) for both outputs.

### Phase 3: Layout + Reduction Strategy

1. Ensure each reduced row has complete `V` coverage in epilogue context.
2. If row is split across CTAs, add multi-CTA reduction strategy:
   - preferred: cluster/shared reduction path
   - fallback: two-stage partial-stat reduction (still no full logits write)
3. Validate numerical stability for large `V` (e.g. 128256).

Exit criteria:
- Correctness maintained with production tile shapes.

### Phase 4: End-to-End Level 5 API

1. Add `level5` entrypoint in `fast_reduction/kernel.py` (or parallel module).
2. Wire into benchmark script as a fifth measured variant.
3. Document constraints (supported dtypes/shapes, required architecture).

Exit criteria:
- Benchmarks report level 5 next to levels 1-4.

## Validation Matrix

- Dtypes: fp32, bf16 input paths.
- Shapes: small correctness + large benchmark shapes.
- Features: bias/no-bias.
- Metrics:
  - numerical parity vs level 4
  - wall time
  - peak memory
  - model bandwidth estimate

## Main Risks

- Epilogue state lifetime and synchronization when rows span multiple blocks.
- Register pressure from added row statistics reducing occupancy.
- Compile/runtime complexity in CuTe/Quack integration.

## Suggested Order of Work

1. CE-only epilogue
2. Joint CE+entropy epilogue
3. Multi-CTA row reduction robustness
4. Benchmark and tune

