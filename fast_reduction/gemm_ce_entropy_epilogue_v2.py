"""
GEMM epilogue mixin v2 for fused CE+entropy (Level 5.2).

Two-pass epilogue for improved numerical precision over v5.1:

Pass 1 — max + target extraction (no exp):
  For each epi_n subtile, load accumulator → SMEM, threads read rows and update
  row_max via fmax (exact) and extract target_logit.

Pass 2 — sums using final CTA max (no merge):
  For each epi_n subtile, reload accumulator → SMEM, threads compute
  exp(val - row_max) using the true CTA-wide max. No online merge needed,
  no fastmath on exp2.

This eliminates the two main error sources from v5.1:
  1. Online softmax merge adjustment factors (exp of differences)
  2. fastmath=True approximation on exp2

Cost: 2× R2S copies + barriers (16 vs 8 for n_epi_n=8). The epilogue is not
on the critical path for performance.

Writes the same [N_tiles, M, 4] partials buffer as v5.1, reusing the same
finalization kernel.
"""

import math
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_og
from cutlass import Int32, Float32, Boolean, const_expr
from cutlass.utils import LayoutEnum

from quack.cute_dsl_utils import ArgumentsBase, ParamsBase
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_sm90 import GemmSm90
from quack.varlen_utils import VarlenManager
import quack.sm90_utils as sm90_utils
import quack.copy_utils as copy_utils


class GemmCEEntropyV2EpiMixin(GemmDefaultEpiMixin):
    """Custom epilogue v2: two-pass reduction for improved numerical precision."""

    num_epi_tensormaps: int = 0

    @dataclass
    class EpilogueArguments(ArgumentsBase):
        mTarget: cute.Tensor      # (M,) int64 target indices
        mPartials: cute.Tensor    # (N_tiles, M, 4) fp32 partial buffer
        vocab_size: Int32         # V (for N-boundary checking)
        total_M: Int32            # M (for M-boundary checking)

    @dataclass
    class EpilogueParams(ParamsBase):
        mTarget: cute.Tensor
        mPartials: cute.Tensor
        vocab_size: Int32
        total_M: Int32
        scratch_layout_staged: cute.ComposedLayout

    def epi_to_underlying_arguments(
        self, args: EpilogueArguments, *, loc=None, ip=None
    ) -> EpilogueParams:
        scratch_layout_staged = sm90_utils.make_smem_layout(
            Float32, LayoutEnum.ROW_MAJOR, self.epi_tile, stage=self.epi_stage
        )
        return self.EpilogueParams(
            mTarget=args.mTarget,
            mPartials=args.mPartials,
            vocab_size=args.vocab_size,
            total_M=args.total_M,
            scratch_layout_staged=scratch_layout_staged,
        )

    @staticmethod
    def epi_smem_bytes_per_stage(
        args: "GemmCEEntropyV2EpiMixin.EpilogueArguments",
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: cute.Tile,
    ) -> int:
        return cute.size(epi_tile) * (Float32.width // 8)

    def epi_get_smem_struct(self, params: EpilogueParams):
        scratch_size = cute.cosize(params.scratch_layout_staged)

        @cute.struct
        class EpiSharedStorage:
            sScratch: cute.struct.Align[
                cute.struct.MemRange[Float32, scratch_size],
                self.buffer_align_bytes,
            ]

        return EpiSharedStorage

    def epi_get_smem_tensors(
        self, params: EpilogueParams, storage
    ) -> Tuple[cute.Tensor, ...]:
        sScratch = storage.epi.sScratch.get_tensor(
            params.scratch_layout_staged.outer,
            swizzle=params.scratch_layout_staged.inner,
        )
        return (sScratch,)

    def epi_get_tma_atoms(
        self, params: EpilogueParams, *, loc=None, ip=None
    ) -> list[cute.CopyAtom]:
        return []

    def epi_get_tensormap_update_shapes_orders(
        self,
        params: EpilogueParams,
        cu_seqlens_m: Optional[cute.Tensor],
        batch_idx: Int32,
        *,
        loc=None,
        ip=None,
    ) -> tuple[list[Int32], list[int]]:
        return [], []

    @cute.jit
    def epilogue(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Tuple[cute.Tensor, ...],
        tma_desc_epi_ptrs: list[Optional[cute.Pointer]],
        epi_pipeline: cutlass.pipeline.PipelineAsync,
        epi_store_pipeline: cutlass.pipeline.PipelineAsync,
        epi_read_state: cutlass.pipeline.PipelineState,
        epi_producer_state: Optional[cutlass.pipeline.PipelineState],
        epi_tile: cute.Tile,
        load_acc_subtile: Callable,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor],
        tiled_copy_t2r: Optional[cute.TiledCopy],  # Sm100 only
        tiled_copy_r2s: cute.TiledCopy,
        tRS_sD: Optional[cute.Tensor],
        tiled_copy_s2r: Optional[cute.TiledCopy],
        tSR_rC: Optional[cute.Tensor],
        tSR_sC: Optional[cute.Tensor],
        copy_D: Optional[Callable],
        copy_C: Optional[Callable],
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tile_scheduler,
        tidx: Int32,
        is_tma_warp: Boolean,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:

        (sScratch,) = epi_smem_tensors

        # ---- Set up Float32 R2S copy targeting scratch SMEM ----
        copy_atom_r2s = sm90_utils_og.sm90_get_smem_store_op(
            LayoutEnum.ROW_MAJOR, elem_ty_d=Float32, elem_ty_acc=self.acc_dtype
        )
        tiled_copy_scratch_r2s = cute.make_tiled_copy_S(
            copy_atom_r2s, tiled_copy_r2s
        )
        thr_copy_scratch = tiled_copy_scratch_r2s.get_slice(tidx)
        tRS_sScratch = thr_copy_scratch.partition_D(sScratch)

        # ---- Tile iteration setup ----
        tile_M = self.cta_tile_shape_mnk[0]
        tile_N = self.cta_tile_shape_mnk[1]
        epi_M = epi_tile[0]
        epi_N = epi_tile[1]
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        n_epi_m = epi_tile_shape[0]   # number of M chunks
        n_epi_n = epi_tile_shape[1]   # number of N chunks per M range

        m_tile = tile_coord_mnkl[0]
        n_tile = tile_coord_mnkl[1]
        m_offset = m_tile * tile_M    # global M start for this CTA
        n_offset = n_tile * tile_N    # global N start for this CTA

        V = params.vocab_size
        total_M = params.total_M
        log2_e = Float32(math.log2(math.e))

        # ---- Nested loop: M ranges × two passes over N subtiles ----
        for epi_m in cutlass.range_constexpr(n_epi_m):

            # ============================================================
            # PASS 1: Find true row max + extract target logit (no exp)
            # ============================================================
            row_max = -Float32.inf
            row_target_logit = Float32(0.0)

            for epi_n in cutlass.range_constexpr(n_epi_n):
                epi_idx = epi_m * n_epi_n + epi_n

                # Load accumulator subtile → D registers
                load_acc_subtile(tRS_rD, epi_idx)

                # R2S copy: D registers → scratch SMEM (stage 0)
                tRS_rD_retiled = tiled_copy_scratch_r2s.retile(tRS_rD)
                cute.copy(
                    tiled_copy_scratch_r2s,
                    tRS_rD_retiled,
                    tRS_sScratch[None, None, None, 0],
                )
                cute.arch.fence_view_async_shared()
                epilogue_barrier.arrive_and_wait()

                # ---- Each thread reads one row from SMEM: max + target ----
                if tidx < epi_M:
                    local_m = tidx
                    global_m = m_offset + epi_m * epi_M + local_m
                    global_n_start = n_offset + epi_n * epi_N

                    if global_m < total_M:
                        target_idx = Int32(params.mTarget[global_m])

                        for j in cutlass.range(epi_N, unroll=8):
                            global_n = global_n_start + j
                            if global_n < V:
                                val = Float32(sScratch[local_m, j, 0])
                                row_max = cute.arch.fmax(row_max, val)
                                if global_n == target_idx:
                                    row_target_logit = val

                # Barrier: protect scratch SMEM before next subtile
                epilogue_barrier.arrive_and_wait()

            # ============================================================
            # PASS 2: Compute sums using true CTA-wide max (no merge)
            # ============================================================
            row_sum_exp = Float32(0.0)
            row_sum_x_exp = Float32(0.0)

            for epi_n in cutlass.range_constexpr(n_epi_n):
                epi_idx = epi_m * n_epi_n + epi_n

                # Reload accumulator subtile → D registers
                load_acc_subtile(tRS_rD, epi_idx)

                # R2S copy: D registers → scratch SMEM (stage 0)
                tRS_rD_retiled = tiled_copy_scratch_r2s.retile(tRS_rD)
                cute.copy(
                    tiled_copy_scratch_r2s,
                    tRS_rD_retiled,
                    tRS_sScratch[None, None, None, 0],
                )
                cute.arch.fence_view_async_shared()
                epilogue_barrier.arrive_and_wait()

                # ---- Each thread reads one row: accumulate sums ----
                if tidx < epi_M:
                    local_m = tidx
                    global_m = m_offset + epi_m * epi_M + local_m
                    global_n_start = n_offset + epi_n * epi_N

                    if global_m < total_M:
                        for j in cutlass.range(epi_N, unroll=8):
                            global_n = global_n_start + j
                            if global_n < V:
                                val = Float32(sScratch[local_m, j, 0])
                                # No fastmath — use precise exp2
                                exp_val = cute.math.exp2(
                                    (val - row_max) * log2_e,
                                )
                                row_sum_exp += exp_val
                                row_sum_x_exp += val * exp_val

                # Barrier: protect scratch SMEM before next subtile
                epilogue_barrier.arrive_and_wait()

            # ---- After both passes: write partials to GMEM ----
            if tidx < epi_M:
                local_m = tidx
                global_m = m_offset + epi_m * epi_M + local_m
                if global_m < total_M:
                    params.mPartials[n_tile, global_m, 0] = row_max
                    params.mPartials[n_tile, global_m, 1] = row_sum_exp
                    params.mPartials[n_tile, global_m, 2] = row_sum_x_exp
                    params.mPartials[n_tile, global_m, 3] = row_target_logit

        return epi_read_state, epi_producer_state


class GemmCEEntropyV2Sm90(GemmCEEntropyV2EpiMixin, GemmSm90):
    pass
