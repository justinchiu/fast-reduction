"""
GEMM epilogue mixin for fused CE+entropy (Level 5).

Instead of writing the full [CTA_M, CTA_N] logits tile to GMEM, this epilogue
reduces the accumulator to 4 partial stats per row and writes them to a small
GMEM buffer.  Each CTA writes [CTA_M, 4] fp32 values:

    partials[n_tile, m, 0] = max_x
    partials[n_tile, m, 1] = sum_exp  (relative to max_x)
    partials[n_tile, m, 2] = sum_x_exp  (sum of x * exp(x - max))
    partials[n_tile, m, 3] = target_logit  (nonzero only for the CTA containing target)

A separate finalization kernel merges partials across N_tiles via online softmax
to produce loss[M] and entropy[M].

Design:
  - Overrides epilogue() to iterate epi subtiles (N-first within each M range).
  - Uses R2S copy to move accumulator values into scratch SMEM.
  - Threads cooperatively reduce rows from SMEM with two passes (max, then sums).
  - Uses online softmax to merge across N epi subtiles.
  - No TMA stores — writes partials via regular global stores.
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


class GemmCEEntropyEpiMixin(GemmDefaultEpiMixin):
    """Custom epilogue: reduces acc to partial CE+entropy stats per row."""

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
        args: "GemmCEEntropyEpiMixin.EpilogueArguments",
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: cute.Tile,
    ) -> int:
        # One epi subtile of fp32 scratch per stage
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

        # ---- Nested loop: M ranges × N subtiles (N-first matches acc order) ----
        for epi_m in cutlass.range_constexpr(n_epi_m):
            # Initialize per-thread running stats (1 row per thread)
            row_max = -Float32.inf
            row_sum_exp = Float32(0.0)
            row_sum_x_exp = Float32(0.0)
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

                # ---- Each thread reads one row from SMEM and reduces ----
                if tidx < epi_M:
                    local_m = tidx
                    global_m = m_offset + epi_m * epi_M + local_m
                    global_n_start = n_offset + epi_n * epi_N

                    if global_m < total_M:
                        target_idx = Int32(params.mTarget[global_m])

                        # Pass 1: row max + target logit extraction
                        partial_max = -Float32.inf
                        partial_target = Float32(0.0)
                        for j in cutlass.range(epi_N, unroll=8):
                            global_n = global_n_start + j
                            if global_n < V:
                                val = Float32(sScratch[local_m, j, 0])
                                partial_max = cute.arch.fmax(partial_max, val)
                                if global_n == target_idx:
                                    partial_target = val

                        # Pass 2: sum_exp and sum_x_exp
                        partial_sum_exp = Float32(0.0)
                        partial_sum_x_exp = Float32(0.0)
                        for j in cutlass.range(epi_N, unroll=8):
                            global_n = global_n_start + j
                            if global_n < V:
                                val = Float32(sScratch[local_m, j, 0])
                                exp_val = cute.math.exp2(
                                    (val - partial_max) * log2_e,
                                    fastmath=True,
                                )
                                partial_sum_exp += exp_val
                                partial_sum_x_exp += val * exp_val

                        # Online softmax merge with running stats
                        new_max = cute.arch.fmax(row_max, partial_max)
                        adj_old = cute.math.exp2(
                            (row_max - new_max) * log2_e, fastmath=True
                        )
                        adj_new = cute.math.exp2(
                            (partial_max - new_max) * log2_e, fastmath=True
                        )
                        row_sum_exp = (
                            row_sum_exp * adj_old + partial_sum_exp * adj_new
                        )
                        row_sum_x_exp = (
                            row_sum_x_exp * adj_old
                            + partial_sum_x_exp * adj_new
                        )
                        row_max = new_max
                        row_target_logit += partial_target

                # Barrier: protect scratch SMEM before next subtile
                epilogue_barrier.arrive_and_wait()

            # ---- After all N subtiles: write partials to GMEM ----
            if tidx < epi_M:
                local_m = tidx
                global_m = m_offset + epi_m * epi_M + local_m
                if global_m < total_M:
                    params.mPartials[n_tile, global_m, 0] = row_max
                    params.mPartials[n_tile, global_m, 1] = row_sum_exp
                    params.mPartials[n_tile, global_m, 2] = row_sum_x_exp
                    params.mPartials[n_tile, global_m, 3] = row_target_logit

        return epi_read_state, epi_producer_state


class GemmCEEntropySm90(GemmCEEntropyEpiMixin, GemmSm90):
    pass
