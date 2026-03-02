from fast_reduction.baseline import (
    baseline_linear_xent,
    baseline_linear_entropy,
    baseline_linear_xent_entropy,
    baseline_linear_xent_entropy_backward,
    chunked_linear_xent,
    chunked_linear_entropy,
    chunked_linear_xent_entropy,
)
from fast_reduction.kernel import (
    fused_linear_xent_entropy,
    fused_linear_xent_entropy_backward,
    fused_linear_xent_entropy_differentiable,
)

__all__ = [
    "baseline_linear_xent",
    "baseline_linear_entropy",
    "baseline_linear_xent_entropy",
    "baseline_linear_xent_entropy_backward",
    "chunked_linear_xent",
    "chunked_linear_entropy",
    "chunked_linear_xent_entropy",
    "fused_linear_xent_entropy",
    "fused_linear_xent_entropy_backward",
    "fused_linear_xent_entropy_differentiable",
    "gemm_fused_ce_entropy",
    "gemm_fused_ce_entropy_v2",
    "gemm_fused_ce_entropy_differentiable",
]


def __getattr__(name):
    if name == "gemm_fused_ce_entropy":
        from fast_reduction.gemm_kernel import gemm_fused_ce_entropy
        return gemm_fused_ce_entropy
    if name == "gemm_fused_ce_entropy_v2":
        from fast_reduction.gemm_kernel import gemm_fused_ce_entropy_v2
        return gemm_fused_ce_entropy_v2
    if name == "gemm_fused_ce_entropy_differentiable":
        from fast_reduction.gemm_kernel import gemm_fused_ce_entropy_differentiable
        return gemm_fused_ce_entropy_differentiable
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
