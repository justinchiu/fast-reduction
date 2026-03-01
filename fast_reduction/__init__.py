from fast_reduction.baseline import (
    baseline_linear_xent,
    baseline_linear_entropy,
    baseline_linear_xent_entropy,
    chunked_linear_xent,
    chunked_linear_entropy,
    chunked_linear_xent_entropy,
)
from fast_reduction.kernel import fused_linear_xent_entropy

__all__ = [
    "baseline_linear_xent",
    "baseline_linear_entropy",
    "baseline_linear_xent_entropy",
    "chunked_linear_xent",
    "chunked_linear_entropy",
    "chunked_linear_xent_entropy",
    "fused_linear_xent_entropy",
    "gemm_fused_ce_entropy",
]


def __getattr__(name):
    if name == "gemm_fused_ce_entropy":
        from fast_reduction.gemm_kernel import gemm_fused_ce_entropy
        return gemm_fused_ce_entropy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
