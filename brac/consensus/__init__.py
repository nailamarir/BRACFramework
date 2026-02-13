"""Byzantine-resilient consensus algorithms for BRAC.

This package implements the core consensus mechanisms including:
- Fisher-Rao geometry on the probability simplex
- Riemannian Weiszfeld algorithm for geometric median
- Trust-bootstrapped reliability estimation
- Conformal prediction for uncertainty quantification
- Shapley attribution for explainability
"""

from brac.consensus.fisher_rao import (
    fisher_rao_distance,
    sqrt_embedding,
    sqrt_embedding_inv,
    exp_map,
    log_map,
    frechet_mean_sphere,
)
from brac.consensus.geometric_median import (
    riemannian_weiszfeld,
    geometric_median_consensus,
)
from brac.consensus.trust import TrustEstimator
from brac.consensus.conformal import ConformalPredictor
from brac.consensus.shapley import ShapleyAttributor

__all__ = [
    # Fisher-Rao geometry
    "fisher_rao_distance",
    "sqrt_embedding",
    "sqrt_embedding_inv",
    "exp_map",
    "log_map",
    "frechet_mean_sphere",
    # Geometric median
    "riemannian_weiszfeld",
    "geometric_median_consensus",
    # Trust
    "TrustEstimator",
    # Conformal
    "ConformalPredictor",
    # Shapley
    "ShapleyAttributor",
]
