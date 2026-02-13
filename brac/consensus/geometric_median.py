"""Riemannian Weiszfeld algorithm for geometric median on probability simplex.

This module implements the Byzantine-resilient geometric median consensus
using the Weiszfeld algorithm adapted to the Riemannian geometry of the
probability simplex under the Fisher-Rao metric.

The geometric median is defined as:
    b* = argmin_b sum_i r_i * d_FR(b, b_i)

Unlike the arithmetic mean, the geometric median has a breakdown point of 50%,
meaning it remains robust even when nearly half the inputs are adversarial.
"""

import torch
import logging
from typing import Optional
from dataclasses import dataclass

from brac.consensus.fisher_rao import (
    fisher_rao_distance,
    sqrt_embedding,
    sqrt_embedding_inv,
    exp_map,
    log_map,
    frechet_mean_simplex,
    EPS,
)

logger = logging.getLogger(__name__)


@dataclass
class WeiszfeldResult:
    """Result from the Riemannian Weiszfeld algorithm.

    Attributes:
        consensus: Final consensus belief on simplex
        iterations: Number of iterations until convergence
        converged: Whether the algorithm converged within max_iter
        convergence_trace: List of d_FR(b^(ell), b^(ell-1)) at each iteration
        final_residual: Final convergence residual
    """
    consensus: torch.Tensor
    iterations: int
    converged: bool
    convergence_trace: list[float]
    final_residual: float


def riemannian_weiszfeld(
    beliefs: torch.Tensor,
    reliabilities: torch.Tensor,
    max_iter: int = 50,
    tol: float = 1e-8,
    smoothing_eps: float = 1e-6,
) -> WeiszfeldResult:
    """Riemannian Weiszfeld algorithm for geometric median on the probability simplex.

    This implements the iterative reweighted least squares approach adapted to
    Riemannian geometry. At each iteration:
    1. Compute weights w_i = r_i / d_FR(b^(ell), b_i)
    2. Compute weighted tangent mean in the tangent space at b^(ell)
    3. Take a step via the exponential map

    Args:
        beliefs: Agent belief distributions, shape (N, K)
        reliabilities: Agent reliability weights, shape (N,)
        max_iter: Maximum number of iterations (L)
        tol: Convergence tolerance (epsilon)
        smoothing_eps: Smoothing constant to avoid division by zero

    Returns:
        WeiszfeldResult containing consensus and convergence information
    """
    N, K = beliefs.shape
    device = beliefs.device
    dtype = beliefs.dtype

    # Normalize reliabilities
    reliabilities = reliabilities / reliabilities.sum()

    # Initialize: weighted Fréchet mean
    b_current, _ = frechet_mean_simplex(beliefs, reliabilities, max_iter=20)
    z_current = sqrt_embedding(b_current)

    convergence_trace = []
    converged = False

    for iteration in range(max_iter):
        # Compute distances from current estimate to all beliefs
        distances = torch.zeros(N, device=device, dtype=dtype)
        for i in range(N):
            distances[i] = fisher_rao_distance(b_current.unsqueeze(0), beliefs[i:i+1])

        # Compute Weiszfeld weights: w_i = r_i / d_i (with smoothing)
        # Avoid division by zero with smoothing
        weights = reliabilities / (distances + smoothing_eps)
        weights = weights / weights.sum()  # Normalize

        # Compute weighted tangent mean
        tangent_sum = torch.zeros(K, device=device, dtype=dtype)
        for i in range(N):
            # Map to sphere
            z_i = sqrt_embedding(beliefs[i])
            # Log map: tangent vector from z_current to z_i
            v_i = log_map(z_current.unsqueeze(0), z_i.unsqueeze(0)).squeeze()
            tangent_sum = tangent_sum + weights[i] * v_i

        # Step on sphere via exponential map
        z_new = exp_map(z_current.unsqueeze(0), tangent_sum.unsqueeze(0)).squeeze()

        # Project back to simplex
        b_new = sqrt_embedding_inv(z_new)

        # Check convergence
        residual = fisher_rao_distance(b_new.unsqueeze(0), b_current.unsqueeze(0)).item()
        convergence_trace.append(residual)

        logger.debug(f"Weiszfeld iteration {iteration}: residual = {residual:.2e}")

        if residual < tol:
            converged = True
            b_current = b_new
            z_current = z_new
            break

        b_current = b_new
        z_current = sqrt_embedding(b_current)

    return WeiszfeldResult(
        consensus=b_current,
        iterations=iteration + 1,
        converged=converged,
        convergence_trace=convergence_trace,
        final_residual=convergence_trace[-1] if convergence_trace else 0.0,
    )


def geometric_median_consensus(
    beliefs: dict,
    reliabilities: dict,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[torch.Tensor, WeiszfeldResult]:
    """Compute geometric median consensus from dictionary of agent beliefs.

    Convenience wrapper for riemannian_weiszfeld that accepts dictionaries
    keyed by modality.

    Args:
        beliefs: Dict mapping Modality -> belief tensor (K,)
        reliabilities: Dict mapping Modality -> reliability float
        max_iter: Maximum Weiszfeld iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (consensus belief tensor, WeiszfeldResult)
    """
    from brac.types import Modality

    # Extract beliefs and reliabilities in consistent order
    modalities = list(beliefs.keys())
    belief_list = [beliefs[m] for m in modalities]
    rel_list = [reliabilities.get(m, 1.0) for m in modalities]

    # Stack into tensors
    belief_tensor = torch.stack(belief_list)
    rel_tensor = torch.tensor(rel_list, dtype=belief_tensor.dtype, device=belief_tensor.device)

    # Run Weiszfeld
    result = riemannian_weiszfeld(belief_tensor, rel_tensor, max_iter, tol)

    return result.consensus, result


def iterative_consensus_round(
    beliefs: torch.Tensor,
    reliabilities: torch.Tensor,
    consensus: torch.Tensor,
    lambda_0: float = 0.3,
) -> torch.Tensor:
    """Update agent beliefs toward consensus (one round).

    Each agent updates its belief by moving toward the current consensus,
    with less reliable agents deferring more:

        b_i^{t+1} = (1 - lambda_i) * b_i^t + lambda_i * b*
        lambda_i = lambda_0 * (1 - r_i)

    Args:
        beliefs: Current agent beliefs, shape (N, K)
        reliabilities: Agent reliabilities, shape (N,)
        consensus: Current consensus belief, shape (K,)
        lambda_0: Base receptivity parameter

    Returns:
        Updated beliefs, shape (N, K)
    """
    N, K = beliefs.shape

    # Compute receptivity for each agent (unreliable agents defer more)
    lambdas = lambda_0 * (1 - reliabilities)

    # Update beliefs
    updated_beliefs = torch.zeros_like(beliefs)
    for i in range(N):
        updated_beliefs[i] = (1 - lambdas[i]) * beliefs[i] + lambdas[i] * consensus
        # Re-normalize to simplex
        updated_beliefs[i] = updated_beliefs[i] / updated_beliefs[i].sum()
        updated_beliefs[i] = torch.clamp(updated_beliefs[i], min=EPS)

    return updated_beliefs


def full_consensus_protocol(
    initial_beliefs: torch.Tensor,
    reliabilities: torch.Tensor,
    max_outer_rounds: int = 10,
    max_weiszfeld_iters: int = 50,
    outer_tol: float = 1e-4,
    weiszfeld_tol: float = 1e-8,
    lambda_0: float = 0.3,
) -> tuple[torch.Tensor, int, list[WeiszfeldResult]]:
    """Run full iterative consensus protocol with belief refinement.

    This implements Algorithm 1 from the BRAC paper:
    1. Compute geometric median consensus b*
    2. Update agent beliefs toward consensus
    3. Repeat until convergence or max rounds

    Args:
        initial_beliefs: Initial agent beliefs, shape (N, K)
        reliabilities: Agent reliabilities, shape (N,)
        max_outer_rounds: Maximum number of outer rounds (T)
        max_weiszfeld_iters: Maximum Weiszfeld iterations per round (L)
        outer_tol: Convergence tolerance for outer loop
        weiszfeld_tol: Convergence tolerance for Weiszfeld
        lambda_0: Base receptivity parameter

    Returns:
        Tuple of:
        - Final consensus belief, shape (K,)
        - Number of outer rounds
        - List of WeiszfeldResult from each round
    """
    N, K = initial_beliefs.shape
    beliefs = initial_beliefs.clone()
    round_results = []

    for round_idx in range(max_outer_rounds):
        # Compute geometric median
        result = riemannian_weiszfeld(
            beliefs, reliabilities,
            max_iter=max_weiszfeld_iters,
            tol=weiszfeld_tol
        )
        round_results.append(result)
        consensus = result.consensus

        # Check outer convergence: max distance from any agent to consensus
        max_distance = 0.0
        for i in range(N):
            d = fisher_rao_distance(beliefs[i:i+1], consensus.unsqueeze(0)).item()
            max_distance = max(max_distance, d)

        logger.debug(f"Consensus round {round_idx}: max_distance = {max_distance:.4f}")

        if max_distance < outer_tol:
            logger.info(f"Consensus converged at round {round_idx}")
            return consensus, round_idx + 1, round_results

        # Update beliefs toward consensus
        beliefs = iterative_consensus_round(beliefs, reliabilities, consensus, lambda_0)

    logger.warning(f"Consensus did not converge within {max_outer_rounds} rounds")
    return consensus, max_outer_rounds, round_results


def compute_consensus_objective(
    beliefs: torch.Tensor,
    reliabilities: torch.Tensor,
    consensus: torch.Tensor,
) -> float:
    """Compute the geometric median objective function value.

    The objective is: sum_i r_i * d_FR(b*, b_i)

    Args:
        beliefs: Agent beliefs, shape (N, K)
        reliabilities: Agent reliabilities, shape (N,)
        consensus: Consensus belief, shape (K,)

    Returns:
        Objective function value (weighted sum of distances)
    """
    N = beliefs.shape[0]
    total = 0.0

    for i in range(N):
        d = fisher_rao_distance(consensus.unsqueeze(0), beliefs[i:i+1]).item()
        total += reliabilities[i].item() * d

    return total


def breakdown_point_analysis(
    honest_beliefs: torch.Tensor,
    byzantine_beliefs: torch.Tensor,
    reliabilities: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Analyze breakdown point by comparing honest vs corrupted consensus.

    The breakdown point is the maximum fraction of corrupted inputs
    the estimator can tolerate before becoming arbitrarily bad.

    Args:
        honest_beliefs: Beliefs from honest agents, shape (N_h, K)
        byzantine_beliefs: Beliefs from Byzantine agents, shape (N_b, K)
        reliabilities: Optional reliability weights

    Returns:
        Dictionary with analysis results including breakdown metrics
    """
    N_h = honest_beliefs.shape[0]
    N_b = byzantine_beliefs.shape[0]
    N = N_h + N_b
    K = honest_beliefs.shape[1]

    if reliabilities is None:
        reliabilities = torch.ones(N) / N

    # Compute honest-only consensus
    honest_rels = reliabilities[:N_h] / reliabilities[:N_h].sum()
    honest_result = riemannian_weiszfeld(honest_beliefs, honest_rels)
    honest_consensus = honest_result.consensus

    # Compute combined consensus
    all_beliefs = torch.cat([honest_beliefs, byzantine_beliefs], dim=0)
    combined_result = riemannian_weiszfeld(all_beliefs, reliabilities)
    combined_consensus = combined_result.consensus

    # Measure displacement
    displacement = fisher_rao_distance(
        honest_consensus.unsqueeze(0),
        combined_consensus.unsqueeze(0)
    ).item()

    # Maximum possible distance (for normalization)
    max_distance = torch.pi  # Fisher-Rao distance is bounded by pi

    return {
        "byzantine_fraction": N_b / N,
        "displacement": displacement,
        "normalized_displacement": displacement / max_distance,
        "honest_consensus_entropy": -torch.sum(
            honest_consensus * torch.log(honest_consensus + EPS)
        ).item(),
        "combined_consensus_entropy": -torch.sum(
            combined_consensus * torch.log(combined_consensus + EPS)
        ).item(),
    }
