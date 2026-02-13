"""Baseline aggregation methods for comparison with BRAC geometric median.

This module implements several aggregation baselines from the Byzantine
fault tolerance literature for comparison in experiments:

1. Weighted Average - Standard approach, breakdown point = 1/n
2. Coordinate-wise Median - Per-dimension median, breakdown = 50% per dim only
3. Krum - Selects belief closest to neighbors (Blanchard et al. 2017)
4. Multi-Krum - Average of m Krum-selected beliefs
5. Trimmed Mean - Coordinate-wise trimmed mean

These serve as baselines for Table 1 in the BRAC paper.
"""

import torch
from typing import Optional, Union
import logging
import warnings
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Track warnings to avoid spam (Issue #6)
_warned_conditions = set()

EPS = 1e-10


@dataclass
class AggregatorResult:
    """Result from an aggregator with metadata.

    Attributes:
        consensus: The aggregated belief on simplex
        valid: Whether the aggregator conditions were met
        method_used: Actual method used (may differ if fallback)
        message: Optional message about the aggregation
    """
    consensus: torch.Tensor
    valid: bool = True
    method_used: str = ""
    message: str = ""


def _warn_once(key: str, message: str):
    """Log a warning only once per unique key."""
    if key not in _warned_conditions:
        _warned_conditions.add(key)
        logger.warning(message)


def _ensure_simplex(p: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is on probability simplex."""
    p = torch.clamp(p, min=EPS)
    return p / p.sum(dim=-1, keepdim=True)


def weighted_average(
    beliefs: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute weighted arithmetic average of beliefs.

    This is the standard approach but has breakdown point = 1/n,
    meaning a single Byzantine agent can arbitrarily corrupt the result.

    Args:
        beliefs: Agent beliefs, shape (N, K)
        weights: Optional weights, shape (N,). Default: uniform

    Returns:
        Aggregated belief, shape (K,)
    """
    N, K = beliefs.shape

    if weights is None:
        weights = torch.ones(N, device=beliefs.device, dtype=beliefs.dtype) / N
    else:
        weights = weights / weights.sum()

    # Weighted sum
    result = torch.sum(weights.unsqueeze(-1) * beliefs, dim=0)

    return _ensure_simplex(result)


def coordinate_median(
    beliefs: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute coordinate-wise weighted median of beliefs.

    Takes the median independently for each coordinate. This has
    breakdown point = 50% per dimension, but does not respect the
    geometry of the probability simplex.

    Args:
        beliefs: Agent beliefs, shape (N, K)
        weights: Optional weights (used for weighted median). Default: uniform

    Returns:
        Aggregated belief, shape (K,)
    """
    N, K = beliefs.shape
    device = beliefs.device
    dtype = beliefs.dtype

    result = torch.zeros(K, device=device, dtype=dtype)

    for k in range(K):
        # Get k-th coordinate from all agents
        values = beliefs[:, k]

        if weights is None:
            # Simple median
            result[k] = torch.median(values)
        else:
            # Weighted median
            sorted_indices = torch.argsort(values)
            sorted_values = values[sorted_indices]
            sorted_weights = weights[sorted_indices]

            # Find weighted median: first index where cumsum >= 0.5
            cumsum = torch.cumsum(sorted_weights / sorted_weights.sum(), dim=0)
            median_idx = torch.searchsorted(cumsum, 0.5)
            median_idx = min(median_idx.item(), N - 1)

            result[k] = sorted_values[median_idx]

    return _ensure_simplex(result)


def _euclidean_distance_matrix(beliefs: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distance matrix.

    Args:
        beliefs: Shape (N, K)

    Returns:
        Distance matrix, shape (N, N)
    """
    N = beliefs.shape[0]
    dist_matrix = torch.zeros(N, N, device=beliefs.device, dtype=beliefs.dtype)

    for i in range(N):
        for j in range(i + 1, N):
            d = torch.norm(beliefs[i] - beliefs[j]).item()
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def krum(
    beliefs: torch.Tensor,
    f: int = 1,
    return_result: bool = False,
) -> Union[torch.Tensor, AggregatorResult]:
    """Krum selection aggregation (Blanchard et al. 2017).

    Selects the belief that is closest to its n-f-2 nearest neighbors.
    This is Byzantine-resilient for f < n/2 - 1.

    The intuition is that Byzantine agents may be far from the honest
    majority, so the belief with smallest total distance to neighbors
    is likely honest.

    Args:
        beliefs: Agent beliefs, shape (N, K)
        f: Number of Byzantine agents to tolerate
        return_result: If True, return AggregatorResult with metadata

    Returns:
        Selected belief (K,) or AggregatorResult if return_result=True
    """
    N, K = beliefs.shape

    if N <= 2 * f + 2:
        _warn_once(f"krum_N{N}_f{f}", f"Krum requires N > 2f+2. Got N={N}, f={f}. Condition not met.")
        fallback = weighted_average(beliefs)
        if return_result:
            return AggregatorResult(
                consensus=fallback,
                valid=False,
                method_used="weighted_average",
                message=f"Krum N>2f+2 not satisfied (N={N}, f={f})"
            )
        return fallback

    # Number of neighbors to consider
    num_neighbors = N - f - 2

    # Compute distance matrix
    dist_matrix = _euclidean_distance_matrix(beliefs)

    # For each agent, compute sum of distances to nearest neighbors
    scores = torch.zeros(N, device=beliefs.device, dtype=beliefs.dtype)

    for i in range(N):
        # Get distances to all other agents
        distances = dist_matrix[i].clone()
        distances[i] = float('inf')  # Exclude self

        # Sort and sum smallest num_neighbors distances
        sorted_distances, _ = torch.sort(distances)
        scores[i] = sorted_distances[:num_neighbors].sum()

    # Select agent with minimum score
    selected_idx = torch.argmin(scores).item()
    logger.debug(f"Krum selected agent {selected_idx} with score {scores[selected_idx]:.4f}")

    result = beliefs[selected_idx].clone()
    if return_result:
        return AggregatorResult(
            consensus=result,
            valid=True,
            method_used="krum",
            message=f"Selected agent {selected_idx}"
        )
    return result


def multi_krum(
    beliefs: torch.Tensor,
    f: int = 1,
    m: int = 3,
    return_result: bool = False,
) -> Union[torch.Tensor, AggregatorResult]:
    """Multi-Krum aggregation: average of m Krum-selected beliefs.

    Iteratively selects m beliefs using the Krum criterion, then
    averages them. This is more stable than single Krum.

    Args:
        beliefs: Agent beliefs, shape (N, K)
        f: Number of Byzantine agents to tolerate
        m: Number of beliefs to select and average
        return_result: If True, return AggregatorResult with metadata

    Returns:
        Aggregated belief (K,) or AggregatorResult if return_result=True
    """
    N, K = beliefs.shape

    if N <= 2 * f + 2:
        _warn_once(f"multi_krum_N{N}_f{f}", f"Multi-Krum requires N > 2f+2. Got N={N}, f={f}. Condition not met.")
        fallback = weighted_average(beliefs)
        if return_result:
            return AggregatorResult(
                consensus=fallback,
                valid=False,
                method_used="weighted_average",
                message=f"Multi-Krum N>2f+2 not satisfied (N={N}, f={f})"
            )
        return fallback

    m = min(m, N - f)  # Can't select more than honest agents
    num_neighbors = N - f - 2

    # Compute distance matrix
    dist_matrix = _euclidean_distance_matrix(beliefs)

    # Track selected indices
    selected_beliefs = []
    remaining_indices = list(range(N))

    for _ in range(m):
        if len(remaining_indices) <= 2:
            break

        # Compute scores for remaining agents
        scores = {}
        for i in remaining_indices:
            distances = [dist_matrix[i, j].item() for j in remaining_indices if j != i]
            distances.sort()
            scores[i] = sum(distances[:min(num_neighbors, len(distances))])

        # Select agent with minimum score
        selected_idx = min(scores, key=scores.get)
        selected_beliefs.append(beliefs[selected_idx])
        remaining_indices.remove(selected_idx)

    # Average selected beliefs
    if not selected_beliefs:
        fallback = weighted_average(beliefs)
        if return_result:
            return AggregatorResult(
                consensus=fallback,
                valid=False,
                method_used="weighted_average",
                message="No beliefs selected"
            )
        return fallback

    stacked = torch.stack(selected_beliefs)
    result = _ensure_simplex(stacked.mean(dim=0))
    if return_result:
        return AggregatorResult(
            consensus=result,
            valid=True,
            method_used="multi_krum",
            message=f"Selected {len(selected_beliefs)} agents"
        )
    return result


def trimmed_mean(
    beliefs: torch.Tensor,
    trim_fraction: float = 0.25,
) -> torch.Tensor:
    """Coordinate-wise trimmed mean.

    For each coordinate, removes the top and bottom trim_fraction of values,
    then computes the mean of the remaining values.

    Args:
        beliefs: Agent beliefs, shape (N, K)
        trim_fraction: Fraction to trim from each end (0 to 0.5)

    Returns:
        Aggregated belief, shape (K,)
    """
    N, K = beliefs.shape
    device = beliefs.device
    dtype = beliefs.dtype

    trim_count = int(N * trim_fraction)

    if 2 * trim_count >= N:
        _warn_once(f"trimmed_mean_N{N}_f{trim_fraction}",
                   f"Trim fraction {trim_fraction} too large for N={N}. Using full mean.")
        trim_count = 0

    result = torch.zeros(K, device=device, dtype=dtype)

    for k in range(K):
        values = beliefs[:, k]
        sorted_values, _ = torch.sort(values)

        # Trim and average
        if trim_count > 0:
            trimmed = sorted_values[trim_count:-trim_count]
        else:
            trimmed = sorted_values

        result[k] = trimmed.mean()

    return _ensure_simplex(result)


def bulyan(
    beliefs: torch.Tensor,
    f: int = 1,
) -> torch.Tensor:
    """Bulyan aggregation (El Mhamdi et al. 2018).

    A two-stage robust aggregation:
    1. Use Multi-Krum to select n-2f candidates
    2. Apply coordinate-wise trimmed mean on candidates

    Provides stronger Byzantine resilience than Krum alone.

    Args:
        beliefs: Agent beliefs, shape (N, K)
        f: Number of Byzantine agents to tolerate

    Returns:
        Aggregated belief, shape (K,)
    """
    N, K = beliefs.shape

    if N <= 4 * f:
        _warn_once(f"bulyan_N{N}_f{f}", f"Bulyan requires N > 4f. Got N={N}, f={f}. Using Multi-Krum.")
        return multi_krum(beliefs, f, m=max(1, N - f))

    # Stage 1: Multi-Krum to select n-2f candidates
    num_candidates = N - 2 * f
    num_neighbors = N - f - 2

    dist_matrix = _euclidean_distance_matrix(beliefs)

    # Select candidates
    candidate_beliefs = []
    remaining_indices = list(range(N))

    for _ in range(num_candidates):
        if len(remaining_indices) <= 2:
            break

        scores = {}
        for i in remaining_indices:
            distances = [dist_matrix[i, j].item() for j in remaining_indices if j != i]
            distances.sort()
            scores[i] = sum(distances[:min(num_neighbors, len(distances))])

        selected_idx = min(scores, key=scores.get)
        candidate_beliefs.append(beliefs[selected_idx])
        remaining_indices.remove(selected_idx)

    if not candidate_beliefs:
        return weighted_average(beliefs)

    candidates = torch.stack(candidate_beliefs)

    # Stage 2: Trimmed mean on candidates (trim f from each end)
    trim_count = min(f, len(candidates) // 2 - 1)
    if trim_count <= 0:
        return _ensure_simplex(candidates.mean(dim=0))

    result = torch.zeros(K, device=beliefs.device, dtype=beliefs.dtype)
    for k in range(K):
        values = candidates[:, k]
        sorted_values, _ = torch.sort(values)
        trimmed = sorted_values[trim_count:-trim_count] if trim_count > 0 else sorted_values
        result[k] = trimmed.mean()

    return _ensure_simplex(result)


def get_aggregator(name: str):
    """Get aggregator function by name.

    Args:
        name: One of "weighted_average", "coordinate_median", "krum",
              "multi_krum", "trimmed_mean", "bulyan"

    Returns:
        Aggregator function
    """
    aggregators = {
        "weighted_average": weighted_average,
        "coordinate_median": coordinate_median,
        "krum": krum,
        "multi_krum": multi_krum,
        "trimmed_mean": trimmed_mean,
        "bulyan": bulyan,
    }

    if name not in aggregators:
        raise ValueError(f"Unknown aggregator: {name}. Available: {list(aggregators.keys())}")

    return aggregators[name]


def compare_aggregators(
    beliefs: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    f: int = 1,
) -> dict[str, torch.Tensor]:
    """Compare all aggregation methods on the same input.

    Useful for experiments and visualization.

    Args:
        beliefs: Agent beliefs, shape (N, K)
        weights: Optional weights for weighted methods
        f: Number of Byzantine agents for robust methods

    Returns:
        Dictionary mapping aggregator name -> result
    """
    results = {
        "weighted_average": weighted_average(beliefs, weights),
        "coordinate_median": coordinate_median(beliefs, weights),
        "krum": krum(beliefs, f),
        "multi_krum": multi_krum(beliefs, f),
        "trimmed_mean": trimmed_mean(beliefs),
        "bulyan": bulyan(beliefs, f),
    }

    return results
