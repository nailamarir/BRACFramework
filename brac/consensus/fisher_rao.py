"""Fisher-Rao geometry on the probability simplex.

This module implements the mathematical foundation for Byzantine-resilient
consensus using the Fisher-Rao metric, which is the natural Riemannian
metric on the space of probability distributions.

The key insight is that the probability simplex Delta^{K-1} can be mapped
isometrically to the positive orthant of the sphere S^{K-1}_+ via the
square-root embedding psi(p) = (sqrt(p_1), ..., sqrt(p_K)).

This allows us to perform Riemannian computations (geodesics, means, medians)
using spherical geometry while preserving the information-geometric structure.
"""

import torch
import numpy as np
from typing import Optional

# Numerical stability constants
EPS = 1e-10


def _ensure_simplex(p: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Ensure tensor is on the probability simplex.

    Args:
        p: Input tensor, shape (..., K)
        eps: Small value to prevent zeros

    Returns:
        Normalized tensor on simplex
    """
    p = torch.clamp(p, min=eps)
    return p / p.sum(dim=-1, keepdim=True)


def _ensure_sphere(z: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Ensure tensor is on the unit sphere.

    Args:
        z: Input tensor, shape (..., K)
        eps: Small value to prevent division by zero

    Returns:
        Normalized tensor on sphere
    """
    norm = torch.norm(z, dim=-1, keepdim=True)
    return z / torch.clamp(norm, min=eps)


def fisher_rao_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute the Fisher-Rao distance between probability distributions.

    The Fisher-Rao distance is the geodesic distance on the probability simplex
    under the Fisher information metric:

        d_FR(p, q) = 2 * arccos(sum_k sqrt(p_k * q_k))

    This is equivalent to the arc length on the sphere after square-root embedding.

    Args:
        p: First distribution(s), shape (..., K)
        q: Second distribution(s), shape (..., K)

    Returns:
        Distance(s), shape (...) or scalar
    """
    # Ensure inputs are on the simplex
    p = _ensure_simplex(p)
    q = _ensure_simplex(q)

    # Compute Bhattacharyya coefficient: sum_k sqrt(p_k * q_k)
    # This is the inner product of the square-root embeddings
    bc = torch.sum(torch.sqrt(p * q + EPS), dim=-1)

    # Clamp for numerical stability (arccos domain is [-1, 1])
    bc = torch.clamp(bc, -1.0 + EPS, 1.0 - EPS)

    # Fisher-Rao distance = 2 * arccos(BC)
    return 2.0 * torch.arccos(bc)


def fisher_rao_distance_batched(
    p: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """Batched Fisher-Rao distance computation.

    Args:
        p: Shape (B, K) - batch of distributions
        q: Shape (B, K) - batch of distributions

    Returns:
        Distances, shape (B,)
    """
    return fisher_rao_distance(p, q)


def sqrt_embedding(p: torch.Tensor) -> torch.Tensor:
    """Map probability simplex to positive orthant of sphere.

    The square-root embedding psi: Delta^{K-1} -> S^{K-1}_+ is defined as:
        psi(p) = (sqrt(p_1), ..., sqrt(p_K))

    This is an isometric embedding under the Fisher-Rao metric.

    Args:
        p: Probability distribution(s), shape (..., K)

    Returns:
        Point(s) on sphere, shape (..., K)
    """
    p = _ensure_simplex(p)
    z = torch.sqrt(p + EPS)
    return _ensure_sphere(z)


def sqrt_embedding_inv(z: torch.Tensor) -> torch.Tensor:
    """Map from sphere back to probability simplex.

    The inverse embedding psi^{-1}: S^{K-1}_+ -> Delta^{K-1} is:
        psi^{-1}(z) = z^2 / ||z||^2 (element-wise square, then normalize)

    Args:
        z: Point(s) on sphere, shape (..., K)

    Returns:
        Probability distribution(s), shape (..., K)
    """
    z = _ensure_sphere(z)
    p = z ** 2
    return _ensure_simplex(p)


def exp_map(z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Exponential map on the sphere at point z in direction v.

    The exponential map takes a tangent vector v at z and returns the
    point reached by following the geodesic in that direction:

        Exp_z(v) = cos(||v||) * z + sin(||v||) * v / ||v||

    Args:
        z: Base point on sphere, shape (..., K)
        v: Tangent vector at z, shape (..., K)

    Returns:
        Point on sphere, shape (..., K)
    """
    z = _ensure_sphere(z)

    # Compute norm of tangent vector
    v_norm = torch.norm(v, dim=-1, keepdim=True)

    # Handle small tangent vectors (stay at z)
    small_mask = (v_norm < EPS).squeeze(-1)

    # Normalize v for direction
    v_normalized = v / torch.clamp(v_norm, min=EPS)

    # Exponential map formula
    result = torch.cos(v_norm) * z + torch.sin(v_norm) * v_normalized

    # For small tangent vectors, return z
    if small_mask.any():
        result = torch.where(small_mask.unsqueeze(-1), z, result)

    return _ensure_sphere(result)


def log_map(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Logarithmic map on the sphere from z to w.

    The logarithmic map is the inverse of the exponential map. It returns
    the tangent vector v at z such that Exp_z(v) = w:

        Log_z(w) = (theta / sin(theta)) * (w - cos(theta) * z)
        where theta = arccos(<z, w>)

    Args:
        z: Base point on sphere, shape (..., K)
        w: Target point on sphere, shape (..., K)

    Returns:
        Tangent vector at z pointing toward w, shape (..., K)
    """
    z = _ensure_sphere(z)
    w = _ensure_sphere(w)

    # Compute angle between z and w
    inner = torch.sum(z * w, dim=-1, keepdim=True)
    inner = torch.clamp(inner, -1.0 + EPS, 1.0 - EPS)
    theta = torch.arccos(inner)

    # Handle small angles (first-order approximation)
    small_mask = (theta < EPS).squeeze(-1)

    # Standard log map formula
    sin_theta = torch.sin(theta)
    sin_theta = torch.clamp(sin_theta, min=EPS)  # Avoid division by zero

    # Project w onto tangent space at z
    w_tangent = w - inner * z
    v = (theta / sin_theta) * w_tangent

    # For small angles, use first-order approximation: v ≈ w - z
    if small_mask.any():
        v_approx = w - z
        v = torch.where(small_mask.unsqueeze(-1), v_approx, v)

    return v


def parallel_transport(z: torch.Tensor, w: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Parallel transport tangent vector v from z to w along geodesic.

    Args:
        z: Source point on sphere, shape (..., K)
        w: Target point on sphere, shape (..., K)
        v: Tangent vector at z to transport, shape (..., K)

    Returns:
        Transported tangent vector at w, shape (..., K)
    """
    z = _ensure_sphere(z)
    w = _ensure_sphere(w)

    # Compute the geodesic direction
    inner_zw = torch.sum(z * w, dim=-1, keepdim=True)
    inner_zw = torch.clamp(inner_zw, -1.0 + EPS, 1.0 - EPS)
    theta = torch.arccos(inner_zw)

    # Handle small angles
    small_mask = (theta < EPS).squeeze(-1)

    sin_theta = torch.sin(theta)
    sin_theta = torch.clamp(sin_theta, min=EPS)

    # Unit vector in direction of geodesic at z
    u = (w - inner_zw * z) / sin_theta

    # Parallel transport formula
    inner_vu = torch.sum(v * u, dim=-1, keepdim=True)
    inner_vz = torch.sum(v * z, dim=-1, keepdim=True)

    v_transported = (
        v
        - (inner_vz + inner_vu * torch.cos(theta)) * (z + w) / (1 + inner_zw)
        + inner_vu * (torch.cos(theta) * z + w)
    )

    # Simplified for small angles
    if small_mask.any():
        v_transported = torch.where(small_mask.unsqueeze(-1), v, v_transported)

    return v_transported


def frechet_mean_sphere(
    points: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[torch.Tensor, int]:
    """Compute weighted Fréchet mean on the sphere.

    The Fréchet mean minimizes the weighted sum of squared geodesic distances:
        argmin_m sum_i w_i * d(m, z_i)^2

    Uses gradient descent on the sphere with exponential/logarithmic maps.

    Args:
        points: Points on sphere, shape (N, K)
        weights: Optional weights, shape (N,). Default: uniform
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (mean on sphere, iterations until convergence)
    """
    N, K = points.shape
    device = points.device
    dtype = points.dtype

    # Default to uniform weights
    if weights is None:
        weights = torch.ones(N, device=device, dtype=dtype) / N
    else:
        weights = weights / weights.sum()  # Normalize

    # Initialize at weighted Euclidean mean, projected to sphere
    mean = torch.sum(weights.unsqueeze(-1) * points, dim=0)
    mean = _ensure_sphere(mean)

    for iteration in range(max_iter):
        # Compute weighted average of log maps (tangent vectors)
        tangent_sum = torch.zeros(K, device=device, dtype=dtype)
        for i in range(N):
            v_i = log_map(mean.unsqueeze(0), points[i:i+1].unsqueeze(0)).squeeze()
            tangent_sum = tangent_sum + weights[i] * v_i

        # Check convergence
        step_size = torch.norm(tangent_sum).item()
        if step_size < tol:
            return mean, iteration

        # Take step in tangent direction
        mean_new = exp_map(mean.unsqueeze(0), tangent_sum.unsqueeze(0)).squeeze()
        mean = _ensure_sphere(mean_new)

    return mean, max_iter


def frechet_mean_simplex(
    beliefs: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[torch.Tensor, int]:
    """Compute weighted Fréchet mean on the probability simplex.

    This is a wrapper that maps to the sphere, computes the Fréchet mean,
    and maps back to the simplex.

    Args:
        beliefs: Probability distributions, shape (N, K)
        weights: Optional weights, shape (N,). Default: uniform
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (mean on simplex, iterations until convergence)
    """
    # Map to sphere
    points_sphere = sqrt_embedding(beliefs)

    # Compute Fréchet mean on sphere
    mean_sphere, iterations = frechet_mean_sphere(points_sphere, weights, max_iter, tol)

    # Map back to simplex
    mean_simplex = sqrt_embedding_inv(mean_sphere)

    return mean_simplex, iterations


def geodesic_interpolation(
    p: torch.Tensor, q: torch.Tensor, t: float
) -> torch.Tensor:
    """Interpolate along geodesic between two distributions.

    Args:
        p: Start distribution, shape (K,)
        q: End distribution, shape (K,)
        t: Interpolation parameter in [0, 1]

    Returns:
        Interpolated distribution at parameter t
    """
    z_p = sqrt_embedding(p)
    z_q = sqrt_embedding(q)

    # Log map from p to q
    v = log_map(z_p.unsqueeze(0), z_q.unsqueeze(0)).squeeze()

    # Scale tangent vector by t
    v_scaled = t * v

    # Exponential map
    z_t = exp_map(z_p.unsqueeze(0), v_scaled.unsqueeze(0)).squeeze()

    return sqrt_embedding_inv(z_t)


def verify_metric_axioms(p: torch.Tensor, q: torch.Tensor, r: torch.Tensor) -> dict[str, bool]:
    """Verify that Fisher-Rao distance satisfies metric axioms.

    Args:
        p, q, r: Three probability distributions

    Returns:
        Dictionary with verification results for each axiom
    """
    d_pp = fisher_rao_distance(p, p).item()
    d_pq = fisher_rao_distance(p, q).item()
    d_qp = fisher_rao_distance(q, p).item()
    d_pr = fisher_rao_distance(p, r).item()
    d_qr = fisher_rao_distance(q, r).item()

    return {
        "identity": abs(d_pp) < 1e-6,                    # d(p, p) = 0
        "symmetry": abs(d_pq - d_qp) < 1e-6,            # d(p, q) = d(q, p)
        "triangle": d_pr <= d_pq + d_qr + 1e-6,         # d(p, r) <= d(p, q) + d(q, r)
        "positivity": d_pq >= 0,                         # d(p, q) >= 0
    }
