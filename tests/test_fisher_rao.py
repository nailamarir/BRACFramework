"""Unit tests for Fisher-Rao geometry module.

Tests verify:
- Metric axioms (identity, symmetry, triangle inequality, positivity)
- Round-trip properties of embeddings
- Known distance values
- Numerical stability
"""

import pytest
import torch
import numpy as np

from brac.consensus.fisher_rao import (
    fisher_rao_distance,
    sqrt_embedding,
    sqrt_embedding_inv,
    exp_map,
    log_map,
    frechet_mean_simplex,
    verify_metric_axioms,
)


class TestFisherRaoDistance:
    """Tests for Fisher-Rao distance computation."""

    def test_identity_property(self):
        """d(p, p) = 0 for any distribution p."""
        for _ in range(10):
            p = torch.softmax(torch.randn(9), dim=0)
            d = fisher_rao_distance(p, p)
            assert abs(d.item()) < 1e-6, f"d(p,p) = {d.item()}, expected 0"

    def test_symmetry_property(self):
        """d(p, q) = d(q, p) for any distributions p, q."""
        for _ in range(10):
            p = torch.softmax(torch.randn(9), dim=0)
            q = torch.softmax(torch.randn(9), dim=0)
            d_pq = fisher_rao_distance(p, q)
            d_qp = fisher_rao_distance(q, p)
            assert abs(d_pq.item() - d_qp.item()) < 1e-6

    def test_triangle_inequality(self):
        """d(p, r) <= d(p, q) + d(q, r) for any p, q, r."""
        for _ in range(10):
            p = torch.softmax(torch.randn(9), dim=0)
            q = torch.softmax(torch.randn(9), dim=0)
            r = torch.softmax(torch.randn(9), dim=0)

            d_pr = fisher_rao_distance(p, r)
            d_pq = fisher_rao_distance(p, q)
            d_qr = fisher_rao_distance(q, r)

            assert d_pr <= d_pq + d_qr + 1e-6

    def test_positivity(self):
        """d(p, q) >= 0 for any distributions."""
        for _ in range(10):
            p = torch.softmax(torch.randn(9), dim=0)
            q = torch.softmax(torch.randn(9), dim=0)
            d = fisher_rao_distance(p, q)
            assert d.item() >= -1e-10

    def test_known_distance_uniform_to_one_hot(self):
        """Distance from uniform to one-hot should be 2*arccos(1/sqrt(K))."""
        K = 9
        uniform = torch.ones(K) / K
        one_hot = torch.zeros(K)
        one_hot[0] = 1.0

        d = fisher_rao_distance(uniform, one_hot)
        expected = 2 * np.arccos(1 / np.sqrt(K))

        assert abs(d.item() - expected) < 1e-5

    def test_batched_computation(self):
        """Batched distance computation should match individual."""
        batch_size = 5
        K = 9

        p_batch = torch.softmax(torch.randn(batch_size, K), dim=-1)
        q_batch = torch.softmax(torch.randn(batch_size, K), dim=-1)

        # Batched
        d_batch = fisher_rao_distance(p_batch, q_batch)

        # Individual
        d_individual = torch.stack([
            fisher_rao_distance(p_batch[i], q_batch[i])
            for i in range(batch_size)
        ])

        assert torch.allclose(d_batch, d_individual, atol=1e-6)

    def test_verify_metric_axioms_helper(self):
        """Test the metric verification helper function."""
        p = torch.softmax(torch.randn(9), dim=0)
        q = torch.softmax(torch.randn(9), dim=0)
        r = torch.softmax(torch.randn(9), dim=0)

        results = verify_metric_axioms(p, q, r)

        assert results["identity"], "Identity axiom failed"
        assert results["symmetry"], "Symmetry axiom failed"
        assert results["triangle"], "Triangle inequality failed"
        assert results["positivity"], "Positivity failed"


class TestSqrtEmbedding:
    """Tests for square-root embedding and inverse."""

    def test_roundtrip_simplex(self):
        """psi_inv(psi(p)) = p for distributions on simplex."""
        for _ in range(10):
            p = torch.softmax(torch.randn(9), dim=0)
            z = sqrt_embedding(p)
            p_recovered = sqrt_embedding_inv(z)
            assert torch.allclose(p, p_recovered, atol=1e-5)

    def test_embedding_on_sphere(self):
        """psi(p) should be on unit sphere."""
        for _ in range(10):
            p = torch.softmax(torch.randn(9), dim=0)
            z = sqrt_embedding(p)
            norm = torch.norm(z)
            assert abs(norm.item() - 1.0) < 1e-5

    def test_embedding_positive_orthant(self):
        """psi(p) should be in positive orthant."""
        for _ in range(10):
            p = torch.softmax(torch.randn(9), dim=0)
            z = sqrt_embedding(p)
            assert (z >= 0).all()


class TestExpLogMaps:
    """Tests for exponential and logarithmic maps on sphere."""

    def test_exp_log_roundtrip(self):
        """Exp_z(Log_z(w)) = w on sphere."""
        K = 9
        for _ in range(10):
            # Two random points on the positive orthant of sphere
            p1 = torch.softmax(torch.randn(K), dim=0)
            p2 = torch.softmax(torch.randn(K), dim=0)

            z = sqrt_embedding(p1)
            w = sqrt_embedding(p2)

            # Log map: tangent vector from z to w
            v = log_map(z.unsqueeze(0), w.unsqueeze(0)).squeeze()

            # Exp map: should recover w
            w_recovered = exp_map(z.unsqueeze(0), v.unsqueeze(0)).squeeze()

            # Compare on simplex (more stable)
            p2_recovered = sqrt_embedding_inv(w_recovered)
            assert torch.allclose(p2, p2_recovered, atol=1e-4)

    def test_exp_map_zero_tangent(self):
        """Exp_z(0) = z."""
        K = 9
        p = torch.softmax(torch.randn(K), dim=0)
        z = sqrt_embedding(p)

        v = torch.zeros(K)
        z_result = exp_map(z.unsqueeze(0), v.unsqueeze(0)).squeeze()

        assert torch.allclose(z, z_result, atol=1e-6)


class TestFrechetMean:
    """Tests for Fréchet mean computation."""

    def test_single_point_mean(self):
        """Fréchet mean of single point is itself."""
        p = torch.softmax(torch.randn(9), dim=0)
        mean, iters = frechet_mean_simplex(p.unsqueeze(0))
        assert torch.allclose(p, mean, atol=1e-5)

    def test_uniform_weights(self):
        """Fréchet mean with uniform weights."""
        N, K = 4, 9
        beliefs = torch.softmax(torch.randn(N, K), dim=-1)

        mean, iters = frechet_mean_simplex(beliefs)

        # Mean should be on simplex
        assert abs(mean.sum().item() - 1.0) < 1e-5
        assert (mean >= 0).all()

    def test_convergence(self):
        """Algorithm should converge."""
        N, K = 4, 9
        beliefs = torch.softmax(torch.randn(N, K), dim=-1)

        mean, iters = frechet_mean_simplex(beliefs, max_iter=100, tol=1e-8)

        # Should converge before max_iter
        assert iters < 100


class TestNumericalStability:
    """Tests for numerical stability edge cases."""

    def test_near_one_hot_distributions(self):
        """Handle near-one-hot distributions without NaN."""
        K = 9
        p = torch.zeros(K)
        p[0] = 0.9999
        p[1:] = 0.0001 / (K - 1)

        q = torch.zeros(K)
        q[1] = 0.9999
        q[0] = 0.0001
        q[2:] = 0.00001 / (K - 2)

        d = fisher_rao_distance(p, q)
        assert not torch.isnan(d)
        assert not torch.isinf(d)

    def test_uniform_distribution(self):
        """Handle uniform distributions."""
        K = 9
        uniform = torch.ones(K) / K

        d = fisher_rao_distance(uniform, uniform)
        assert abs(d.item()) < 1e-6

    def test_very_small_values(self):
        """Handle distributions with very small values."""
        K = 9
        p = torch.softmax(torch.randn(K) * 10, dim=0)  # More peaked
        q = torch.softmax(torch.randn(K) * 10, dim=0)

        d = fisher_rao_distance(p, q)
        assert not torch.isnan(d)
        assert not torch.isinf(d)
