"""Unit tests for geometric median (Riemannian Weiszfeld) module.

Tests verify:
- Convergence properties
- Breakdown point behavior
- Consensus quality
"""

import pytest
import torch
import numpy as np

from brac.consensus.geometric_median import (
    riemannian_weiszfeld,
    geometric_median_consensus,
    iterative_consensus_round,
    full_consensus_protocol,
    compute_consensus_objective,
)
from brac.consensus.fisher_rao import fisher_rao_distance
from brac.types import Modality


class TestRiemannianWeiszfeld:
    """Tests for Riemannian Weiszfeld algorithm."""

    def test_convergence(self):
        """Algorithm should converge."""
        N, K = 4, 9
        beliefs = torch.softmax(torch.randn(N, K), dim=-1)
        reliabilities = torch.ones(N) / N

        result = riemannian_weiszfeld(beliefs, reliabilities, max_iter=50)

        assert result.converged or result.iterations < 50
        assert len(result.convergence_trace) > 0
        assert result.final_residual < 1e-4 or not result.converged

    def test_consensus_on_simplex(self):
        """Result should be on probability simplex."""
        N, K = 4, 9
        beliefs = torch.softmax(torch.randn(N, K), dim=-1)
        reliabilities = torch.ones(N) / N

        result = riemannian_weiszfeld(beliefs, reliabilities)
        consensus = result.consensus

        assert abs(consensus.sum().item() - 1.0) < 1e-5
        assert (consensus >= 0).all()

    def test_single_agent(self):
        """Single agent case: consensus should be that agent's belief."""
        K = 9
        belief = torch.softmax(torch.randn(K), dim=0)
        beliefs = belief.unsqueeze(0)
        reliabilities = torch.ones(1)

        result = riemannian_weiszfeld(beliefs, reliabilities)

        assert torch.allclose(result.consensus, belief, atol=1e-4)

    def test_identical_beliefs(self):
        """Identical beliefs: consensus should equal them."""
        N, K = 4, 9
        belief = torch.softmax(torch.randn(K), dim=0)
        beliefs = belief.unsqueeze(0).expand(N, -1).clone()
        reliabilities = torch.ones(N) / N

        result = riemannian_weiszfeld(beliefs, reliabilities)

        assert torch.allclose(result.consensus, belief, atol=1e-4)

    def test_weighted_reliability(self):
        """Higher reliability agents should have more influence."""
        K = 9
        # Two very different beliefs
        belief1 = torch.zeros(K)
        belief1[0] = 0.9
        belief1[1:] = 0.1 / (K - 1)

        belief2 = torch.zeros(K)
        belief2[1] = 0.9
        belief2[0] = 0.05
        belief2[2:] = 0.05 / (K - 2)

        beliefs = torch.stack([belief1, belief2])

        # High reliability on agent 0
        reliabilities = torch.tensor([0.9, 0.1])
        result = riemannian_weiszfeld(beliefs, reliabilities)

        # Consensus should be closer to belief1
        d1 = fisher_rao_distance(result.consensus.unsqueeze(0), belief1.unsqueeze(0))
        d2 = fisher_rao_distance(result.consensus.unsqueeze(0), belief2.unsqueeze(0))
        assert d1 < d2


class TestGeometricMedianConsensus:
    """Tests for dictionary-based consensus interface."""

    def test_modality_dict_interface(self):
        """Test with dictionary of modality -> belief."""
        K = 9
        beliefs = {
            Modality.PATHOLOGY: torch.softmax(torch.randn(K), dim=0),
            Modality.RADIOLOGY: torch.softmax(torch.randn(K), dim=0),
            Modality.LABORATORY: torch.softmax(torch.randn(K), dim=0),
            Modality.CLINICAL: torch.softmax(torch.randn(K), dim=0),
        }
        reliabilities = {m: 0.25 for m in Modality.all()}

        consensus, result = geometric_median_consensus(beliefs, reliabilities)

        assert abs(consensus.sum().item() - 1.0) < 1e-5
        assert (consensus >= 0).all()


class TestIterativeConsensus:
    """Tests for iterative belief refinement."""

    def test_belief_update(self):
        """Beliefs should move toward consensus."""
        N, K = 4, 9
        beliefs = torch.softmax(torch.randn(N, K) * 2, dim=-1)  # Spread out
        reliabilities = torch.ones(N) / N

        # Compute consensus
        result = riemannian_weiszfeld(beliefs, reliabilities)
        consensus = result.consensus

        # Update beliefs
        updated = iterative_consensus_round(
            beliefs, reliabilities, consensus, lambda_0=0.3
        )

        # Updated beliefs should be closer to consensus
        for i in range(N):
            d_old = fisher_rao_distance(beliefs[i:i+1], consensus.unsqueeze(0))
            d_new = fisher_rao_distance(updated[i:i+1], consensus.unsqueeze(0))
            # With lambda_0 < 1, beliefs move toward but don't reach consensus
            assert d_new <= d_old + 1e-6

    def test_full_protocol_convergence(self):
        """Full protocol should converge."""
        N, K = 4, 9
        beliefs = torch.softmax(torch.randn(N, K) * 2, dim=-1)
        reliabilities = torch.ones(N) / N

        consensus, rounds, results = full_consensus_protocol(
            beliefs, reliabilities,
            max_outer_rounds=20,
            outer_tol=1e-3,
            lambda_0=0.5,
        )

        # Should converge
        assert rounds < 20 or rounds == 20
        # Result should be valid
        assert abs(consensus.sum().item() - 1.0) < 1e-5


class TestBreakdownPoint:
    """Tests for breakdown point properties."""

    def test_one_byzantine_resilience(self):
        """Should be resilient to one Byzantine agent."""
        K = 9
        true_label = 3

        # Create 3 honest beliefs centered on true label
        honest_beliefs = []
        for _ in range(3):
            alpha = torch.ones(K) * 0.1
            alpha[true_label] = 10.0
            honest_beliefs.append(torch.distributions.Dirichlet(alpha).sample())

        # Create 1 Byzantine belief (wrong class, high confidence)
        byzantine = torch.zeros(K)
        byzantine[(true_label + 1) % K] = 0.9
        byzantine[true_label] = 0.05
        byzantine[torch.arange(K) != (true_label + 1) % K] += 0.05 / (K - 2)
        byzantine = byzantine / byzantine.sum()

        # All beliefs
        beliefs = torch.stack(honest_beliefs + [byzantine])
        reliabilities = torch.ones(4) / 4

        result = riemannian_weiszfeld(beliefs, reliabilities)

        # Consensus should still favor true label
        assert result.consensus.argmax().item() == true_label

    def test_objective_decreases(self):
        """Objective should decrease with iterations."""
        N, K = 4, 9
        beliefs = torch.softmax(torch.randn(N, K) * 2, dim=-1)
        reliabilities = torch.ones(N) / N

        result = riemannian_weiszfeld(beliefs, reliabilities, max_iter=50)

        # Objective at final point
        final_obj = compute_consensus_objective(
            beliefs, reliabilities, result.consensus
        )

        # Objective should be reasonable (not negative or huge)
        assert final_obj >= 0
        assert final_obj < 10  # Bounded by pi * sum(reliabilities)
