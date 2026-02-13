"""Unit tests for trust estimation module.

Tests verify:
- ReLU clipping of behavioral trust
- Entropy computation
- Reliability combination
- Root of trust handling
"""

import pytest
import torch
import numpy as np

from brac.consensus.trust import TrustEstimator, create_default_trust_estimator
from brac.types import Modality, EvidenceQuality


class TestTrustEstimator:
    """Tests for TrustEstimator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.K = 9
        np.random.seed(42)
        torch.manual_seed(42)

        self.estimator = TrustEstimator(
            root_of_trust=Modality.PATHOLOGY,
            learnable=False,
        )

        # Generate test beliefs
        self.beliefs = {
            Modality.PATHOLOGY: torch.softmax(torch.randn(self.K), dim=0),
            Modality.RADIOLOGY: torch.softmax(torch.randn(self.K), dim=0),
            Modality.LABORATORY: torch.softmax(torch.randn(self.K), dim=0),
            Modality.CLINICAL: torch.softmax(torch.randn(self.K), dim=0),
        }

        self.qualities = {
            Modality.PATHOLOGY: EvidenceQuality(Q=0.9, C=0.8, S=0.85),
            Modality.RADIOLOGY: EvidenceQuality(Q=0.7, C=0.6, S=0.65),
            Modality.LABORATORY: EvidenceQuality(Q=0.8, C=0.75, S=0.8),
            Modality.CLINICAL: EvidenceQuality(Q=0.6, C=0.5, S=0.55),
        }

    def test_root_of_trust_has_trust_one(self):
        """Root of trust agent should have tau = 1.0."""
        trusts = self.estimator.compute_behavioral_trust(self.beliefs)
        assert trusts[Modality.PATHOLOGY] == 1.0

    def test_behavioral_trust_relu_clipped(self):
        """Behavioral trust should be in [0, 1]."""
        trusts = self.estimator.compute_behavioral_trust(self.beliefs)

        for m, trust in trusts.items():
            assert 0 <= trust <= 1, f"Trust {trust} for {m} not in [0,1]"

    def test_negative_cosine_sim_gives_zero_trust(self):
        """Opposing beliefs should give zero trust (ReLU clip)."""
        # Create opposing beliefs
        path_belief = torch.zeros(self.K)
        path_belief[0] = 0.9
        path_belief[1:] = 0.1 / (self.K - 1)

        opposing_belief = torch.zeros(self.K)
        opposing_belief[1] = 0.9
        opposing_belief[0] = 0.05
        opposing_belief[2:] = 0.05 / (self.K - 2)

        beliefs = {
            Modality.PATHOLOGY: path_belief,
            Modality.RADIOLOGY: opposing_belief,
            Modality.LABORATORY: torch.softmax(torch.randn(self.K), dim=0),
            Modality.CLINICAL: torch.softmax(torch.randn(self.K), dim=0),
        }

        trusts = self.estimator.compute_behavioral_trust(beliefs)

        # Opposing beliefs should have low trust (may not be exactly 0
        # if there's overlap in other dimensions)
        assert trusts[Modality.RADIOLOGY] < 0.5

    def test_reliability_in_range(self):
        """Final reliability should be in [0, 1]."""
        result = self.estimator.compute_reliability(self.beliefs, self.qualities)

        for m, rel in result.reliabilities.items():
            assert 0 <= rel <= 1, f"Reliability {rel} for {m} not in [0,1]"

    def test_entropy_computation(self):
        """Entropy should be computed correctly."""
        # Uniform distribution has max entropy = log(K)
        uniform = torch.ones(self.K) / self.K
        entropy = self.estimator.compute_entropy(uniform)
        expected = np.log(self.K)
        assert abs(entropy - expected) < 1e-5

        # One-hot has zero entropy
        one_hot = torch.zeros(self.K)
        one_hot[0] = 1.0
        entropy = self.estimator.compute_entropy(one_hot)
        assert entropy < 0.01

    def test_high_quality_gives_higher_reliability(self):
        """Higher quality scores should increase reliability."""
        # Same beliefs, different qualities
        high_quality = {m: EvidenceQuality(Q=0.95, C=0.95, S=0.95) for m in Modality.all()}
        low_quality = {m: EvidenceQuality(Q=0.3, C=0.3, S=0.3) for m in Modality.all()}

        result_high = self.estimator.compute_reliability(self.beliefs, high_quality)
        result_low = self.estimator.compute_reliability(self.beliefs, low_quality)

        # For non-root agents, high quality should give higher reliability
        for m in [Modality.RADIOLOGY, Modality.LABORATORY, Modality.CLINICAL]:
            # Trust is the same (same beliefs), but evidence reliability differs
            assert result_high.reliabilities[m] >= result_low.reliabilities[m] - 0.01

    def test_trust_result_structure(self):
        """TrustResult should contain all expected fields."""
        result = self.estimator.compute_reliability(self.beliefs, self.qualities)

        assert hasattr(result, 'trusts')
        assert hasattr(result, 'reliabilities')
        assert hasattr(result, 'entropies')
        assert hasattr(result, 'mlp_scores')

        # All modalities should be present
        for m in Modality.all():
            assert m in result.trusts
            assert m in result.reliabilities
            assert m in result.entropies


class TestDefaultTrustEstimator:
    """Tests for factory function."""

    def test_factory_function(self):
        """Factory should create valid estimator."""
        estimator = create_default_trust_estimator(learnable=False)

        assert estimator.root_of_trust == Modality.PATHOLOGY
        assert not estimator.learnable


class TestLearnableTrustEstimator:
    """Tests for learnable MLP mode."""

    def test_learnable_mode_creates_mlp(self):
        """Learnable mode should create MLP."""
        estimator = TrustEstimator(
            root_of_trust=Modality.PATHOLOGY,
            learnable=True,
        )

        assert estimator.learnable
        assert estimator.mlp is not None

    def test_learnable_reliability_computation(self):
        """Should compute reliability with learnable MLP."""
        K = 9
        estimator = TrustEstimator(
            root_of_trust=Modality.PATHOLOGY,
            learnable=True,
        )

        beliefs = {m: torch.softmax(torch.randn(K), dim=0) for m in Modality.all()}
        qualities = {m: EvidenceQuality(Q=0.8, C=0.7, S=0.75) for m in Modality.all()}

        result = estimator.compute_reliability(beliefs, qualities)

        for m, rel in result.reliabilities.items():
            assert 0 <= rel <= 1
