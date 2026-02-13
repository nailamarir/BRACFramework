"""Unit tests for conformal prediction module.

Tests verify:
- Coverage guarantee holds empirically
- Calibration correctness
- Prediction set properties
"""

import pytest
import torch
import numpy as np

from brac.consensus.conformal import (
    ConformalPredictor,
    split_calibration_test,
)
from brac.types import NHLSubtype


class TestConformalPredictor:
    """Tests for ConformalPredictor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.K = 9
        self.N = 500
        np.random.seed(42)
        torch.manual_seed(42)

        # Generate synthetic consensus beliefs
        self.true_labels = torch.randint(0, self.K, (self.N,))

        # Create beliefs centered on true labels
        self.beliefs = torch.zeros(self.N, self.K)
        for i in range(self.N):
            alpha = torch.ones(self.K) * 0.5
            alpha[self.true_labels[i]] = 10.0
            self.beliefs[i] = torch.distributions.Dirichlet(alpha).sample()

    def test_calibration(self):
        """Calibration should set q_hat."""
        predictor = ConformalPredictor(alpha=0.05)
        q_hat = predictor.calibrate(self.beliefs, self.true_labels)

        assert predictor.is_calibrated
        assert predictor.q_hat is not None
        assert 0 <= q_hat <= 1

    def test_coverage_guarantee(self):
        """Empirical coverage should be >= 1-alpha."""
        # Split data
        cal_beliefs, cal_labels, test_beliefs, test_labels = split_calibration_test(
            self.beliefs, self.true_labels, cal_fraction=0.5, seed=42
        )

        for alpha in [0.05, 0.10, 0.20]:
            predictor = ConformalPredictor(alpha=alpha)
            predictor.calibrate(cal_beliefs, cal_labels)
            coverage = predictor.marginal_coverage(test_beliefs, test_labels)

            # Coverage should be at least 1-alpha (with small margin for finite sample)
            assert coverage >= (1 - alpha - 0.05), \
                f"Coverage {coverage:.2%} < target {1-alpha:.2%} for alpha={alpha}"

    def test_prediction_set_contains_elements(self):
        """Prediction sets should contain at least one element."""
        predictor = ConformalPredictor(alpha=0.05)
        predictor.calibrate(self.beliefs, self.true_labels)

        for i in range(min(50, self.N)):
            result = predictor.predict_set(self.beliefs[i])
            assert result.set_size >= 1
            assert len(result.prediction_set) >= 1

    def test_prediction_set_valid_subtypes(self):
        """Prediction sets should contain valid NHLSubtype enums."""
        predictor = ConformalPredictor(alpha=0.05)
        predictor.calibrate(self.beliefs, self.true_labels)

        result = predictor.predict_set(self.beliefs[0])

        for subtype in result.prediction_set:
            assert isinstance(subtype, NHLSubtype)
            assert 0 <= subtype.value < self.K

    def test_smaller_alpha_larger_sets(self):
        """Smaller alpha should give larger prediction sets on average."""
        cal_beliefs, cal_labels, test_beliefs, test_labels = split_calibration_test(
            self.beliefs, self.true_labels, cal_fraction=0.5, seed=42
        )

        avg_sizes = {}
        for alpha in [0.20, 0.10, 0.05, 0.01]:
            predictor = ConformalPredictor(alpha=alpha)
            predictor.calibrate(cal_beliefs, cal_labels)
            avg_sizes[alpha] = predictor.average_set_size(test_beliefs)

        # Smaller alpha should give larger sets (more conservative)
        assert avg_sizes[0.01] >= avg_sizes[0.05] - 0.1
        assert avg_sizes[0.05] >= avg_sizes[0.10] - 0.1
        assert avg_sizes[0.10] >= avg_sizes[0.20] - 0.1

    def test_set_size_distribution(self):
        """Set size distribution should be valid."""
        predictor = ConformalPredictor(alpha=0.05)
        predictor.calibrate(self.beliefs, self.true_labels)

        dist = predictor.set_size_distribution(self.beliefs)

        # Check distribution sums to N
        assert sum(dist.values()) == self.N

        # All sizes should be positive integers
        for size, count in dist.items():
            assert size >= 1
            assert size <= self.K
            assert count >= 0

    def test_not_calibrated_raises(self):
        """Prediction without calibration should raise error."""
        predictor = ConformalPredictor(alpha=0.05)

        with pytest.raises(RuntimeError):
            predictor.predict_set(self.beliefs[0])

    def test_decision_field(self):
        """Decision should be 'accept' or 'escalate'."""
        predictor = ConformalPredictor(alpha=0.05, max_set_size_accept=2)
        predictor.calibrate(self.beliefs, self.true_labels)

        result = predictor.predict_set(self.beliefs[0])

        assert result.decision in ["accept", "escalate"]
        if result.set_size <= 2:
            assert result.decision == "accept"
        else:
            assert result.decision == "escalate"


class TestSplitCalibrationTest:
    """Tests for data splitting utility."""

    def test_split_sizes(self):
        """Split should produce correct sizes."""
        N = 100
        data = torch.randn(N, 9)
        labels = torch.randint(0, 9, (N,))

        cal_data, cal_labels, test_data, test_labels = split_calibration_test(
            data, labels, cal_fraction=0.5, seed=42
        )

        assert cal_data.shape[0] == 50
        assert test_data.shape[0] == 50
        assert cal_labels.shape[0] == 50
        assert test_labels.shape[0] == 50

    def test_no_overlap(self):
        """Calibration and test sets should not overlap."""
        N = 100
        # Create data with unique row identifiers
        data = torch.arange(N).unsqueeze(1).float()
        labels = torch.randint(0, 9, (N,))

        cal_data, _, test_data, _ = split_calibration_test(
            data, labels, cal_fraction=0.5, seed=42
        )

        # Convert to sets of row IDs
        cal_ids = set(cal_data.squeeze().tolist())
        test_ids = set(test_data.squeeze().tolist())

        assert len(cal_ids & test_ids) == 0, "Calibration and test sets overlap"

    def test_deterministic_with_seed(self):
        """Same seed should produce same split."""
        N = 100
        data = torch.randn(N, 9)
        labels = torch.randint(0, 9, (N,))

        split1 = split_calibration_test(data, labels, cal_fraction=0.5, seed=42)
        split2 = split_calibration_test(data, labels, cal_fraction=0.5, seed=42)

        assert torch.equal(split1[0], split2[0])
        assert torch.equal(split1[1], split2[1])
