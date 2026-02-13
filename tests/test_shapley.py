"""Unit tests for Shapley attribution module.

Tests verify:
- Efficiency axiom: sum(phi_i) = v(A) - v(empty)
- Coalition enumeration correctness
- Interaction index properties
"""

import pytest
import torch
import numpy as np

from brac.consensus.shapley import ShapleyAttributor, aggregate_shapley_across_cases
from brac.consensus.geometric_median import geometric_median_consensus
from brac.types import Modality, NHLSubtype


class TestShapleyAttributor:
    """Tests for ShapleyAttributor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.K = 9
        np.random.seed(42)
        torch.manual_seed(42)

        self.attributor = ShapleyAttributor(
            agents=Modality.all(),
            geometric_median_fn=geometric_median_consensus,
            num_classes=self.K,
        )

        # Generate test beliefs
        self.beliefs = {
            Modality.PATHOLOGY: torch.softmax(torch.randn(self.K), dim=0),
            Modality.RADIOLOGY: torch.softmax(torch.randn(self.K), dim=0),
            Modality.LABORATORY: torch.softmax(torch.randn(self.K), dim=0),
            Modality.CLINICAL: torch.softmax(torch.randn(self.K), dim=0),
        }
        self.reliabilities = {m: 0.25 for m in Modality.all()}

    def test_coalition_enumeration(self):
        """Should enumerate all 2^4 = 16 coalitions."""
        coalitions = self.attributor.coalitions
        assert len(coalitions) == 16

        # Check empty set is included
        assert frozenset() in coalitions

        # Check all singletons
        for m in Modality.all():
            assert frozenset([m]) in coalitions

        # Check grand coalition
        assert frozenset(Modality.all()) in coalitions

    def test_efficiency_axiom(self):
        """sum(phi_i) should equal v(A) - v(empty)."""
        coalition_values = self.attributor.compute_coalition_values(
            self.beliefs, self.reliabilities
        )
        shapley_values = self.attributor.shapley_values(coalition_values)

        error = self.attributor.verify_efficiency(shapley_values, coalition_values)

        assert error < 1e-6, f"Efficiency axiom violated with error {error}"

    def test_shapley_values_computed(self):
        """Should compute Shapley value for each agent."""
        result = self.attributor.compute_full_attribution(
            self.beliefs, self.reliabilities
        )

        # Should have value for each modality
        for m in Modality.all():
            assert m in result.shapley_values
            # Values should be bounded (v in [0,1], so phi in [-1, 1])
            assert -1.5 <= result.shapley_values[m] <= 1.5

    def test_interaction_indices_symmetric(self):
        """I_ij should equal I_ji."""
        result = self.attributor.compute_full_attribution(
            self.beliefs, self.reliabilities
        )

        for m1 in Modality.all():
            for m2 in Modality.all():
                if m1 != m2:
                    I_12 = result.interaction_indices.get((m1, m2), 0)
                    I_21 = result.interaction_indices.get((m2, m1), 0)
                    assert abs(I_12 - I_21) < 1e-6

    def test_empty_coalition_uses_uniform(self):
        """Empty coalition should use uniform prior."""
        coalition_values = self.attributor.compute_coalition_values(
            self.beliefs, self.reliabilities
        )

        empty_value = coalition_values[frozenset()]
        # Uniform prior has max_prob = 1/K
        expected = 1.0 / self.K

        assert abs(empty_value - expected) < 1e-5

    def test_subtype_shapley(self):
        """Should compute Shapley values per subtype."""
        subtype_shapley = self.attributor.subtype_shapley(
            self.beliefs, self.reliabilities
        )

        # Should have entry for each subtype
        for subtype in NHLSubtype:
            assert subtype in subtype_shapley
            # Should have value for each modality
            for m in Modality.all():
                assert m in subtype_shapley[subtype]


class TestShapleyWithTrueLabel:
    """Tests for Shapley with true label (true_prob value function)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.K = 9
        np.random.seed(42)
        torch.manual_seed(42)

        self.attributor = ShapleyAttributor(
            agents=Modality.all(),
            geometric_median_fn=geometric_median_consensus,
            num_classes=self.K,
        )

    def test_efficiency_with_true_label(self):
        """Efficiency should hold with true_prob value function."""
        true_label = 3

        # Create beliefs centered on true label
        beliefs = {}
        for m in Modality.all():
            alpha = torch.ones(self.K) * 0.5
            alpha[true_label] = 5.0
            beliefs[m] = torch.distributions.Dirichlet(alpha).sample()

        reliabilities = {m: 0.25 for m in Modality.all()}

        result = self.attributor.compute_full_attribution(
            beliefs, reliabilities, true_label=true_label
        )

        assert result.efficiency_error < 1e-6


class TestAggregateShapley:
    """Tests for aggregating Shapley values across cases."""

    def test_aggregate_statistics(self):
        """Should compute mean, std, min, max across cases."""
        K = 9
        attributor = ShapleyAttributor(
            agents=Modality.all(),
            geometric_median_fn=geometric_median_consensus,
            num_classes=K,
        )

        # Generate multiple results
        results = []
        for _ in range(10):
            beliefs = {
                m: torch.softmax(torch.randn(K), dim=0)
                for m in Modality.all()
            }
            reliabilities = {m: 0.25 for m in Modality.all()}
            result = attributor.compute_full_attribution(beliefs, reliabilities)
            results.append(result)

        # Aggregate
        stats = aggregate_shapley_across_cases(results)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

        # Check structure
        for stat_name in ["mean", "std", "min", "max"]:
            for m in Modality.all():
                assert m in stats[stat_name]
