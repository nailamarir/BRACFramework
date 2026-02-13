"""Unit tests for BRAC orchestrator.

Tests verify:
- End-to-end pipeline execution
- Configuration loading
- Result structure
"""

import pytest
import torch
import numpy as np

from brac.orchestrator import BRACOrchestrator, BRACConfig, create_orchestrator_from_dict
from brac.types import Modality, NHLSubtype, AgentOutput, EvidenceQuality, BRACResult
from brac.agents.mock_agent import MockAgentFactory


class TestBRACConfig:
    """Tests for BRACConfig class."""

    def test_default_config(self):
        """Default config should have valid values."""
        config = BRACConfig()

        assert config.num_classes == 9
        assert config.num_agents == 4
        assert config.root_of_trust == Modality.PATHOLOGY
        assert config.max_outer_rounds > 0
        assert 0 < config.alpha < 1

    def test_config_from_dict(self):
        """Should create config from dictionary."""
        config_dict = {
            "num_classes": 5,
            "alpha": 0.10,
            "max_outer_rounds": 15,
        }
        config = BRACConfig(**config_dict)

        assert config.num_classes == 5
        assert config.alpha == 0.10
        assert config.max_outer_rounds == 15


class TestBRACOrchestrator:
    """Tests for BRACOrchestrator class."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        torch.manual_seed(42)

        self.config = BRACConfig(
            num_classes=9,
            alpha=0.05,
            compute_shapley=True,
        )
        self.orchestrator = BRACOrchestrator(self.config)

        # Create mock agent outputs
        self.factory = MockAgentFactory(num_classes=9, seed=42)
        self.agents = self.factory.create_all_agents()
        self.true_label = 3
        self.agent_outputs = self.factory.generate_case(
            self.agents, true_label=self.true_label
        )

    def test_run_returns_brac_result(self):
        """Run should return BRACResult."""
        result = self.orchestrator.run(self.agent_outputs, calibrated=False)

        assert isinstance(result, BRACResult)
        assert isinstance(result.diagnosis, NHLSubtype)
        assert isinstance(result.consensus_belief, torch.Tensor)

    def test_consensus_on_simplex(self):
        """Consensus belief should be valid distribution."""
        result = self.orchestrator.run(self.agent_outputs, calibrated=False)

        consensus = result.consensus_belief
        assert abs(consensus.sum().item() - 1.0) < 1e-5
        assert (consensus >= 0).all()

    def test_shapley_computed(self):
        """Shapley values should be computed when enabled."""
        result = self.orchestrator.run(self.agent_outputs, calibrated=False)

        # Should have Shapley value for each modality
        for m in Modality.all():
            assert m in result.shapley_values

    def test_reliabilities_computed(self):
        """Agent reliabilities should be computed."""
        result = self.orchestrator.run(self.agent_outputs, calibrated=False)

        for m in Modality.all():
            assert m in result.agent_reliabilities
            assert 0 <= result.agent_reliabilities[m] <= 1

    def test_trusts_computed(self):
        """Agent trusts should be computed."""
        result = self.orchestrator.run(self.agent_outputs, calibrated=False)

        for m in Modality.all():
            assert m in result.agent_trusts
            assert 0 <= result.agent_trusts[m] <= 1

        # Pathology (root of trust) should have trust 1.0
        assert result.agent_trusts[Modality.PATHOLOGY] == 1.0

    def test_prediction_set_structure(self):
        """Prediction set should contain NHLSubtype enums."""
        result = self.orchestrator.run(self.agent_outputs, calibrated=False)

        assert len(result.prediction_set) > 0
        for subtype in result.prediction_set:
            assert isinstance(subtype, NHLSubtype)

    def test_convergence_tracked(self):
        """Should track convergence rounds."""
        result = self.orchestrator.run(self.agent_outputs, calibrated=False)

        assert result.convergence_rounds > 0
        assert result.convergence_rounds <= self.config.max_outer_rounds

    def test_summary_string(self):
        """Summary should be a valid string."""
        result = self.orchestrator.run(self.agent_outputs, calibrated=False)
        summary = result.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert result.diagnosis.name in summary


class TestOrchestratorCalibration:
    """Tests for orchestrator calibration."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        torch.manual_seed(42)

        self.config = BRACConfig(num_classes=9, alpha=0.05)
        self.orchestrator = BRACOrchestrator(self.config)
        self.factory = MockAgentFactory(num_classes=9, seed=42)

    def test_calibration(self):
        """Should calibrate conformal predictor."""
        # Generate calibration data
        agents = self.factory.create_all_agents()
        cal_data = []
        for i in range(50):
            true_label = i % 9
            outputs = self.factory.generate_case(agents, true_label)
            cal_data.append((outputs, true_label))

        q_hat = self.orchestrator.calibrate(cal_data)

        assert self.orchestrator.conformal.is_calibrated
        assert 0 <= q_hat <= 1

    def test_evaluation(self):
        """Should evaluate on test data after calibration."""
        agents = self.factory.create_all_agents()

        # Calibration data
        cal_data = []
        for i in range(50):
            true_label = i % 9
            outputs = self.factory.generate_case(agents, true_label)
            cal_data.append((outputs, true_label))

        self.orchestrator.calibrate(cal_data)

        # Test data
        test_data = []
        for i in range(50):
            true_label = (i + 5) % 9
            outputs = self.factory.generate_case(agents, true_label)
            test_data.append((outputs, true_label))

        metrics = self.orchestrator.evaluate(test_data)

        assert "accuracy" in metrics
        assert "coverage" in metrics
        assert "avg_set_size" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["coverage"] <= 1


class TestOrchestratorFactory:
    """Tests for orchestrator factory functions."""

    def test_from_dict(self):
        """Should create orchestrator from dictionary."""
        config_dict = {
            "num_classes": 9,
            "alpha": 0.10,
            "compute_shapley": False,
        }
        orchestrator = create_orchestrator_from_dict(config_dict)

        assert orchestrator.config.num_classes == 9
        assert orchestrator.config.alpha == 0.10
        assert not orchestrator.config.compute_shapley
