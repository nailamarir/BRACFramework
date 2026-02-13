"""Main BRAC orchestrator implementing Algorithm 1.

This module integrates all components of the BRAC framework:
1. Trust-bootstrapped reliability estimation
2. Riemannian geometric median consensus
3. Conformal prediction for uncertainty quantification
4. Shapley attribution for explainability

The orchestrator coordinates the full diagnostic pipeline from agent
outputs to final diagnosis with coverage guarantees.
"""

import torch
import yaml
import logging
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from brac.types import (
    Modality, NHLSubtype, AgentOutput, BRACResult, EvidenceQuality
)
from brac.consensus.trust import TrustEstimator, TrustResult
from brac.consensus.geometric_median import (
    geometric_median_consensus, full_consensus_protocol, WeiszfeldResult
)
from brac.consensus.conformal import ConformalPredictor, ConformalResult
from brac.consensus.shapley import ShapleyAttributor, ShapleyResult

logger = logging.getLogger(__name__)


@dataclass
class BRACConfig:
    """Configuration for the BRAC orchestrator.

    Attributes:
        num_classes: Number of NHL subtypes (K)
        num_agents: Number of diagnostic agents (n)
        root_of_trust: Modality to use as trust anchor
        max_outer_rounds: Maximum consensus rounds (T)
        max_weiszfeld_iters: Maximum Weiszfeld iterations per round (L)
        outer_tol: Convergence tolerance for outer loop
        weiszfeld_tol: Weiszfeld convergence tolerance
        lambda_0: Base receptivity parameter
        alpha: Conformal prediction coverage target (1-alpha)
        max_set_size_accept: Maximum prediction set size for acceptance
        compute_shapley: Whether to compute Shapley values
        compute_interactions: Whether to compute interaction indices
        learnable_trust: Whether to use learnable trust MLP
    """
    num_classes: int = 9
    num_agents: int = 4
    root_of_trust: Modality = Modality.PATHOLOGY
    max_outer_rounds: int = 10
    max_weiszfeld_iters: int = 50
    outer_tol: float = 1e-4
    weiszfeld_tol: float = 1e-8
    lambda_0: float = 0.3
    alpha: float = 0.05
    max_set_size_accept: int = 2
    compute_shapley: bool = True
    compute_interactions: bool = True
    learnable_trust: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "BRACConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            BRACConfig instance
        """
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Extract relevant sections
        kwargs = {}

        if "num_subtypes" in config_dict:
            kwargs["num_classes"] = config_dict["num_subtypes"]
        if "num_agents" in config_dict:
            kwargs["num_agents"] = config_dict["num_agents"]
        if "root_of_trust" in config_dict:
            kwargs["root_of_trust"] = Modality(config_dict["root_of_trust"])

        # Consensus config
        if "consensus" in config_dict:
            cc = config_dict["consensus"]
            if "max_outer_rounds" in cc:
                kwargs["max_outer_rounds"] = cc["max_outer_rounds"]
            if "max_weiszfeld_iters" in cc:
                kwargs["max_weiszfeld_iters"] = cc["max_weiszfeld_iters"]
            if "convergence_threshold" in cc:
                kwargs["outer_tol"] = cc["convergence_threshold"]
            if "weiszfeld_tol" in cc:
                kwargs["weiszfeld_tol"] = cc["weiszfeld_tol"]
            if "lambda_0" in cc:
                kwargs["lambda_0"] = cc["lambda_0"]

        # Conformal config
        if "conformal" in config_dict:
            cf = config_dict["conformal"]
            if "alpha" in cf:
                kwargs["alpha"] = cf["alpha"]

        # Decision config
        if "decision" in config_dict:
            dc = config_dict["decision"]
            if "max_set_size_accept" in dc:
                kwargs["max_set_size_accept"] = dc["max_set_size_accept"]

        # Shapley config
        if "shapley" in config_dict:
            sc = config_dict["shapley"]
            if "compute_interactions" in sc:
                kwargs["compute_interactions"] = sc["compute_interactions"]

        # Trust config
        if "trust" in config_dict:
            tc = config_dict["trust"]
            if "learnable" in tc:
                kwargs["learnable_trust"] = tc["learnable"]

        return cls(**kwargs)


class BRACOrchestrator:
    """Main orchestrator for the BRAC framework.

    Implements Algorithm 1 from the paper, coordinating:
    1. Trust estimation
    2. Iterative consensus
    3. Conformal prediction
    4. Shapley attribution
    """

    def __init__(self, config: BRACConfig):
        """Initialize the BRAC orchestrator.

        Args:
            config: BRACConfig with all hyperparameters
        """
        self.config = config

        # Initialize components
        self.trust_estimator = TrustEstimator(
            root_of_trust=config.root_of_trust,
            learnable=config.learnable_trust,
        )

        self.conformal = ConformalPredictor(
            alpha=config.alpha,
            max_set_size_accept=config.max_set_size_accept,
        )

        # Shapley attributor initialized lazily (needs geometric median fn)
        self._shapley = None

        logger.info(f"BRACOrchestrator initialized with config: {config}")

    @classmethod
    def from_yaml(cls, path: str) -> "BRACOrchestrator":
        """Create orchestrator from YAML config file.

        Args:
            path: Path to YAML configuration

        Returns:
            Configured BRACOrchestrator
        """
        config = BRACConfig.from_yaml(path)
        return cls(config)

    def _get_shapley_attributor(self) -> ShapleyAttributor:
        """Get or create Shapley attributor."""
        if self._shapley is None:
            self._shapley = ShapleyAttributor(
                agents=Modality.all(),
                geometric_median_fn=geometric_median_consensus,
                num_classes=self.config.num_classes,
            )
        return self._shapley

    def run(
        self,
        agent_outputs: dict[Modality, AgentOutput],
        calibrated: bool = True,
    ) -> BRACResult:
        """Execute the full BRAC pipeline.

        Implements Algorithm 1 from the paper:
        1. Extract beliefs and qualities from agent outputs
        2. Compute trust-bootstrapped reliabilities
        3. Run iterative geometric median consensus
        4. Generate conformal prediction set
        5. Compute Shapley attributions

        Args:
            agent_outputs: Dictionary mapping Modality -> AgentOutput
            calibrated: If False, skip conformal prediction (not calibrated yet)

        Returns:
            BRACResult with full diagnostic output
        """
        config = self.config

        # Step 1: Extract beliefs and qualities
        beliefs = {m: out.belief for m, out in agent_outputs.items()}
        qualities = {m: out.quality for m, out in agent_outputs.items()}

        # Step 2: Compute trust-bootstrapped reliabilities
        trust_result = self.trust_estimator.compute_reliability(beliefs, qualities)
        reliabilities = trust_result.reliabilities

        logger.debug(f"Trusts: {trust_result.trusts}")
        logger.debug(f"Reliabilities: {reliabilities}")

        # Step 3: Iterative geometric median consensus
        belief_tensor = torch.stack([beliefs[m] for m in Modality.all()])
        rel_tensor = torch.tensor([reliabilities[m] for m in Modality.all()])

        consensus, num_rounds, round_results = full_consensus_protocol(
            initial_beliefs=belief_tensor,
            reliabilities=rel_tensor,
            max_outer_rounds=config.max_outer_rounds,
            max_weiszfeld_iters=config.max_weiszfeld_iters,
            outer_tol=config.outer_tol,
            weiszfeld_tol=config.weiszfeld_tol,
            lambda_0=config.lambda_0,
        )

        logger.debug(f"Consensus converged in {num_rounds} rounds")

        # Step 4: Conformal prediction set
        if calibrated and self.conformal.is_calibrated:
            conformal_result = self.conformal.predict_set(consensus)
            prediction_set = conformal_result.prediction_set
            accepted = conformal_result.decision == "accept"
        else:
            # Not calibrated: return top-k as prediction set
            sorted_probs, sorted_indices = torch.sort(consensus, descending=True)
            prediction_set = [
                NHLSubtype.from_index(sorted_indices[i].item())
                for i in range(min(2, config.num_classes))
            ]
            accepted = True  # Default accept when not calibrated

        # Step 5: Shapley attribution
        if config.compute_shapley:
            shapley_attributor = self._get_shapley_attributor()
            shapley_result = shapley_attributor.compute_full_attribution(
                beliefs=beliefs,
                reliabilities=reliabilities,
            )
            shapley_values = shapley_result.shapley_values
            interaction_indices = shapley_result.interaction_indices if config.compute_interactions else {}
        else:
            shapley_values = {m: 0.0 for m in Modality.all()}
            interaction_indices = {}

        # Step 6: Final diagnosis
        diagnosis_idx = consensus.argmax().item()
        diagnosis = NHLSubtype.from_index(diagnosis_idx)
        confidence = consensus.max().item()

        return BRACResult(
            diagnosis=diagnosis,
            consensus_belief=consensus,
            prediction_set=prediction_set,
            prediction_set_size=len(prediction_set),
            shapley_values=shapley_values,
            interaction_indices=interaction_indices,
            agent_reliabilities=reliabilities,
            agent_trusts=trust_result.trusts,
            convergence_rounds=num_rounds,
            accepted=accepted,
            confidence=confidence,
        )

    def calibrate(
        self,
        calibration_data: list[tuple[dict[Modality, AgentOutput], int]],
    ) -> float:
        """Calibrate conformal predictor on labeled data.

        Args:
            calibration_data: List of (agent_outputs, true_label) tuples

        Returns:
            Calibrated threshold q_hat
        """
        # Run BRAC on each calibration case to get consensus beliefs
        consensus_beliefs = []
        true_labels = []

        for agent_outputs, true_label in calibration_data:
            # Run without conformal (not calibrated yet)
            result = self.run(agent_outputs, calibrated=False)
            consensus_beliefs.append(result.consensus_belief)
            true_labels.append(true_label)

        beliefs_tensor = torch.stack(consensus_beliefs)
        labels_tensor = torch.tensor(true_labels)

        # Calibrate conformal predictor
        q_hat = self.conformal.calibrate(beliefs_tensor, labels_tensor)

        logger.info(f"Conformal predictor calibrated with q_hat = {q_hat:.4f}")

        return q_hat

    def evaluate(
        self,
        test_data: list[tuple[dict[Modality, AgentOutput], int]],
    ) -> dict[str, float]:
        """Evaluate BRAC on test data.

        Args:
            test_data: List of (agent_outputs, true_label) tuples

        Returns:
            Dictionary with evaluation metrics
        """
        correct = 0
        covered = 0
        total_set_size = 0
        accepted_count = 0

        for agent_outputs, true_label in test_data:
            result = self.run(agent_outputs, calibrated=True)

            # Accuracy
            if result.diagnosis.value == true_label:
                correct += 1

            # Coverage
            if any(s.value == true_label for s in result.prediction_set):
                covered += 1

            total_set_size += result.prediction_set_size
            if result.accepted:
                accepted_count += 1

        n = len(test_data)
        return {
            "accuracy": correct / n,
            "coverage": covered / n,
            "avg_set_size": total_set_size / n,
            "acceptance_rate": accepted_count / n,
        }


def create_orchestrator_from_dict(config_dict: dict) -> BRACOrchestrator:
    """Create orchestrator from configuration dictionary.

    Args:
        config_dict: Dictionary with configuration parameters

    Returns:
        Configured BRACOrchestrator
    """
    config = BRACConfig(
        num_classes=config_dict.get("num_classes", 9),
        num_agents=config_dict.get("num_agents", 4),
        max_outer_rounds=config_dict.get("max_outer_rounds", 10),
        max_weiszfeld_iters=config_dict.get("max_weiszfeld_iters", 50),
        outer_tol=config_dict.get("outer_tol", 1e-4),
        weiszfeld_tol=config_dict.get("weiszfeld_tol", 1e-8),
        lambda_0=config_dict.get("lambda_0", 0.3),
        alpha=config_dict.get("alpha", 0.05),
        max_set_size_accept=config_dict.get("max_set_size_accept", 2),
        compute_shapley=config_dict.get("compute_shapley", True),
        compute_interactions=config_dict.get("compute_interactions", True),
    )
    return BRACOrchestrator(config)
