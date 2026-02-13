"""Conformal prediction for uncertainty quantification in BRAC.

This module implements Innovation 4 of the BRAC framework: distribution-free
prediction sets with guaranteed coverage using Adaptive Prediction Sets (APS).

Conformal prediction provides the guarantee:
    P(h_true in C_alpha) >= 1 - alpha

without any distributional assumptions, only requiring exchangeability of
calibration and test data.

The key insight is that we can use non-conformity scores from calibration
data to construct prediction sets that include the true label with
probability at least 1-alpha.
"""

import torch
import numpy as np
from typing import Optional
import logging
from dataclasses import dataclass

from brac.types import NHLSubtype

logger = logging.getLogger(__name__)

EPS = 1e-10


@dataclass
class ConformalResult:
    """Result from conformal prediction.

    Attributes:
        prediction_set: List of NHL subtypes in the prediction set
        set_size: Size of the prediction set |C_alpha|
        threshold: The non-conformity threshold q_hat used
        max_probability: Maximum probability in the consensus belief
        decision: "accept" if |C_alpha| <= max_accept, else "escalate"
    """
    prediction_set: list[NHLSubtype]
    set_size: int
    threshold: float
    max_probability: float
    decision: str


class ConformalPredictor:
    """Conformal prediction for BRAC uncertainty quantification.

    Implements Adaptive Prediction Sets (APS) which creates prediction sets
    by including classes in order of decreasing probability until the
    cumulative probability exceeds a calibrated threshold.

    The calibration process computes non-conformity scores on held-out data:
        s_j = 1 - p*(h_j^true)

    The prediction threshold is the (1-alpha)(1 + 1/N) quantile of these scores.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_set_size_accept: int = 2,
    ):
        """Initialize the conformal predictor.

        Args:
            alpha: Target miscoverage rate (1-alpha is target coverage)
            max_set_size_accept: Maximum set size for "accept" decision
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.alpha = alpha
        self.max_set_size_accept = max_set_size_accept
        self.q_hat: Optional[float] = None
        self.calibration_scores: Optional[torch.Tensor] = None
        self.is_calibrated = False

    def calibrate(
        self,
        consensus_beliefs: torch.Tensor,
        true_labels: torch.Tensor,
    ) -> float:
        """Calibrate the conformal predictor on held-out data.

        Computes non-conformity scores and the threshold q_hat such that
        the prediction sets will have coverage >= 1-alpha.

        Args:
            consensus_beliefs: Consensus beliefs for calibration cases, shape (N, K)
            true_labels: True class indices, shape (N,)

        Returns:
            The calibrated threshold q_hat
        """
        N = consensus_beliefs.shape[0]
        device = consensus_beliefs.device

        # Compute non-conformity scores: s_j = 1 - p*(h_j^true)
        scores = torch.zeros(N, device=device)
        for j in range(N):
            true_prob = consensus_beliefs[j, true_labels[j].long()].item()
            scores[j] = 1 - true_prob

        self.calibration_scores = scores

        # Compute quantile: ceil((N+1)(1-alpha)) / N quantile
        # This is the finite-sample correction for coverage guarantee
        quantile_level = np.ceil((N + 1) * (1 - self.alpha)) / N
        quantile_level = min(quantile_level, 1.0)  # Cap at 1

        self.q_hat = torch.quantile(scores, quantile_level).item()
        self.is_calibrated = True

        logger.info(f"Calibrated conformal predictor: q_hat = {self.q_hat:.4f} "
                   f"(alpha={self.alpha}, N={N})")

        return self.q_hat

    def predict_set(
        self,
        consensus_belief: torch.Tensor,
    ) -> ConformalResult:
        """Construct the adaptive prediction set for a new case.

        The APS algorithm:
        1. Sort classes by decreasing probability
        2. Include classes until cumulative probability >= 1 - q_hat
        3. Return the set of included classes

        Args:
            consensus_belief: Consensus belief for the new case, shape (K,)

        Returns:
            ConformalResult with prediction set and metadata
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalPredictor must be calibrated before prediction")

        K = consensus_belief.shape[0]

        # Sort classes by decreasing probability
        sorted_probs, sorted_indices = torch.sort(consensus_belief, descending=True)

        # Cumulative probability threshold
        threshold = 1 - self.q_hat

        # Include classes until cumsum >= threshold
        cumsum = 0.0
        prediction_set = []

        for i in range(K):
            class_idx = sorted_indices[i].item()
            prediction_set.append(NHLSubtype.from_index(class_idx))
            cumsum += sorted_probs[i].item()

            if cumsum >= threshold:
                break

        # Always include at least one class
        if not prediction_set:
            top_class = sorted_indices[0].item()
            prediction_set = [NHLSubtype.from_index(top_class)]

        set_size = len(prediction_set)
        decision = "accept" if set_size <= self.max_set_size_accept else "escalate"

        return ConformalResult(
            prediction_set=prediction_set,
            set_size=set_size,
            threshold=self.q_hat,
            max_probability=sorted_probs[0].item(),
            decision=decision,
        )

    def predict_set_indices(
        self,
        consensus_belief: torch.Tensor,
    ) -> list[int]:
        """Convenience method to get prediction set as class indices.

        Args:
            consensus_belief: Consensus belief, shape (K,)

        Returns:
            List of class indices in the prediction set
        """
        result = self.predict_set(consensus_belief)
        return [s.value for s in result.prediction_set]

    def marginal_coverage(
        self,
        consensus_beliefs: torch.Tensor,
        true_labels: torch.Tensor,
    ) -> float:
        """Compute empirical marginal coverage on a test set.

        The marginal coverage should be >= 1 - alpha by the conformal guarantee.

        Args:
            consensus_beliefs: Consensus beliefs for test cases, shape (N, K)
            true_labels: True class indices, shape (N,)

        Returns:
            Empirical coverage rate
        """
        N = consensus_beliefs.shape[0]
        covered = 0

        for j in range(N):
            pred_set = self.predict_set_indices(consensus_beliefs[j])
            true_label = true_labels[j].item()

            if true_label in pred_set:
                covered += 1

        coverage = covered / N
        logger.info(f"Marginal coverage: {coverage:.2%} (target: {1-self.alpha:.2%})")

        return coverage

    def conditional_coverage(
        self,
        consensus_beliefs: torch.Tensor,
        true_labels: torch.Tensor,
    ) -> dict[int, float]:
        """Compute coverage conditional on true class.

        Useful for detecting if coverage varies across classes
        (conformal guarantees marginal, not conditional coverage).

        Args:
            consensus_beliefs: Consensus beliefs, shape (N, K)
            true_labels: True class indices, shape (N,)

        Returns:
            Dictionary mapping class index -> conditional coverage
        """
        N = consensus_beliefs.shape[0]
        K = consensus_beliefs.shape[1]

        # Count per class
        class_total = {k: 0 for k in range(K)}
        class_covered = {k: 0 for k in range(K)}

        for j in range(N):
            pred_set = self.predict_set_indices(consensus_beliefs[j])
            true_label = true_labels[j].item()

            class_total[true_label] += 1
            if true_label in pred_set:
                class_covered[true_label] += 1

        # Compute conditional coverage
        conditional = {}
        for k in range(K):
            if class_total[k] > 0:
                conditional[k] = class_covered[k] / class_total[k]

        return conditional

    def average_set_size(
        self,
        consensus_beliefs: torch.Tensor,
    ) -> float:
        """Compute average prediction set size.

        Smaller sets are more informative while maintaining coverage.

        Args:
            consensus_beliefs: Consensus beliefs, shape (N, K)

        Returns:
            Average set size
        """
        N = consensus_beliefs.shape[0]
        total_size = 0

        for j in range(N):
            result = self.predict_set(consensus_beliefs[j])
            total_size += result.set_size

        return total_size / N

    def set_size_distribution(
        self,
        consensus_beliefs: torch.Tensor,
    ) -> dict[int, int]:
        """Compute distribution of prediction set sizes.

        Args:
            consensus_beliefs: Consensus beliefs, shape (N, K)

        Returns:
            Dictionary mapping set size -> count
        """
        N = consensus_beliefs.shape[0]
        size_counts = {}

        for j in range(N):
            result = self.predict_set(consensus_beliefs[j])
            size = result.set_size
            size_counts[size] = size_counts.get(size, 0) + 1

        return dict(sorted(size_counts.items()))

    def calibrate_and_evaluate(
        self,
        cal_beliefs: torch.Tensor,
        cal_labels: torch.Tensor,
        test_beliefs: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> dict:
        """Calibrate on calibration set and evaluate on test set.

        Convenience method for experiments.

        Args:
            cal_beliefs: Calibration consensus beliefs, shape (N_cal, K)
            cal_labels: Calibration true labels, shape (N_cal,)
            test_beliefs: Test consensus beliefs, shape (N_test, K)
            test_labels: Test true labels, shape (N_test,)

        Returns:
            Dictionary with evaluation metrics
        """
        # Calibrate
        self.calibrate(cal_beliefs, cal_labels)

        # Evaluate
        coverage = self.marginal_coverage(test_beliefs, test_labels)
        avg_size = self.average_set_size(test_beliefs)
        size_dist = self.set_size_distribution(test_beliefs)
        cond_coverage = self.conditional_coverage(test_beliefs, test_labels)

        return {
            "q_hat": self.q_hat,
            "alpha": self.alpha,
            "target_coverage": 1 - self.alpha,
            "empirical_coverage": coverage,
            "coverage_gap": coverage - (1 - self.alpha),
            "average_set_size": avg_size,
            "set_size_distribution": size_dist,
            "conditional_coverage": cond_coverage,
        }


def split_calibration_test(
    data: torch.Tensor,
    labels: torch.Tensor,
    cal_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split data into calibration and test sets.

    Args:
        data: Input data, shape (N, ...)
        labels: Labels, shape (N,)
        cal_fraction: Fraction for calibration set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (cal_data, cal_labels, test_data, test_labels)
    """
    N = data.shape[0]
    torch.manual_seed(seed)

    perm = torch.randperm(N)
    n_cal = int(N * cal_fraction)

    cal_idx = perm[:n_cal]
    test_idx = perm[n_cal:]

    return (
        data[cal_idx],
        labels[cal_idx],
        data[test_idx],
        labels[test_idx],
    )
