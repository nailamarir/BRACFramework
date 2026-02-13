"""Trust-bootstrapped reliability estimation for BRAC agents.

This module implements Innovation 1 of the BRAC framework: trust-bootstrapped
reliability estimation that anchors trust to a designated root-of-trust agent
(typically pathology, as it is the gold standard for lymphoma diagnosis).

The reliability r_i for each agent is computed as:
    r_i = tau_i * sigma(MLP(Q_i, C_i, S_i, H(b_i)))

where:
    tau_i = ReLU(cos_sim(b_i, b_root))  # behavioral trust
    H(b_i) = -sum_k b_ik * log(b_ik)     # belief entropy
    sigma = sigmoid
    MLP: R^4 -> R^1 with learnable parameters (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional
from dataclasses import dataclass

from brac.types import Modality, EvidenceQuality

logger = logging.getLogger(__name__)

EPS = 1e-10


@dataclass
class TrustResult:
    """Results from trust and reliability estimation.

    Attributes:
        trusts: Behavioral trust tau_i per agent
        reliabilities: Final reliability r_i per agent
        entropies: Belief entropy H(b_i) per agent
        mlp_scores: Raw MLP output (before sigmoid) per agent
    """
    trusts: dict[Modality, float]
    reliabilities: dict[Modality, float]
    entropies: dict[Modality, float]
    mlp_scores: dict[Modality, float]


class ReliabilityMLP(nn.Module):
    """Small MLP for learning reliability from evidence quality and entropy.

    Architecture: [4] -> [32, ReLU] -> [32, ReLU] -> [1]

    Input features:
        - Q: Data quality [0, 1]
        - C: Coverage [0, 1]
        - S: Consistency [0, 1]
        - H: Belief entropy [0, log(K)]
    """

    def __init__(self, hidden_size: int = 32):
        """Initialize the reliability MLP.

        Args:
            hidden_size: Number of units in hidden layers
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [Q, C, S, H], shape (..., 4)

        Returns:
            Raw reliability score (before sigmoid), shape (...)
        """
        return self.layers(x).squeeze(-1)


class TrustEstimator:
    """Trust-bootstrapped reliability estimator for BRAC agents.

    This class implements the hierarchical trust model where:
    1. Behavioral trust (tau_i) measures alignment with root-of-trust agent
    2. Evidence-based reliability combines quality scores with belief uncertainty
    3. Final reliability is the product of trust and evidence reliability

    The root-of-trust is typically the pathology agent, which serves as the
    anchor for diagnostic accuracy in lymphoma classification.
    """

    def __init__(
        self,
        root_of_trust: Modality = Modality.PATHOLOGY,
        learnable: bool = False,
        hidden_size: int = 32,
        default_weights: Optional[list[float]] = None,
    ):
        """Initialize the trust estimator.

        Args:
            root_of_trust: The modality to use as trust anchor
            learnable: Whether to use learnable MLP (vs. fixed linear combination)
            hidden_size: Hidden layer size for MLP if learnable=True
            default_weights: Weights [w_Q, w_C, w_S, w_H] for non-learnable mode.
                            Default: [0.4, 0.3, 0.2, -0.5]
        """
        self.root_of_trust = root_of_trust
        self.learnable = learnable

        if learnable:
            self.mlp = ReliabilityMLP(hidden_size)
        else:
            self.mlp = None
            # Default weights: Q, C, S have positive impact, entropy has negative
            self.weights = torch.tensor(
                default_weights or [0.4, 0.3, 0.2, -0.5],
                dtype=torch.float32
            )
            # Bias term for centering
            self.bias = 0.0

    def compute_entropy(self, belief: torch.Tensor) -> float:
        """Compute Shannon entropy of a belief distribution.

        H(b) = -sum_k b_k * log(b_k)

        Args:
            belief: Probability distribution, shape (K,)

        Returns:
            Entropy value (higher = more uncertain)
        """
        return -torch.sum(belief * torch.log(belief + EPS)).item()

    def compute_behavioral_trust(
        self,
        beliefs: dict[Modality, torch.Tensor],
    ) -> dict[Modality, float]:
        """Compute behavioral trust based on alignment with root-of-trust.

        Behavioral trust is the ReLU-clipped cosine similarity between
        each agent's belief and the root-of-trust agent's belief:

            tau_i = ReLU(cos_sim(b_i, b_root))

        This ensures:
        - Agents aligned with pathology get high trust
        - Agents contradicting pathology get zero trust (not negative)
        - The root-of-trust itself gets tau = 1.0

        Args:
            beliefs: Dictionary mapping Modality -> belief tensor

        Returns:
            Dictionary mapping Modality -> behavioral trust [0, 1]
        """
        if self.root_of_trust not in beliefs:
            raise ValueError(f"Root of trust {self.root_of_trust} not in beliefs")

        root_belief = beliefs[self.root_of_trust]
        trusts = {}

        for modality, belief in beliefs.items():
            if modality == self.root_of_trust:
                # Root of trust always has trust 1.0
                trusts[modality] = 1.0
            else:
                # Cosine similarity
                cos_sim = F.cosine_similarity(
                    belief.unsqueeze(0),
                    root_belief.unsqueeze(0)
                ).item()
                # ReLU clip: zero out negative similarities
                trusts[modality] = max(0.0, cos_sim)

        return trusts

    def compute_evidence_reliability(
        self,
        quality: EvidenceQuality,
        entropy: float,
    ) -> float:
        """Compute evidence-based reliability component.

        Using either learnable MLP or fixed linear combination:
            score = MLP([Q, C, S, H]) or w_Q*Q + w_C*C + w_S*S + w_H*H

        The final reliability component is sigmoid(score).

        Args:
            quality: Evidence quality factors (Q, C, S)
            entropy: Belief entropy H(b)

        Returns:
            Evidence reliability in [0, 1]
        """
        features = torch.tensor(
            [quality.Q, quality.C, quality.S, entropy],
            dtype=torch.float32
        )

        if self.learnable and self.mlp is not None:
            with torch.no_grad():
                score = self.mlp(features).item()
        else:
            score = torch.dot(self.weights, features).item() + self.bias

        return torch.sigmoid(torch.tensor(score)).item()

    def compute_reliability(
        self,
        beliefs: dict[Modality, torch.Tensor],
        qualities: dict[Modality, EvidenceQuality],
    ) -> TrustResult:
        """Compute full reliability scores for all agents.

        The reliability r_i combines behavioral trust with evidence reliability:
            r_i = tau_i * sigmoid(MLP(Q_i, C_i, S_i, H(b_i)))

        Args:
            beliefs: Dictionary mapping Modality -> belief tensor
            qualities: Dictionary mapping Modality -> EvidenceQuality

        Returns:
            TrustResult containing trusts, reliabilities, and intermediate values
        """
        # Compute behavioral trusts
        trusts = self.compute_behavioral_trust(beliefs)

        # Compute entropies and evidence reliabilities
        entropies = {}
        mlp_scores = {}
        reliabilities = {}

        for modality in beliefs.keys():
            belief = beliefs[modality]
            quality = qualities.get(modality, EvidenceQuality(Q=0.5, C=0.5, S=0.5))

            # Compute entropy
            entropy = self.compute_entropy(belief)
            entropies[modality] = entropy

            # Compute evidence reliability
            evidence_rel = self.compute_evidence_reliability(quality, entropy)

            # Store MLP score (for debugging)
            features = torch.tensor(
                [quality.Q, quality.C, quality.S, entropy],
                dtype=torch.float32
            )
            if self.learnable and self.mlp is not None:
                with torch.no_grad():
                    mlp_scores[modality] = self.mlp(features).item()
            else:
                mlp_scores[modality] = torch.dot(self.weights, features).item()

            # Final reliability = trust * evidence_reliability
            reliabilities[modality] = trusts[modality] * evidence_rel

        logger.debug(f"Trusts: {trusts}")
        logger.debug(f"Reliabilities: {reliabilities}")

        return TrustResult(
            trusts=trusts,
            reliabilities=reliabilities,
            entropies=entropies,
            mlp_scores=mlp_scores,
        )

    def get_reliability_dict(
        self,
        beliefs: dict[Modality, torch.Tensor],
        qualities: dict[Modality, EvidenceQuality],
    ) -> dict[Modality, float]:
        """Convenience method to get just the reliability dictionary.

        Args:
            beliefs: Dictionary mapping Modality -> belief tensor
            qualities: Dictionary mapping Modality -> EvidenceQuality

        Returns:
            Dictionary mapping Modality -> reliability score
        """
        return self.compute_reliability(beliefs, qualities).reliabilities

    def train_mlp(
        self,
        training_data: list[tuple[dict[Modality, torch.Tensor], dict[Modality, EvidenceQuality], int]],
        epochs: int = 100,
        lr: float = 0.01,
    ) -> list[float]:
        """Train the reliability MLP on labeled calibration data.

        The MLP is trained to maximize the reliability of agents that
        contribute to correct consensus predictions.

        Args:
            training_data: List of (beliefs, qualities, true_label) tuples
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            List of loss values per epoch
        """
        if not self.learnable or self.mlp is None:
            raise ValueError("Cannot train MLP when learnable=False")

        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0

            for beliefs, qualities, true_label in training_data:
                optimizer.zero_grad()

                # Compute reliabilities
                result = self.compute_reliability(beliefs, qualities)

                # Loss: encourage reliable agents to be confident on true class
                loss = 0.0
                for modality, reliability in result.reliabilities.items():
                    belief = beliefs[modality]
                    # Cross-entropy weighted by reliability
                    ce = -torch.log(belief[true_label] + EPS)
                    loss += reliability * ce

                loss_tensor = torch.tensor(loss, requires_grad=True)
                loss_tensor.backward()
                optimizer.step()

                epoch_loss += loss

            losses.append(epoch_loss / len(training_data))
            logger.debug(f"Epoch {epoch}: loss = {losses[-1]:.4f}")

        return losses


def create_default_trust_estimator(
    learnable: bool = False,
) -> TrustEstimator:
    """Factory function to create a trust estimator with default settings.

    Args:
        learnable: Whether to use learnable MLP

    Returns:
        Configured TrustEstimator instance
    """
    return TrustEstimator(
        root_of_trust=Modality.PATHOLOGY,
        learnable=learnable,
        default_weights=[0.4, 0.3, 0.2, -0.5],
    )
