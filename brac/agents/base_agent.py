"""Abstract base class for BRAC diagnostic agents.

All agents in the BRAC framework inherit from BaseAgent and implement
the inference method to produce AgentOutput from modality-specific evidence.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import torch

from brac.types import AgentOutput, Modality, EvidenceQuality, SemanticEvidence


class BaseAgent(ABC):
    """Abstract base class for BRAC diagnostic agents.

    Each agent processes evidence from a specific modality and produces
    a belief distribution over NHL subtypes along with quality metrics.

    Subclasses must implement:
    - forward(): Process raw evidence and return AgentOutput
    """

    def __init__(
        self,
        modality: Modality,
        num_classes: int = 9,
        device: str = "cpu",
    ):
        """Initialize the base agent.

        Args:
            modality: The diagnostic modality this agent represents
            num_classes: Number of NHL subtypes (K)
            device: Device for tensor computations
        """
        self.modality = modality
        self.num_classes = num_classes
        self.device = device

    @abstractmethod
    def forward(self, evidence: Any) -> AgentOutput:
        """Process evidence and return agent output.

        This is the main inference method. Subclasses implement this
        with modality-specific processing.

        Args:
            evidence: Raw evidence data (type depends on modality)

        Returns:
            AgentOutput with belief distribution and quality metrics
        """
        pass

    def __call__(self, evidence: Any) -> AgentOutput:
        """Make agent callable."""
        return self.forward(evidence)

    def to(self, device: str) -> "BaseAgent":
        """Move agent to specified device."""
        self.device = device
        return self

    def _create_uniform_belief(self) -> torch.Tensor:
        """Create uniform belief distribution (maximum entropy)."""
        return torch.ones(self.num_classes, device=self.device) / self.num_classes

    def _normalize_belief(self, belief: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Normalize belief to probability simplex."""
        belief = torch.clamp(belief, min=eps)
        return belief / belief.sum()

    def _compute_entropy(self, belief: torch.Tensor, eps: float = 1e-10) -> float:
        """Compute entropy of belief distribution."""
        return -torch.sum(belief * torch.log(belief + eps)).item()

    def _default_quality(self) -> EvidenceQuality:
        """Return default evidence quality (medium confidence)."""
        return EvidenceQuality(Q=0.5, C=0.5, S=0.5)


class NeuralAgent(BaseAgent):
    """Base class for neural network-based agents.

    Extends BaseAgent with common neural network functionality
    like parameter management and training mode.
    """

    def __init__(
        self,
        modality: Modality,
        num_classes: int = 9,
        device: str = "cpu",
    ):
        super().__init__(modality, num_classes, device)
        self._model = None
        self._training = False

    @property
    def model(self):
        """Access the underlying neural network model."""
        return self._model

    def train(self, mode: bool = True) -> "NeuralAgent":
        """Set training mode."""
        self._training = mode
        if self._model is not None:
            self._model.train(mode)
        return self

    def eval(self) -> "NeuralAgent":
        """Set evaluation mode."""
        return self.train(False)

    def parameters(self):
        """Return model parameters for optimization."""
        if self._model is not None:
            return self._model.parameters()
        return iter([])

    def to(self, device: str) -> "NeuralAgent":
        """Move agent and model to device."""
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self


class EnsembleAgent(BaseAgent):
    """Agent that combines multiple sub-agents.

    Useful for creating agent ensembles or multi-view processing
    within a single modality.
    """

    def __init__(
        self,
        modality: Modality,
        sub_agents: list[BaseAgent],
        num_classes: int = 9,
        aggregation: str = "mean",
    ):
        """Initialize ensemble agent.

        Args:
            modality: The diagnostic modality
            sub_agents: List of sub-agents to ensemble
            num_classes: Number of NHL subtypes
            aggregation: How to combine sub-agent beliefs ("mean", "max", "weighted")
        """
        super().__init__(modality, num_classes)
        self.sub_agents = sub_agents
        self.aggregation = aggregation

    def forward(self, evidence: Any) -> AgentOutput:
        """Combine outputs from all sub-agents."""
        outputs = [agent(evidence) for agent in self.sub_agents]

        # Combine beliefs
        beliefs = torch.stack([out.belief for out in outputs])

        if self.aggregation == "mean":
            combined_belief = beliefs.mean(dim=0)
        elif self.aggregation == "max":
            combined_belief = beliefs.max(dim=0)[0]
        else:
            combined_belief = beliefs.mean(dim=0)

        combined_belief = self._normalize_belief(combined_belief)

        # Average quality scores
        avg_Q = sum(out.quality.Q for out in outputs) / len(outputs)
        avg_C = sum(out.quality.C for out in outputs) / len(outputs)
        avg_S = sum(out.quality.S for out in outputs) / len(outputs)

        # Combine evidence
        all_evidence = []
        for out in outputs:
            all_evidence.extend(out.evidence)

        return AgentOutput(
            belief=combined_belief,
            quality=EvidenceQuality(Q=avg_Q, C=avg_C, S=avg_S),
            evidence=all_evidence,
            modality=self.modality,
        )
