"""Core type definitions for the BRAC framework.

This module defines all dataclasses, enums, and type aliases used throughout
the BRAC framework for Byzantine-resilient multimodal NHL subtyping.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
import numpy as np


class NHLSubtype(Enum):
    """WHO-classified B-cell NHL subtypes. K=9 classes.

    These subtypes follow the WHO classification of mature B-cell neoplasms,
    focusing on the most common non-Hodgkin lymphoma subtypes.
    """
    FL = 0          # Follicular Lymphoma
    MCL = 1         # Mantle Cell Lymphoma
    CLL_SLL = 2     # Chronic Lymphocytic Leukemia / Small Lymphocytic Lymphoma
    DLBCL_GCB = 3   # Diffuse Large B-Cell Lymphoma, Germinal Center B-cell
    DLBCL_ABC = 4   # Diffuse Large B-Cell Lymphoma, Activated B-cell
    MZL = 5         # Marginal Zone Lymphoma
    BL = 6          # Burkitt Lymphoma
    LPL = 7         # Lymphoplasmacytic Lymphoma
    HCL = 8         # Hairy Cell Leukemia

    @classmethod
    def num_classes(cls) -> int:
        """Return the number of NHL subtypes (K)."""
        return len(cls)

    @classmethod
    def from_index(cls, idx: int) -> "NHLSubtype":
        """Get NHLSubtype from integer index."""
        for subtype in cls:
            if subtype.value == idx:
                return subtype
        raise ValueError(f"Invalid subtype index: {idx}")

    @property
    def short_name(self) -> str:
        """Return abbreviated name for display."""
        return self.name.replace("_", "/")


class Modality(Enum):
    """Diagnostic modalities corresponding to the four BRAC agents.

    Each modality represents a distinct source of diagnostic evidence
    that contributes to NHL subtyping.
    """
    PATHOLOGY = "pathology"     # Histopathology, IHC, molecular markers
    RADIOLOGY = "radiology"     # PET/CT imaging
    LABORATORY = "laboratory"   # Flow cytometry, biochemistry
    CLINICAL = "clinical"       # Demographics, staging, symptoms

    @classmethod
    def all(cls) -> list["Modality"]:
        """Return all modalities in standard order."""
        return [cls.PATHOLOGY, cls.RADIOLOGY, cls.LABORATORY, cls.CLINICAL]

    @property
    def agent_index(self) -> int:
        """Return the index of this modality's agent (0-3)."""
        return list(Modality).index(self)


class ByzantineType(Enum):
    """Byzantine failure modes for experimental evaluation.

    These types model different ways an agent can fail or become adversarial,
    following the Byzantine fault tolerance literature.
    """
    HONEST = "honest"           # Behaves correctly according to protocol
    TYPE_I = "type_i"           # Data fault: uncertain, low quality evidence
    TYPE_II = "type_ii"         # Model fault: confident but wrong beliefs
    STRATEGIC = "strategic"     # Adversarial: optimally misleading


@dataclass
class SemanticEvidence:
    """Ontology-grounded evidence unit.

    Represents a single piece of diagnostic evidence normalized to
    standard medical ontologies (SNOMED-CT, ICD-O-3).

    Attributes:
        finding_type: Category of finding (e.g., "morphology", "immunophenotype")
        code: Ontology code (SNOMED-CT or ICD-O-3)
        value: The observed value (e.g., "CD20+", "Ki-67 > 90%")
        confidence: Confidence in this observation [0, 1]
        provenance: Source modality identifier
        quality_score: Quality assessment of this evidence [0, 1]
    """
    finding_type: str
    code: str
    value: str
    confidence: float
    provenance: str
    quality_score: float

    def __post_init__(self):
        """Validate confidence and quality_score are in [0, 1]."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not 0 <= self.quality_score <= 1:
            raise ValueError(f"quality_score must be in [0, 1], got {self.quality_score}")


@dataclass
class EvidenceQuality:
    """Per-agent evidence quality factors.

    These factors are used in the trust estimation module to compute
    agent reliability scores.

    Attributes:
        Q: Data quality [0, 1] - technical quality of input data
        C: Coverage [0, 1] - fraction of expected evidence present
        S: Consistency [0, 1] - internal coherence of evidence
    """
    Q: float    # Data quality
    C: float    # Coverage
    S: float    # Consistency

    def __post_init__(self):
        """Validate all scores are in [0, 1]."""
        for name, val in [("Q", self.Q), ("C", self.C), ("S", self.S)]:
            if not 0 <= val <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {val}")

    def as_tensor(self) -> torch.Tensor:
        """Return quality factors as a tensor [Q, C, S]."""
        return torch.tensor([self.Q, self.C, self.S], dtype=torch.float32)


@dataclass
class AgentOutput:
    """Output from a single diagnostic agent.

    Contains the agent's belief distribution over NHL subtypes along with
    supporting quality metrics and evidence.

    Attributes:
        belief: Probability distribution over K subtypes (on simplex)
        quality: Evidence quality factors (Q, C, S)
        evidence: List of semantic evidence items supporting the belief
        modality: The modality this agent represents
    """
    belief: torch.Tensor            # Shape: (K,) on probability simplex
    quality: EvidenceQuality
    evidence: list[SemanticEvidence] = field(default_factory=list)
    modality: Modality = Modality.PATHOLOGY

    def __post_init__(self):
        """Validate belief is a valid probability distribution."""
        if self.belief.dim() != 1:
            raise ValueError(f"belief must be 1D tensor, got shape {self.belief.shape}")
        if not torch.allclose(self.belief.sum(), torch.tensor(1.0), atol=1e-5):
            raise ValueError(f"belief must sum to 1, got {self.belief.sum().item()}")
        if (self.belief < 0).any():
            raise ValueError("belief must be non-negative")

    @property
    def predicted_class(self) -> int:
        """Return the argmax of the belief distribution."""
        return self.belief.argmax().item()

    @property
    def confidence(self) -> float:
        """Return the maximum probability (confidence in top prediction)."""
        return self.belief.max().item()

    @property
    def entropy(self) -> float:
        """Return the entropy of the belief distribution."""
        eps = 1e-10
        return -torch.sum(self.belief * torch.log(self.belief + eps)).item()


@dataclass
class BRACResult:
    """Complete output of the BRAC framework.

    Contains the final diagnosis, uncertainty estimates, explainability
    metrics, and intermediate values from the consensus process.

    Attributes:
        diagnosis: The predicted NHL subtype (argmax of consensus belief)
        consensus_belief: Final consensus distribution b* on simplex
        prediction_set: Conformal prediction set C_alpha
        prediction_set_size: Size of the prediction set |C_alpha|
        shapley_values: Attribution phi_i per agent modality
        interaction_indices: Pairwise interaction I_ij between agents
        agent_reliabilities: Computed reliability r_i per agent
        agent_trusts: Behavioral trust tau_i per agent
        convergence_rounds: Number of rounds until convergence
        accepted: True if |C_alpha| <= max_set_size_accept
        confidence: Maximum probability in consensus belief
    """
    diagnosis: NHLSubtype
    consensus_belief: torch.Tensor          # b* on simplex
    prediction_set: list[NHLSubtype]        # C_alpha
    prediction_set_size: int
    shapley_values: dict[Modality, float]   # phi_i per agent
    interaction_indices: dict[tuple[Modality, Modality], float]  # I_ij
    agent_reliabilities: dict[Modality, float]   # r_i
    agent_trusts: dict[Modality, float]          # tau_i
    convergence_rounds: int
    accepted: bool                          # True if |C_alpha| <= threshold
    confidence: float                       # max(b*)

    def __post_init__(self):
        """Ensure prediction_set_size matches prediction_set length."""
        if self.prediction_set_size != len(self.prediction_set):
            self.prediction_set_size = len(self.prediction_set)

    def summary(self) -> str:
        """Return a human-readable summary of the result."""
        lines = [
            f"BRAC Diagnosis: {self.diagnosis.name}",
            f"Confidence: {self.confidence:.2%}",
            f"Prediction Set: {[s.short_name for s in self.prediction_set]}",
            f"Set Size: {self.prediction_set_size}",
            f"Decision: {'ACCEPTED' if self.accepted else 'ESCALATE TO REVIEW'}",
            f"Convergence Rounds: {self.convergence_rounds}",
            "",
            "Agent Contributions (Shapley):",
        ]
        for mod, phi in sorted(self.shapley_values.items(), key=lambda x: -x[1]):
            trust = self.agent_trusts.get(mod, 0)
            rel = self.agent_reliabilities.get(mod, 0)
            lines.append(f"  {mod.value:12s}: phi={phi:+.3f}, tau={trust:.3f}, r={rel:.3f}")
        return "\n".join(lines)


# Type aliases for convenience
BeliefTensor = torch.Tensor  # Shape: (K,) on probability simplex
BeliefDict = dict[Modality, torch.Tensor]
ReliabilityDict = dict[Modality, float]
QualityDict = dict[Modality, EvidenceQuality]
