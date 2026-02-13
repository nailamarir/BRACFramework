"""BRAC: Byzantine-Resilient Agentic Consensus Framework.

A multimodal AI framework for B-cell Non-Hodgkin Lymphoma (NHL) subtyping
with Byzantine fault tolerance, uncertainty quantification, and explainability.
"""

__version__ = "0.1.0"
__author__ = "BRAC Team"

from brac.types import (
    NHLSubtype,
    Modality,
    ByzantineType,
    SemanticEvidence,
    EvidenceQuality,
    AgentOutput,
    BRACResult,
)

__all__ = [
    "NHLSubtype",
    "Modality",
    "ByzantineType",
    "SemanticEvidence",
    "EvidenceQuality",
    "AgentOutput",
    "BRACResult",
]
