"""BRAC diagnostic agents for multimodal NHL subtyping.

This package contains the agent implementations:
- BaseAgent: Abstract base class for all agents
- MockAgent: Deterministic/stochastic mock agent for testing
- PathologyAgent: ViT-based histopathology agent (stub)
- RadiologyAgent: 3D CNN for PET/CT (stub)
- LaboratoryAgent: Transformer for flow cytometry (stub)
- ClinicalAgent: Transformer for clinical data (stub)
"""

from brac.agents.base_agent import BaseAgent
from brac.agents.mock_agent import MockAgent, MockAgentConfig

__all__ = [
    "BaseAgent",
    "MockAgent",
    "MockAgentConfig",
]
