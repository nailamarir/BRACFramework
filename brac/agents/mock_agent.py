"""Mock agents for BRAC consensus layer experiments.

Mock agents generate synthetic belief distributions from known ground truth
with controlled noise. This enables rigorous testing of the consensus
mechanism without requiring real clinical data.

The mock agents support various noise models:
- Dirichlet: Samples from Dirichlet distribution with controlled concentration
- Gaussian: Adds Gaussian noise in log-space then normalizes
- Deterministic: Returns fixed belief based on true label

FIXES Applied:
- Issue #3: Lowered concentration (8-15 -> 2-5) for meaningful non-conformity scores
- Issue #4: Added subtype-specific expertise profiles for varied Shapley values
- Issue #5: Added cross-modal synergy modeling for positive interaction indices
"""

import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

from brac.types import (
    AgentOutput, Modality, EvidenceQuality, SemanticEvidence, NHLSubtype
)
from brac.agents.base_agent import BaseAgent


@dataclass
class MockAgentConfig:
    """Configuration for mock agent behavior.

    Attributes:
        accuracy: Probability of being correct (argmax = true label)
        concentration: Dirichlet concentration parameter (higher = more confident)
        noise_type: Type of noise model ("dirichlet", "gaussian", "deterministic")
        quality_mean: Mean of quality scores [Q, C, S]
        quality_std: Std of quality scores
    """
    accuracy: float = 0.85
    concentration: float = 3.0  # Issue #3: Lowered from 10.0
    noise_type: str = "dirichlet"
    quality_mean: tuple[float, float, float] = (0.8, 0.7, 0.75)
    quality_std: float = 0.1


# Issue #3: Lowered concentrations (8-15 -> 2-5) for meaningful uncertainty
# Issue #4: Base configurations - subtype-specific overrides below
DEFAULT_CONFIGS = {
    Modality.PATHOLOGY: MockAgentConfig(
        accuracy=0.85,
        concentration=4.0,  # Lowered from 15.0
        quality_mean=(0.85, 0.80, 0.82),
    ),
    Modality.RADIOLOGY: MockAgentConfig(
        accuracy=0.75,
        concentration=2.5,  # Lowered from 8.0
        quality_mean=(0.75, 0.70, 0.72),
    ),
    Modality.LABORATORY: MockAgentConfig(
        accuracy=0.80,
        concentration=3.0,  # Lowered from 10.0
        quality_mean=(0.88, 0.75, 0.80),
    ),
    Modality.CLINICAL: MockAgentConfig(
        accuracy=0.65,
        concentration=2.0,  # Lowered from 5.0
        quality_mean=(0.70, 0.65, 0.68),
    ),
}


# Issue #4 & #5: Subtype-specific expertise profiles
# Maps (subtype, modality) -> concentration multiplier
# Higher multiplier = this modality is more informative for this subtype
SUBTYPE_EXPERTISE = {
    # FL: Pathology dominates (follicular pattern morphologically distinctive)
    (NHLSubtype.FL, Modality.PATHOLOGY): 3.0,
    (NHLSubtype.FL, Modality.RADIOLOGY): 0.8,
    (NHLSubtype.FL, Modality.LABORATORY): 1.2,
    (NHLSubtype.FL, Modality.CLINICAL): 0.6,

    # MCL: Path + Lab synergy (cyclin D1 + morphology)
    (NHLSubtype.MCL, Modality.PATHOLOGY): 2.5,
    (NHLSubtype.MCL, Modality.RADIOLOGY): 0.8,
    (NHLSubtype.MCL, Modality.LABORATORY): 2.2,
    (NHLSubtype.MCL, Modality.CLINICAL): 0.7,

    # CLL/SLL: Laboratory dominates (flow cytometry diagnostic)
    (NHLSubtype.CLL_SLL, Modality.PATHOLOGY): 1.0,
    (NHLSubtype.CLL_SLL, Modality.RADIOLOGY): 0.6,
    (NHLSubtype.CLL_SLL, Modality.LABORATORY): 3.5,
    (NHLSubtype.CLL_SLL, Modality.CLINICAL): 1.2,

    # DLBCL_GCB: Path + Lab synergy (IHC + flow for cell-of-origin)
    (NHLSubtype.DLBCL_GCB, Modality.PATHOLOGY): 2.5,
    (NHLSubtype.DLBCL_GCB, Modality.RADIOLOGY): 1.0,
    (NHLSubtype.DLBCL_GCB, Modality.LABORATORY): 2.0,
    (NHLSubtype.DLBCL_GCB, Modality.CLINICAL): 0.8,

    # DLBCL_ABC: Path + Lab synergy (similar to GCB)
    (NHLSubtype.DLBCL_ABC, Modality.PATHOLOGY): 2.5,
    (NHLSubtype.DLBCL_ABC, Modality.RADIOLOGY): 1.0,
    (NHLSubtype.DLBCL_ABC, Modality.LABORATORY): 2.0,
    (NHLSubtype.DLBCL_ABC, Modality.CLINICAL): 0.8,

    # MZL: Pathology important (marginal zone pattern)
    (NHLSubtype.MZL, Modality.PATHOLOGY): 2.8,
    (NHLSubtype.MZL, Modality.RADIOLOGY): 0.9,
    (NHLSubtype.MZL, Modality.LABORATORY): 1.5,
    (NHLSubtype.MZL, Modality.CLINICAL): 1.0,

    # BL: Radiology high (PET avidity characteristic)
    (NHLSubtype.BL, Modality.PATHOLOGY): 1.5,
    (NHLSubtype.BL, Modality.RADIOLOGY): 2.8,
    (NHLSubtype.BL, Modality.LABORATORY): 1.8,
    (NHLSubtype.BL, Modality.CLINICAL): 0.7,

    # LPL: Laboratory dominates (IgM paraprotein)
    (NHLSubtype.LPL, Modality.PATHOLOGY): 1.2,
    (NHLSubtype.LPL, Modality.RADIOLOGY): 0.7,
    (NHLSubtype.LPL, Modality.LABORATORY): 3.0,
    (NHLSubtype.LPL, Modality.CLINICAL): 1.5,

    # HCL: Laboratory dominates (hairy cell markers)
    (NHLSubtype.HCL, Modality.PATHOLOGY): 1.5,
    (NHLSubtype.HCL, Modality.RADIOLOGY): 0.6,
    (NHLSubtype.HCL, Modality.LABORATORY): 3.2,
    (NHLSubtype.HCL, Modality.CLINICAL): 1.0,
}


# Issue #5: Cross-modal synergy pairs
# When both modalities agree, boost confidence for these subtypes
SYNERGY_PAIRS = {
    # DLBCL: Path + Lab synergy for cell-of-origin determination
    NHLSubtype.DLBCL_GCB: [(Modality.PATHOLOGY, Modality.LABORATORY)],
    NHLSubtype.DLBCL_ABC: [(Modality.PATHOLOGY, Modality.LABORATORY)],
    # MCL: Path + Lab for cyclin D1 confirmation
    NHLSubtype.MCL: [(Modality.PATHOLOGY, Modality.LABORATORY)],
    # BL: Path + Rad for Ki-67 + PET correlation
    NHLSubtype.BL: [(Modality.PATHOLOGY, Modality.RADIOLOGY)],
}


def get_subtype_concentration(
    subtype: NHLSubtype,
    modality: Modality,
    base_concentration: float,
) -> float:
    """Get subtype-specific concentration for an agent.

    Args:
        subtype: The NHL subtype
        modality: The agent modality
        base_concentration: Base concentration from config

    Returns:
        Adjusted concentration based on subtype expertise
    """
    multiplier = SUBTYPE_EXPERTISE.get((subtype, modality), 1.0)
    return base_concentration * multiplier


class MockAgent(BaseAgent):
    """Mock agent for consensus layer experiments.

    Generates synthetic beliefs from known ground truth with controlled
    noise and quality characteristics. Supports subtype-specific expertise
    profiles for realistic Shapley attribution patterns.
    """

    def __init__(
        self,
        modality: Modality,
        config: Optional[MockAgentConfig] = None,
        num_classes: int = 9,
        seed: Optional[int] = None,
        use_subtype_expertise: bool = True,
    ):
        """Initialize mock agent.

        Args:
            modality: The diagnostic modality
            config: Agent configuration (uses default for modality if None)
            num_classes: Number of NHL subtypes (K)
            seed: Random seed for reproducibility
            use_subtype_expertise: If True, use subtype-specific concentrations
        """
        super().__init__(modality, num_classes)

        self.config = config or DEFAULT_CONFIGS.get(modality, MockAgentConfig())
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.use_subtype_expertise = use_subtype_expertise

        if seed is not None:
            torch.manual_seed(seed)

    def forward(self, evidence: int) -> AgentOutput:
        """Generate mock output for a given true label.

        Args:
            evidence: True class label (integer index)

        Returns:
            AgentOutput with synthetic belief and quality
        """
        return self.generate_belief(true_label=evidence)

    def generate_belief(
        self,
        true_label: int,
        byzantine: bool = False,
    ) -> AgentOutput:
        """Generate a belief distribution centered on true label.

        Args:
            true_label: The ground truth class index
            byzantine: If True, generate adversarial belief

        Returns:
            AgentOutput with belief and quality metrics
        """
        K = self.num_classes

        if byzantine:
            belief = self._generate_byzantine_belief(true_label)
        else:
            belief = self._generate_honest_belief(true_label)

        quality = self._generate_quality()
        evidence = self._generate_evidence(true_label, belief)

        return AgentOutput(
            belief=belief,
            quality=quality,
            evidence=evidence,
            modality=self.modality,
        )

    def _generate_honest_belief(self, true_label: int) -> torch.Tensor:
        """Generate honest belief centered on true label."""
        K = self.num_classes
        config = self.config

        # Get subtype-specific concentration (Issue #4)
        if self.use_subtype_expertise:
            subtype = NHLSubtype.from_index(true_label)
            concentration = get_subtype_concentration(
                subtype, self.modality, config.concentration
            )
        else:
            concentration = config.concentration

        if config.noise_type == "deterministic":
            # Peaked distribution on true class
            belief = torch.full((K,), 0.01)
            belief[true_label] = 0.91
            return belief / belief.sum()

        elif config.noise_type == "gaussian":
            # Log-normal noise
            logits = torch.randn(K) * 0.5
            logits[true_label] += concentration * 0.3
            belief = torch.softmax(logits, dim=0)
            return belief

        else:  # dirichlet
            # Dirichlet with concentration on true class
            # Issue #3: Use lower base alpha for more uncertainty
            alpha = torch.ones(K) * 0.5  # Increased from 0.1 for more spread
            alpha[true_label] = concentration

            # Sample from Dirichlet
            belief = torch.distributions.Dirichlet(alpha).sample()

            # Occasionally flip to wrong class (1 - accuracy)
            if self._rng.random() > config.accuracy:
                # Pick wrong class
                wrong_classes = [i for i in range(K) if i != true_label]
                wrong_label = self._rng.choice(wrong_classes)
                # Swap concentrations
                belief_wrong = belief.clone()
                belief_wrong[wrong_label] = belief[true_label]
                belief_wrong[true_label] = belief[wrong_label]
                belief = belief_wrong

            return belief

    def _generate_byzantine_belief(self, true_label: int) -> torch.Tensor:
        """Generate adversarial belief (wrong with high confidence)."""
        K = self.num_classes

        # Pick wrong class
        wrong_classes = [i for i in range(K) if i != true_label]
        wrong_label = self._rng.choice(wrong_classes)

        # High confidence on wrong class
        belief = torch.full((K,), 0.01)
        belief[wrong_label] = 0.91
        return belief / belief.sum()

    def _generate_quality(self) -> EvidenceQuality:
        """Generate quality scores with noise."""
        config = self.config
        mean = config.quality_mean
        std = config.quality_std

        Q = np.clip(self._rng.normal(mean[0], std), 0, 1)
        C = np.clip(self._rng.normal(mean[1], std), 0, 1)
        S = np.clip(self._rng.normal(mean[2], std), 0, 1)

        return EvidenceQuality(Q=Q, C=C, S=S)

    def _generate_evidence(
        self,
        true_label: int,
        belief: torch.Tensor,
    ) -> list[SemanticEvidence]:
        """Generate mock semantic evidence."""
        subtype = NHLSubtype.from_index(true_label)

        # Create a few mock evidence items
        evidence_items = [
            SemanticEvidence(
                finding_type="mock_finding",
                code=f"MOCK-{self.modality.value[:3].upper()}-001",
                value=f"Consistent with {subtype.name}",
                confidence=belief.max().item(),
                provenance=self.modality.value,
                quality_score=0.8,
            )
        ]

        return evidence_items


class MockAgentWithSynergy(MockAgent):
    """Mock agent that models cross-modal synergies (Issue #5).

    When paired with a complementary modality for certain subtypes,
    both agents receive a confidence boost, creating positive interaction indices.
    """

    def __init__(
        self,
        modality: Modality,
        partner_beliefs: Optional[dict[Modality, torch.Tensor]] = None,
        **kwargs,
    ):
        """Initialize synergy-aware mock agent.

        Args:
            modality: The diagnostic modality
            partner_beliefs: Beliefs from other agents (for synergy calculation)
            **kwargs: Passed to MockAgent
        """
        super().__init__(modality, **kwargs)
        self.partner_beliefs = partner_beliefs or {}

    def _generate_honest_belief(self, true_label: int) -> torch.Tensor:
        """Generate belief with synergy boost when applicable."""
        # Get base belief
        belief = super()._generate_honest_belief(true_label)

        # Check for synergy (Issue #5)
        subtype = NHLSubtype.from_index(true_label)
        synergy_pairs = SYNERGY_PAIRS.get(subtype, [])

        for m1, m2 in synergy_pairs:
            if self.modality in (m1, m2):
                # Find partner modality
                partner = m2 if self.modality == m1 else m1

                if partner in self.partner_beliefs:
                    partner_belief = self.partner_beliefs[partner]
                    partner_pred = partner_belief.argmax().item()

                    # If partner agrees on true label, boost confidence
                    if partner_pred == true_label:
                        synergy_boost = 1.3  # 30% boost
                        belief[true_label] *= synergy_boost
                        belief = belief / belief.sum()  # Renormalize

        return belief


class MockAgentFactory:
    """Factory for creating sets of mock agents."""

    def __init__(
        self,
        num_classes: int = 9,
        seed: int = 42,
        use_subtype_expertise: bool = True,
        use_synergy: bool = True,
    ):
        """Initialize factory.

        Args:
            num_classes: Number of NHL subtypes
            seed: Base random seed
            use_subtype_expertise: Enable subtype-specific concentrations (Issue #4)
            use_synergy: Enable cross-modal synergy modeling (Issue #5)
        """
        self.num_classes = num_classes
        self.base_seed = seed
        self.use_subtype_expertise = use_subtype_expertise
        self.use_synergy = use_synergy

    def create_all_agents(
        self,
        configs: Optional[dict[Modality, MockAgentConfig]] = None,
    ) -> dict[Modality, MockAgent]:
        """Create mock agents for all modalities.

        Args:
            configs: Optional custom configurations per modality

        Returns:
            Dictionary mapping Modality -> MockAgent
        """
        configs = configs or {}
        agents = {}

        for i, modality in enumerate(Modality.all()):
            config = configs.get(modality, DEFAULT_CONFIGS.get(modality))
            agents[modality] = MockAgent(
                modality=modality,
                config=config,
                num_classes=self.num_classes,
                seed=self.base_seed + i,
                use_subtype_expertise=self.use_subtype_expertise,
            )

        return agents

    def generate_case(
        self,
        agents: dict[Modality, MockAgent],
        true_label: int,
        byzantine_modality: Optional[Modality] = None,
    ) -> dict[Modality, AgentOutput]:
        """Generate agent outputs for a single case.

        With synergy modeling, generates beliefs in two passes:
        1. Generate initial beliefs
        2. Re-generate with synergy information

        Args:
            agents: Dictionary of mock agents
            true_label: Ground truth class index
            byzantine_modality: If set, this agent behaves as Byzantine

        Returns:
            Dictionary mapping Modality -> AgentOutput
        """
        outputs = {}

        if self.use_synergy:
            # Pass 1: Generate initial beliefs
            initial_beliefs = {}
            for modality, agent in agents.items():
                is_byzantine = (modality == byzantine_modality)
                output = agent.generate_belief(
                    true_label=true_label,
                    byzantine=is_byzantine,
                )
                initial_beliefs[modality] = output.belief
                outputs[modality] = output

            # Pass 2: Re-generate with synergy (for synergistic subtypes)
            subtype = NHLSubtype.from_index(true_label)
            if subtype in SYNERGY_PAIRS:
                for modality, agent in agents.items():
                    if modality == byzantine_modality:
                        continue  # Don't modify Byzantine beliefs

                    # Create synergy-aware agent for this generation
                    synergy_agent = MockAgentWithSynergy(
                        modality=modality,
                        partner_beliefs=initial_beliefs,
                        config=agent.config,
                        num_classes=agent.num_classes,
                        seed=agent.seed,
                        use_subtype_expertise=agent.use_subtype_expertise,
                    )
                    outputs[modality] = synergy_agent.generate_belief(
                        true_label=true_label,
                        byzantine=False,
                    )
        else:
            # Simple generation without synergy
            for modality, agent in agents.items():
                is_byzantine = (modality == byzantine_modality)
                outputs[modality] = agent.generate_belief(
                    true_label=true_label,
                    byzantine=is_byzantine,
                )

        return outputs

    def generate_dataset(
        self,
        num_cases: int,
        byzantine_fraction: float = 0.0,
        byzantine_modality: Optional[Modality] = None,
        configs: Optional[dict[Modality, MockAgentConfig]] = None,
    ) -> tuple[list[dict[Modality, AgentOutput]], torch.Tensor]:
        """Generate a synthetic dataset.

        Args:
            num_cases: Number of cases to generate
            byzantine_fraction: Fraction of cases with Byzantine agent
            byzantine_modality: Which modality is Byzantine (random if None)
            configs: Custom agent configurations

        Returns:
            Tuple of (list of agent outputs per case, labels tensor)
        """
        agents = self.create_all_agents(configs)
        rng = np.random.default_rng(self.base_seed)

        cases = []
        labels = []

        for i in range(num_cases):
            # Random true label
            true_label = rng.integers(0, self.num_classes)
            labels.append(true_label)

            # Determine if this case has a Byzantine agent
            is_byzantine_case = rng.random() < byzantine_fraction
            if is_byzantine_case:
                if byzantine_modality is None:
                    # Random Byzantine modality (not pathology - root of trust)
                    byz_mod = rng.choice([Modality.RADIOLOGY, Modality.LABORATORY, Modality.CLINICAL])
                else:
                    byz_mod = byzantine_modality
            else:
                byz_mod = None

            case_outputs = self.generate_case(agents, true_label, byz_mod)
            cases.append(case_outputs)

        return cases, torch.tensor(labels)
