"""Exact Shapley values for agent attribution in BRAC.

This module implements Innovation 5 of the BRAC framework: axiomatic
explainability using Shapley values from cooperative game theory.

For n=4 agents, we compute EXACT Shapley values by enumerating all
2^4 = 16 coalitions. This provides:
1. Agent importance scores (phi_i) - how much each agent contributes
2. Interaction indices (I_ij) - synergy (+) or redundancy (-) between agents
3. Subtype-specific attributions - which agents matter for each diagnosis

The Shapley value satisfies four desirable axioms:
- Efficiency: sum(phi_i) = v(A) - v(empty)
- Symmetry: agents with equal contributions get equal values
- Linearity: Shapley values compose linearly
- Null player: agents that never contribute get zero value
"""

import torch
from itertools import combinations
from math import factorial
from typing import Optional, Callable
import logging
from dataclasses import dataclass

from brac.types import Modality, NHLSubtype

logger = logging.getLogger(__name__)

EPS = 1e-10


@dataclass
class ShapleyResult:
    """Results from Shapley attribution.

    Attributes:
        shapley_values: Dictionary mapping Modality -> phi_i
        interaction_indices: Dictionary mapping (Modality, Modality) -> I_ij
        coalition_values: Dictionary mapping frozenset -> v(S)
        efficiency_error: |sum(phi) - (v(A) - v(empty))|, should be ~0
    """
    shapley_values: dict[Modality, float]
    interaction_indices: dict[tuple[Modality, Modality], float]
    coalition_values: dict[frozenset, float]
    efficiency_error: float


class ShapleyAttributor:
    """Exact Shapley value computation for BRAC agent attribution.

    With n=4 agents, we enumerate all 16 coalitions explicitly.
    The value function v(S) is the consensus confidence (or probability
    of true class) when using only agents in coalition S.

    The geometric median is used as the consensus mechanism for each
    coalition, respecting the Fisher-Rao geometry.
    """

    def __init__(
        self,
        agents: list[Modality],
        geometric_median_fn: Callable,
        uniform_prior: Optional[torch.Tensor] = None,
        num_classes: int = 9,
    ):
        """Initialize the Shapley attributor.

        Args:
            agents: List of agent modalities (typically all 4)
            geometric_median_fn: Function to compute geometric median
                Should have signature: (beliefs_dict, reliabilities_dict) -> consensus_tensor
            uniform_prior: Prior belief for empty coalition, default: uniform
            num_classes: Number of classes K (for uniform prior)
        """
        self.agents = agents
        self.n = len(agents)
        self.gm_fn = geometric_median_fn
        self.num_classes = num_classes

        if uniform_prior is None:
            self.uniform_prior = torch.ones(num_classes) / num_classes
        else:
            self.uniform_prior = uniform_prior

        # Precompute all coalitions (subsets of agents)
        self.coalitions = self._enumerate_coalitions()

        logger.debug(f"ShapleyAttributor initialized with {self.n} agents, "
                    f"{len(self.coalitions)} coalitions")

    def _enumerate_coalitions(self) -> list[frozenset]:
        """Enumerate all 2^n coalitions (subsets of agents).

        Returns:
            List of frozensets, each representing a coalition
        """
        coalitions = []
        for r in range(self.n + 1):
            for subset in combinations(range(self.n), r):
                coalition = frozenset(self.agents[i] for i in subset)
                coalitions.append(coalition)
        return coalitions

    def compute_coalition_values(
        self,
        beliefs: dict[Modality, torch.Tensor],
        reliabilities: dict[Modality, float],
        true_label: Optional[int] = None,
        value_type: str = "max_prob",
    ) -> dict[frozenset, float]:
        """Compute value v(S) for all 2^n coalitions.

        The value function measures the "value" of a coalition:
        - "max_prob": Maximum probability (confidence) in consensus
        - "true_prob": Probability of true class (if true_label provided)
        - "entropy": Negative entropy (higher = more certain)

        Args:
            beliefs: Dictionary mapping Modality -> belief tensor (K,)
            reliabilities: Dictionary mapping Modality -> reliability float
            true_label: Optional true class index for "true_prob" value type
            value_type: One of "max_prob", "true_prob", "entropy"

        Returns:
            Dictionary mapping coalition (frozenset) -> value
        """
        coalition_values = {}

        for coalition in self.coalitions:
            if len(coalition) == 0:
                # Empty coalition: use uniform prior
                consensus = self.uniform_prior
            else:
                # Non-empty coalition: compute geometric median
                coalition_beliefs = {m: beliefs[m] for m in coalition}
                coalition_rels = {m: reliabilities.get(m, 1.0) for m in coalition}

                # Call geometric median function
                consensus, _ = self.gm_fn(coalition_beliefs, coalition_rels)

            # Compute value based on value_type
            if value_type == "max_prob":
                value = consensus.max().item()
            elif value_type == "true_prob":
                if true_label is None:
                    raise ValueError("true_label required for value_type='true_prob'")
                value = consensus[true_label].item()
            elif value_type == "entropy":
                entropy = -torch.sum(consensus * torch.log(consensus + EPS)).item()
                value = -entropy  # Negative because we want to maximize certainty
            else:
                raise ValueError(f"Unknown value_type: {value_type}")

            coalition_values[coalition] = value

        return coalition_values

    def shapley_values(
        self,
        coalition_values: dict[frozenset, float],
    ) -> dict[Modality, float]:
        """Compute exact Shapley values from coalition values.

        The Shapley value for agent i is:
            phi_i = sum_{S ⊆ A\\{i}} [|S|! * (n-|S|-1)! / n!] * [v(S∪{i}) - v(S)]

        Args:
            coalition_values: Dictionary mapping coalition -> value

        Returns:
            Dictionary mapping Modality -> Shapley value
        """
        shapley = {}

        for agent in self.agents:
            phi_i = 0.0

            # Sum over all coalitions not containing agent
            for coalition in self.coalitions:
                if agent in coalition:
                    continue

                s = len(coalition)
                # Shapley weight: |S|! * (n-|S|-1)! / n!
                weight = (factorial(s) * factorial(self.n - s - 1)) / factorial(self.n)

                # Marginal contribution: v(S ∪ {i}) - v(S)
                coalition_with_i = coalition | {agent}
                marginal = coalition_values[coalition_with_i] - coalition_values[coalition]

                phi_i += weight * marginal

            shapley[agent] = phi_i

        return shapley

    def interaction_indices(
        self,
        coalition_values: dict[frozenset, float],
    ) -> dict[tuple[Modality, Modality], float]:
        """Compute pairwise interaction indices between agents.

        The interaction index I_ij measures synergy (+) or redundancy (-):
            I_ij = sum_{S ⊆ A\\{i,j}} [|S|!*(n-|S|-2)! / (n-1)!]
                   * [v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S)]

        Args:
            coalition_values: Dictionary mapping coalition -> value

        Returns:
            Dictionary mapping (Modality_i, Modality_j) -> interaction index
        """
        interactions = {}

        for i, agent_i in enumerate(self.agents):
            for j, agent_j in enumerate(self.agents):
                if i >= j:
                    continue  # Only compute for i < j

                I_ij = 0.0

                # Sum over coalitions not containing either i or j
                for coalition in self.coalitions:
                    if agent_i in coalition or agent_j in coalition:
                        continue

                    s = len(coalition)
                    # Interaction weight: |S|! * (n-|S|-2)! / (n-1)!
                    weight = (factorial(s) * factorial(self.n - s - 2)) / factorial(self.n - 1)

                    # Interaction effect
                    v_S = coalition_values[coalition]
                    v_Si = coalition_values[coalition | {agent_i}]
                    v_Sj = coalition_values[coalition | {agent_j}]
                    v_Sij = coalition_values[coalition | {agent_i, agent_j}]

                    I_ij += weight * (v_Sij - v_Si - v_Sj + v_S)

                interactions[(agent_i, agent_j)] = I_ij
                interactions[(agent_j, agent_i)] = I_ij  # Symmetric

        return interactions

    def verify_efficiency(
        self,
        shapley_values: dict[Modality, float],
        coalition_values: dict[frozenset, float],
    ) -> float:
        """Verify the efficiency axiom: sum(phi_i) = v(A) - v(empty).

        Args:
            shapley_values: Computed Shapley values
            coalition_values: Coalition values

        Returns:
            Absolute error (should be ~0)
        """
        # Sum of Shapley values
        phi_sum = sum(shapley_values.values())

        # v(grand coalition) - v(empty set)
        grand_coalition = frozenset(self.agents)
        empty_coalition = frozenset()
        expected = coalition_values[grand_coalition] - coalition_values[empty_coalition]

        error = abs(phi_sum - expected)
        logger.debug(f"Efficiency check: sum(phi)={phi_sum:.6f}, "
                    f"v(A)-v(∅)={expected:.6f}, error={error:.2e}")

        return error

    def compute_full_attribution(
        self,
        beliefs: dict[Modality, torch.Tensor],
        reliabilities: dict[Modality, float],
        true_label: Optional[int] = None,
    ) -> ShapleyResult:
        """Compute full Shapley attribution including values and interactions.

        Args:
            beliefs: Dictionary mapping Modality -> belief tensor
            reliabilities: Dictionary mapping Modality -> reliability
            true_label: Optional true class for value computation

        Returns:
            ShapleyResult with all attribution information
        """
        # Determine value type
        value_type = "true_prob" if true_label is not None else "max_prob"

        # Compute coalition values
        coalition_values = self.compute_coalition_values(
            beliefs, reliabilities, true_label, value_type
        )

        # Compute Shapley values
        shapley = self.shapley_values(coalition_values)

        # Compute interaction indices
        interactions = self.interaction_indices(coalition_values)

        # Verify efficiency
        efficiency_error = self.verify_efficiency(shapley, coalition_values)

        return ShapleyResult(
            shapley_values=shapley,
            interaction_indices=interactions,
            coalition_values=coalition_values,
            efficiency_error=efficiency_error,
        )

    def subtype_shapley(
        self,
        beliefs: dict[Modality, torch.Tensor],
        reliabilities: dict[Modality, float],
    ) -> dict[NHLSubtype, dict[Modality, float]]:
        """Compute Shapley values for each subtype (class).

        For each subtype k, uses v(S) = p*_S(h_k) as the value function.
        This reveals which agents are most important for diagnosing each subtype.

        Args:
            beliefs: Dictionary mapping Modality -> belief tensor
            reliabilities: Dictionary mapping Modality -> reliability

        Returns:
            Dictionary mapping NHLSubtype -> {Modality -> phi_i^(k)}
        """
        subtype_shapley = {}

        for subtype in NHLSubtype:
            # Use this subtype as "true label" for value computation
            coalition_values = self.compute_coalition_values(
                beliefs, reliabilities,
                true_label=subtype.value,
                value_type="true_prob"
            )
            shapley = self.shapley_values(coalition_values)
            subtype_shapley[subtype] = shapley

        return subtype_shapley


def create_shapley_attributor(
    geometric_median_fn: Callable,
    num_classes: int = 9,
) -> ShapleyAttributor:
    """Factory function to create a ShapleyAttributor with default settings.

    Args:
        geometric_median_fn: Geometric median consensus function
        num_classes: Number of NHL subtypes

    Returns:
        Configured ShapleyAttributor
    """
    return ShapleyAttributor(
        agents=Modality.all(),
        geometric_median_fn=geometric_median_fn,
        num_classes=num_classes,
    )


def aggregate_shapley_across_cases(
    shapley_results: list[ShapleyResult],
) -> dict[str, dict[Modality, float]]:
    """Aggregate Shapley values across multiple cases.

    Args:
        shapley_results: List of ShapleyResult from multiple cases

    Returns:
        Dictionary with "mean", "std", "min", "max" Shapley values per agent
    """
    import numpy as np

    # Collect values per agent
    agent_values = {m: [] for m in Modality.all()}

    for result in shapley_results:
        for agent, value in result.shapley_values.items():
            agent_values[agent].append(value)

    # Compute statistics
    stats = {}
    for stat_name, stat_fn in [
        ("mean", np.mean),
        ("std", np.std),
        ("min", np.min),
        ("max", np.max),
    ]:
        stats[stat_name] = {
            agent: stat_fn(values) for agent, values in agent_values.items()
        }

    return stats
