"""Byzantine attack models for experimental evaluation.

This module implements various Byzantine failure modes for testing the
robustness of the BRAC consensus mechanism:

1. Type I (Data Fault): Uncertain agent with noisy, low-quality beliefs
2. Type II (Model Fault): Confident but wrong predictions
3. Strategic Attack: Adversary knows other beliefs and optimally misleads
4. Label Flip: Reverses the belief vector

These attacks are used in experiments to verify Byzantine resilience.
"""

import torch
import numpy as np
from typing import Optional
from enum import Enum

from brac.types import ByzantineType, Modality, AgentOutput, EvidenceQuality

EPS = 1e-10


class ByzantineAttack:
    """Byzantine attack models for BRAC experiments.

    Provides static methods to transform honest beliefs into various
    types of Byzantine (faulty/adversarial) beliefs.
    """

    @staticmethod
    def apply_attack(
        honest_belief: torch.Tensor,
        attack_type: ByzantineType,
        K: int,
        honest_beliefs: Optional[list[torch.Tensor]] = None,
        target_class: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply a Byzantine attack to an honest belief.

        Args:
            honest_belief: The original honest belief
            attack_type: Type of Byzantine attack
            K: Number of classes
            honest_beliefs: Other honest beliefs (for strategic attack)
            target_class: Target class for strategic attack
            **kwargs: Additional attack-specific parameters

        Returns:
            Corrupted belief tensor
        """
        if attack_type == ByzantineType.HONEST:
            return honest_belief

        elif attack_type == ByzantineType.TYPE_I:
            return ByzantineAttack.type_i_data_fault(
                honest_belief,
                noise_scale=kwargs.get("noise_scale", 0.3),
            )

        elif attack_type == ByzantineType.TYPE_II:
            return ByzantineAttack.type_ii_model_fault(honest_belief, K)

        elif attack_type == ByzantineType.STRATEGIC:
            if honest_beliefs is None:
                raise ValueError("Strategic attack requires honest_beliefs")
            return ByzantineAttack.strategic_attack(
                honest_beliefs,
                target_class=target_class or 0,
                K=K,
            )

        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

    @staticmethod
    def type_i_data_fault(
        honest_belief: torch.Tensor,
        noise_scale: float = 0.3,
    ) -> torch.Tensor:
        """Type I Byzantine: Data fault with uncertain, noisy beliefs.

        Models scenarios where the agent has low-quality input data,
        resulting in high-entropy, uncertain predictions.

        Args:
            honest_belief: The original belief
            noise_scale: Scale of Dirichlet noise (lower = more noise)

        Returns:
            Noisy belief with high uncertainty
        """
        K = honest_belief.shape[0]

        # Sample from diffuse Dirichlet to add noise
        concentration = torch.ones(K) * (1.0 / noise_scale)
        noise = torch.distributions.Dirichlet(concentration).sample()

        # Mix honest belief with noise
        noisy = 0.5 * honest_belief + 0.5 * noise

        # Normalize
        noisy = torch.clamp(noisy, min=EPS)
        return noisy / noisy.sum()

    @staticmethod
    def type_ii_model_fault(
        honest_belief: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Type II Byzantine: Model fault with confident wrong predictions.

        Models scenarios where the agent's model is faulty, producing
        high-confidence predictions on the wrong class.

        Args:
            honest_belief: The original belief
            K: Number of classes

        Returns:
            High-confidence belief on wrong class
        """
        true_class = honest_belief.argmax().item()

        # Pick a random wrong class
        wrong_classes = [i for i in range(K) if i != true_class]
        wrong_class = np.random.choice(wrong_classes)

        # Create peaked belief on wrong class
        belief = torch.full((K,), 0.01)
        belief[wrong_class] = 0.91

        return belief / belief.sum()

    @staticmethod
    def strategic_attack(
        honest_beliefs: list[torch.Tensor],
        target_class: int,
        K: int,
        strength: float = 2.0,
    ) -> torch.Tensor:
        """Strategic Byzantine: Adversary optimally misleads consensus.

        The adversary knows other agents' beliefs and tries to maximally
        displace the consensus toward a target class.

        Args:
            honest_beliefs: List of honest agent beliefs
            target_class: Class the adversary wants to promote
            K: Number of classes
            strength: How aggressively to push toward target

        Returns:
            Adversarial belief designed to shift consensus
        """
        # Compute honest centroid
        honest_stack = torch.stack(honest_beliefs)
        centroid = honest_stack.mean(dim=0)

        # Direction from centroid to target one-hot
        target_one_hot = torch.zeros(K)
        target_one_hot[target_class] = 1.0

        direction = target_one_hot - centroid

        # Push in that direction (scaled by strength)
        adversarial = centroid + strength * direction

        # Project to simplex
        adversarial = torch.clamp(adversarial, min=EPS)
        return adversarial / adversarial.sum()

    @staticmethod
    def label_flip(honest_belief: torch.Tensor) -> torch.Tensor:
        """Label flip attack: Reverse the belief vector.

        Simple attack that reverses class ordering, making the least
        likely class most likely.

        Args:
            honest_belief: The original belief

        Returns:
            Flipped belief
        """
        flipped = honest_belief.flip(0)
        return flipped / flipped.sum()

    @staticmethod
    def random_attack(K: int) -> torch.Tensor:
        """Random attack: Completely random belief.

        Args:
            K: Number of classes

        Returns:
            Random belief sampled uniformly from simplex
        """
        alpha = torch.ones(K)
        return torch.distributions.Dirichlet(alpha).sample()

    @staticmethod
    def targeted_attack(
        target_class: int,
        K: int,
        confidence: float = 0.9,
    ) -> torch.Tensor:
        """Targeted attack: High confidence on specific target class.

        Args:
            target_class: Class to promote
            K: Number of classes
            confidence: Confidence level for target class

        Returns:
            Belief peaked on target class
        """
        belief = torch.full((K,), (1 - confidence) / (K - 1))
        belief[target_class] = confidence
        return belief


class ByzantineScenario:
    """Scenario generator for Byzantine experiments.

    Creates controlled experimental setups with specified numbers
    of honest and Byzantine agents.
    """

    def __init__(
        self,
        num_classes: int = 9,
        seed: int = 42,
    ):
        """Initialize scenario generator.

        Args:
            num_classes: Number of NHL subtypes
            seed: Random seed
        """
        self.K = num_classes
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def create_scenario(
        self,
        num_agents: int,
        num_byzantine: int,
        true_label: int,
        attack_type: ByzantineType = ByzantineType.TYPE_II,
        honest_concentration: float = 10.0,
    ) -> tuple[torch.Tensor, torch.Tensor, list[bool]]:
        """Create a Byzantine scenario with specified parameters.

        Args:
            num_agents: Total number of agents (n)
            num_byzantine: Number of Byzantine agents (f)
            true_label: Ground truth class
            attack_type: Type of Byzantine attack
            honest_concentration: Dirichlet concentration for honest agents

        Returns:
            Tuple of:
            - beliefs: Shape (n, K)
            - reliabilities: Shape (n,) - equal weights initially
            - is_byzantine: List of booleans indicating which agents are Byzantine
        """
        beliefs = []
        is_byzantine = []

        # Decide which agents are Byzantine
        byzantine_indices = set(np.random.choice(
            num_agents, size=num_byzantine, replace=False
        ))

        # Generate beliefs
        for i in range(num_agents):
            if i in byzantine_indices:
                is_byzantine.append(True)

                # Generate honest belief first (for some attack types)
                alpha = torch.ones(self.K) * 0.1
                alpha[true_label] = honest_concentration
                honest_belief = torch.distributions.Dirichlet(alpha).sample()

                # Apply attack
                if attack_type == ByzantineType.TYPE_I:
                    belief = ByzantineAttack.type_i_data_fault(honest_belief)
                elif attack_type == ByzantineType.TYPE_II:
                    belief = ByzantineAttack.type_ii_model_fault(honest_belief, self.K)
                elif attack_type == ByzantineType.STRATEGIC:
                    # Collect honest beliefs so far for strategic attack
                    honest_so_far = [b for b, byz in zip(beliefs, is_byzantine) if not byz]
                    if honest_so_far:
                        target = np.random.choice([c for c in range(self.K) if c != true_label])
                        belief = ByzantineAttack.strategic_attack(honest_so_far, target, self.K)
                    else:
                        belief = ByzantineAttack.random_attack(self.K)
                else:
                    belief = ByzantineAttack.random_attack(self.K)

                beliefs.append(belief)

            else:
                is_byzantine.append(False)
                # Honest agent: Dirichlet centered on true label
                alpha = torch.ones(self.K) * 0.1
                alpha[true_label] = honest_concentration
                belief = torch.distributions.Dirichlet(alpha).sample()
                beliefs.append(belief)

        beliefs_tensor = torch.stack(beliefs)
        reliabilities = torch.ones(num_agents) / num_agents

        return beliefs_tensor, reliabilities, is_byzantine

    def evaluate_attack_effectiveness(
        self,
        consensus: torch.Tensor,
        true_label: int,
        oracle_consensus: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Evaluate how effective an attack was.

        Args:
            consensus: The achieved consensus belief
            true_label: Ground truth class
            oracle_consensus: Consensus that would be achieved without Byzantine

        Returns:
            Dictionary with effectiveness metrics
        """
        from brac.consensus.fisher_rao import fisher_rao_distance

        metrics = {
            "predicted_correct": int(consensus.argmax().item() == true_label),
            "true_class_prob": consensus[true_label].item(),
            "max_prob": consensus.max().item(),
            "entropy": -torch.sum(consensus * torch.log(consensus + EPS)).item(),
        }

        if oracle_consensus is not None:
            metrics["displacement"] = fisher_rao_distance(
                consensus.unsqueeze(0),
                oracle_consensus.unsqueeze(0)
            ).item()

        return metrics


def corrupt_agent_output(
    output: AgentOutput,
    attack_type: ByzantineType,
    K: int = 9,
) -> AgentOutput:
    """Apply Byzantine attack to an AgentOutput.

    Args:
        output: Original agent output
        attack_type: Type of attack to apply
        K: Number of classes

    Returns:
        Corrupted AgentOutput
    """
    corrupted_belief = ByzantineAttack.apply_attack(
        output.belief,
        attack_type,
        K,
    )

    # Optionally corrupt quality scores for Type I attacks
    if attack_type == ByzantineType.TYPE_I:
        quality = EvidenceQuality(
            Q=max(0.1, output.quality.Q - 0.3),
            C=max(0.1, output.quality.C - 0.2),
            S=max(0.1, output.quality.S - 0.3),
        )
    else:
        quality = output.quality

    return AgentOutput(
        belief=corrupted_belief,
        quality=quality,
        evidence=output.evidence,
        modality=output.modality,
    )
