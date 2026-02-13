"""Experiment 1: Byzantine Resilience Analysis.

Demonstrates that geometric median resists Byzantine attacks while
weighted average fails.

Setup:
- K=9 subtypes, n=4 agents
- Generate synthetic cases
- Vary Byzantine agent count and attack type
- Compare aggregation methods

Metrics:
- Top-1 and Top-3 accuracy
- Consensus displacement from oracle
- Breakdown verification

FIXES:
- Issue #1: Pass actual Byzantine count to Krum/Multi-Krum
- Issue #2: Exclude pathology (root-of-trust) from Byzantine selection
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Optional
import logging
from tqdm import tqdm

from brac.types import Modality, ByzantineType
from brac.agents.mock_agent import MockAgentFactory
from brac.attacks import ByzantineAttack, ByzantineScenario
from brac.consensus.geometric_median import riemannian_weiszfeld
from brac.consensus.aggregators import (
    weighted_average, coordinate_median, krum, multi_krum, trimmed_mean,
    AggregatorResult
)
from brac.consensus.fisher_rao import fisher_rao_distance
from brac.visualization import plot_byzantine_resilience

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_single_case(
    beliefs: torch.Tensor,
    reliabilities: torch.Tensor,
    true_label: int,
    aggregator_name: str,
    num_byzantine: int = 0,
) -> dict:
    """Run a single case with specified aggregator.

    Args:
        beliefs: Agent beliefs, shape (N, K)
        reliabilities: Agent reliabilities, shape (N,)
        true_label: Ground truth class
        aggregator_name: Name of aggregation method
        num_byzantine: Actual number of Byzantine agents (for Krum/Multi-Krum)

    Returns:
        Dictionary with metrics
    """
    valid = True  # Track if aggregator conditions were met

    if aggregator_name == "geometric_median":
        result = riemannian_weiszfeld(beliefs, reliabilities)
        consensus = result.consensus
    elif aggregator_name == "weighted_average":
        consensus = weighted_average(beliefs, reliabilities)
    elif aggregator_name == "coordinate_median":
        consensus = coordinate_median(beliefs, reliabilities)
    elif aggregator_name == "krum":
        # Pass actual Byzantine count (Issue #1)
        agg_result = krum(beliefs, f=num_byzantine, return_result=True)
        consensus = agg_result.consensus
        valid = agg_result.valid
    elif aggregator_name == "multi_krum":
        # Pass actual Byzantine count (Issue #1)
        agg_result = multi_krum(beliefs, f=num_byzantine, m=3, return_result=True)
        consensus = agg_result.consensus
        valid = agg_result.valid
    elif aggregator_name == "trimmed_mean":
        consensus = trimmed_mean(beliefs, trim_fraction=0.25)
    else:
        raise ValueError(f"Unknown aggregator: {aggregator_name}")

    # Compute metrics
    predicted = consensus.argmax().item()
    sorted_indices = torch.argsort(consensus, descending=True)
    top3 = sorted_indices[:3].tolist()

    return {
        "correct": int(predicted == true_label),
        "top3_correct": int(true_label in top3),
        "true_prob": consensus[true_label].item(),
        "max_prob": consensus.max().item(),
        "valid": valid,  # Track if conditions were met
    }


class ByzantineScenarioFixed:
    """Fixed Byzantine scenario that excludes pathology from Byzantine selection.

    Issue #2: Pathology is the root-of-trust and should not become Byzantine,
    otherwise tau_path=1.0 gives corrupted agent maximum influence.
    """

    def __init__(self, num_classes: int = 9, seed: int = 42):
        self.K = num_classes
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def create_scenario(
        self,
        num_agents: int,
        num_byzantine: int,
        true_label: int,
        attack_type: ByzantineType = ByzantineType.TYPE_II,
        honest_concentration: float = 10.0,
        protect_root_of_trust: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, list[bool]]:
        """Create a Byzantine scenario.

        Args:
            num_agents: Total number of agents (n)
            num_byzantine: Number of Byzantine agents (f)
            true_label: Ground truth class
            attack_type: Type of Byzantine attack
            honest_concentration: Dirichlet concentration for honest agents
            protect_root_of_trust: If True, agent 0 (pathology) is never Byzantine

        Returns:
            Tuple of (beliefs, reliabilities, is_byzantine)
        """
        beliefs = []
        is_byzantine = []

        # Determine eligible agents for Byzantine corruption
        if protect_root_of_trust:
            # Agent 0 is pathology (root-of-trust), exclude from Byzantine selection
            eligible_indices = list(range(1, num_agents))
            num_byzantine = min(num_byzantine, len(eligible_indices))
        else:
            eligible_indices = list(range(num_agents))

        # Select Byzantine agents from eligible pool
        byzantine_indices = set(
            self.rng.choice(eligible_indices, size=num_byzantine, replace=False)
        ) if num_byzantine > 0 else set()

        # Generate beliefs
        for i in range(num_agents):
            if i in byzantine_indices:
                is_byzantine.append(True)

                # Generate honest belief first
                alpha = torch.ones(self.K) * 0.1
                alpha[true_label] = honest_concentration
                honest_belief = torch.distributions.Dirichlet(alpha).sample()

                # Apply attack
                if attack_type == ByzantineType.TYPE_I:
                    belief = ByzantineAttack.type_i_data_fault(honest_belief)
                elif attack_type == ByzantineType.TYPE_II:
                    belief = ByzantineAttack.type_ii_model_fault(honest_belief, self.K)
                elif attack_type == ByzantineType.STRATEGIC:
                    honest_so_far = [b for b, byz in zip(beliefs, is_byzantine) if not byz]
                    if honest_so_far:
                        target = self.rng.choice([c for c in range(self.K) if c != true_label])
                        belief = ByzantineAttack.strategic_attack(honest_so_far, target, self.K)
                    else:
                        belief = ByzantineAttack.random_attack(self.K)
                else:
                    belief = ByzantineAttack.random_attack(self.K)
                beliefs.append(belief)
            else:
                is_byzantine.append(False)
                alpha = torch.ones(self.K) * 0.1
                alpha[true_label] = honest_concentration
                belief = torch.distributions.Dirichlet(alpha).sample()
                beliefs.append(belief)

        beliefs_tensor = torch.stack(beliefs)
        reliabilities = torch.ones(num_agents) / num_agents

        return beliefs_tensor, reliabilities, is_byzantine


def run_experiment(
    num_cases: int = 1000,
    byzantine_counts: list[int] = [0, 1, 2, 3],
    attack_type: ByzantineType = ByzantineType.TYPE_II,
    aggregators: list[str] = None,
    seed: int = 42,
    output_dir: Optional[str] = None,
    protect_root_of_trust: bool = True,
) -> dict:
    """Run Byzantine resilience experiment.

    Args:
        num_cases: Number of test cases
        byzantine_counts: List of Byzantine agent counts to test
        attack_type: Type of Byzantine attack
        aggregators: List of aggregator names to compare
        seed: Random seed
        output_dir: Optional output directory for results
        protect_root_of_trust: If True, pathology is never Byzantine (Issue #2)

    Returns:
        Dictionary with experiment results
    """
    if aggregators is None:
        aggregators = [
            "weighted_average",
            "coordinate_median",
            "krum",
            "multi_krum",
            "trimmed_mean",
            "geometric_median",
        ]

    np.random.seed(seed)
    torch.manual_seed(seed)

    K = 9  # Number of classes
    N = 4  # Number of agents

    # Use fixed scenario that protects root-of-trust (Issue #2)
    scenario = ByzantineScenarioFixed(num_classes=K, seed=seed)

    results = {agg: {bc: [] for bc in byzantine_counts} for agg in aggregators}
    validity = {agg: {bc: {"valid": 0, "invalid": 0} for bc in byzantine_counts} for agg in aggregators}

    for num_byz in byzantine_counts:
        # Cap Byzantine count to max eligible (N-1 if protecting root-of-trust)
        actual_byz = min(num_byz, N - 1) if protect_root_of_trust else num_byz
        if actual_byz != num_byz:
            logger.info(f"Capping Byzantine count from {num_byz} to {actual_byz} (protecting root-of-trust)")

        logger.info(f"Testing with {actual_byz} Byzantine agents...")

        for case_idx in tqdm(range(num_cases), desc=f"Byzantine={actual_byz}"):
            # Generate random true label
            true_label = np.random.randint(0, K)

            # Create scenario with protection (Issue #2)
            beliefs, reliabilities, is_byzantine = scenario.create_scenario(
                num_agents=N,
                num_byzantine=actual_byz,
                true_label=true_label,
                attack_type=attack_type,
                honest_concentration=10.0,
                protect_root_of_trust=protect_root_of_trust,
            )

            # Test each aggregator
            for agg_name in aggregators:
                metrics = run_single_case(
                    beliefs, reliabilities, true_label, agg_name,
                    num_byzantine=actual_byz  # Pass actual count (Issue #1)
                )
                results[agg_name][num_byz].append(metrics)

                # Track validity
                if metrics.get("valid", True):
                    validity[agg_name][num_byz]["valid"] += 1
                else:
                    validity[agg_name][num_byz]["invalid"] += 1

    # Aggregate results
    summary = {}
    for agg_name in aggregators:
        summary[agg_name] = {}
        for num_byz in byzantine_counts:
            case_results = results[agg_name][num_byz]
            valid_results = [r for r in case_results if r.get("valid", True)]

            if valid_results:
                summary[agg_name][num_byz] = {
                    "accuracy": np.mean([r["correct"] for r in valid_results]),
                    "top3_accuracy": np.mean([r["top3_correct"] for r in valid_results]),
                    "avg_true_prob": np.mean([r["true_prob"] for r in valid_results]),
                    "avg_max_prob": np.mean([r["max_prob"] for r in valid_results]),
                    "valid_fraction": len(valid_results) / len(case_results),
                }
            else:
                # All conditions invalid - report N/A
                summary[agg_name][num_byz] = {
                    "accuracy": None,
                    "top3_accuracy": None,
                    "avg_true_prob": None,
                    "avg_max_prob": None,
                    "valid_fraction": 0.0,
                }

    # Print summary table
    print("\n" + "=" * 70)
    print(f"Byzantine Resilience Results ({attack_type.value})")
    print(f"Root-of-trust protection: {protect_root_of_trust}")
    print("=" * 70)
    print(f"{'Aggregator':<20} " + " ".join(f"Byz={bc:>6}" for bc in byzantine_counts))
    print("-" * 70)

    for agg_name in aggregators:
        acc_strs = []
        for bc in byzantine_counts:
            acc = summary[agg_name][bc]["accuracy"]
            if acc is None:
                acc_strs.append("   N/A")
            else:
                acc_strs.append(f"{acc:>6.1%}")
        print(f"{agg_name:<20} " + " ".join(acc_strs))

    # Print validity info for Krum/Multi-Krum
    print("\n" + "-" * 70)
    print("Aggregator Validity (conditions met):")
    for agg_name in ["krum", "multi_krum"]:
        if agg_name in aggregators:
            valid_info = [f"Byz={bc}: {validity[agg_name][bc]['valid']}/{num_cases}"
                         for bc in byzantine_counts]
            print(f"  {agg_name}: " + ", ".join(valid_info))

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save summary as YAML
        save_summary = {}
        for agg_name in aggregators:
            save_summary[agg_name] = {}
            for bc in byzantine_counts:
                s = summary[agg_name][bc]
                save_summary[agg_name][bc] = {
                    k: (float(v) if v is not None else None)
                    for k, v in s.items()
                }

        with open(output_path / "summary.yaml", "w") as f:
            yaml.dump(save_summary, f)

        # Create visualization (skip N/A values)
        accs_by_method = {}
        for agg in aggregators:
            accs = []
            for bc in byzantine_counts:
                acc = summary[agg][bc]["accuracy"]
                accs.append(acc if acc is not None else 0.0)
            accs_by_method[agg] = accs

        plot_byzantine_resilience(
            byzantine_counts,
            accs_by_method,
            title=f"Byzantine Resilience ({attack_type.value})",
            save_path=str(output_path / "byzantine_resilience.png"),
        )

    return summary


def main():
    """Main entry point for experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="Byzantine Resilience Experiment")
    parser.add_argument("--num-cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/byzantine_resilience")
    parser.add_argument("--attack", type=str, default="type_ii",
                       choices=["type_i", "type_ii", "strategic"])
    parser.add_argument("--no-protect-root", action="store_true",
                       help="Allow pathology to be Byzantine (not recommended)")
    args = parser.parse_args()

    attack_map = {
        "type_i": ByzantineType.TYPE_I,
        "type_ii": ByzantineType.TYPE_II,
        "strategic": ByzantineType.STRATEGIC,
    }

    run_experiment(
        num_cases=args.num_cases,
        attack_type=attack_map[args.attack],
        seed=args.seed,
        output_dir=args.output_dir,
        protect_root_of_trust=not args.no_protect_root,
    )


if __name__ == "__main__":
    main()
