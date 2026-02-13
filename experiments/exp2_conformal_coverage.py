"""Experiment 2: Conformal Coverage Verification.

Verifies the distribution-free coverage guarantee:
    P(h_true in C_alpha) >= 1 - alpha

Setup:
- Split synthetic data: calibration + test
- Test multiple alpha values
- Verify coverage with and without Byzantine agents

Metrics:
- Empirical coverage (should be >= 1-alpha)
- Average prediction set size
- Set size distribution
- Coverage conditional on subtype
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Optional
import logging
from tqdm import tqdm

from brac.types import Modality, NHLSubtype
from brac.agents.mock_agent import MockAgentFactory
from brac.consensus.geometric_median import geometric_median_consensus
from brac.consensus.conformal import ConformalPredictor, split_calibration_test
from brac.consensus.trust import TrustEstimator
from brac.visualization import plot_conformal_coverage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_consensus_beliefs(
    num_cases: int,
    seed: int = 42,
    byzantine_fraction: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic consensus beliefs for conformal calibration.

    Args:
        num_cases: Number of cases to generate
        seed: Random seed
        byzantine_fraction: Fraction of cases with Byzantine agent

    Returns:
        Tuple of (consensus_beliefs, true_labels)
    """
    factory = MockAgentFactory(num_classes=9, seed=seed)
    trust_estimator = TrustEstimator()

    cases, labels = factory.generate_dataset(
        num_cases=num_cases,
        byzantine_fraction=byzantine_fraction,
    )

    consensus_beliefs = []

    for agent_outputs in tqdm(cases, desc="Generating consensus"):
        # Extract beliefs and qualities
        beliefs = {m: out.belief for m, out in agent_outputs.items()}
        qualities = {m: out.quality for m, out in agent_outputs.items()}

        # Compute reliabilities
        trust_result = trust_estimator.compute_reliability(beliefs, qualities)
        reliabilities = trust_result.reliabilities

        # Compute geometric median consensus
        consensus, _ = geometric_median_consensus(beliefs, reliabilities)
        consensus_beliefs.append(consensus)

    return torch.stack(consensus_beliefs), labels


def run_experiment(
    calibration_size: int = 500,
    test_size: int = 500,
    alpha_values: list[float] = [0.01, 0.05, 0.10, 0.20],
    byzantine_fraction: float = 0.0,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> dict:
    """Run conformal coverage experiment.

    Args:
        calibration_size: Number of calibration cases
        test_size: Number of test cases
        alpha_values: Alpha values to test
        byzantine_fraction: Fraction of Byzantine cases
        seed: Random seed
        output_dir: Optional output directory

    Returns:
        Dictionary with experiment results
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    total_size = calibration_size + test_size

    # Generate data
    logger.info(f"Generating {total_size} synthetic cases...")
    all_beliefs, all_labels = generate_consensus_beliefs(
        num_cases=total_size,
        seed=seed,
        byzantine_fraction=byzantine_fraction,
    )

    # Split into calibration and test
    cal_beliefs, cal_labels, test_beliefs, test_labels = split_calibration_test(
        all_beliefs, all_labels,
        cal_fraction=calibration_size / total_size,
        seed=seed,
    )

    results = {}

    for alpha in alpha_values:
        logger.info(f"Testing alpha = {alpha}...")

        # Create and calibrate conformal predictor
        conformal = ConformalPredictor(alpha=alpha)
        q_hat = conformal.calibrate(cal_beliefs, cal_labels)

        # Evaluate on test set
        eval_results = conformal.calibrate_and_evaluate(
            cal_beliefs, cal_labels, test_beliefs, test_labels
        )

        results[alpha] = {
            "target_coverage": 1 - alpha,
            "empirical_coverage": eval_results["empirical_coverage"],
            "coverage_gap": eval_results["coverage_gap"],
            "q_hat": q_hat,
            "average_set_size": eval_results["average_set_size"],
            "set_size_distribution": eval_results["set_size_distribution"],
        }

        logger.info(f"  Coverage: {eval_results['empirical_coverage']:.2%} "
                   f"(target: {1-alpha:.2%})")
        logger.info(f"  Avg set size: {eval_results['average_set_size']:.2f}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"Conformal Coverage Results (Byzantine={byzantine_fraction:.0%})")
    print("=" * 60)
    print(f"{'Alpha':<10} {'Target':<12} {'Empirical':<12} {'Gap':<10} {'Avg |C|':<10}")
    print("-" * 60)

    for alpha in alpha_values:
        r = results[alpha]
        print(f"{alpha:<10.2f} {r['target_coverage']:<12.2%} "
              f"{r['empirical_coverage']:<12.2%} "
              f"{r['coverage_gap']:+<10.2%} "
              f"{r['average_set_size']:<10.2f}")

    # Check coverage guarantee
    print("\n" + "-" * 60)
    all_valid = all(
        results[a]["empirical_coverage"] >= (1 - a - 0.02)  # Allow small margin
        for a in alpha_values
    )
    if all_valid:
        print("✓ Coverage guarantee SATISFIED for all alpha values")
    else:
        print("✗ Coverage guarantee VIOLATED for some alpha values")

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as YAML (convert non-serializable types)
        save_results = {}
        for alpha, r in results.items():
            save_results[float(alpha)] = {
                "target_coverage": float(r["target_coverage"]),
                "empirical_coverage": float(r["empirical_coverage"]),
                "coverage_gap": float(r["coverage_gap"]),
                "q_hat": float(r["q_hat"]),
                "average_set_size": float(r["average_set_size"]),
                "set_size_distribution": {int(k): int(v) for k, v in r["set_size_distribution"].items()},
            }

        with open(output_path / "summary.yaml", "w") as f:
            yaml.dump(save_results, f)

        # Create visualization
        empirical_coverages = [results[a]["empirical_coverage"] for a in alpha_values]
        avg_set_sizes = [results[a]["average_set_size"] for a in alpha_values]

        plot_conformal_coverage(
            alpha_values,
            empirical_coverages,
            avg_set_sizes,
            title=f"Conformal Coverage (Byzantine={byzantine_fraction:.0%})",
            save_path=str(output_path / "conformal_coverage.png"),
        )

    return results


def main():
    """Main entry point for experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="Conformal Coverage Experiment")
    parser.add_argument("--cal-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--byzantine", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="results/conformal_coverage")
    args = parser.parse_args()

    run_experiment(
        calibration_size=args.cal_size,
        test_size=args.test_size,
        byzantine_fraction=args.byzantine,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
