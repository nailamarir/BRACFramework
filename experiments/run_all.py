"""Master experiment runner for BRAC framework.

Runs all experiments and generates paper figures and tables.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run all BRAC experiments."""
    parser = argparse.ArgumentParser(description="Run all BRAC experiments")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Base output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick version with fewer cases")
    parser.add_argument("--experiments", nargs="+",
                       default=["byzantine", "conformal", "shapley"],
                       choices=["byzantine", "conformal", "shapley", "convergence", "ablation", "case_studies"],
                       help="Which experiments to run")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set case counts based on quick mode
    num_cases = 100 if args.quick else 1000
    cal_size = 100 if args.quick else 500
    test_size = 100 if args.quick else 500
    cases_per_subtype = 20 if args.quick else 100

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting BRAC experiments at {timestamp}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Quick mode: {args.quick}")

    results = {}

    # Experiment 1: Byzantine Resilience
    if "byzantine" in args.experiments:
        logger.info("\n" + "=" * 60)
        logger.info("Running Experiment 1: Byzantine Resilience")
        logger.info("=" * 60)

        from experiments.exp1_byzantine_resilience import run_experiment as run_byzantine

        results["byzantine"] = run_byzantine(
            num_cases=num_cases,
            seed=args.seed,
            output_dir=str(output_dir / "byzantine_resilience"),
        )

    # Experiment 2: Conformal Coverage
    if "conformal" in args.experiments:
        logger.info("\n" + "=" * 60)
        logger.info("Running Experiment 2: Conformal Coverage")
        logger.info("=" * 60)

        from experiments.exp2_conformal_coverage import run_experiment as run_conformal

        results["conformal"] = run_conformal(
            calibration_size=cal_size,
            test_size=test_size,
            seed=args.seed,
            output_dir=str(output_dir / "conformal_coverage"),
        )

    # Experiment 3: Shapley Attribution
    if "shapley" in args.experiments:
        logger.info("\n" + "=" * 60)
        logger.info("Running Experiment 3: Shapley Attribution")
        logger.info("=" * 60)

        from experiments.exp3_shapley_attribution import run_experiment as run_shapley

        results["shapley"] = run_shapley(
            cases_per_subtype=cases_per_subtype,
            seed=args.seed,
            output_dir=str(output_dir / "shapley_attribution"),
        )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("All experiments completed!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")

    return results


if __name__ == "__main__":
    main()
