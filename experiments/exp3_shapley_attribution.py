"""Experiment 3: Shapley Attribution Analysis.

Demonstrates that Shapley values reveal clinically meaningful patterns
for NHL subtype diagnosis.

Expected patterns (from paper):
- FL: pathology dominates (phi_path ≈ 0.45) - follicular pattern distinctive
- BL: radiology high (phi_rad ≈ 0.30) - PET avidity characteristic
- DLBCL: positive I_{path,lab} - IHC + flow cytometry synergy for GCB vs ABC
- CLL/SLL: laboratory dominates - flow cytometry is diagnostic
- MCL: path + lab synergy - cyclin D1 + morphology

Metrics:
- Shapley values per subtype
- Interaction indices
- Efficiency axiom verification
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
from brac.consensus.shapley import ShapleyAttributor, aggregate_shapley_across_cases
from brac.consensus.trust import TrustEstimator
from brac.visualization import plot_shapley_heatmap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_experiment(
    cases_per_subtype: int = 100,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> dict:
    """Run Shapley attribution experiment.

    Args:
        cases_per_subtype: Number of cases per NHL subtype
        seed: Random seed
        output_dir: Optional output directory

    Returns:
        Dictionary with experiment results
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    factory = MockAgentFactory(num_classes=9, seed=seed)
    trust_estimator = TrustEstimator()
    shapley_attributor = ShapleyAttributor(
        agents=Modality.all(),
        geometric_median_fn=geometric_median_consensus,
        num_classes=9,
    )

    # Results storage
    subtype_results = {subtype: [] for subtype in NHLSubtype}
    efficiency_errors = []

    # Generate cases for each subtype
    for subtype in NHLSubtype:
        logger.info(f"Processing {subtype.name}...")

        for case_idx in tqdm(range(cases_per_subtype), desc=subtype.name):
            # Generate case with this subtype as true label
            agents = factory.create_all_agents()
            agent_outputs = factory.generate_case(
                agents, true_label=subtype.value
            )

            # Extract beliefs and compute reliabilities
            beliefs = {m: out.belief for m, out in agent_outputs.items()}
            qualities = {m: out.quality for m, out in agent_outputs.items()}

            trust_result = trust_estimator.compute_reliability(beliefs, qualities)
            reliabilities = trust_result.reliabilities

            # Compute Shapley values
            shapley_result = shapley_attributor.compute_full_attribution(
                beliefs=beliefs,
                reliabilities=reliabilities,
                true_label=subtype.value,
            )

            subtype_results[subtype].append(shapley_result)
            efficiency_errors.append(shapley_result.efficiency_error)

    # Aggregate results per subtype
    aggregated = {}
    for subtype in NHLSubtype:
        shapley_values_list = [r.shapley_values for r in subtype_results[subtype]]
        interaction_list = [r.interaction_indices for r in subtype_results[subtype]]

        # Average Shapley values
        avg_shapley = {m: 0.0 for m in Modality.all()}
        for sv in shapley_values_list:
            for m, val in sv.items():
                avg_shapley[m] += val / len(shapley_values_list)

        # Average interactions
        avg_interactions = {}
        for m1 in Modality.all():
            for m2 in Modality.all():
                if m1 != m2:
                    key = (m1, m2)
                    vals = [il.get(key, 0) for il in interaction_list]
                    avg_interactions[key] = np.mean(vals)

        aggregated[subtype] = {
            "shapley_values": avg_shapley,
            "interactions": avg_interactions,
            "dominant_agent": max(avg_shapley, key=avg_shapley.get),
        }

    # Print summary
    print("\n" + "=" * 70)
    print("Shapley Attribution Results")
    print("=" * 70)
    print(f"{'Subtype':<12} " + " ".join(f"{m.value[:4]:>8}" for m in Modality.all()) + "  Dominant")
    print("-" * 70)

    for subtype in NHLSubtype:
        agg = aggregated[subtype]
        vals = [f"{agg['shapley_values'][m]:+.3f}" for m in Modality.all()]
        dominant = agg["dominant_agent"].value[:4]
        print(f"{subtype.short_name:<12} " + " ".join(f"{v:>8}" for v in vals) + f"  {dominant}")

    # Print interaction analysis
    print("\n" + "-" * 70)
    print("Key Interactions (I_ij > 0 = synergy, I_ij < 0 = redundancy):")
    print("-" * 70)

    for subtype in [NHLSubtype.DLBCL_GCB, NHLSubtype.DLBCL_ABC, NHLSubtype.MCL]:
        agg = aggregated[subtype]
        path_lab = agg["interactions"].get((Modality.PATHOLOGY, Modality.LABORATORY), 0)
        print(f"  {subtype.short_name}: I_{{path,lab}} = {path_lab:+.4f} "
              f"({'synergy' if path_lab > 0 else 'redundancy'})")

    # Efficiency axiom check
    print("\n" + "-" * 70)
    print("Efficiency Axiom Verification:")
    avg_error = np.mean(efficiency_errors)
    max_error = np.max(efficiency_errors)
    print(f"  Average error: {avg_error:.2e}")
    print(f"  Maximum error: {max_error:.2e}")
    print(f"  {'✓ PASSED' if max_error < 1e-5 else '✗ FAILED'}")

    # Prepare for visualization
    subtype_shapley = {
        subtype: aggregated[subtype]["shapley_values"]
        for subtype in NHLSubtype
    }

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as YAML
        save_results = {}
        for subtype in NHLSubtype:
            agg = aggregated[subtype]
            save_results[subtype.name] = {
                "shapley_values": {m.value: float(v) for m, v in agg["shapley_values"].items()},
                "dominant_agent": agg["dominant_agent"].value,
            }

        save_results["efficiency"] = {
            "average_error": float(avg_error),
            "max_error": float(max_error),
        }

        with open(output_path / "summary.yaml", "w") as f:
            yaml.dump(save_results, f)

        # Create heatmap
        plot_shapley_heatmap(
            subtype_shapley,
            title="Agent Importance by NHL Subtype",
            save_path=str(output_path / "shapley_heatmap.png"),
        )

    return aggregated


def main():
    """Main entry point for experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="Shapley Attribution Experiment")
    parser.add_argument("--cases-per-subtype", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/shapley_attribution")
    args = parser.parse_args()

    run_experiment(
        cases_per_subtype=args.cases_per_subtype,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
