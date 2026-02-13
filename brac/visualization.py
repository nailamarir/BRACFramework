"""Visualization utilities for BRAC framework.

This module provides plotting functions for:
1. Simplex plots - agent beliefs and consensus on probability simplex
2. Convergence curves - Weiszfeld algorithm convergence
3. Shapley heatmaps - agent importance per subtype
4. Shapley bar charts - per-case attribution
5. Conformal coverage plots - empirical vs target coverage
6. Byzantine resilience plots - accuracy vs number of Byzantine agents
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from typing import Optional
import logging

from brac.types import Modality, NHLSubtype, BRACResult

logger = logging.getLogger(__name__)

# Color palette from paper specification
COLORS = {
    Modality.PATHOLOGY: "#DC5050",   # Red
    Modality.RADIOLOGY: "#508CDC",   # Blue
    Modality.LABORATORY: "#50B450",  # Green
    Modality.CLINICAL: "#DCB43C",    # Yellow
    "geometric_median": "#3CB4B4",   # Teal
    "weighted_average": "#C85050",   # Dark red
    "orchestrator": "#A050C8",       # Purple
}

# Plot style configuration
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _barycentric_to_cartesian(b: np.ndarray) -> tuple[float, float]:
    """Convert barycentric (simplex) coordinates to 2D Cartesian.

    Maps 3D barycentric coordinates to equilateral triangle vertices:
    - Vertex 0 at (0, 0)
    - Vertex 1 at (1, 0)
    - Vertex 2 at (0.5, sqrt(3)/2)

    Args:
        b: Barycentric coordinates [b0, b1, b2], sum to 1

    Returns:
        Tuple (x, y) in Cartesian coordinates
    """
    # Triangle vertices
    v0 = np.array([0, 0])
    v1 = np.array([1, 0])
    v2 = np.array([0.5, np.sqrt(3) / 2])

    # Convert
    x = b[0] * v0[0] + b[1] * v1[0] + b[2] * v2[0]
    y = b[0] * v0[1] + b[1] * v1[1] + b[2] * v2[1]

    return x, y


def plot_simplex(
    beliefs: dict[Modality, torch.Tensor],
    consensus: Optional[torch.Tensor] = None,
    weighted_avg: Optional[torch.Tensor] = None,
    class_indices: tuple[int, int, int] = (0, 1, 2),
    class_names: Optional[tuple[str, str, str]] = None,
    title: str = "Agent Beliefs on Simplex",
    ax: Optional[plt.Axes] = None,
    show_legend: bool = True,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot agent beliefs on a 2D simplex (3-class projection).

    Creates an equilateral triangle with vertices at the 3 selected classes.
    Each agent's belief is projected onto this simplex and plotted.

    Args:
        beliefs: Dictionary mapping Modality -> belief tensor (K,)
        consensus: Optional geometric median consensus to highlight
        weighted_avg: Optional weighted average for comparison
        class_indices: Which 3 classes to use for simplex vertices
        class_names: Names for the 3 classes (default: use subtype names)
        title: Plot title
        ax: Matplotlib axes (creates new figure if None)
        show_legend: Whether to show legend
        save_path: Optional path to save figure

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    # Get class names
    if class_names is None:
        class_names = tuple(
            NHLSubtype.from_index(i).short_name for i in class_indices
        )

    # Draw triangle
    triangle = plt.Polygon(
        [[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]],
        fill=False, edgecolor="black", linewidth=2,
    )
    ax.add_patch(triangle)

    # Label vertices
    ax.text(0, -0.05, class_names[0], ha="center", va="top", fontweight="bold")
    ax.text(1, -0.05, class_names[1], ha="center", va="top", fontweight="bold")
    ax.text(0.5, np.sqrt(3) / 2 + 0.05, class_names[2], ha="center", va="bottom", fontweight="bold")

    # Plot agent beliefs
    for modality, belief in beliefs.items():
        # Extract relevant components and normalize
        b = belief[list(class_indices)].numpy()
        b = b / b.sum()  # Renormalize to 3-simplex

        x, y = _barycentric_to_cartesian(b)
        ax.scatter(
            x, y, c=COLORS[modality], s=200, label=modality.value.capitalize(),
            edgecolors="white", linewidths=2, zorder=5,
        )

    # Plot consensus
    if consensus is not None:
        b = consensus[list(class_indices)].numpy()
        b = b / b.sum()
        x, y = _barycentric_to_cartesian(b)
        ax.scatter(
            x, y, c=COLORS["geometric_median"], s=300, marker="*",
            label="Geometric Median", edgecolors="black", linewidths=1, zorder=6,
        )

    # Plot weighted average for comparison
    if weighted_avg is not None:
        b = weighted_avg[list(class_indices)].numpy()
        b = b / b.sum()
        x, y = _barycentric_to_cartesian(b)
        ax.scatter(
            x, y, c="none", s=250, marker="o",
            label="Weighted Average", edgecolors=COLORS["weighted_average"],
            linewidths=3, zorder=4,
        )

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)

    if show_legend:
        ax.legend(loc="upper right", framealpha=0.9)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")

    return ax


def plot_convergence(
    convergence_traces: list[list[float]],
    labels: Optional[list[str]] = None,
    title: str = "Weiszfeld Convergence",
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot convergence curves for Weiszfeld algorithm.

    Args:
        convergence_traces: List of convergence traces (one per experiment)
        labels: Optional labels for each trace
        title: Plot title
        ax: Matplotlib axes
        log_scale: Whether to use log scale on y-axis
        save_path: Optional path to save figure

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(convergence_traces))]

    for trace, label in zip(convergence_traces, labels):
        iterations = range(1, len(trace) + 1)
        ax.plot(iterations, trace, label=label, linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("$d_{FR}(b^{(\ell)}, b^{(\ell-1)})$")
    ax.set_title(title)

    if log_scale:
        ax.set_yscale("log")

    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return ax


def plot_shapley_heatmap(
    subtype_shapley: dict[NHLSubtype, dict[Modality, float]],
    title: str = "Agent Importance by Subtype",
    ax: Optional[plt.Axes] = None,
    annotate: bool = True,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot heatmap of Shapley values per subtype.

    Rows = subtypes, Columns = agents
    Color intensity = Shapley value (importance)

    Args:
        subtype_shapley: Dict mapping NHLSubtype -> {Modality -> phi}
        title: Plot title
        ax: Matplotlib axes
        annotate: Whether to show values in cells
        save_path: Optional path to save figure

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))

    # Build matrix
    subtypes = list(NHLSubtype)
    modalities = Modality.all()

    matrix = np.zeros((len(subtypes), len(modalities)))
    for i, subtype in enumerate(subtypes):
        if subtype in subtype_shapley:
            for j, modality in enumerate(modalities):
                matrix[i, j] = subtype_shapley[subtype].get(modality, 0)

    # Plot heatmap
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")

    # Labels
    ax.set_xticks(range(len(modalities)))
    ax.set_xticklabels([m.value.capitalize() for m in modalities], rotation=45, ha="right")
    ax.set_yticks(range(len(subtypes)))
    ax.set_yticklabels([s.short_name for s in subtypes])

    # Annotate cells
    if annotate:
        for i in range(len(subtypes)):
            for j in range(len(modalities)):
                val = matrix[i, j]
                color = "white" if abs(val) > 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Shapley Value")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return ax


def plot_shapley_bar(
    result: BRACResult,
    title: str = "Agent Attribution",
    ax: Optional[plt.Axes] = None,
    show_trust: bool = True,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot horizontal bar chart of Shapley values for a single case.

    Args:
        result: BRACResult from orchestrator
        title: Plot title
        ax: Matplotlib axes
        show_trust: Whether to show trust/reliability alongside
        save_path: Optional path to save figure

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    modalities = Modality.all()
    y_pos = np.arange(len(modalities))

    # Get Shapley values
    shapley_vals = [result.shapley_values.get(m, 0) for m in modalities]

    # Bar colors
    colors = [COLORS[m] for m in modalities]

    # Plot bars
    bars = ax.barh(y_pos, shapley_vals, color=colors, edgecolor="black")

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.value.capitalize() for m in modalities])
    ax.set_xlabel("Shapley Value ($\\phi_i$)")
    ax.set_title(title)

    # Add trust/reliability annotations
    if show_trust:
        for i, (m, val) in enumerate(zip(modalities, shapley_vals)):
            trust = result.agent_trusts.get(m, 0)
            rel = result.agent_reliabilities.get(m, 0)
            ax.text(
                max(shapley_vals) * 1.02, i,
                f"$\\tau$={trust:.2f}, r={rel:.2f}",
                va="center", fontsize=9,
            )

    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return ax


def plot_conformal_coverage(
    alphas: list[float],
    empirical_coverages: list[float],
    avg_set_sizes: Optional[list[float]] = None,
    title: str = "Conformal Prediction Coverage",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot empirical coverage vs target coverage.

    Args:
        alphas: List of alpha values
        empirical_coverages: Empirical coverage rates
        avg_set_sizes: Optional average set sizes
        title: Plot title
        ax: Matplotlib axes
        save_path: Optional path to save figure

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    target_coverages = [1 - a for a in alphas]

    # Plot target line
    ax.plot(alphas, target_coverages, "k--", label="Target (1-$\\alpha$)", linewidth=2)

    # Plot empirical
    ax.plot(alphas, empirical_coverages, "o-", color=COLORS["geometric_median"],
            label="Empirical", linewidth=2, markersize=8)

    # Shade valid region
    ax.fill_between(alphas, target_coverages, 1, alpha=0.2, color="green",
                    label="Valid Region")

    ax.set_xlabel("$\\alpha$ (Miscoverage Rate)")
    ax.set_ylabel("Coverage")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Secondary axis for set size
    if avg_set_sizes is not None:
        ax2 = ax.twinx()
        ax2.plot(alphas, avg_set_sizes, "s--", color=COLORS["orchestrator"],
                 label="Avg Set Size", linewidth=2, markersize=6)
        ax2.set_ylabel("Average Set Size", color=COLORS["orchestrator"])
        ax2.tick_params(axis="y", labelcolor=COLORS["orchestrator"])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return ax


def plot_byzantine_resilience(
    num_byzantine: list[int],
    accuracies: dict[str, list[float]],
    title: str = "Byzantine Resilience",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot accuracy vs number of Byzantine agents for different methods.

    Args:
        num_byzantine: List of Byzantine agent counts
        accuracies: Dict mapping method name -> list of accuracies
        title: Plot title
        ax: Matplotlib axes
        save_path: Optional path to save figure

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Color map for methods
    method_colors = {
        "geometric_median": COLORS["geometric_median"],
        "weighted_average": COLORS["weighted_average"],
        "krum": "#FFA500",
        "multi_krum": "#FF6347",
        "trimmed_mean": "#9370DB",
        "coordinate_median": "#20B2AA",
    }

    for method, accs in accuracies.items():
        color = method_colors.get(method, "gray")
        label = method.replace("_", " ").title()
        ax.plot(num_byzantine, accs, "o-", label=label, color=color,
                linewidth=2, markersize=8)

    ax.set_xlabel("Number of Byzantine Agents")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(num_byzantine)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return ax


def create_diagnostic_report(
    result: BRACResult,
    beliefs: dict[Modality, torch.Tensor],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create comprehensive diagnostic report figure.

    Combines multiple visualizations:
    - Simplex plot
    - Shapley bar chart
    - Summary statistics

    Args:
        result: BRACResult from orchestrator
        beliefs: Original agent beliefs
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(14, 8))

    # Simplex plot
    ax1 = fig.add_subplot(2, 2, 1)
    plot_simplex(
        beliefs, consensus=result.consensus_belief,
        title="Agent Beliefs", ax=ax1, show_legend=True,
    )

    # Shapley bar chart
    ax2 = fig.add_subplot(2, 2, 2)
    plot_shapley_bar(result, title="Agent Attribution", ax=ax2)

    # Text summary
    ax3 = fig.add_subplot(2, 2, (3, 4))
    ax3.axis("off")

    summary_text = f"""
BRAC Diagnostic Report
{'='*50}

Diagnosis: {result.diagnosis.name} ({result.diagnosis.value})
Confidence: {result.confidence:.1%}

Prediction Set: {[s.short_name for s in result.prediction_set]}
Set Size: {result.prediction_set_size}
Decision: {'ACCEPTED' if result.accepted else 'ESCALATE TO REVIEW'}

Convergence: {result.convergence_rounds} rounds

Agent Reliabilities:
{chr(10).join(f'  {m.value}: tau={result.agent_trusts.get(m, 0):.3f}, r={result.agent_reliabilities.get(m, 0):.3f}' for m in Modality.all())}

Shapley Values:
{chr(10).join(f'  {m.value}: phi={result.shapley_values.get(m, 0):+.4f}' for m in Modality.all())}
"""

    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
             fontfamily="monospace", fontsize=10, verticalalignment="top")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")

    return fig


def save_all_formats(fig: plt.Figure, base_path: str):
    """Save figure in both PNG and PDF formats.

    Args:
        fig: Matplotlib figure
        base_path: Base path without extension
    """
    fig.savefig(f"{base_path}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight")
    logger.info(f"Saved figure to {base_path}.png and {base_path}.pdf")
