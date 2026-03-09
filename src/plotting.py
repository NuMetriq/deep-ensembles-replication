from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_reliability_diagram(
    stats: dict[str, torch.Tensor],
    save_path: str | Path,
    title: str = "Reliability Diagram",
) -> None:
    """
    Plot and save an improved reliability diagram with:
    - top panel: average accuracy by confidence bin
    - diagonal reference line for perfect calibration
    - confidence line overlay
    - bottom panel: bin counts

    Parameters
    ----------
    stats : dict[str, torch.Tensor]
        Output from reliability_diagram_stats(...)
    save_path : str | Path
        Path where the figure will be saved.
    title : str
        Title for the figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    bin_edges = stats["bin_edges"]
    bin_centers = stats["bin_centers"]
    bin_counts = stats["bin_counts"]
    avg_confidence = stats["avg_confidence"]
    avg_accuracy = stats["avg_accuracy"]

    widths = (bin_edges[1:] - bin_edges[:-1]).numpy()
    centers = bin_centers.numpy()
    counts = bin_counts.numpy()
    conf = avg_confidence.numpy()
    acc = avg_accuracy.numpy()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_top, ax_bottom = axes

    # Top: calibration bars + lines
    ax_top.bar(
        centers,
        acc,
        width=widths,
        align="center",
        alpha=0.7,
        edgecolor="black",
        label="Accuracy",
    )
    ax_top.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax_top.plot(
        centers,
        conf,
        marker="o",
        linewidth=1.5,
        label="Avg confidence",
    )

    # Optional visual gap markers
    for x, y_conf, y_acc in zip(centers, conf, acc):
        ax_top.plot([x, x], [y_acc, y_conf], linewidth=1)

    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel("Accuracy / Confidence")
    ax_top.set_title(title)
    ax_top.legend()
    ax_top.grid(alpha=0.3)

    # Bottom: counts per bin
    ax_bottom.bar(
        centers,
        counts,
        width=widths,
        align="center",
        alpha=0.7,
        edgecolor="black",
    )
    ax_bottom.set_xlim(0, 1)
    ax_bottom.set_xlabel("Confidence")
    ax_bottom.set_ylabel("Count")
    ax_bottom.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
