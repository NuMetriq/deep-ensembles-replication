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


def top_class_confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Extract top-class confidence from logits.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (N, C), raw model outputs or log-probabilities.

    Returns
    -------
    torch.Tensor
        Shape (N,), max predicted probability for each sample.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (N, C), got {tuple(logits.shape)}")

    probs = torch.softmax(logits, dim=1)
    confidences, _ = probs.max(dim=1)
    return confidences.cpu()


def plot_confidence_histogram(
    confidences: torch.Tensor,
    save_path: str | Path,
    title: str = "Prediction Confidence Histogram",
    n_bins: int = 20,
) -> None:
    """
    Plot and save a histogram of top-class prediction confidences.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.hist(confidences.numpy(), bins=n_bins, range=(0.0, 1.0), edgecolor="black")
    plt.xlim(0, 1)
    plt.xlabel("Top-class confidence")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confidence_histogram_overlay(
    baseline_confidences: torch.Tensor,
    ensemble_confidences: torch.Tensor,
    save_path: str | Path,
    title: str = "Baseline vs Ensemble Confidence Histogram",
    n_bins: int = 20,
) -> None:
    """
    Plot and save an overlaid confidence histogram comparing baseline and ensemble.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.hist(
        baseline_confidences.numpy(),
        bins=n_bins,
        range=(0.0, 1.0),
        alpha=0.6,
        label="Baseline",
        edgecolor="black",
    )
    plt.hist(
        ensemble_confidences.numpy(),
        bins=n_bins,
        range=(0.0, 1.0),
        alpha=0.6,
        label="Ensemble",
        edgecolor="black",
    )
    plt.xlim(0, 1)
    plt.xlabel("Top-class confidence")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
