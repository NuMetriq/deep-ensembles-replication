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
    Plot and save a reliability diagram from precomputed bin statistics.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    bin_centers = stats["bin_centers"]
    avg_accuracy = stats["avg_accuracy"]
    bin_edges = stats["bin_edges"]

    widths = bin_edges[1:] - bin_edges[:-1]

    plt.figure(figsize=(6, 6))
    plt.bar(
        bin_centers.numpy(),
        avg_accuracy.numpy(),
        width=widths.numpy(),
        align="center",
        alpha=0.7,
        edgecolor="black",
    )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    dummy_stats = {
        "bin_edges": torch.linspace(0, 1, 11),
        "bin_centers": torch.linspace(0.05, 0.95, 10),
        "bin_counts": torch.randint(0, 100, (10,)),
        "avg_confidence": torch.linspace(0.05, 0.95, 10),
        "avg_accuracy": torch.linspace(0.10, 0.90, 10),
    }

    plot_reliability_diagram(
        dummy_stats,
        save_path="results/figures/generated/reliability_diagram_dummy.png",
    )
    print("Saved dummy reliability diagram.")
