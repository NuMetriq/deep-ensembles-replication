from pathlib import Path

import torch
from src.plotting import plot_reliability_diagram


def test_plot_reliability_diagram_saves_file(tmp_path: Path):
    stats = {
        "bin_edges": torch.linspace(0, 1, 11),
        "bin_centers": torch.linspace(0.05, 0.95, 10),
        "bin_counts": torch.randint(0, 100, (10,)),
        "avg_confidence": torch.linspace(0.05, 0.95, 10),
        "avg_accuracy": torch.linspace(0.10, 0.90, 10),
    }

    save_path = tmp_path / "reliability.png"
    plot_reliability_diagram(stats, save_path=save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0
