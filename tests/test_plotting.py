from pathlib import Path

import torch
from src.plotting import (
    plot_confidence_histogram,
    plot_confidence_histogram_overlay,
    plot_reliability_diagram,
    top_class_confidence_from_logits,
)


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


def test_top_class_confidence_from_logits_returns_valid_probabilities():
    logits = torch.tensor(
        [
            [2.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
        ]
    )

    confidences = top_class_confidence_from_logits(logits)

    assert confidences.shape == (2,)
    assert torch.all(confidences >= 0.0)
    assert torch.all(confidences <= 1.0)


def test_plot_confidence_histogram_saves_file(tmp_path: Path):
    confidences = torch.rand(100)
    save_path = tmp_path / "confidence_hist.png"

    plot_confidence_histogram(confidences, save_path=save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_confidence_histogram_overlay_saves_file(tmp_path: Path):
    baseline_confidences = torch.rand(100)
    ensemble_confidences = torch.rand(100)
    save_path = tmp_path / "confidence_hist_overlay.png"

    plot_confidence_histogram_overlay(
        baseline_confidences,
        ensemble_confidences,
        save_path=save_path,
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0
