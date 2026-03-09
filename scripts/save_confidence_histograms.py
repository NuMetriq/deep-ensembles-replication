from __future__ import annotations

from pathlib import Path

import torch
from src.data import get_mnist_dataloaders
from src.ensemble import ensemble_predict_logits, load_ensemble_models
from src.evaluate import collect_logits_and_targets
from src.model import MNISTClassifier
from src.plotting import (
    plot_confidence_histogram,
    plot_confidence_histogram_overlay,
    top_class_confidence_from_logits,
)


def main() -> None:
    data_dir = "data"
    batch_size = 128
    num_workers = 0
    pin_memory = False

    baseline_checkpoint = Path("checkpoints/mnist_baseline.pt")
    ensemble_checkpoint_dir = Path("checkpoints/ensemble")

    baseline_hist_path = Path(
        "results/figures/generated/baseline_confidence_histogram.png"
    )
    ensemble_hist_path = Path(
        "results/figures/generated/ensemble_confidence_histogram.png"
    )
    overlay_hist_path = Path(
        "results/figures/generated/baseline_vs_ensemble_confidence_histogram.png"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not baseline_checkpoint.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {baseline_checkpoint}. "
            "Run `python -m scripts.train_single` first."
        )

    _, test_loader = get_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Baseline logits
    baseline_model = MNISTClassifier().to(device)
    baseline_state_dict = torch.load(baseline_checkpoint, map_location=device)
    baseline_model.load_state_dict(baseline_state_dict)

    baseline_logits, _ = collect_logits_and_targets(
        model=baseline_model,
        dataloader=test_loader,
        device=device,
    )
    baseline_confidences = top_class_confidence_from_logits(baseline_logits)

    # Ensemble logits
    ensemble_models = load_ensemble_models(
        model_factory=MNISTClassifier,
        checkpoint_dir=ensemble_checkpoint_dir,
        device=device,
    )
    ensemble_logits, _ = ensemble_predict_logits(
        models=ensemble_models,
        dataloader=test_loader,
        device=device,
    )
    ensemble_confidences = top_class_confidence_from_logits(ensemble_logits)

    # Save figures
    plot_confidence_histogram(
        baseline_confidences,
        save_path=baseline_hist_path,
        title="Baseline Confidence Histogram",
    )
    plot_confidence_histogram(
        ensemble_confidences,
        save_path=ensemble_hist_path,
        title="Ensemble Confidence Histogram",
    )
    plot_confidence_histogram_overlay(
        baseline_confidences,
        ensemble_confidences,
        save_path=overlay_hist_path,
        title="Baseline vs Ensemble Confidence Histogram",
    )

    print(f"Saved baseline histogram to: {baseline_hist_path}")
    print(f"Saved ensemble histogram to: {ensemble_hist_path}")
    print(f"Saved overlay histogram to: {overlay_hist_path}")


if __name__ == "__main__":
    main()
