from __future__ import annotations

import json
from pathlib import Path

import torch
from src.data import get_mnist_dataloaders
from src.ensemble import ensemble_predict_logits, load_ensemble_models
from src.evaluate import evaluate_from_logits
from src.model import MNISTClassifier
from src.plotting import plot_reliability_diagram


def main() -> None:
    # Config
    data_dir = "data"
    batch_size = 128
    num_workers = 0
    pin_memory = False

    checkpoint_dir = Path("checkpoints/ensemble")
    metrics_path = Path("results/tables/generated/ensemble_metrics.json")
    figure_path = Path("results/figures/generated/ensemble_reliability_diagram.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, test_loader = get_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    models = load_ensemble_models(
        model_factory=MNISTClassifier,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
    print(f"Loaded {len(models)} ensemble members from {checkpoint_dir}")

    ensemble_logits, targets = ensemble_predict_logits(
        models=models,
        dataloader=test_loader,
        device=device,
    )

    results = evaluate_from_logits(ensemble_logits, targets, n_bins=10)
    metrics = results["metrics"]
    reliability_stats = results["reliability_stats"]

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_reliability_diagram(
        reliability_stats,
        save_path=figure_path,
        title="MNIST Ensemble Reliability Diagram",
    )

    print("Ensemble evaluation complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved reliability diagram to: {figure_path}")


if __name__ == "__main__":
    main()
