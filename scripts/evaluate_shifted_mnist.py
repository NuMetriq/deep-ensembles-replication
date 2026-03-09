from __future__ import annotations

import json
from pathlib import Path

import torch
from src.data import get_mnist_shifted_test_loader
from src.ensemble import ensemble_predict_logits, load_ensemble_models
from src.evaluate import collect_logits_and_targets, evaluate_from_logits
from src.model import MNISTClassifier
from src.plotting import (
    plot_confidence_histogram,
    plot_reliability_diagram,
    top_class_confidence_from_logits,
)


def main() -> None:
    data_dir = "data"
    batch_size = 128
    num_workers = 0
    pin_memory = False
    noise_std = 0.2

    baseline_checkpoint = Path("checkpoints/mnist_baseline.pt")
    ensemble_checkpoint_dir = Path("checkpoints/ensemble")

    baseline_metrics_path = Path(
        "results/tables/generated/shifted_baseline_metrics.json"
    )
    ensemble_metrics_path = Path(
        "results/tables/generated/shifted_ensemble_metrics.json"
    )

    baseline_reliability_path = Path(
        "results/figures/generated/shifted_baseline_reliability_diagram.png"
    )
    ensemble_reliability_path = Path(
        "results/figures/generated/shifted_ensemble_reliability_diagram.png"
    )

    baseline_conf_hist_path = Path(
        "results/figures/generated/shifted_baseline_confidence_histogram.png"
    )
    ensemble_conf_hist_path = Path(
        "results/figures/generated/shifted_ensemble_confidence_histogram.png"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating shifted MNIST with Gaussian noise std={noise_std}")

    if not baseline_checkpoint.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {baseline_checkpoint}. "
            "Run `python -m scripts.train_single` first."
        )

    shifted_test_loader = get_mnist_shifted_test_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        noise_std=noise_std,
    )

    # Baseline
    baseline_model = MNISTClassifier().to(device)
    baseline_state_dict = torch.load(baseline_checkpoint, map_location=device)
    baseline_model.load_state_dict(baseline_state_dict)

    baseline_logits, baseline_targets = collect_logits_and_targets(
        model=baseline_model,
        dataloader=shifted_test_loader,
        device=device,
    )
    baseline_results = evaluate_from_logits(
        baseline_logits,
        baseline_targets,
        n_bins=10,
    )

    baseline_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_metrics_path, "w", encoding="utf-8") as f:
        json.dump(baseline_results["metrics"], f, indent=2)

    plot_reliability_diagram(
        baseline_results["reliability_stats"],
        save_path=baseline_reliability_path,
        title="Shifted MNIST Baseline Reliability Diagram",
    )

    baseline_confidences = top_class_confidence_from_logits(baseline_logits)
    plot_confidence_histogram(
        baseline_confidences,
        save_path=baseline_conf_hist_path,
        title="Shifted MNIST Baseline Confidence Histogram",
    )

    # Ensemble
    ensemble_models = load_ensemble_models(
        model_factory=MNISTClassifier,
        checkpoint_dir=ensemble_checkpoint_dir,
        device=device,
    )
    ensemble_logits, ensemble_targets = ensemble_predict_logits(
        models=ensemble_models,
        dataloader=shifted_test_loader,
        device=device,
    )
    ensemble_results = evaluate_from_logits(
        ensemble_logits,
        ensemble_targets,
        n_bins=10,
    )

    with open(ensemble_metrics_path, "w", encoding="utf-8") as f:
        json.dump(ensemble_results["metrics"], f, indent=2)

    plot_reliability_diagram(
        ensemble_results["reliability_stats"],
        save_path=ensemble_reliability_path,
        title="Shifted MNIST Ensemble Reliability Diagram",
    )

    ensemble_confidences = top_class_confidence_from_logits(ensemble_logits)
    plot_confidence_histogram(
        ensemble_confidences,
        save_path=ensemble_conf_hist_path,
        title="Shifted MNIST Ensemble Confidence Histogram",
    )

    print("\nShifted baseline metrics:")
    print(json.dumps(baseline_results["metrics"], indent=2))

    print("\nShifted ensemble metrics:")
    print(json.dumps(ensemble_results["metrics"], indent=2))

    print(f"\nSaved shifted baseline metrics to: {baseline_metrics_path}")
    print(f"Saved shifted ensemble metrics to: {ensemble_metrics_path}")
    print(f"Saved shifted baseline reliability diagram to: {baseline_reliability_path}")
    print(f"Saved shifted ensemble reliability diagram to: {ensemble_reliability_path}")
    print(f"Saved shifted baseline confidence histogram to: {baseline_conf_hist_path}")
    print(f"Saved shifted ensemble confidence histogram to: {ensemble_conf_hist_path}")


if __name__ == "__main__":
    main()
