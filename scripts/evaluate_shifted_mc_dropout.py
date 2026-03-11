from __future__ import annotations

import json
from pathlib import Path

import torch
from src.data import get_mnist_shifted_test_loader
from src.evaluate import evaluate_from_logits
from src.mc_dropout import mc_dropout_predict_logits
from src.model import MNISTDropoutClassifier
from src.plotting import (
    plot_confidence_histogram,
    plot_reliability_diagram,
    top_class_confidence_from_logits,
)


def main() -> None:
    # Config
    data_dir = "data"
    batch_size = 128
    num_workers = 0
    pin_memory = False

    dropout_p = 0.2
    n_passes = 20
    noise_std = 0.2

    checkpoint_path = Path("checkpoints/mc_dropout.pt")
    metrics_path = Path("results/tables/generated/shifted_mc_dropout_metrics.json")
    reliability_path = Path(
        "results/figures/generated/shifted_mc_dropout_reliability_diagram.png"
    )
    confidence_hist_path = Path(
        "results/figures/generated/shifted_mc_dropout_confidence_histogram.png"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        f"Evaluating shifted MC Dropout with dropout_p={dropout_p}, "
        f"n_passes={n_passes}, noise_std={noise_std}"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Run `python -m scripts.train_mc_dropout` first."
        )

    shifted_test_loader = get_mnist_shifted_test_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        noise_std=noise_std,
    )

    model = MNISTDropoutClassifier(dropout_p=dropout_p).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    mc_logits, targets = mc_dropout_predict_logits(
        model=model,
        dataloader=shifted_test_loader,
        device=device,
        n_passes=n_passes,
    )

    results = evaluate_from_logits(mc_logits, targets, n_bins=10)
    metrics = results["metrics"]
    reliability_stats = results["reliability_stats"]

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_reliability_diagram(
        reliability_stats,
        save_path=reliability_path,
        title="Shifted MC Dropout Reliability Diagram",
    )

    confidences = top_class_confidence_from_logits(mc_logits)
    plot_confidence_histogram(
        confidences,
        save_path=confidence_hist_path,
        title="Shifted MC Dropout Confidence Histogram",
    )

    print("Shifted MC Dropout evaluation complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved reliability diagram to: {reliability_path}")
    print(f"Saved confidence histogram to: {confidence_hist_path}")


if __name__ == "__main__":
    main()
