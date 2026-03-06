from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from src.data import get_mnist_dataloaders
from src.evaluate import evaluate_model
from src.model import MNISTClassifier
from src.plotting import plot_reliability_diagram


def main() -> None:
    # Config
    data_dir = "data"
    batch_size = 128
    num_workers = 0
    pin_memory = False
    checkpoint_path = Path("checkpoints/mnist_baseline.pt")
    metrics_path = Path("results/tables/generated/baseline_metrics.json")
    figure_path = Path("results/figures/generated/baseline_reliability_diagram.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Run `python scripts/train_single.py` first."
        )

    _, test_loader = get_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = MNISTClassifier().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    results = evaluate_model(model, test_loader, device, n_bins=10)
    metrics = results["metrics"]
    reliability_stats = results["reliability_stats"]

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_reliability_diagram(
        reliability_stats,
        save_path=figure_path,
        title="MNIST Baseline Reliability Diagram",
    )

    print("Evaluation complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved reliability diagram to: {figure_path}")


if __name__ == "__main__":
    main()
