from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from src.data import get_mnist_dataloaders
from src.evaluate import evaluate_model
from src.model import MNISTClassifier
from src.train import save_checkpoint, train_one_epoch


def main() -> None:
    # Config
    data_dir = "data"
    batch_size = 128
    num_workers = 0
    pin_memory = False
    epochs = 5
    learning_rate = 1e-3
    checkpoint_path = Path("checkpoints/mnist_baseline.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = MNISTClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        eval_results = evaluate_model(model, test_loader, device)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"test_acc={eval_results['metrics']['accuracy']:.4f} | "
            f"test_nll={eval_results['metrics']['nll']:.4f} | "
            f"test_brier={eval_results['metrics']['brier']:.4f}"
        )

    save_checkpoint(model, checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
