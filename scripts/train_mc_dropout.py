from __future__ import annotations

from pathlib import Path

import torch
from src.data import get_mnist_dataloaders
from src.evaluate import evaluate_model
from src.model import MNISTDropoutClassifier
from src.train import save_checkpoint, set_seed, train_one_epoch


def main() -> None:
    # Config
    data_dir = "data"
    batch_size = 128
    num_workers = 0
    pin_memory = False

    seed = 42
    epochs = 5
    learning_rate = 1e-3
    dropout_p = 0.2

    checkpoint_path = Path("checkpoints/mc_dropout.pt")

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training MC Dropout model with dropout_p={dropout_p}")

    train_loader, test_loader = get_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = MNISTDropoutClassifier(dropout_p=dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        eval_results = evaluate_model(model, test_loader, device, n_bins=10)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"test_acc={eval_results['metrics']['accuracy']:.4f} | "
            f"test_nll={eval_results['metrics']['nll']:.4f} | "
            f"test_brier={eval_results['metrics']['brier']:.4f} | "
            f"test_ece={eval_results['metrics']['ece']:.4f}"
        )

    save_checkpoint(model, checkpoint_path)
    print(f"Saved MC Dropout checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
