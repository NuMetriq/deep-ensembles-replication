from __future__ import annotations

from pathlib import Path

import torch
from src.data import get_mnist_dataloaders
from src.ensemble import save_ensemble_member_checkpoint
from src.evaluate import evaluate_model
from src.model import MNISTClassifier
from src.train import set_seed, train_one_epoch


def main() -> None:
    # Config
    data_dir = "data"
    batch_size = 128
    num_workers = 0
    pin_memory = False

    ensemble_size = 5
    base_seed = 42

    epochs = 5
    learning_rate = 1e-3
    checkpoint_dir = Path("checkpoints/ensemble")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training ensemble with {ensemble_size} members")

    train_loader, test_loader = get_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    for member_index in range(ensemble_size):
        seed = base_seed + member_index
        set_seed(seed)

        print("\n" + "=" * 60)
        print(
            f"Training ensemble member {member_index + 1}/{ensemble_size} (seed={seed})"
        )
        print("=" * 60)

        model = MNISTClassifier().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(1, epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, device)
            eval_results = evaluate_model(model, test_loader, device)

            print(
                f"Member {member_index} | "
                f"Epoch {epoch:02d}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"test_acc={eval_results['metrics']['accuracy']:.4f} | "
                f"test_nll={eval_results['metrics']['nll']:.4f} | "
                f"test_brier={eval_results['metrics']['brier']:.4f}"
            )

        save_path = save_ensemble_member_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            member_index=member_index,
        )
        print(f"Saved ensemble member checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
