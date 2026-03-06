from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train the model for one epoch and return average training loss/accuracy.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == y).sum().item()
        total += batch_size

    return {
        "loss": running_loss / total,
        "accuracy": running_correct / total,
    }


def save_checkpoint(
    model: nn.Module,
    save_path: str | Path,
) -> None:
    """
    Save model state dict to disk.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
