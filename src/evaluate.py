from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import compute_metrics, reliability_diagram_stats


@torch.no_grad()
def collect_logits_and_targets(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run model on a dataloader and collect all logits and targets.
    """
    model.eval()

    all_logits = []
    all_targets = []

    for x, y in dataloader:
        x = x.to(device)
        logits = model(x)

        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_bins: int = 10,
) -> Dict[str, object]:
    """
    Evaluate model and return scalar metrics plus reliability stats.
    """
    logits, targets = collect_logits_and_targets(model, dataloader, device)

    metrics = compute_metrics(logits, targets)
    reliability_stats = reliability_diagram_stats(logits, targets, n_bins=n_bins)

    return {
        "metrics": metrics,
        "logits": logits,
        "targets": targets,
        "reliability_stats": reliability_stats,
    }
