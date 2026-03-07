from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


def get_ensemble_checkpoint_path(
    checkpoint_dir: str | Path,
    member_index: int,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    return checkpoint_dir / f"member_{member_index}.pt"


def save_ensemble_member_checkpoint(
    model: nn.Module,
    checkpoint_dir: str | Path,
    member_index: int,
) -> Path:
    save_path = get_ensemble_checkpoint_path(checkpoint_dir, member_index)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return save_path


def load_ensemble_member_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def find_ensemble_checkpoint_paths(
    checkpoint_dir: str | Path,
) -> list[Path]:
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    paths = sorted(checkpoint_dir.glob("member_*.pt"), key=_member_index_from_path)

    if not paths:
        raise FileNotFoundError(
            f"No ensemble checkpoints found in directory: {checkpoint_dir}"
        )

    return paths


def load_ensemble_models(
    model_factory: Callable[[], nn.Module],
    checkpoint_dir: str | Path,
    device: torch.device,
) -> list[nn.Module]:
    checkpoint_paths = find_ensemble_checkpoint_paths(checkpoint_dir)

    models: list[nn.Module] = []
    for path in checkpoint_paths:
        model = model_factory()
        model = load_ensemble_member_checkpoint(model, path, device)
        models.append(model)

    return models


@torch.no_grad()
def ensemble_predict_proba(
    models: list[nn.Module],
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ensemble probabilities by averaging member probabilities.

    Parameters
    ----------
    models : list[nn.Module]
        Loaded ensemble member models.
    dataloader : DataLoader
        Dataloader providing (x, y) batches.
    device : torch.device
        Device for inference.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        mean_probs: shape (N, C)
        targets: shape (N,)
    """
    if not models:
        raise ValueError("models must be a non-empty list")

    for model in models:
        model.eval()

    all_mean_probs = []
    all_targets = []

    for x, y in dataloader:
        x = x.to(device)

        member_probs = []
        for model in models:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            member_probs.append(probs)

        stacked_probs = torch.stack(member_probs, dim=0)  # (M, B, C)
        mean_probs = stacked_probs.mean(dim=0)  # (B, C)

        all_mean_probs.append(mean_probs.cpu())
        all_targets.append(y.cpu())

    return torch.cat(all_mean_probs, dim=0), torch.cat(all_targets, dim=0)


def probs_to_logits(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Convert probabilities to log-probabilities for metric compatibility.

    This is useful because your existing metric functions expect logits.
    Log-probabilities are valid logits for softmax-based metrics.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must have shape (N, C), got {tuple(probs.shape)}")

    probs = probs.clamp_min(eps)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return torch.log(probs)


@torch.no_grad()
def ensemble_predict_logits(
    models: list[nn.Module],
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ensemble predictions and return log-probabilities as logits.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ensemble_logits: shape (N, C)
        targets: shape (N,)
    """
    mean_probs, targets = ensemble_predict_proba(models, dataloader, device)
    ensemble_logits = probs_to_logits(mean_probs)
    return ensemble_logits, targets


def _member_index_from_path(path: Path) -> int:
    stem = path.stem
    prefix = "member_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected checkpoint filename: {path.name}")
    return int(stem[len(prefix) :])
