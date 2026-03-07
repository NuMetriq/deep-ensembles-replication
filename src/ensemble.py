from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import nn


def get_ensemble_checkpoint_path(
    checkpoint_dir: str | Path,
    member_index: int,
) -> Path:
    """
    Return the checkpoint path for a specific ensemble member.

    Example
    -------
    checkpoints/ensemble/member_0.pt
    """
    checkpoint_dir = Path(checkpoint_dir)
    return checkpoint_dir / f"member_{member_index}.pt"


def save_ensemble_member_checkpoint(
    model: nn.Module,
    checkpoint_dir: str | Path,
    member_index: int,
) -> Path:
    """
    Save a single ensemble member checkpoint and return the saved path.
    """
    save_path = get_ensemble_checkpoint_path(checkpoint_dir, member_index)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return save_path


def load_ensemble_member_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    """
    Load a checkpoint into an existing model instance.
    """
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
    """
    Find ensemble checkpoints in a directory, sorted by member index.

    Expected filenames:
        member_0.pt
        member_1.pt
        ...
    """
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
    """
    Create and load all ensemble member models from a checkpoint directory.

    Parameters
    ----------
    model_factory : Callable[[], nn.Module]
        Zero-argument function that returns a fresh model instance.
    checkpoint_dir : str | Path
        Directory containing member_*.pt checkpoints.
    device : torch.device
        Device to load models onto.

    Returns
    -------
    list[nn.Module]
        Loaded models in sorted member order.
    """
    checkpoint_paths = find_ensemble_checkpoint_paths(checkpoint_dir)

    models: list[nn.Module] = []
    for path in checkpoint_paths:
        model = model_factory()
        model = load_ensemble_member_checkpoint(model, path, device)
        models.append(model)

    return models


def _member_index_from_path(path: Path) -> int:
    """
    Extract the member index from a path like 'member_3.pt'.
    """
    stem = path.stem  # member_3
    prefix = "member_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected checkpoint filename: {path.name}")
    return int(stem[len(prefix) :])
