from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def enable_dropout_during_inference(model: nn.Module) -> None:
    """
    Put dropout layers in training mode while leaving the rest of the model unchanged.

    This is the standard trick used for MC Dropout inference.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


@torch.no_grad()
def mc_dropout_predict_proba(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_passes: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform MC Dropout inference by averaging probabilities over multiple
    stochastic forward passes.

    Parameters
    ----------
    model : nn.Module
        Trained dropout-enabled model.
    dataloader : DataLoader
        Dataloader yielding (x, y) batches.
    device : torch.device
        Device for inference.
    n_passes : int
        Number of stochastic forward passes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        mean_probs: shape (N, C)
        targets: shape (N,)
    """
    if n_passes <= 0:
        raise ValueError("n_passes must be positive")

    model.eval()
    enable_dropout_during_inference(model)

    all_mean_probs = []
    all_targets = []

    for x, y in dataloader:
        x = x.to(device)

        pass_probs = []
        for _ in range(n_passes):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pass_probs.append(probs)

        stacked_probs = torch.stack(pass_probs, dim=0)  # (T, B, C)
        mean_probs = stacked_probs.mean(dim=0)  # (B, C)

        all_mean_probs.append(mean_probs.cpu())
        all_targets.append(y.cpu())

    return torch.cat(all_mean_probs, dim=0), torch.cat(all_targets, dim=0)


def probs_to_logits(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Convert probabilities to log-probabilities for metric compatibility.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must have shape (N, C), got {tuple(probs.shape)}")

    probs = probs.clamp_min(eps)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return torch.log(probs)


@torch.no_grad()
def mc_dropout_predict_logits(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_passes: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform MC Dropout inference and return metric-compatible logits.
    """
    mean_probs, targets = mc_dropout_predict_proba(
        model=model,
        dataloader=dataloader,
        device=device,
        n_passes=n_passes,
    )
    logits = probs_to_logits(mean_probs)
    return logits, targets
