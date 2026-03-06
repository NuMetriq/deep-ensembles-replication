from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy from raw logits.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (N, C), raw model outputs.
    targets : torch.Tensor
        Shape (N,), integer class labels.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (N, C), got {tuple(logits.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets must have shape (N,), got {tuple(targets.shape)}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have the same batch size")

    preds = logits.argmax(dim=1)
    acc = (preds == targets).float().mean()
    return float(acc.item())


def nll_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute mean negative log-likelihood from raw logits.

    For multiclass classification, this is just cross-entropy loss.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (N, C), raw model outputs.
    targets : torch.Tensor
        Shape (N,), integer class labels.

    Returns
    -------
    float
        Mean NLL over the batch.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (N, C), got {tuple(logits.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets must have shape (N,), got {tuple(targets.shape)}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have the same batch size")

    loss = F.cross_entropy(logits, targets, reduction="mean")
    return float(loss.item())


def brier_score_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute the multiclass Brier score from raw logits.

    Brier score is the mean squared error between predicted probabilities
    and one-hot encoded targets, averaged over samples.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (N, C), raw model outputs.
    targets : torch.Tensor
        Shape (N,), integer class labels.

    Returns
    -------
    float
        Mean multiclass Brier score over the batch.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (N, C), got {tuple(logits.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets must have shape (N,), got {tuple(targets.shape)}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have the same batch size")

    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=num_classes).float()

    brier = torch.sum((probs - one_hot) ** 2, dim=1).mean()
    return float(brier.item())


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute the main classification metrics from raw logits.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (N, C), raw model outputs.
    targets : torch.Tensor
        Shape (N,), integer class labels.

    Returns
    -------
    dict[str, float]
        Dictionary with accuracy, nll, and brier score.
    """
    return {
        "accuracy": accuracy_from_logits(logits, targets),
        "nll": nll_from_logits(logits, targets),
        "brier": brier_score_from_logits(logits, targets),
    }


if __name__ == "__main__":
    torch.manual_seed(0)

    logits = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))

    metrics = compute_metrics(logits, targets)
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
