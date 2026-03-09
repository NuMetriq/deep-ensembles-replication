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


def reliability_diagram_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Compute bin-wise statistics for a reliability diagram.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (N, C), raw model outputs.
    targets : torch.Tensor
        Shape (N,), integer class labels.
    n_bins : int
        Number of confidence bins.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing:
        - bin_edges: shape (n_bins + 1,)
        - bin_centers: shape (n_bins,)
        - bin_counts: shape (n_bins,)
        - avg_confidence: shape (n_bins,)
        - avg_accuracy: shape (n_bins,)
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (N, C), got {tuple(logits.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets must have shape (N,), got {tuple(targets.shape)}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have the same batch size")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    probs = torch.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    correctness = (predictions == targets).float()

    bin_edges = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=logits.device)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_counts = torch.zeros(n_bins, dtype=torch.long, device=logits.device)
    avg_confidence = torch.zeros(n_bins, dtype=torch.float32, device=logits.device)
    avg_accuracy = torch.zeros(n_bins, dtype=torch.float32, device=logits.device)

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == 0:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences > left) & (confidences <= right)

        count = in_bin.sum()
        bin_counts[i] = count

        if count > 0:
            avg_confidence[i] = confidences[in_bin].mean()
            avg_accuracy[i] = correctness[in_bin].mean()

    return {
        "bin_edges": bin_edges.detach().cpu(),
        "bin_centers": bin_centers.detach().cpu(),
        "bin_counts": bin_counts.detach().cpu(),
        "avg_confidence": avg_confidence.detach().cpu(),
        "avg_accuracy": avg_accuracy.detach().cpu(),
    }


def expected_calibration_error_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) from raw logits.

    ECE is the weighted average of the absolute difference between
    average confidence and average accuracy across confidence bins.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (N, C), raw model outputs.
    targets : torch.Tensor
        Shape (N,), integer class labels.
    n_bins : int
        Number of confidence bins.

    Returns
    -------
    float
        Expected calibration error in [0, 1].
    """
    stats = reliability_diagram_stats(logits, targets, n_bins=n_bins)

    bin_counts = stats["bin_counts"].float()
    avg_confidence = stats["avg_confidence"]
    avg_accuracy = stats["avg_accuracy"]

    total = bin_counts.sum()
    if total == 0:
        return 0.0

    gaps = torch.abs(avg_confidence - avg_accuracy)
    ece = torch.sum((bin_counts / total) * gaps)

    return float(ece.item())


def calibration_gap_stats_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
) -> Dict[str, object]:
    """
    Compute calibration gap summaries from reliability bins.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (N, C), raw model outputs.
    targets : torch.Tensor
        Shape (N,), integer class labels.
    n_bins : int
        Number of confidence bins.

    Returns
    -------
    dict[str, object]
        Dictionary containing:
        - gap: signed per-bin gap = avg_confidence - avg_accuracy
        - abs_gap: absolute per-bin gap
        - max_gap: maximum absolute calibration gap (MCE-style)
        - mean_abs_gap: mean absolute bin gap across all bins
    """
    stats = reliability_diagram_stats(logits, targets, n_bins=n_bins)

    gap = stats["avg_confidence"] - stats["avg_accuracy"]
    abs_gap = torch.abs(gap)

    return {
        "gap": gap,
        "abs_gap": abs_gap,
        "max_gap": float(abs_gap.max().item()),
        "mean_abs_gap": float(abs_gap.mean().item()),
    }


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute the main classification and calibration metrics from raw logits.
    """
    return {
        "accuracy": accuracy_from_logits(logits, targets),
        "nll": nll_from_logits(logits, targets),
        "brier": brier_score_from_logits(logits, targets),
        "ece": expected_calibration_error_from_logits(logits, targets, n_bins=n_bins),
    }


if __name__ == "__main__":
    torch.manual_seed(0)

    logits = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))

    metrics = compute_metrics(logits, targets)
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
