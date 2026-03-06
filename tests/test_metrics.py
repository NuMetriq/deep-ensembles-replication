import math

import torch
import torch.nn.functional as F
from src.metrics import (
    accuracy_from_logits,
    brier_score_from_logits,
    compute_metrics,
    nll_from_logits,
)


def test_accuracy_from_logits_perfect():
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
    )
    targets = torch.tensor([0, 1, 2])

    acc = accuracy_from_logits(logits, targets)
    assert acc == 1.0


def test_nll_matches_torch_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(128, 10)
    targets = torch.randint(0, 10, (128,))

    a = nll_from_logits(logits, targets)
    b = float(F.cross_entropy(logits, targets).item())

    assert abs(a - b) < 1e-12


def test_brier_score_nonnegative():
    torch.manual_seed(0)
    logits = torch.randn(64, 10)
    targets = torch.randint(0, 10, (64,))

    score = brier_score_from_logits(logits, targets)
    assert score >= 0.0
    assert math.isfinite(score)


def test_compute_metrics_returns_expected_keys():
    torch.manual_seed(0)
    logits = torch.randn(16, 10)
    targets = torch.randint(0, 10, (16,))

    metrics = compute_metrics(logits, targets)

    assert set(metrics.keys()) == {"accuracy", "nll", "brier"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["nll"] >= 0.0
    assert metrics["brier"] >= 0.0
