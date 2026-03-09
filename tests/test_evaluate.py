import torch
from src.evaluate import evaluate_from_logits


def test_evaluate_from_logits_returns_expected_keys():
    torch.manual_seed(0)
    logits = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))

    results = evaluate_from_logits(logits, targets, n_bins=10)

    assert set(results.keys()) == {
        "metrics",
        "logits",
        "targets",
        "reliability_stats",
        "calibration_gap_stats",
    }
    assert set(results["metrics"].keys()) == {"accuracy", "nll", "brier", "ece"}

    stats = results["reliability_stats"]
    assert set(stats.keys()) == {
        "bin_edges",
        "bin_centers",
        "bin_counts",
        "avg_confidence",
        "avg_accuracy",
    }

    gap_stats = results["calibration_gap_stats"]
    assert set(gap_stats.keys()) == {"gap", "abs_gap", "max_gap", "mean_abs_gap"}
