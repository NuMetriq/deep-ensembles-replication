import torch
from src.evaluate import evaluate_from_logits


def test_evaluate_from_logits_handles_mc_dropout_style_logits():
    torch.manual_seed(0)
    probs = torch.rand(32, 10)
    probs = probs / probs.sum(dim=1, keepdim=True)
    logits = torch.log(probs)
    targets = torch.randint(0, 10, (32,))

    results = evaluate_from_logits(logits, targets, n_bins=10)

    assert set(results["metrics"].keys()) == {"accuracy", "nll", "brier", "ece"}
