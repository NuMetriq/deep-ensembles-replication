from pathlib import Path

import torch
from src.mc_dropout import (
    enable_dropout_during_inference,
    mc_dropout_predict_logits,
    mc_dropout_predict_proba,
    probs_to_logits,
)
from src.model import MNISTDropoutClassifier
from torch.utils.data import DataLoader, TensorDataset


class ConstantLogitDropoutModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.register_buffer("fixed_logits", logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        logits = self.fixed_logits.unsqueeze(0).repeat(batch_size, 1)
        return self.dropout(logits)


def test_enable_dropout_during_inference_sets_dropout_to_train_mode():
    model = MNISTDropoutClassifier(dropout_p=0.2)
    model.eval()

    enable_dropout_during_inference(model)

    dropout_modules = [
        module
        for module in model.modules()
        if isinstance(
            module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)
        )
    ]
    assert len(dropout_modules) > 0
    assert all(module.training for module in dropout_modules)


def test_probs_to_logits_round_trip_softmax():
    probs = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
        ],
        dtype=torch.float32,
    )

    logits = probs_to_logits(probs)
    recovered = torch.softmax(logits, dim=1)

    assert torch.allclose(recovered, probs, atol=1e-6)


def test_mc_dropout_predict_proba_returns_expected_shapes():
    model = MNISTDropoutClassifier(dropout_p=0.2)

    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    dataloader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    probs, targets = mc_dropout_predict_proba(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
        n_passes=5,
    )

    assert probs.shape == (8, 10)
    assert targets.shape == (8,)
    assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)


def test_mc_dropout_predict_logits_returns_metric_compatible_logits():
    model = MNISTDropoutClassifier(dropout_p=0.2)

    x = torch.randn(6, 1, 28, 28)
    y = torch.randint(0, 10, (6,))
    dataloader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)

    logits, targets = mc_dropout_predict_logits(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
        n_passes=5,
    )

    probs = torch.softmax(logits, dim=1)

    assert logits.shape == (6, 10)
    assert targets.shape == (6,)
    assert torch.allclose(probs.sum(dim=1), torch.ones(6), atol=1e-5)
