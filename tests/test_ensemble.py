from pathlib import Path

import torch
from src.ensemble import (
    ensemble_predict_logits,
    ensemble_predict_proba,
    find_ensemble_checkpoint_paths,
    get_ensemble_checkpoint_path,
    load_ensemble_models,
    probs_to_logits,
    save_ensemble_member_checkpoint,
)
from src.model import MNISTClassifier
from torch.utils.data import DataLoader, TensorDataset


class ConstantLogitModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return self.fixed_logits.unsqueeze(0).repeat(batch_size, 1)


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


def test_ensemble_predict_proba_averages_member_probabilities():
    model_a = ConstantLogitModel(torch.tensor([2.0, 0.0]))
    model_b = ConstantLogitModel(torch.tensor([0.0, 2.0]))

    x = torch.randn(4, 1, 28, 28)
    y = torch.tensor([0, 1, 0, 1])
    dataloader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

    probs, targets = ensemble_predict_proba(
        models=[model_a, model_b],
        dataloader=dataloader,
        device=torch.device("cpu"),
    )

    probs_a = torch.softmax(torch.tensor([2.0, 0.0]), dim=0)
    probs_b = torch.softmax(torch.tensor([0.0, 2.0]), dim=0)
    expected = (probs_a + probs_b) / 2.0
    expected = expected.unsqueeze(0).repeat(4, 1)

    assert probs.shape == (4, 2)
    assert torch.allclose(probs, expected, atol=1e-6)
    assert torch.equal(targets, y)


def test_ensemble_predict_logits_returns_metric_compatible_logits():
    model_a = ConstantLogitModel(torch.tensor([3.0, 1.0, 0.0]))
    model_b = ConstantLogitModel(torch.tensor([1.0, 3.0, 0.0]))

    x = torch.randn(5, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 0, 1])
    dataloader = DataLoader(TensorDataset(x, y), batch_size=5, shuffle=False)

    logits, targets = ensemble_predict_logits(
        models=[model_a, model_b],
        dataloader=dataloader,
        device=torch.device("cpu"),
    )

    probs = torch.softmax(logits, dim=1)

    assert logits.shape == (5, 3)
    assert probs.shape == (5, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-6)
    assert torch.equal(targets, y)


def test_get_ensemble_checkpoint_path():
    path = get_ensemble_checkpoint_path("checkpoints/ensemble", member_index=3)
    assert path == Path("checkpoints/ensemble/member_3.pt")


def test_save_and_find_ensemble_checkpoints(tmp_path: Path):
    checkpoint_dir = tmp_path / "ensemble"

    for member_index in range(3):
        model = MNISTClassifier()
        save_path = save_ensemble_member_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            member_index=member_index,
        )
        assert save_path.exists()

    paths = find_ensemble_checkpoint_paths(checkpoint_dir)
    assert len(paths) == 3
    assert paths[0].name == "member_0.pt"
    assert paths[1].name == "member_1.pt"
    assert paths[2].name == "member_2.pt"


def test_load_ensemble_models(tmp_path: Path):
    checkpoint_dir = tmp_path / "ensemble"

    for member_index in range(2):
        model = MNISTClassifier()
        save_ensemble_member_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            member_index=member_index,
        )

    models = load_ensemble_models(
        model_factory=MNISTClassifier,
        checkpoint_dir=checkpoint_dir,
        device=torch.device("cpu"),
    )

    assert len(models) == 2
    assert all(isinstance(model, MNISTClassifier) for model in models)
