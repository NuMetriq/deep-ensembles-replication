from pathlib import Path

import torch
from src.ensemble import (
    find_ensemble_checkpoint_paths,
    get_ensemble_checkpoint_path,
    load_ensemble_models,
    save_ensemble_member_checkpoint,
)
from src.model import MNISTClassifier


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
