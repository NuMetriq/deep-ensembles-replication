import torch
from src.model import MNISTClassifier
from src.train import set_seed, train_one_epoch


def test_set_seed_reproducible_torch_randomness():
    set_seed(123)
    a = torch.randn(4)

    set_seed(123)
    b = torch.randn(4)

    assert torch.allclose(a, b)


def test_train_one_epoch_returns_metrics():
    model = MNISTClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    dataloader = [(x, y)]

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert set(metrics.keys()) == {"loss", "accuracy"}
    assert metrics["loss"] >= 0.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
