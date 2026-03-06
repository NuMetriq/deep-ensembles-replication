from src.data import get_mnist_dataloaders


def test_mnist_dataloaders_return_expected_shapes():
    train_loader, test_loader = get_mnist_dataloaders(batch_size=32)

    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    assert x_train.shape == (32, 1, 28, 28)
    assert y_train.shape == (32,)
    assert x_test.shape == (32, 1, 28, 28)
    assert y_test.shape == (32,)
