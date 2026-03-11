import torch
from src.model import MNISTClassifier, MNISTDropoutClassifier


def test_mnist_classifier_output_shape():
    model = MNISTClassifier()
    x = torch.randn(16, 1, 28, 28)
    logits = model(x)

    assert logits.shape == (16, 10)


def test_mnist_classifier_output_is_finite():
    model = MNISTClassifier()
    x = torch.randn(16, 1, 28, 28)
    logits = model(x)

    assert torch.isfinite(logits).all()


def test_mnist_dropout_classifier_output_shape():
    model = MNISTDropoutClassifier(dropout_p=0.2)
    x = torch.randn(16, 1, 28, 28)
    logits = model(x)

    assert logits.shape == (16, 10)


def test_mnist_dropout_classifier_output_is_finite():
    model = MNISTDropoutClassifier(dropout_p=0.2)
    x = torch.randn(16, 1, 28, 28)
    logits = model(x)

    assert torch.isfinite(logits).all()
