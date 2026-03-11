from __future__ import annotations

import torch
from torch import nn


class MNISTClassifier(nn.Module):
    """
    Simple CNN baseline for MNIST classification.

    Input:  (N, 1, 28, 28)
    Output: (N, 10) logits
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 28x28 -> 14x14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class MNISTDropoutClassifier(nn.Module):
    """
    CNN classifier for MNIST with dropout layers for MC Dropout.

    Input:  (N, 1, 28, 28)
    Output: (N, 10) logits
    """

    def __init__(self, dropout_p: float = 0.2) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 28x28 -> 14x14
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 14x14 -> 7x7
            nn.Dropout2d(p=dropout_p),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    baseline_model = MNISTClassifier()
    dropout_model = MNISTDropoutClassifier(dropout_p=0.2)

    x = torch.randn(8, 1, 28, 28)

    baseline_logits = baseline_model(x)
    dropout_logits = dropout_model(x)

    print("Baseline model:")
    print(baseline_model)
    print("Baseline output shape:", baseline_logits.shape)

    print("\nDropout model:")
    print(dropout_model)
    print("Dropout output shape:", dropout_logits.shape)
