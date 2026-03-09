from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class AddGaussianNoise:
    """
    Add Gaussian noise to a tensor image and clamp back to a valid range.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.2) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noisy = tensor + torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(noisy, 0.0, 1.0)


def get_mnist_shifted_test_dataset(
    data_dir: str | Path = "data",
    noise_std: float = 0.2,
) -> datasets.MNIST:
    """
    Create a shifted MNIST test dataset using Gaussian noise.
    """
    data_dir = Path(data_dir)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            AddGaussianNoise(std=noise_std),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return test_dataset


def get_mnist_shifted_test_loader(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    noise_std: float = 0.2,
) -> DataLoader:
    """
    Create a shifted MNIST test dataloader using Gaussian noise.
    """
    test_dataset = get_mnist_shifted_test_dataset(
        data_dir=data_dir,
        noise_std=noise_std,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return test_loader


def get_mnist_transforms() -> transforms.Compose:
    """
    Return the basic transform pipeline for MNIST.

    For the first pass, keep preprocessing minimal:
    - convert PIL image to tensor
    - normalize using standard MNIST mean/std
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def get_mnist_datasets(
    data_dir: str | Path = "data",
) -> Tuple[datasets.MNIST, datasets.MNIST]:
    """
    Create the MNIST training and test datasets.

    Parameters
    ----------
    data_dir : str | Path
        Directory where MNIST should be stored/downloaded.

    Returns
    -------
    train_dataset, test_dataset : tuple[datasets.MNIST, datasets.MNIST]
    """
    data_dir = Path(data_dir)
    transform = get_mnist_transforms()

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset


def get_mnist_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for MNIST.

    Parameters
    ----------
    data_dir : str | Path
        Directory where MNIST is stored/downloaded.
    batch_size : int
        Batch size for both train and test loaders.
    num_workers : int
        Number of subprocesses used for data loading.
    pin_memory : bool
        Whether to pin memory in the DataLoader.

    Returns
    -------
    train_loader, test_loader : tuple[DataLoader, DataLoader]
    """
    train_dataset, test_dataset = get_mnist_datasets(data_dir=data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_mnist_dataloaders()

    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    print("Train batch shapes:", x_train.shape, y_train.shape)
    print("Test batch shapes:", x_test.shape, y_test.shape)
    print("Train dtype:", x_train.dtype)
    print("Pixel range:", float(x_train.min()), float(x_train.max()))
