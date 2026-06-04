"""Dataset utilities for MNIST / CIFAR-10 at 32x32 resolution, normalised to [-1, 1]."""
from __future__ import annotations

import pathlib

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def _mnist_transform(img_size: int = 32):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [0,1] -> [-1,1]
    ])


def _cifar_transform(img_size: int = 32):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])


def get_mnist(root: str | pathlib.Path, img_size: int = 32, batch_size: int = 128, num_workers: int = 0):
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train = datasets.MNIST(root, train=True, download=True, transform=_mnist_transform(img_size))
    test = datasets.MNIST(root, train=False, download=True, transform=_mnist_transform(img_size))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def get_cifar10(root: str | pathlib.Path, img_size: int = 32, batch_size: int = 128, num_workers: int = 0):
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train = datasets.CIFAR10(root, train=True, download=True, transform=_cifar_transform(img_size))
    test = datasets.CIFAR10(root, train=False, download=True, transform=_cifar_transform(img_size))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def to_image(x: torch.Tensor) -> torch.Tensor:
    """Map a sample tensor from [-1, 1] back to [0, 1] for visualisation."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)
