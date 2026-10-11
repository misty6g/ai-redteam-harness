"""MNIST loading with fixed subsets for reproducible, CPU-fast runs.

Pixels are kept in [0,1] (ToTensor) so FGSM/PGD epsilons are in pixel units.
"""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_TF = transforms.ToTensor()  # -> [0,1], shape (1,28,28)


def _subset(ds, n, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return Subset(ds, idx)


def get_loaders(root: str = "data", n_train: int = 12000, n_test: int = 2000,
                batch_size: int = 128, seed: int = 1337):
    train = datasets.MNIST(root, train=True, download=True, transform=_TF)
    test = datasets.MNIST(root, train=False, download=True, transform=_TF)
    train = _subset(train, n_train, seed)
    test = _subset(test, n_test, seed + 1)
    tl = DataLoader(train, batch_size=batch_size, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    vl = DataLoader(test, batch_size=256, shuffle=False)
    return tl, vl
