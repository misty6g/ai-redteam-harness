"""Training: standard and PGD adversarial training."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import SmallCNN
from .attacks import pgd


def _set_seed(seed: int):
    torch.manual_seed(seed)


def train_standard(train_loader, epochs: int = 2, lr: float = 1e-3, seed: int = 1337):
    _set_seed(seed)
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        for x, y in train_loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward(); opt.step()
    model.eval()
    return model


def train_adversarial(train_loader, epochs: int = 3, lr: float = 1e-3,
                      eps: float = 0.2, steps: int = 7, seed: int = 1337):
    _set_seed(seed)
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x_adv = pgd(model, x, y, eps=eps, steps=steps)
            opt.zero_grad()
            loss = F.cross_entropy(model(x_adv), y)
            loss.backward(); opt.step()
    model.eval()
    return model
