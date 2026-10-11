"""FGSM and PGD L-infinity attacks on [0,1] images."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def fgsm(model, x, y, eps: float):
    if eps == 0:
        return x.clone()
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = x + eps * grad.sign()
    return x_adv.clamp(0, 1).detach()


def pgd(model, x, y, eps: float, alpha: float | None = None, steps: int = 10,
        rand_init: bool = True):
    if eps == 0:
        return x.clone()
    if alpha is None:
        alpha = max(eps / 4.0, 0.01)
    x0 = x.clone().detach()
    if rand_init:
        x_adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1).detach()
    else:
        x_adv = x0.clone()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps).clamp(0, 1).detach()
    return x_adv
