"""Evaluate clean and robust accuracy under FGSM/PGD, and run adv training."""
from __future__ import annotations
import json, time
from pathlib import Path
import torch
from .data import get_loaders
from .train import train_standard, train_adversarial
from .attacks import fgsm, pgd


@torch.no_grad()
def _clean_acc(model, loader):
    correct = total = 0
    for x, y in loader:
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
    return correct / total


def _robust_acc(model, loader, attack, **kw):
    correct = total = 0
    for x, y in loader:
        x_adv = attack(model, x, y, **kw)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
    return correct / total


def run(out_path: str = "results/adv_results.json", seed: int = 1337,
        epsilons=(0.05, 0.1, 0.15, 0.2, 0.3), pgd_steps: int = 10,
        n_train: int = 12000, n_test: int = 2000, std_epochs: int = 2,
        adv_epochs: int = 3, adv_eps: float = 0.2) -> dict:
    torch.manual_seed(seed)
    tl, vl = get_loaders(n_train=n_train, n_test=n_test, seed=seed)
    t0 = time.time()

    std = train_standard(tl, epochs=std_epochs, seed=seed)
    std_clean = _clean_acc(std, vl)
    std_fgsm = {str(e): _robust_acc(std, vl, fgsm, eps=e) for e in epsilons}
    std_pgd = {str(e): _robust_acc(std, vl, pgd, eps=e, steps=pgd_steps) for e in epsilons}
    t_std = time.time() - t0

    t1 = time.time()
    adv = train_adversarial(tl, epochs=adv_epochs, eps=adv_eps, steps=7, seed=seed)
    adv_clean = _clean_acc(adv, vl)
    adv_fgsm = {str(e): _robust_acc(adv, vl, fgsm, eps=e) for e in epsilons}
    adv_pgd = {str(e): _robust_acc(adv, vl, pgd, eps=e, steps=pgd_steps) for e in epsilons}
    t_adv = time.time() - t1

    e = str(adv_eps)
    result = {
        "seed": seed, "n_train": n_train, "n_test": n_test,
        "pgd_steps_eval": pgd_steps, "epsilons": list(epsilons),
        "std_epochs": std_epochs, "adv_epochs": adv_epochs, "adv_train_eps": adv_eps,
        "standard_model": {
            "clean_acc": round(std_clean, 4),
            "fgsm_acc": {k: round(v, 4) for k, v in std_fgsm.items()},
            "pgd_acc": {k: round(v, 4) for k, v in std_pgd.items()},
            "train_sec": round(t_std, 1),
        },
        "adv_trained_model": {
            "clean_acc": round(adv_clean, 4),
            "fgsm_acc": {k: round(v, 4) for k, v in adv_fgsm.items()},
            "pgd_acc": {k: round(v, 4) for k, v in adv_pgd.items()},
            "train_sec": round(t_adv, 1),
        },
        "headline": {
            "clean_cost": round(std_clean - adv_clean, 4),
            "pgd_robust_gain_at_eps": {e: round(adv_pgd[e] - std_pgd[e], 4)},
            "std_pgd_at_eps": {e: round(std_pgd[e], 4)},
            "adv_pgd_at_eps": {e: round(adv_pgd[e], 4)},
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"[adv] wrote {out_path}")
    return result
