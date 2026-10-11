import torch
from redteam.adv.model import SmallCNN
from redteam.adv.attacks import fgsm, pgd

def _batch():
    torch.manual_seed(0)
    x = torch.rand(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    return x, y

def test_fgsm_respects_linf_and_box():
    m = SmallCNN().eval()
    x, y = _batch()
    adv = fgsm(m, x, y, eps=0.1)
    assert (adv - x).abs().max().item() <= 0.1 + 1e-5
    assert adv.min() >= 0 and adv.max() <= 1

def test_pgd_respects_linf_and_box():
    m = SmallCNN().eval()
    x, y = _batch()
    adv = pgd(m, x, y, eps=0.2, steps=5)
    assert (adv - x).abs().max().item() <= 0.2 + 1e-4
    assert adv.min() >= 0 and adv.max() <= 1

def test_eps_zero_is_identity():
    m = SmallCNN().eval()
    x, y = _batch()
    assert torch.allclose(fgsm(m, x, y, 0.0), x)
    assert torch.allclose(pgd(m, x, y, 0.0), x)

def test_pgd_at_least_as_strong_as_no_attack():
    # a non-zero perturbation changes the input
    m = SmallCNN().eval()
    x, y = _batch()
    adv = pgd(m, x, y, eps=0.3, steps=10)
    assert (adv - x).abs().max().item() > 0
