import torch
from redteam.adv.model import SmallCNN

def test_forward_shape():
    m = SmallCNN()
    out = m(torch.rand(4, 1, 28, 28))
    assert out.shape == (4, 10)

def test_deterministic_init_with_seed():
    torch.manual_seed(5); a = SmallCNN()
    torch.manual_seed(5); b = SmallCNN()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb)
