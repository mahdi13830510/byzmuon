import torch
from conftest import tiny_model
from torch import nn

from byzmuon.optim.local_factory import make_local_optimizer
from byzmuon.optim.muon_local import MuonLocal
from byzmuon.optim.orthogonalize import ExactPolar


def test_groups_split_matrix_and_other_params():
    m = tiny_model()
    opt = MuonLocal(m, orthogonalizer=ExactPolar())
    muon_group, other_group = opt.param_groups
    assert muon_group["use_muon"] and not other_group["use_muon"]
    assert len(muon_group["params"]) == 1  # 0.weight only; 2.weight is the head
    assert len(other_group["params"]) == 3  # 0.bias, 2.weight, 2.bias


def test_muon_step_is_orthogonal_direction():
    m = tiny_model()
    opt = MuonLocal(m, lr=0.1, momentum=0.0, nesterov=False, orthogonalizer=ExactPolar())
    before = m[0].weight.detach().clone()
    x = torch.randn(16, 8)
    nn.functional.cross_entropy(m(x), torch.randint(0, 3, (16,))).backward()
    opt.step()
    step = (m[0].weight.detach() - before) / (-0.1 * max(1.0, 16 / 8) ** 0.5)
    sv = torch.linalg.svdvals(step)
    assert torch.allclose(sv, torch.ones_like(sv), atol=1e-4)


def test_muon_local_decreases_loss():
    torch.manual_seed(0)
    m = tiny_model()
    opt = MuonLocal(m, lr=0.05)
    x = torch.randn(256, 8)
    y = (x[:, 0] > 0).long() + (x[:, 1] > 0).long()
    first = None
    for _ in range(50):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss.detach())
    assert float(loss.detach()) < 0.7 * first


def test_factory_builds_all_three():
    for spec in [{"name": "sgd", "lr": 0.1}, {"name": "adamw", "lr": 1e-3}, {"name": "muon", "lr": 0.02}]:
        opt = make_local_optimizer(spec)(tiny_model())
        assert isinstance(opt, torch.optim.Optimizer)
