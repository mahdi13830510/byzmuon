import pytest
import torch
from torch import nn

from byzmuon.core.update import Update


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def low_rank_update(rows=64, cols=48, rank=4, noise=1e-3, seed=0, name="w") -> Update:
    g = torch.Generator().manual_seed(seed)
    u = torch.linalg.qr(torch.randn(rows, rank, generator=g))[0]
    v = torch.linalg.qr(torch.randn(cols, rank, generator=g))[0]
    s = torch.linspace(1.0, 0.5, rank)
    m = (u * s) @ v.T + noise * torch.randn(rows, cols, generator=g)
    return Update(tensors={name: m, "b": torch.zeros(rows)}, meta={"exclude_from_muon": set()})


def tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
