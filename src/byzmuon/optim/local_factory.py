from collections.abc import Callable

import torch
from torch import nn

from byzmuon.optim.muon_local import MuonLocal
from byzmuon.optim.server_opt import build_orthogonalizer


def make_local_optimizer(spec: dict) -> Callable[[nn.Module], torch.optim.Optimizer]:
    """spec: {name: sgd|adamw|muon, lr: ..., ...}. Returns a factory the Client calls each round."""
    spec = dict(spec)
    name = spec.pop("name")
    if name == "sgd":
        return lambda m: torch.optim.SGD(m.parameters(), **spec)
    if name == "adamw":
        return lambda m: torch.optim.AdamW(m.parameters(), **spec)
    if name == "muon":
        orth = build_orthogonalizer(spec.pop("orthogonalizer", None))
        return lambda m: MuonLocal(m, orthogonalizer=orth, **spec)
    raise KeyError(f"unknown client optimizer '{name}' (sgd, adamw, muon)")
