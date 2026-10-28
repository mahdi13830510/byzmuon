"""Server-side optimizers. The aggregated Update is new − old, so the pseudo-gradient is g = −agg."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import Tensor

from byzmuon.core.update import Update
from byzmuon.optim.orthogonalize import NewtonSchulz, Orthogonalizer
from byzmuon.registry import REGISTRY, register


@dataclass
class ServerState:
    params: dict[str, Tensor]
    round: int = 0
    momentum: dict[str, Tensor] = field(default_factory=dict)
    adam_m: dict[str, Tensor] = field(default_factory=dict)
    adam_v: dict[str, Tensor] = field(default_factory=dict)
    exclude_from_muon: set[str] = field(default_factory=set)
    history: list[dict] = field(default_factory=list)  # per-round metrics, read-only for attacks

    def as_update(self) -> Update:
        return Update(
            tensors={n: p.clone() for n, p in self.params.items()},
            meta={"exclude_from_muon": set(self.exclude_from_muon)},
        )


class ServerOptimizer(ABC):
    def __init__(self, lr: float = 1.0, weight_decay: float = 0.0) -> None:
        self.lr = lr
        self.weight_decay = weight_decay

    @abstractmethod
    def step(self, state: ServerState, agg: Update) -> ServerState: ...

    def _decay(self, state: ServerState, name: str) -> None:
        if self.weight_decay:
            state.params[name].mul_(1 - self.lr * self.weight_decay)


def _adam_step(state: ServerState, name: str, g: Tensor, lr: float, betas: tuple[float, float], eps: float) -> None:
    b1, b2 = betas
    m = state.adam_m.get(name)
    v = state.adam_v.get(name)
    if m is None:
        m, v = torch.zeros_like(g), torch.zeros_like(g)
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    state.adam_m[name], state.adam_v[name] = m, v
    t = state.round + 1
    m_hat = m / (1 - b1**t)
    v_hat = v / (1 - b2**t)
    state.params[name].addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


@register("server_opt", "sgd")
class ServerSGD(ServerOptimizer):
    """params += lr * agg (+ optional heavy-ball momentum). lr=1, momentum=0 is FedAvg."""

    def __init__(self, lr: float = 1.0, momentum: float = 0.0, weight_decay: float = 0.0) -> None:
        super().__init__(lr, weight_decay)
        self.momentum = momentum

    def step(self, state: ServerState, agg: Update) -> ServerState:
        for n, delta in agg.tensors.items():
            g = -delta
            if self.momentum:
                buf = state.momentum.get(n)
                buf = g.clone() if buf is None else buf.mul_(self.momentum).add_(g)
                state.momentum[n] = buf
                g = buf
            self._decay(state, n)
            state.params[n].add_(g, alpha=-self.lr)
        state.round += 1
        return state


@register("server_opt", "adam")
class ServerAdam(ServerOptimizer):
    """FedAdam (Reddi et al. 2021): Adam on the pseudo-gradient."""

    def __init__(
        self, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.99), eps: float = 1e-3, weight_decay: float = 0.0
    ) -> None:
        super().__init__(lr, weight_decay)
        self.betas, self.eps = tuple(betas), eps

    def step(self, state: ServerState, agg: Update) -> ServerState:
        for n, delta in agg.tensors.items():
            self._decay(state, n)
            _adam_step(state, n, -delta, self.lr, self.betas, self.eps)
        state.round += 1
        return state


@register("server_opt", "muon")
class ServerMuon(ServerOptimizer):
    """Mode A (aggregate-then-polarize): Muon on the aggregated pseudo-gradient.

    Matrix layers: EMA momentum, optional Nesterov, orthogonalize, scale by sqrt(max(1, rows/cols)).
    Other parameters (biases, norms, embeddings, head): Adam with adam_lr.
    """

    def __init__(
        self,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        orthogonalizer: Orthogonalizer | dict | str | None = None,
        weight_decay: float = 0.0,
        adam_lr: float = 3e-4,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
    ) -> None:
        super().__init__(lr, weight_decay)
        self.momentum, self.nesterov = momentum, nesterov
        self.orthogonalizer = build_orthogonalizer(orthogonalizer)
        self.adam_lr, self.adam_betas, self.adam_eps = adam_lr, tuple(adam_betas), adam_eps

    def step(self, state: ServerState, agg: Update) -> ServerState:
        matrix = set(agg.matrix_names())
        for n, delta in agg.tensors.items():
            g = -delta
            self._decay(state, n)
            if n not in matrix:
                _adam_step(state, n, g, self.adam_lr, self.adam_betas, self.adam_eps)
                continue
            buf = state.momentum.get(n)
            buf = g.clone() if buf is None else buf.lerp_(g, 1 - self.momentum)
            state.momentum[n] = buf
            upd = g.lerp(buf, self.momentum) if self.nesterov else buf
            m2d = upd.reshape(upd.shape[0], -1)
            o = self.orthogonalizer(m2d).reshape(upd.shape)
            scale = max(1.0, m2d.shape[0] / m2d.shape[1]) ** 0.5
            state.params[n].add_(o.to(state.params[n].dtype), alpha=-self.lr * scale)
        state.round += 1
        return state


@register("server_opt", "normsgd")
class ServerNormSGD(ServerOptimizer):
    """Momentum SGD with Muon's step size but without orthogonalization: the control for FedMuon.

    Identical to ServerMuon (EMA momentum, Nesterov, Adam for non-matrix parameters, sqrt(rows/cols) scale) except
    that each matrix momentum is rescaled to the Frobenius norm of Muon's semi-orthogonal step, sqrt(min(rows, cols)),
    instead of being replaced by its polar factor. The per-round step norm is therefore the same as FedMuon's, so
    the step size no longer depends on how much the client deltas cancel, and the only remaining difference to
    FedMuon is the direction (polar factor vs. the momentum itself).
    """

    def __init__(
        self,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        adam_lr: float = 3e-4,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(lr, weight_decay)
        self.momentum, self.nesterov, self.eps = momentum, nesterov, eps
        self.adam_lr, self.adam_betas, self.adam_eps = adam_lr, tuple(adam_betas), adam_eps

    def step(self, state: ServerState, agg: Update) -> ServerState:
        matrix = set(agg.matrix_names())
        for n, delta in agg.tensors.items():
            g = -delta
            self._decay(state, n)
            if n not in matrix:
                _adam_step(state, n, g, self.adam_lr, self.adam_betas, self.adam_eps)
                continue
            buf = state.momentum.get(n)
            buf = g.clone() if buf is None else buf.lerp_(g, 1 - self.momentum)
            state.momentum[n] = buf
            upd = g.lerp(buf, self.momentum) if self.nesterov else buf
            m2d = upd.reshape(upd.shape[0], -1)
            target = min(m2d.shape) ** 0.5  # ||UV^T||_F of a rank-min(m,k) polar factor
            o = (m2d * (target / (m2d.norm() + self.eps))).reshape(upd.shape)
            scale = max(1.0, m2d.shape[0] / m2d.shape[1]) ** 0.5
            state.params[n].add_(o.to(state.params[n].dtype), alpha=-self.lr * scale)
        state.round += 1
        return state


def build_orthogonalizer(spec: Orthogonalizer | dict | str | None) -> Orthogonalizer:
    if spec is None:
        return NewtonSchulz()
    if isinstance(spec, Orthogonalizer):
        return spec
    if isinstance(spec, str):
        return REGISTRY.build("orthogonalizer", spec)
    spec = dict(spec)
    return REGISTRY.build("orthogonalizer", spec.pop("name"), **spec)


def build_server_opt(spec: dict) -> ServerOptimizer:
    spec = dict(spec)
    return REGISTRY.build("server_opt", spec.pop("name"), **spec)
