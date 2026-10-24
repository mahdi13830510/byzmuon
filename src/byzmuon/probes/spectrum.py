"""Research probes: singular spectrum of the aggregate and the amplification gain of orthogonalization."""

import torch
from torch import Tensor

from byzmuon.aggregators.mean import Mean
from byzmuon.core.update import Update
from byzmuon.optim.orthogonalize import ExactPolar, Orthogonalizer
from byzmuon.probes.base import Probe
from byzmuon.registry import register


def spectrum_stats(m: Tensor, tail_frac: float = 0.01) -> dict[str, float]:
    s = torch.linalg.svdvals(m.float().cpu())
    s1 = float(s[0]) if s.numel() else 0.0
    erank = float(s.sum() ** 2 / (s.pow(2).sum() + 1e-30))  # participation ratio
    tail = float((s < tail_frac * s1).float().mean()) if s1 > 0 else 0.0
    return {"s1": s1, "s_min": float(s[-1]), "erank": erank, "tail_frac": tail}


def server_orthogonalizer(server) -> Orthogonalizer:
    return getattr(server.server_opt, "orthogonalizer", None) or ExactPolar()


@register("probe", "spectrum")
class Spectrum(Probe):
    """Per matrix layer: top singular value, smallest, effective rank, fraction of near-zero directions.
    Logged for the aggregate ('spec/') and, when available, the honest mean ('hspec/')."""

    def __init__(self, every: int = 10, layers: list[str] | None = None, max_layers: int = 8) -> None:
        super().__init__(every)
        self.layers, self.max_layers = layers, max_layers
        self._pending: dict[str, float] = {}

    def _layers(self, u: Update) -> list[str]:
        names = self.layers or u.matrix_names()
        return names[: self.max_layers]

    def on_aggregated(self, server, r: int, agg: Update) -> None:
        if not self.due(r):
            return
        self._pending = {}
        for n in self._layers(agg):
            for k, v in spectrum_stats(agg.as_matrix(n)).items():
                self._pending[f"spec/{n}/{k}"] = v
        if server.last_honest:
            hm = Mean().aggregate(server.last_honest)
            for n in self._layers(hm):
                for k, v in spectrum_stats(hm.as_matrix(n)).items():
                    self._pending[f"hspec/{n}/{k}"] = v

    def on_round_end(self, server, r: int, metrics: dict) -> None:
        metrics.update(self._pending)
        self._pending = {}


def amplification_gain(
    m: Tensor, orth: Orthogonalizer, eps: float, mode: str, rank_frac: float = 0.1, gen: torch.Generator | None = None
) -> float:
    """(relative change of orth(M) under a perturbation of relative size eps) / eps. Linear maps give 1."""
    m = m.float().cpu()
    base = orth(m).float()
    z = torch.randn(m.shape, generator=gen)
    if mode == "tail":
        u, s, vt = torch.linalg.svd(m, full_matrices=False)
        k = int((s > rank_frac * s[0]).sum().clamp(max=s.numel() - 1))
        uk, vk = u[:, :k], vt[:k].T
        z = z - uk @ (uk.T @ z)
        z = z - (z @ vk) @ vk.T
    e = z * (eps * m.norm() / (z.norm() + 1e-12))
    pert = orth(m + e).float()
    return float((pert - base).norm() / (base.norm() + 1e-12) / eps)


@register("probe", "amplification")
class Amplification(Probe):
    """Measures the amplification gain of the server's orthogonalizer on the honest mean, per layer,
    for random and trailing-subspace perturbations. The go/no-go measurement of the project."""

    def __init__(
        self,
        every: int = 10,
        eps: tuple[float, ...] = (0.01, 0.05),
        max_layers: int = 8,
        rank_frac: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__(every)
        self.eps, self.max_layers, self.rank_frac = tuple(eps), max_layers, rank_frac
        self.gen = torch.Generator().manual_seed(seed)
        self._pending: dict[str, float] = {}

    def on_aggregated(self, server, r: int, agg: Update) -> None:
        if not self.due(r):
            return
        self._pending = {}
        src = Mean().aggregate(server.last_honest) if server.last_honest else agg
        orth = server_orthogonalizer(server)
        sums: dict[str, list[float]] = {}
        for n in src.matrix_names()[: self.max_layers]:
            m = src.as_matrix(n)
            for eps in self.eps:
                for mode in ("random", "tail"):
                    g = amplification_gain(m, orth, eps, mode, self.rank_frac, self.gen)
                    self._pending[f"amp/{n}/{mode}/{eps}"] = g
                    sums.setdefault(f"amp/mean/{mode}/{eps}", []).append(g)
        for k, v in sums.items():
            self._pending[k] = sum(v) / len(v)

    def on_round_end(self, server, r: int, metrics: dict) -> None:
        metrics.update(self._pending)
        self._pending = {}
