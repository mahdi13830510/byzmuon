"""The attacks this project introduces: they exploit Muon's orthogonalization step."""

import torch
from torch import Tensor

from byzmuon.attacks.base import Attack, AttackContext
from byzmuon.core.update import Update
from byzmuon.registry import REGISTRY, register


def leading_rank(s: Tensor, rank: int | None, rank_frac: float) -> int:
    """Number of leading singular directions treated as 'signal'."""
    if rank is not None:
        return max(0, min(rank, s.numel() - 1))
    return int((s > rank_frac * s[0]).sum().clamp(max=s.numel() - 1))


@register("attack", "subgap")
class SubGapInjection(Attack):
    """A1. Perturb only the trailing (weak) singular subspace of the reference, on both sides:
    E = (I − U_k U_kᵀ) Z (I − V_k V_kᵀ). Small in Frobenius norm, invisible to Euclidean defenses,
    amplified by ~1/(s_i + s_j) after orthogonalization.

    mode='random': Z Gaussian. mode='flip_tail': Z = −(tail of the reference), i.e. reverse the weak
    directions. eps is the per-layer perturbation size relative to the layer's Frobenius norm.
    """

    def __init__(
        self,
        num_byz: int,
        eps: float = 0.05,
        rank: int | None = None,
        rank_frac: float = 0.1,
        mode: str = "random",
        seed: int = 0,
        **kw,
    ) -> None:
        super().__init__(num_byz, **kw)
        self.eps, self.rank, self.rank_frac, self.mode = eps, rank, rank_frac, mode
        self.gen = torch.Generator().manual_seed(seed)

    def perturb_matrix(self, m: Tensor) -> Tensor:
        u, s, vt = torch.linalg.svd(m.float().cpu(), full_matrices=False)
        k = leading_rank(s, self.rank, self.rank_frac)
        uk, vk = u[:, :k], vt[:k].T
        if self.mode == "random":
            z = torch.randn(m.shape, generator=self.gen)
        elif self.mode == "flip_tail":
            z = -(u[:, k:] * s[k:]) @ vt[k:]
        else:
            raise KeyError(f"unknown mode '{self.mode}' (random, flip_tail)")
        e = z - uk @ (uk.T @ z)  # project rows off the leading left subspace
        e = e - (e @ vk) @ vk.T  # project columns off the leading right subspace
        e = e * (self.eps * m.norm().cpu() / (e.norm() + 1e-12))
        return e.to(m.device, m.dtype)

    def craft(self, ctx: AttackContext) -> list[Update]:
        ref = ctx.reference()
        matrix = set(ref.matrix_names())

        def f(n: str, t: Tensor) -> Tensor:
            if n not in matrix:
                return t
            m = t.reshape(t.shape[0], -1)
            return t + self.perturb_matrix(m).reshape(t.shape)

        bad = self.apply_budget(ref.map(f), ctx)
        return [bad for _ in range(self.count(ctx))]


@register("attack", "spike")
class RankOneSpike(Attack):
    """A2. Add a dense rank-1 matrix u vᵀ to every matrix layer. Entrywise tiny (passes coordinate-wise
    defenses), but its spectral norm equals its Frobenius norm, so it dominates the orthogonalized
    update once it exceeds the reference's top singular value. scale is relative to ||ref layer||_F."""

    def __init__(self, num_byz: int, scale: float = 1.0, seed: int = 0, **kw) -> None:
        super().__init__(num_byz, **kw)
        self.scale = scale
        self.gen = torch.Generator().manual_seed(seed)

    def craft(self, ctx: AttackContext) -> list[Update]:
        ref = ctx.reference()
        matrix = set(ref.matrix_names())

        def f(n: str, t: Tensor) -> Tensor:
            if n not in matrix:
                return t
            m = t.reshape(t.shape[0], -1)
            u = torch.randn(m.shape[0], generator=self.gen)
            v = torch.randn(m.shape[1], generator=self.gen)
            spike = torch.outer(u / u.norm(), v / v.norm()).to(m.device, m.dtype) * (self.scale * m.norm())
            return t + spike.reshape(t.shape)

        bad = self.apply_budget(ref.map(f), ctx)
        return [bad for _ in range(self.count(ctx))]


@register("attack", "intermittent")
class Intermittent(Attack):
    """A3 wrapper. Runs `inner` only on the listed rounds (or every k-th round); otherwise the Byzantine
    clients behave honestly. Measures how long a one-shot injection persists through momentum."""

    def __init__(
        self, num_byz: int, inner: dict, rounds: list[int] | None = None, every: int | None = None, start: int = 0, **kw
    ) -> None:
        super().__init__(num_byz, **kw)
        inner = dict(inner)
        self.inner: Attack = REGISTRY.build(
            "attack", inner.pop("name"), num_byz=num_byz, knowledge=self.knowledge, budget=self.budget, **inner
        )
        self.rounds, self.every, self.start = set(rounds or []), every, start
        self.poisons_data = self.inner.poisons_data

    def active(self, r: int) -> bool:
        if r in self.rounds:
            return True
        return self.every is not None and r >= self.start and (r - self.start) % self.every == 0

    def label_map(self, num_classes: int):
        return self.inner.label_map(num_classes)

    def craft(self, ctx: AttackContext) -> list[Update]:
        if self.active(ctx.round):
            return self.inner.craft(ctx)
        return list(ctx.byz_own_updates[: self.count(ctx)])
