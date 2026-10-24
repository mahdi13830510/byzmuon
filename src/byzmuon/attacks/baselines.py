"""Standard model-poisoning baselines from the Byzantine-FL literature."""

import math

import torch
from torch import Tensor

from byzmuon.attacks.base import Attack, AttackContext, Knowledge
from byzmuon.core.update import Update
from byzmuon.registry import register


@register("attack", "none")
class NoAttack(Attack):
    """No attack: any clients marked Byzantine simply send their honest updates."""

    def __init__(self, num_byz: int = 0, **kw) -> None:
        super().__init__(num_byz, **kw)

    def craft(self, ctx: AttackContext) -> list[Update]:
        return list(ctx.byz_own_updates[: self.count(ctx)])


@register("attack", "gaussian")
class Gaussian(Attack):
    """Random noise at the reference's per-layer scale. A sanity-check attack."""

    def __init__(self, num_byz: int, std: float = 1.0, **kw) -> None:
        super().__init__(num_byz, **kw)
        self.std = std

    def craft(self, ctx: AttackContext) -> list[Update]:
        ref = ctx.reference()
        out = []
        for _ in range(self.count(ctx)):
            noise = ref.map(lambda n, t: torch.randn_like(t) * self.std * (t.std() + 1e-12))
            out.append(self.apply_budget(noise, ctx))
        return out


@register("attack", "signflip")
class SignFlip(Attack):
    """Send −scale × reference."""

    def __init__(self, num_byz: int, scale: float = 1.0, **kw) -> None:
        super().__init__(num_byz, **kw)
        self.scale = scale

    def craft(self, ctx: AttackContext) -> list[Update]:
        bad = self.apply_budget(ctx.reference() * (-self.scale), ctx)
        return [bad for _ in range(self.count(ctx))]


@register("attack", "ipm")
class IPM(Attack):
    """Inner Product Manipulation (Xie et al. 2020): −eps × reference. Small eps makes the aggregate
    nearly zero, large eps flips it."""

    def __init__(self, num_byz: int, eps: float = 0.1, **kw) -> None:
        super().__init__(num_byz, **kw)
        self.eps = eps

    def craft(self, ctx: AttackContext) -> list[Update]:
        bad = self.apply_budget(ctx.reference() * (-self.eps), ctx)
        return [bad for _ in range(self.count(ctx))]


def _mean_std(updates: list[Update]) -> tuple[Tensor, Tensor]:
    vecs = torch.stack([u.flat() for u in updates])
    std = vecs.std(dim=0, unbiased=False) if vecs.shape[0] > 1 else torch.zeros_like(vecs[0])
    return vecs.mean(dim=0), std


@register("attack", "lie")
class LIE(Attack):
    """A Little Is Enough (Baruch et al. 2019): mean − z × std per coordinate, z from (n, f) or fixed."""

    def __init__(self, num_byz: int, n_clients: int | None = None, z: float | None = None, **kw) -> None:
        super().__init__(num_byz, **kw)
        self.n_clients, self.z = n_clients, z

    def z_value(self, n: int) -> float:
        if self.z is not None:
            return self.z
        f = self.num_byz
        s = math.floor(n / 2 + 1) - f
        p = max(1e-6, min(1 - 1e-6, (n - f - s) / (n - f)))
        return float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(p)))

    def craft(self, ctx: AttackContext) -> list[Update]:
        refs = ctx.reference_set()
        mean, std = _mean_std(refs)
        n = self.n_clients or (len(refs) + self.count(ctx))
        vec = mean - self.z_value(n) * std
        bad = self.apply_budget(Update.unflat(vec, like=refs[0]), ctx)
        return [bad for _ in range(self.count(ctx))]


@register("attack", "minmax")
class MinMax(Attack):
    """Shejwalkar & Houmansadr 2021. malicious = mean + gamma × p with the largest gamma such that
    the malicious update is no farther from any benign update than benign updates are from each other
    (minmax) or in sum of squared distances (minsum). p ∈ {unit, sign, std}."""

    objective = "minmax"

    def __init__(
        self,
        num_byz: int,
        perturbation: str = "unit",
        gamma_init: float = 10.0,
        tol: float = 1e-3,
        max_iter: int = 30,
        **kw,
    ) -> None:
        super().__init__(num_byz, **kw)
        self.perturbation, self.gamma_init, self.tol, self.max_iter = perturbation, gamma_init, tol, max_iter

    def _direction(self, mean: Tensor, std: Tensor) -> Tensor:
        if self.perturbation == "unit":
            return -mean / (mean.norm() + 1e-12)
        if self.perturbation == "sign":
            return -mean.sign()
        if self.perturbation == "std":
            return -std
        raise KeyError(f"unknown perturbation '{self.perturbation}' (unit, sign, std)")

    def _ok(self, vec: Tensor, benign: Tensor, threshold: float) -> bool:
        d = (benign - vec).norm(dim=1)
        val = float(d.max()) if self.objective == "minmax" else float((d**2).sum())
        return val <= threshold

    def craft(self, ctx: AttackContext) -> list[Update]:
        refs = ctx.reference_set()
        benign = torch.stack([u.flat() for u in refs])
        mean, std = (
            benign.mean(dim=0),
            benign.std(dim=0, unbiased=False) if len(refs) > 1 else torch.zeros_like(benign[0]),
        )
        p = self._direction(mean, std)
        pair = torch.cdist(benign, benign)
        threshold = float(pair.max()) if self.objective == "minmax" else float((pair**2).sum(dim=1).max())
        lo, hi, gamma = 0.0, self.gamma_init, 0.0
        while self._ok(mean + hi * p, benign, threshold) and hi < 1e6:  # grow until infeasible
            lo, hi = hi, hi * 2
        for _ in range(self.max_iter):
            mid = (lo + hi) / 2
            if self._ok(mean + mid * p, benign, threshold):
                lo, gamma = mid, mid
            else:
                hi = mid
            if hi - lo < self.tol:
                break
        bad = self.apply_budget(Update.unflat(mean + gamma * p, like=refs[0]), ctx)
        return [bad for _ in range(self.count(ctx))]


@register("attack", "minsum")
class MinSum(MinMax):
    objective = "minsum"


@register("attack", "labelflip")
class LabelFlip(Attack):
    """Data poisoning: Byzantine clients train on labels y → num_classes − 1 − y and send that honestly."""

    poisons_data = True

    def __init__(self, num_byz: int, **kw) -> None:
        super().__init__(num_byz, **kw)

    def label_map(self, num_classes: int):
        return lambda y: num_classes - 1 - y

    def craft(self, ctx: AttackContext) -> list[Update]:
        return [
            self.apply_budget(u, ctx) if ctx.knowledge is not Knowledge.AGNOSTIC else u
            for u in ctx.byz_own_updates[: self.count(ctx)]
        ]
