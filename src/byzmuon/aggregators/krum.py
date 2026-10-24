import torch
from torch import Tensor

from byzmuon.aggregators.base import FlatAggregator
from byzmuon.registry import register


def krum_scores(vecs: Tensor, f: int) -> Tensor:
    """Score_i = sum of squared distances to the n − f − 2 nearest other clients."""
    n = vecs.shape[0]
    d2 = torch.cdist(vecs, vecs) ** 2
    k = max(1, n - f - 2)
    d2.fill_diagonal_(float("inf"))
    nearest, _ = d2.topk(k, dim=1, largest=False)
    return nearest.sum(dim=1)


@register("aggregator", "krum")
class Krum(FlatAggregator):
    """Krum (m=1) / Multi-Krum (m>1): average the m clients with the lowest Krum score (Blanchard et al. 2017)."""

    def __init__(self, num_byz: int = 0, m: int | None = 1) -> None:
        super().__init__(num_byz)
        self.m = m

    def _agg_flat(self, vecs: Tensor) -> Tensor:
        n = vecs.shape[0]
        m = self.m if self.m is not None else max(1, n - self.num_byz)
        m = min(m, n)
        idx = krum_scores(vecs, self.num_byz).topk(m, largest=False).indices
        return vecs[idx].mean(dim=0)


@register("aggregator", "multikrum")
class MultiKrum(Krum):
    def __init__(self, num_byz: int = 0, m: int | None = None) -> None:
        super().__init__(num_byz, m)
