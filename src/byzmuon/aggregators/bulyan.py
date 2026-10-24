import torch
from torch import Tensor

from byzmuon.aggregators.base import FlatAggregator
from byzmuon.aggregators.krum import krum_scores
from byzmuon.registry import register


@register("aggregator", "bulyan")
class Bulyan(FlatAggregator):
    """Bulyan (El Mhamdi et al. 2018): Multi-Krum selection of θ = n − 2f, then per-coordinate trimmed mean
    keeping the β = θ − 2f values closest to the median. Degrades gracefully when n < 4f + 3."""

    def _agg_flat(self, vecs: Tensor) -> Tensor:
        n, f = vecs.shape[0], self.num_byz
        theta = max(1, n - 2 * f)
        remaining = list(range(n))
        chosen: list[int] = []
        while len(chosen) < theta and remaining:
            sub = vecs[remaining]
            best = int(krum_scores(sub, f).argmin()) if len(remaining) > 2 else 0
            chosen.append(remaining.pop(best))
        sel = vecs[chosen]  # [theta, dim]
        beta = max(1, theta - 2 * f)
        med = sel.median(dim=0).values
        order = (sel - med).abs().argsort(dim=0)[:beta]  # [beta, dim] indices of closest-to-median
        return torch.gather(sel, 0, order).mean(dim=0)
