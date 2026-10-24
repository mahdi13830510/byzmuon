import torch
from torch import Tensor

from byzmuon.aggregators.base import FlatAggregator
from byzmuon.registry import register


@register("aggregator", "geomed")
class GeometricMedian(FlatAggregator):
    """Smoothed Weiszfeld iterations for the geometric median (RFA, Pillutla et al. 2022)."""

    def __init__(self, num_byz: int = 0, iters: int = 10, eps: float = 1e-6) -> None:
        super().__init__(num_byz)
        self.iters, self.eps = iters, eps

    def _agg_flat(self, vecs: Tensor) -> Tensor:
        z = vecs.mean(dim=0)
        for _ in range(self.iters):
            w = 1.0 / torch.clamp((vecs - z).norm(dim=1), min=self.eps)
            z = (w[:, None] * vecs).sum(dim=0) / w.sum()
        return z
