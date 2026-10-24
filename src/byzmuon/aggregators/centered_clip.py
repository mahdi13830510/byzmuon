import torch
from torch import Tensor

from byzmuon.aggregators.base import FlatAggregator
from byzmuon.registry import register


@register("aggregator", "centered_clip")
class CenteredClip(FlatAggregator):
    """Centered clipping (Karimireddy et al. 2021): v ← v + mean_i clip(x_i − v, tau), iterated L times,
    starting from the previous round's aggregate (stateful) or the coordinate median on the first round."""

    def __init__(self, num_byz: int = 0, tau: float | None = None, iters: int = 3) -> None:
        super().__init__(num_byz)
        self.tau, self.iters = tau, iters
        self.v: Tensor | None = None

    def _agg_flat(self, vecs: Tensor) -> Tensor:
        v = self.v if self.v is not None and self.v.shape == vecs.shape[1:] else vecs.median(dim=0).values
        for _ in range(self.iters):
            diff = vecs - v
            norms = diff.norm(dim=1)
            tau = self.tau if self.tau is not None else float(norms.median())
            scale = torch.clamp(tau / (norms + 1e-12), max=1.0)
            v = v + (diff * scale[:, None]).mean(dim=0)
        self.v = v.detach().clone()
        return v
