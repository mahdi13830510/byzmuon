from torch import Tensor

from byzmuon.aggregators.base import PerLayerAggregator
from byzmuon.registry import register


@register("aggregator", "trimmed_mean")
class TrimmedMean(PerLayerAggregator):
    """Per coordinate: drop the num_byz largest and smallest values, average the rest (Yin et al. 2018)."""

    def _agg_layer(self, stack: Tensor) -> Tensor:
        n, f = stack.shape[0], self.num_byz
        if n - 2 * f < 1:
            return stack.median(dim=0).values
        s, _ = stack.sort(dim=0)
        return s[f : n - f].mean(dim=0)
