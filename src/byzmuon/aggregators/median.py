from torch import Tensor

from byzmuon.aggregators.base import PerLayerAggregator
from byzmuon.registry import register


@register("aggregator", "median")
class Median(PerLayerAggregator):
    """Coordinate-wise median: for every single entry, take the median across clients."""

    def _agg_layer(self, stack: Tensor) -> Tensor:
        return stack.median(dim=0).values
