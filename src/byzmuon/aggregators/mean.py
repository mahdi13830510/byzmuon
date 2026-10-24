from typing import override

from torch import Tensor

from byzmuon.aggregators.base import PerLayerAggregator
from byzmuon.registry import register


@register("aggregator", "mean")
class Mean(PerLayerAggregator):
    @override
    def _agg_layer(self, stack: Tensor) -> Tensor:
        return stack.mean(dim=0)
