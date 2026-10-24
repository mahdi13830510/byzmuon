from abc import ABC, abstractmethod
from typing import override

import torch
from torch import Tensor

from byzmuon.core.update import Update


class Aggregator(ABC):
    """Combines the sampled clients' updates into one. num_byz is the assumed upper bound f."""

    def __init__(self, num_byz: int = 0) -> None:
        self.num_byz = num_byz

    @abstractmethod
    def aggregate(self, updates: list[Update]) -> Update: ...


class PerLayerAggregator(Aggregator):
    """Template for rules that treat each layer independently (mean, median, trimmed mean)."""

    def aggregate(self, updates: list[Update]) -> Update:
        template = updates[0]
        out = {n: self._agg_layer(Update.stack(updates, n)) for n in template.names()}
        return Update(
            tensors=out,
            meta={"exclude_from_muon": template.meta.get("exclude_from_muon", set())},
        )

    @abstractmethod
    def _agg_layer(self, stack: Tensor) -> Tensor:
        """stack has shape [n_clients, *layer_shape]; return [*layer_shape]."""


class FlatAggregator(Aggregator):
    """Template for rules that need whole-client distances (Krum, Bulyan, geometric median)."""

    @override
    def aggregate(self, updates: list[Update]) -> Update:
        vecs = torch.stack([u.flat() for u in updates])
        return Update.unflat(self._agg_flat(vecs), like=updates[0])

    @abstractmethod
    def _agg_flat(self, vecs: Tensor) -> Tensor:
        """vecs has shape [n_clients, dim]; return [dim]."""


class PreAgg(ABC):
    """A transform applied to the list of updates before the base rule (clipping, NNM, ...)."""

    def __init__(self, num_byz: int = 0) -> None:
        self.num_byz = num_byz

    @abstractmethod
    def transform(self, updates: list[Update]) -> list[Update]: ...


class Pipeline(Aggregator):
    """pre[0] -> pre[1] -> ... -> base.aggregate"""

    def __init__(self, pre: list[PreAgg], base: Aggregator) -> None:
        super().__init__(num_byz=base.num_byz)
        self.pre = pre
        self.base = base

    def aggregate(self, updates: list[Update]) -> Update:
        for step in self.pre:
            updates = step.transform(updates)
        return self.base.aggregate(updates)
