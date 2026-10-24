from byzmuon.aggregators import bulyan, centered_clip, geomed, krum, mean, median, preagg, trimmed_mean  # noqa: F401
from byzmuon.aggregators.base import Aggregator, FlatAggregator, PerLayerAggregator, Pipeline, PreAgg
from byzmuon.registry import REGISTRY


def build_aggregator(spec: dict, num_byz: int) -> Aggregator:
    """spec: {base: {name: ..., ...}, pre: [{name: ..., ...}, ...]} or just {name: ...}."""
    spec = dict(spec)
    if "base" not in spec:
        spec = {"base": spec, "pre": []}
    base = dict(spec["base"])
    agg = REGISTRY.build("aggregator", base.pop("name"), num_byz=num_byz, **base)
    pres = []
    for p in spec.get("pre", []) or []:
        p = dict(p)
        pres.append(REGISTRY.build("preagg", p.pop("name"), num_byz=num_byz, **p))
    return Pipeline(pres, agg) if pres else agg


__all__ = ["Aggregator", "FlatAggregator", "PerLayerAggregator", "Pipeline", "PreAgg", "build_aggregator"]
