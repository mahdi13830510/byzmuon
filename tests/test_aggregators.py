import pytest
import torch

from byzmuon.aggregators.base import Pipeline, PreAgg
from byzmuon.aggregators.mean import Mean
from byzmuon.aggregators.median import Median
from byzmuon.core.update import Update
from byzmuon.registry import REGISTRY


def const_update(value, seed_shape=((2, 3), (3,))):
    w, b = seed_shape
    return Update(
        tensors={
            "fc.weight": torch.full(w, float(value)),
            "fc.bias": torch.full(b, float(value)),
        },
        meta={"exclude_from_muon": set()},
    )


def test_mean_of_constants():
    out = Mean().aggregate([const_update(1), const_update(3)])
    assert torch.allclose(out.tensors["fc.weight"], torch.full((2, 3), 2.0))


def test_median_ignores_one_outlier():
    honest = [const_update(1) for _ in range(4)]
    byz = [const_update(1000)]
    out = Median(num_byz=1).aggregate(honest + byz)
    assert torch.allclose(out.tensors["fc.weight"], torch.ones(2, 3))


def test_mean_is_broken_by_one_outlier():
    out = Mean().aggregate([const_update(1)] * 4 + [const_update(1000)])
    assert out.tensors["fc.weight"][0, 0] > 100


def test_aggregate_keeps_exclude_meta():
    u = const_update(1)
    u.meta["exclude_from_muon"] = {"fc.weight"}
    out = Mean().aggregate([u, u])
    assert out.meta["exclude_from_muon"] == {"fc.weight"}


class Doubler(PreAgg):
    def transform(self, updates):
        return [u * 2 for u in updates]


def test_pipeline_applies_pre_then_base():
    out = Pipeline(pre=[Doubler()], base=Mean()).aggregate([const_update(1), const_update(1)])
    assert torch.allclose(out.tensors["fc.bias"], torch.full((3,), 2.0))


def test_registry_builds_by_name():
    agg = REGISTRY.build("aggregator", "median", num_byz=2)
    assert isinstance(agg, Median) and agg.num_byz == 2


def test_registry_unknown_name_lists_known():
    with pytest.raises(KeyError, match="mean"):
        REGISTRY.get("aggregator", "does_not_exist")
