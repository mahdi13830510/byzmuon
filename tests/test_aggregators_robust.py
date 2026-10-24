import torch
from conftest import low_rank_update

from byzmuon.aggregators import build_aggregator
from byzmuon.aggregators.base import Pipeline
from byzmuon.aggregators.bulyan import Bulyan
from byzmuon.aggregators.centered_clip import CenteredClip
from byzmuon.aggregators.geomed import GeometricMedian
from byzmuon.aggregators.krum import Krum, MultiKrum
from byzmuon.aggregators.preagg import NNM, FroClip, SpectralClip
from byzmuon.aggregators.trimmed_mean import TrimmedMean
from byzmuon.core.update import Update


def upd(value, seed=0, noise=0.01):
    g = torch.Generator().manual_seed(seed)
    return Update(
        tensors={"w": value + noise * torch.randn(4, 3, generator=g), "b": torch.full((2,), float(value))},
        meta={"exclude_from_muon": set()},
    )


HONEST = [upd(1.0, s) for s in range(7)]
BYZ = [upd(1000.0, 100 + s) for s in range(2)]


def assert_close_to_honest(agg, tol=0.05):
    out = agg.aggregate(HONEST + BYZ)
    assert (out.tensors["w"] - 1.0).abs().max() < tol, out.tensors["w"]
    assert torch.allclose(out.tensors["b"], torch.ones(2), atol=tol)


def test_trimmed_mean_krum_multikrum_bulyan_geomed_centered_clip_resist_outliers():
    for agg in [
        TrimmedMean(num_byz=2),
        Krum(num_byz=2),
        MultiKrum(num_byz=2),
        Bulyan(num_byz=2),
        GeometricMedian(num_byz=2),
        CenteredClip(num_byz=2),
    ]:
        assert_close_to_honest(agg)


def test_krum_picks_a_single_honest_vector():
    out = Krum(num_byz=2).aggregate(HONEST + BYZ)
    assert any(torch.equal(out.tensors["w"], h.tensors["w"]) for h in HONEST)


def test_nnm_pulls_outliers_toward_honest():
    mixed = NNM(num_byz=2).transform(HONEST + BYZ)
    worst = max(u.tensors["b"][0] for u in mixed)
    assert worst < 500  # 1000 averaged with n−f=7 neighbours
    out = Pipeline([NNM(num_byz=2)], TrimmedMean(num_byz=2)).aggregate(HONEST + BYZ)
    assert (out.tensors["w"] - 1.0).abs().max() < 0.05


def test_fro_clip_to_median_norm():
    clipped = FroClip().transform(HONEST + BYZ)
    med = float(torch.tensor([u.fro_norm() for u in HONEST + BYZ]).median())
    assert max(u.fro_norm() for u in clipped) <= med + 1e-4


def test_spectral_clip_removes_spike_but_keeps_honest():
    honest = [low_rank_update(seed=s) for s in range(5)]
    spike = low_rank_update(seed=0)
    u, v = torch.randn(64), torch.randn(48)
    spike = Update(
        tensors={"w": spike.tensors["w"] + 5 * torch.outer(u / u.norm(), v / v.norm()), "b": spike.tensors["b"]},
        meta={"exclude_from_muon": set()},
    )
    out = SpectralClip(tau_rel=1.0).transform(honest + [spike])
    med = float(torch.tensor([x.spec_norms()["w"] for x in honest + [spike]]).median())
    assert out[-1].spec_norms()["w"] <= med + 1e-3
    assert torch.allclose(out[0].tensors["w"], honest[0].tensors["w"])  # below threshold: untouched


def test_centered_clip_is_stateful_and_shape_safe():
    cc = CenteredClip(num_byz=2)
    cc.aggregate(HONEST + BYZ)
    assert cc.v is not None and cc.v.shape == (14,)
    cc.aggregate(HONEST)  # second call reuses v


def test_build_aggregator_specs():
    agg = build_aggregator(
        {"base": {"name": "krum", "m": 3}, "pre": [{"name": "nnm"}, {"name": "fro_clip"}]}, num_byz=2
    )
    assert isinstance(agg, Pipeline) and isinstance(agg.base, Krum) and agg.base.m == 3 and agg.num_byz == 2
    assert isinstance(build_aggregator({"name": "median"}, num_byz=1).aggregate(HONEST).tensors["w"], torch.Tensor)
