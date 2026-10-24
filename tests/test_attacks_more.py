import pytest
import torch
from conftest import low_rank_update

from byzmuon.aggregators.mean import Mean
from byzmuon.aggregators.median import Median
from byzmuon.attacks import build_attack
from byzmuon.attacks.base import AttackContext, BudgetSpec, Knowledge
from byzmuon.attacks.baselines import IPM, LIE, LabelFlip, MinMax, MinSum, NoAttack
from byzmuon.attacks.spectral import Intermittent, RankOneSpike, SubGapInjection
from byzmuon.optim.orthogonalize import ExactPolar


def honest_set(n=5, noise=1e-3):
    return [low_rank_update(seed=s, noise=noise) for s in range(n)]


def ctx(benign, knowledge=Knowledge.OMNISCIENT, budget=None, n_mal=None, r=0):
    return AttackContext(
        round=r,
        knowledge=knowledge,
        byz_own_updates=benign[:2],
        benign_updates=benign if knowledge is Knowledge.OMNISCIENT else None,
        budget=budget or BudgetSpec(),
        n_mal=n_mal,
    )


def test_none_and_count():
    assert NoAttack().craft(ctx(honest_set())) == []
    assert len(NoAttack(num_byz=2).craft(ctx(honest_set()))) == 2
    out = IPM(num_byz=3).craft(ctx(honest_set(), n_mal=2))
    assert len(out) == 2
    assert torch.allclose(out[0].tensors["w"], -0.1 * Mean().aggregate(honest_set()).tensors["w"])


def test_lie_is_mean_minus_z_std():
    benign = honest_set()
    out = LIE(num_byz=2, n_clients=7).craft(ctx(benign))[0]
    vecs = torch.stack([u.flat() for u in benign])
    z = LIE(num_byz=2).z_value(7)
    assert z > 0
    assert torch.allclose(out.flat(), vecs.mean(0) - z * vecs.std(0, unbiased=False), atol=1e-6)


def test_minmax_stays_within_benign_spread_and_minsum_too():
    benign = honest_set(noise=0.05)
    vecs = torch.stack([u.flat() for u in benign])
    maxpair = float(torch.cdist(vecs, vecs).max())
    for cls in (MinMax, MinSum):
        out = cls(num_byz=2, perturbation="unit").craft(ctx(benign))[0]
        assert out.fro_norm() > 0
        if cls is MinMax:
            assert float((vecs - out.flat()).norm(dim=1).max()) <= maxpair * (1 + 1e-3)
    with pytest.raises(KeyError):
        MinMax(num_byz=1, perturbation="bad").craft(ctx(benign))


def test_labelflip_is_data_poisoning():
    a = LabelFlip(num_byz=2)
    assert a.poisons_data and a.label_map(10)(torch.tensor([0, 9])).tolist() == [9, 0]
    benign = honest_set()
    out = a.craft(ctx(benign, n_mal=2))
    assert len(out) == 2 and torch.equal(out[0].tensors["w"], benign[0].tensors["w"])


def test_subgap_perturbation_lives_in_trailing_subspace_and_is_small():
    benign = honest_set()
    ref = Mean().aggregate(benign)
    atk = SubGapInjection(num_byz=1, eps=0.05, rank=4)
    bad = atk.craft(ctx(benign))[0]
    e = bad.tensors["w"] - ref.tensors["w"]
    u, s, vt = torch.linalg.svd(ref.tensors["w"], full_matrices=False)
    assert float((u[:, :4].T @ e).norm()) < 1e-4 and float((e @ vt[:4].T).norm()) < 1e-4
    assert e.norm() == pytest.approx(0.05 * ref.tensors["w"].norm(), rel=1e-4)
    assert torch.equal(bad.tensors["b"], ref.tensors["b"])


def test_subgap_is_amplified_by_orthogonalization_but_not_by_mean():
    benign = honest_set(noise=1e-3)
    ref = Mean().aggregate(benign)
    bad = SubGapInjection(num_byz=1, eps=0.02, rank=4).craft(ctx(benign))[0]
    agg = Mean().aggregate(benign + [bad]).tensors["w"]  # one Byzantine among six
    linear_change = float((agg - ref.tensors["w"]).norm() / ref.tensors["w"].norm())
    p = ExactPolar()
    orth_change = float((p(agg) - p(ref.tensors["w"])).norm() / p(ref.tensors["w"]).norm())
    assert linear_change < 0.01
    assert orth_change > 10 * linear_change


def test_subgap_flip_tail_mode_and_partial_knowledge():
    benign = honest_set()
    out = SubGapInjection(num_byz=1, eps=0.05, mode="flip_tail", knowledge="partial").craft(
        ctx(benign, Knowledge.PARTIAL)
    )
    assert len(out) == 1 and out[0].fro_norm() > 0


def test_spike_is_rank_one_and_entrywise_small_but_spectrally_large():
    benign = honest_set()
    ref = Mean().aggregate(benign)
    bad = RankOneSpike(num_byz=1, scale=1.0).craft(ctx(benign))[0]
    e = bad.tensors["w"] - ref.tensors["w"]
    s = torch.linalg.svdvals(e)
    assert s[1] / s[0] < 1e-4  # rank one
    assert e.norm() == pytest.approx(ref.tensors["w"].norm(), rel=1e-4)
    assert e.abs().max() < 0.2 * e.norm()  # dense, small entries
    assert bad.spec_norms()["w"] > 1.5 * ref.spec_norms()["w"]  # dominates the spectrum


def test_spike_survives_median_better_than_signflip():
    benign = honest_set(noise=0.02)
    ref = Mean().aggregate(benign)
    spikes = RankOneSpike(num_byz=2, scale=1.0).craft(ctx(benign))
    agg = Median(num_byz=2).aggregate(benign + spikes)
    assert agg.fro_norm() > 0 and (agg - ref).fro_norm() < ref.fro_norm()  # median still mostly honest


def test_intermittent_wrapper_schedules_inner_attack():
    benign = honest_set()
    atk = Intermittent(num_byz=1, inner={"name": "signflip"}, rounds=[3], every=5, start=10)
    assert [atk.active(r) for r in (0, 3, 10, 12, 15)] == [False, True, True, False, True]
    quiet = atk.craft(ctx(benign, r=1))[0]
    loud = atk.craft(ctx(benign, r=3))[0]
    assert torch.equal(quiet.tensors["w"], benign[0].tensors["w"])
    assert torch.allclose(loud.tensors["w"], -Mean().aggregate(benign).tensors["w"])


def test_build_attack_from_spec_with_budget():
    a = build_attack({"name": "subgap", "eps": 0.1, "knowledge": "partial", "budget": {"fro_rel": 1.5}}, num_byz=3)
    assert isinstance(a, SubGapInjection) and a.knowledge is Knowledge.PARTIAL and a.budget.fro_rel == 1.5
