import pytest
import torch

from byzmuon.attacks.base import AttackContext, BudgetSpec, Knowledge
from byzmuon.attacks.baselines import Gaussian, SignFlip
from byzmuon.core.update import Update


def honest(value):
    return Update(
        tensors={
            "fc.weight": torch.full((4, 4), float(value)),
            "fc.bias": torch.zeros(4),
        },
        meta={"exclude_from_muon": set()},
    )


def ctx(knowledge, budget=None):
    budget = budget or BudgetSpec()
    return AttackContext(
        round=0,
        knowledge=knowledge,
        byz_own_updates=[honest(2)],
        benign_updates=[honest(1), honest(3)] if knowledge is Knowledge.OMNISCIENT else None,
        budget=budget,
    )


def test_reference_omniscient_is_true_mean():
    assert torch.allclose(
        ctx(Knowledge.OMNISCIENT).reference().tensors["fc.weight"],
        torch.full((4, 4), 2.0),
    )


def test_reference_partial_uses_own_updates_only():
    c = ctx(Knowledge.PARTIAL)
    assert c.benign_updates is None
    assert torch.allclose(c.reference().tensors["fc.weight"], torch.full((4, 4), 2.0))


def test_reference_agnostic_raises():
    with pytest.raises(PermissionError):
        ctx(Knowledge.AGNOSTIC).reference()


def test_signflip_returns_num_byz_negated():
    out = SignFlip(num_byz=3).craft(ctx(Knowledge.OMNISCIENT))
    assert len(out) == 3
    assert torch.allclose(out[0].tensors["fc.weight"], torch.full((4, 4), -2.0))


def test_budget_scales_down_but_keeps_direction():
    c = ctx(Knowledge.OMNISCIENT, BudgetSpec(fro_rel=0.5))
    ref_norm = c.reference().fro_norm()
    out = SignFlip(num_byz=1, scale=10.0).craft(c)[0]
    assert out.fro_norm() == pytest.approx(0.5 * ref_norm, rel=1e-4)
    assert out.tensors["fc.weight"][0, 0] < 0


def test_gaussian_shapes_and_count():
    out = Gaussian(num_byz=2).craft(ctx(Knowledge.PARTIAL))
    assert len(out) == 2 and out[0].tensors["fc.weight"].shape == (4, 4)
