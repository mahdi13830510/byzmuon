import pytest
import torch

from byzmuon.core.update import Update
from byzmuon.optim.orthogonalize import ExactPolar
from byzmuon.optim.server_opt import (
    ServerAdam,
    ServerMuon,
    ServerNormSGD,
    ServerSGD,
    ServerState,
    build_server_opt,
)


def state_and_update():
    params = {"w": torch.zeros(8, 4), "b": torch.zeros(4)}
    st = ServerState(params={k: v.clone() for k, v in params.items()}, exclude_from_muon=set())
    agg = Update(tensors={"w": torch.randn(8, 4), "b": torch.ones(4)}, meta={"exclude_from_muon": set()})
    return st, agg


def test_server_sgd_lr1_is_fedavg():
    st, agg = state_and_update()
    ServerSGD(lr=1.0).step(st, agg)
    assert torch.allclose(st.params["w"], agg.tensors["w"]) and torch.allclose(st.params["b"], agg.tensors["b"])
    assert st.round == 1


def test_server_sgd_momentum_accumulates():
    st, agg = state_and_update()
    opt = ServerSGD(lr=1.0, momentum=0.5)
    opt.step(st, agg)
    opt.step(st, agg)
    assert torch.allclose(st.params["b"], torch.full((4,), 2.5))  # 1 + (1 + 0.5)


def test_server_adam_moves_params_toward_update():
    st, agg = state_and_update()
    ServerAdam(lr=0.1).step(st, agg)
    assert (st.params["b"] > 0).all()


def test_server_muon_matrix_step_is_orthogonal_and_scaled():
    st, agg = state_and_update()
    opt = ServerMuon(lr=0.1, momentum=0.0, nesterov=False, orthogonalizer=ExactPolar())
    opt.step(st, agg)
    w = st.params["w"]
    scale = max(1.0, 8 / 4) ** 0.5
    sv = torch.linalg.svdvals(w)
    assert torch.allclose(sv, torch.full_like(sv, 0.1 * scale), atol=1e-4)  # lr*scale*polar(-agg)
    assert torch.allclose(w, -0.1 * scale * ExactPolar()(-agg.tensors["w"]) * -1 * -1, atol=1e-5)
    assert (st.params["b"] > 0).all()  # non-matrix handled by Adam


def test_server_muon_respects_exclusion():
    st, agg = state_and_update()
    agg.meta["exclude_from_muon"] = {"w"}
    ServerMuon(lr=0.1, adam_lr=3e-4, orthogonalizer=ExactPolar()).step(st, agg)
    w = st.params["w"]
    assert torch.allclose(w.abs(), torch.full_like(w, 3e-4), atol=1e-6)  # Adam's first step is ±adam_lr


def test_server_muon_handles_conv_shapes():
    st = ServerState(params={"c": torch.zeros(6, 3, 3, 3)}, exclude_from_muon=set())
    agg = Update(tensors={"c": torch.randn(6, 3, 3, 3)}, meta={"exclude_from_muon": set()})
    ServerMuon(lr=0.1, orthogonalizer=ExactPolar()).step(st, agg)
    assert st.params["c"].shape == (6, 3, 3, 3)
    sv = torch.linalg.svdvals(st.params["c"].reshape(6, -1))
    assert torch.allclose(sv, torch.full_like(sv, 0.1), atol=1e-4)


def test_build_server_opt_from_spec():
    opt = build_server_opt({"name": "muon", "lr": 0.05, "orthogonalizer": {"name": "regularized", "delta": 0.1}})
    assert isinstance(opt, ServerMuon) and opt.lr == 0.05
    assert type(opt.orthogonalizer).__name__ == "RegularizedPolar"
    with pytest.raises(KeyError):
        build_server_opt({"name": "nope"})


def test_server_normsgd_matches_muon_step_norm_but_keeps_direction():
    st_m, agg = state_and_update()
    st_n = ServerState(params={k: v.clone() for k, v in st_m.params.items()}, exclude_from_muon=set())
    ServerMuon(lr=0.1, momentum=0.0, nesterov=False, orthogonalizer=ExactPolar()).step(st_m, agg)
    ServerNormSGD(lr=0.1, momentum=0.0, nesterov=False).step(st_n, agg)
    assert torch.allclose(st_m.params["w"].norm(), st_n.params["w"].norm(), rtol=1e-4)  # same Frobenius step size
    direction = agg.tensors["w"] / agg.tensors["w"].norm()
    step = st_n.params["w"] / st_n.params["w"].norm()
    assert torch.allclose(direction, step, atol=1e-5)  # along the aggregate, not its polar factor
    assert (st_n.params["b"] > 0).all()  # non-matrix parameters: Adam


def test_server_normsgd_step_is_independent_of_aggregate_scale():
    st1, agg = state_and_update()
    st2 = ServerState(params={k: v.clone() for k, v in st1.params.items()}, exclude_from_muon=set())
    small = Update(tensors={k: 1e-3 * v for k, v in agg.tensors.items()}, meta={"exclude_from_muon": set()})
    ServerNormSGD(lr=0.1, momentum=0.0, nesterov=False).step(st1, agg)
    ServerNormSGD(lr=0.1, momentum=0.0, nesterov=False).step(st2, small)
    assert torch.allclose(st1.params["w"], st2.params["w"], atol=1e-5)


def test_build_normsgd_from_spec():
    assert isinstance(build_server_opt({"name": "normsgd", "lr": 0.02}), ServerNormSGD)
