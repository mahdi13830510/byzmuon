import torch
from conftest import tiny_model

from byzmuon.core.client import Client
from byzmuon.core.update import Update


def batches():
    g = torch.Generator().manual_seed(0)
    while True:
        x = torch.randn(32, 8, generator=g)
        yield x, (x[:, 0] > 0).long() + (x[:, 1] > 0).long()


def make_client(local_steps=5, lr=0.1, label_map=None):
    return Client(
        0,
        tiny_model(),
        batches,
        lambda m: torch.optim.SGD(m.parameters(), lr=lr),
        local_steps,
        torch.device("cpu"),
        label_map=label_map,
    )


def test_update_has_model_shapes_and_meta():
    c = make_client()
    u = c.local_train(c.model.state_dict())
    assert isinstance(u, Update)
    assert set(u.names()) == {n for n, _ in c.model.named_parameters()}
    assert u.tensors["0.weight"].shape == (16, 8)
    assert u.meta["exclude_from_muon"] == {"2.weight"}
    assert u.matrix_names() == ["0.weight"]
    assert u.meta["client_id"] == 0 and u.meta["train_loss"] > 0


def test_zero_steps_gives_zero_update():
    c = make_client(local_steps=0)
    assert c.local_train(c.model.state_dict()).fro_norm() == 0.0


def test_update_does_not_alias_model():
    c = make_client()
    state = {k: v.clone() for k, v in c.model.state_dict().items()}
    u = c.local_train(state)
    before = u.tensors["0.weight"].clone()
    c.local_train(state)
    assert torch.equal(u.tensors["0.weight"], before)


def test_label_map_changes_update():
    state = {k: v.clone() for k, v in tiny_model().state_dict().items()}
    honest = make_client().local_train(state)
    flipped = make_client(label_map=lambda y: 2 - y).local_train(state)
    assert (honest - flipped).fro_norm() > 1e-3


def test_local_momentum_is_persistent_ema_of_deltas():
    plain = make_client(local_steps=3)
    mom = make_client(local_steps=3)
    mom.momentum = 0.9
    state = {k: v.clone() for k, v in plain.model.state_dict().items()}
    mom.model.load_state_dict(state)
    d1 = plain.local_train(state)
    m1 = mom.local_train(state)
    for n in d1.names():  # first round: buffer = 0.1 * delta
        assert torch.allclose(m1.tensors[n], 0.1 * d1.tensors[n], atol=1e-6)
    m2 = mom.local_train(state)
    d2 = plain.local_train(state)
    for n in d1.names():  # second round from the same global state: 0.9 * buf + 0.1 * delta2
        assert torch.allclose(m2.tensors[n], 0.9 * m1.tensors[n] + 0.1 * d2.tensors[n], atol=1e-6)
    first = next(iter(d1.names()))
    assert m2.tensors[first].data_ptr() != mom.buf[first].data_ptr()  # the sent update is a copy of the buffer


def test_local_momentum_from_config(tmp_path):
    from byzmuon.config import load_config
    from byzmuon.runner.run import build_server

    cfg = load_config(
        base={
            "device": "cpu",
            "rounds": 1,
            "n_clients": 3,
            "num_byz": 0,
            "data": {"name": "synthetic", "n": 60},
            "model": {"name": "mlp", "hidden": 8},
            "client_opt": {"name": "sgd", "lr": 0.1, "local_momentum": 0.5},
            "probes": [],
            "out_dir": str(tmp_path),
        }
    )
    server = build_server(cfg)
    assert all(c.momentum == 0.5 for c in server.clients)
