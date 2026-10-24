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
