import torch

from byzmuon.core.update import Update


def make_update():
    return Update(tensors={"fc_weights": torch.ones(2, 3), "fc_bias": torch.zeros(3)})


def test_names_sorted():
    u: Update = make_update()
    assert u.names() == ["fc_bias", "fc_weights"]


def test_flat_unflat_roundtrip():
    u: Update = make_update()
    v = Update.unflat(u.flat(), u)
    for n in u.names():
        assert torch.equal(u.tensors[n], v.tensors[n])
