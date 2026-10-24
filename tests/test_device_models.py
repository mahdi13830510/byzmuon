import torch

from byzmuon.core.device import pick_device
from byzmuon.models.params import muon_excluded, trainable_names
from byzmuon.registry import REGISTRY


def test_pick_device_override_and_auto():
    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type in {"cpu", "cuda", "xpu"}


def test_muon_excluded_marks_head_and_embeddings():
    from torch import nn

    m = nn.Sequential(nn.Embedding(10, 4), nn.Linear(4, 8), nn.Linear(8, 2))
    assert muon_excluded(m) == {"0.weight", "2.weight"}


def test_models_build_and_forward():
    import byzmuon.models  # noqa: F401

    mlp = REGISTRY.build("model", "mlp", in_dim=5, hidden=7, num_classes=3)
    assert mlp(torch.randn(2, 5)).shape == (2, 3)
    cnn = REGISTRY.build("model", "cnn", num_classes=10, width=4)
    assert cnn(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    rn = REGISTRY.build("model", "resnet18", num_classes=10, base_width=8)
    assert rn(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert not list(rn.named_buffers()), "GroupNorm models must have no buffers"
    assert muon_excluded(rn) == {"fc.weight"}
    assert len(trainable_names(rn)) == len(list(rn.parameters()))
