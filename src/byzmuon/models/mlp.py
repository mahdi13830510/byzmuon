from torch import nn

from byzmuon.registry import register


@register("model", "mlp")
class MLP(nn.Module):
    """Small fully connected net for synthetic data and smoke tests."""

    def __init__(self, in_dim: int = 20, hidden: int = 64, num_classes: int = 4, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))
