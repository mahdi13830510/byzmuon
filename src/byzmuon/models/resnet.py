"""CIFAR ResNet-18 with GroupNorm (no running statistics, so the model has no buffers to aggregate)."""

import torch.nn.functional as F
from torch import nn

from byzmuon.registry import register


def _norm(c: int) -> nn.Module:
    return nn.GroupNorm(num_groups=2, num_channels=c)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.n1 = _norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.n2 = _norm(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), _norm(planes))

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


@register("model", "resnet18")
class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, base_width: int = 64, in_channels: int = 3) -> None:
        super().__init__()
        w = base_width
        self.conv1 = nn.Conv2d(in_channels, w, 3, 1, 1, bias=False)
        self.n1 = _norm(w)
        self.in_planes = w
        self.layer1 = self._make(w, 2, 1)
        self.layer2 = self._make(2 * w, 2, 2)
        self.layer3 = self._make(4 * w, 2, 2)
        self.layer4 = self._make(8 * w, 2, 2)
        self.fc = nn.Linear(8 * w, num_classes)

    def _make(self, planes: int, n: int, stride: int) -> nn.Sequential:
        blocks = []
        for s in [stride] + [1] * (n - 1):
            blocks.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


@register("model", "cnn")
class SmallCNN(nn.Module):
    """Fast CIFAR model for development runs on weak GPUs."""

    def __init__(self, num_classes: int = 10, width: int = 32, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, 1, 1, bias=False),
            _norm(width),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, 1, 1, bias=False),
            _norm(2 * width),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * width, 4 * width, 3, 1, 1, bias=False),
            _norm(4 * width),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.hidden = nn.Linear(4 * width, 4 * width)
        self.fc = nn.Linear(4 * width, num_classes)

    def forward(self, x):
        return self.fc(F.relu(self.hidden(self.features(x).flatten(1))))
