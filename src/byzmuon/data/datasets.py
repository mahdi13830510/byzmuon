from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from byzmuon.registry import REGISTRY, register


class SyntheticClassification(Dataset):
    """Gaussian clusters, one per class, with a fixed seed. For tests and smoke runs."""

    def __init__(
        self,
        n: int = 2000,
        in_dim: int = 20,
        num_classes: int = 4,
        noise: float = 1.0,
        seed: int = 0,
        train: bool = True,
    ) -> None:
        g = torch.Generator().manual_seed(seed + (0 if train else 1_000_003))
        means = torch.randn(num_classes, in_dim, generator=torch.Generator().manual_seed(seed)) * 2.0
        self.targets = torch.randint(0, num_classes, (n,), generator=g)
        self.data = means[self.targets] + noise * torch.randn(n, in_dim, generator=g)
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return self.data[i], self.targets[i]


@register("dataset", "synthetic")
def synthetic(train: bool = True, root: str = "", **kw) -> Dataset:
    return SyntheticClassification(train=train, **kw)


class ArrayImageDataset(Dataset):
    """Images stored as a uint8 array [N, H, W, C] plus labels (the .npz written by data/fetch.py)."""

    def __init__(self, npz: str | Path, transform: Callable, num_classes: int) -> None:
        d = np.load(npz)
        self.x, self.targets = d["x"], d["y"].tolist()
        self.transform, self.num_classes = transform, num_classes

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> tuple[Tensor, int]:
        from PIL import Image

        return self.transform(Image.fromarray(self.x[i])), self.targets[i]


def _cifar_transform(train: bool, augment: bool, mean: tuple, std: tuple):
    from torchvision import transforms

    ops = []
    if train and augment:
        ops += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def _cifar(name: str, train: bool, root: str, augment: bool, mean: tuple, std: tuple, num_classes: int) -> Dataset:
    tf = _cifar_transform(train, augment, mean, std)
    npz = Path(root) / name / ("train.npz" if train else "test.npz")
    if npz.exists():  # fast path: arrays fetched from the Hugging Face mirror (`byzmuon fetch cifar10`)
        return ArrayImageDataset(npz, tf, num_classes)
    from torchvision import datasets

    cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    ds = cls(root, train=train, download=True, transform=tf)
    ds.num_classes = num_classes
    return ds


@register("dataset", "cifar10")
def cifar10(train: bool = True, root: str = "data", augment: bool = True, **kw) -> Dataset:
    return _cifar("cifar10", train, root, augment, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10)


@register("dataset", "cifar100")
def cifar100(train: bool = True, root: str = "data", augment: bool = True, **kw) -> Dataset:
    return _cifar("cifar100", train, root, augment, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 100)


def load_dataset(name: str, train: bool, **kw) -> Dataset:
    return REGISTRY.get("dataset", name)(train=train, **kw)


def labels_of(ds: Dataset) -> np.ndarray:
    t = ds.targets
    return t.numpy() if isinstance(t, Tensor) else np.asarray(t)


def cycling_batches(
    ds: Dataset, indices: list[int], batch_size: int, seed: int, num_workers: int = 0
) -> Callable[[], Iterator[tuple[Tensor, Tensor]]]:
    """Returns a function that yields batches forever, reshuffling each pass. One per client."""
    subset = Subset(ds, indices)
    state = {"epoch": 0}

    def gen() -> Iterator[tuple[Tensor, Tensor]]:
        while True:
            g = torch.Generator().manual_seed(seed * 100_003 + state["epoch"])
            state["epoch"] += 1
            loader = DataLoader(
                subset,
                batch_size=min(batch_size, len(subset)),
                shuffle=True,
                generator=g,
                num_workers=num_workers,
                drop_last=False,
            )
            yield from loader

    return gen


def eval_loader(ds: Dataset, batch_size: int = 512, num_workers: int = 0) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
