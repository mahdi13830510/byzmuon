from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class Update:
    tensors: dict[str, Tensor]
    extras: dict[str, dict[str, Tensor]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def names(self) -> list[str]:
        """
        Returns Keys of the object in sorted order.
        """
        return sorted(self.tensors)

    def matrix_names(self) -> list[str]:
        """
        Muon works on 2D matrices so this function returns names of 2D matrices.
        """
        excluded = self.meta.get("exclude_from_muon", set())
        return [n for n in self.names() if self.tensors[n].ndim >= 2 and n not in excluded]

    def map(self, fn: Callable[[str, Tensor], Tensor]) -> "Update":
        """
        For every changes needed to apply to all tensors this function is used.
        """
        new: dict[str, Tensor] = {n: fn(n, t) for n, t in self.tensors.items()}
        return Update(tensors=new, extras=self.extras, meta=self.meta)

    def zip_with(self, other: "Update", fn: Callable[[str, Tensor, Tensor], Tensor]) -> "Update":
        """
        Works on two changes paring tensors by names.
        """
        if self.names() != other.names():
            raise ValueError("updates have diffrent paramter sets")
        new: dict[str, Tensor] = {n: fn(n, self.tensors[n], other.tensors[n]) for n in self.names()}
        return Update(tensors=new, extras=self.extras, meta=self.meta)

    def flat(self) -> Tensor:
        return torch.cat([self.tensors[n].reshape(-1) for n in self.names()])

    @staticmethod
    def unflat(vec: Tensor, like: "Update") -> "Update":
        out: dict[str, Tensor] = {}
        offset = 0
        for n in like.names():
            t: Tensor = like.tensors[n]
            k: int = t.numel()
            out[n] = vec[offset : offset + k].reshape(t.shape)
            offset += k
        if offset != vec.numel():
            raise ValueError("vector length does not match the template")
        return Update(tensors=out, extras=like.extras, meta=like.meta)

    def fro_norm(self) -> float:
        total = sum(t.pow(2).sum() for t in self.tensors.values())
        return float(total.sqrt())

    def as_matrix(self, name: str) -> Tensor:
        """2D view of a matrix-like parameter: conv kernels [out, in, k, k] become [out, in*k*k]."""
        t = self.tensors[name]
        return t.reshape(t.shape[0], -1)

    def spec_norms(self) -> dict[str, float]:
        return {n: float(torch.linalg.matrix_norm(self.as_matrix(n).float(), ord=2)) for n in self.matrix_names()}

    def clone(self) -> "Update":
        return self.map(lambda n, t: t.clone())

    @staticmethod
    def stack(updates: list["Update"], name: str) -> Tensor:
        return torch.stack([u.tensors[name] for u in updates])

    def __add__(self, other: "Update") -> "Update":
        return self.zip_with(other, lambda n, a, b: a + b)

    def __sub__(self, other: "Update") -> "Update":
        return self.zip_with(other, lambda n, a, b: a - b)

    def __mul__(self, scalar: float) -> "Update":
        return self.map(lambda n, t: t * scalar)

    __rmul__ = __mul__
