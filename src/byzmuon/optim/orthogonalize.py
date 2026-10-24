from abc import ABC, abstractmethod

import torch
from torch import Tensor

from byzmuon.registry import register


class Orthogonalizer(ABC):
    """Maps a matrix M = U S V to (approximately) its polar factor U V."""

    @abstractmethod
    def __call__(self, m: Tensor) -> Tensor: ...


@register("orthogonalizer", "exact")
class ExactPolar(Orthogonalizer):
    """Refrence implementaion through the SVD. Exact but slow."""

    def __call__(self, m: Tensor) -> Tensor:
        u, _, vt = torch.linalg.svd(m.float(), full_matrices=False)
        return (u @ vt).to(m.dtype)


@register("orthogonalizer", "ns5")
class NewtonSchulz(Orthogonalizer):
    """Muon's iteration: 5 steps of the quitic p(s) = a s + b s³ + c s⁵, in bfloat16."""

    def __init__(
        self,
        steps: int = 5,
        coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0135),
        eps: float = 1e-7,
    ):
        self.steps = steps
        self.a, self.b, self.c = coeffs
        self.eps = eps

    def __call__(self, m: Tensor) -> Tensor:
        x = m.to(torch.bfloat16)
        tall = x.shape[0] > x.shape[1]
        if tall:
            x = x.T
        x = x / (x.norm() + self.eps)
        for _ in range(self.steps):
            xxt = x @ x.T
            poly = self.b * xxt + self.c * (xxt @ xxt)
            x = self.a * x + poly @ x
        if tall:
            x = x.T
        return x.to(m.dtype)


@register("orthogonalizer", "truncated")
class TruncatedPolar(Orthogonalizer):
    """Defense D2a: directions with singular value ≤ delta are dropped, the rest set to 1."""

    def __init__(self, delta: float):
        self.delta = delta

    def __call__(self, m: Tensor) -> Tensor:
        u, s, vt = torch.linalg.svd(m.float(), full_matrices=False)
        keep = (s > self.delta).to(s.dtype)  # 1.0 where s > delta, else 0.0
        return ((u * keep) @ vt).to(m.dtype)  # U · diag(keep) · Vᵀ


@register("orthogonalizer", "regularized")
class RegularizedPolar(Orthogonalizer):
    """Defense D2b: singular value s is mapped to s / sqrt(s² + delta²) instead of 1."""

    def __init__(self, delta: float):
        self.delta = delta

    def __call__(self, m: Tensor) -> Tensor:
        u, s, vt = torch.linalg.svd(m.float(), full_matrices=False)
        scale = s / torch.sqrt(s * s + self.delta**2)
        return ((u * scale) @ vt).to(m.dtype)  # U · diag(scale) · Vᵀ
