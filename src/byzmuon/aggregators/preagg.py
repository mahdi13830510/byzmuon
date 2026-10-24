"""Pre-aggregation transforms: NNM, Frobenius clipping, and spectral clipping (defense D1)."""

import torch

from byzmuon.aggregators.base import PreAgg
from byzmuon.core.update import Update
from byzmuon.registry import register


@register("preagg", "nnm")
class NNM(PreAgg):
    """Nearest Neighbor Mixing (Allouah et al. 2023): replace each update by the mean of its n − f nearest."""

    def transform(self, updates: list[Update]) -> list[Update]:
        n = len(updates)
        k = max(1, n - self.num_byz)
        vecs = torch.stack([u.flat() for u in updates])
        nn_idx = torch.cdist(vecs, vecs).topk(k, dim=1, largest=False).indices
        mixed = torch.stack([vecs[nn_idx[i]].mean(dim=0) for i in range(n)])
        return [Update.unflat(mixed[i], like=updates[i]) for i in range(n)]


@register("preagg", "fro_clip")
class FroClip(PreAgg):
    """Clip each update's Frobenius norm to tau, or to the median norm across clients if tau is None."""

    def __init__(self, num_byz: int = 0, tau: float | None = None) -> None:
        super().__init__(num_byz)
        self.tau = tau

    def transform(self, updates: list[Update]) -> list[Update]:
        norms = [u.fro_norm() for u in updates]
        tau = self.tau if self.tau is not None else float(torch.tensor(norms).median())
        return [u if nrm <= tau else u * (tau / nrm) for u, nrm in zip(updates, norms, strict=True)]


@register("preagg", "spectral_clip")
class SpectralClip(PreAgg):
    """Defense D1: clip every matrix layer's singular values to at most tau (absolute) or
    tau_rel × the median spectral norm of that layer across clients. Non-matrix layers untouched."""

    def __init__(self, num_byz: int = 0, tau: float | None = None, tau_rel: float | None = 1.0) -> None:
        super().__init__(num_byz)
        self.tau, self.tau_rel = tau, tau_rel

    def transform(self, updates: list[Update]) -> list[Update]:
        names = updates[0].matrix_names()
        spec = {n: torch.tensor([u.spec_norms()[n] for u in updates]) for n in names}
        out = []
        for u in updates:
            new = dict(u.tensors)
            for n in names:
                tau = self.tau if self.tau is not None else self.tau_rel * float(spec[n].median())
                t = u.tensors[n]
                m = u.as_matrix(n).float()
                if float(torch.linalg.matrix_norm(m, ord=2)) <= tau:
                    continue
                uu, s, vt = torch.linalg.svd(m, full_matrices=False)
                new[n] = ((uu * s.clamp(max=tau)) @ vt).reshape(t.shape).to(t.dtype)
            out.append(Update(tensors=new, extras=u.extras, meta=u.meta))
        return out
