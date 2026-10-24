"""Client-side Muon (Mode B: polarize-then-aggregate), following KellerJordan/Muon."""

import torch
from torch import nn

from byzmuon.models.params import muon_excluded
from byzmuon.optim.orthogonalize import NewtonSchulz, Orthogonalizer


class MuonLocal(torch.optim.Optimizer):
    """Muon for matrix params (ndim >= 2, not excluded), AdamW for the rest."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        orthogonalizer: Orthogonalizer | None = None,
        adam_lr: float = 3e-4,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        excluded: set[str] | None = None,
    ) -> None:
        excluded = muon_excluded(model) if excluded is None else excluded
        muon_p, other_p = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (muon_p if p.ndim >= 2 and n not in excluded else other_p).append(p)
        groups = [
            {
                "params": muon_p,
                "use_muon": True,
                "lr": lr,
                "momentum": momentum,
                "nesterov": nesterov,
                "weight_decay": weight_decay,
            },
            {
                "params": other_p,
                "use_muon": False,
                "lr": adam_lr,
                "betas": tuple(adam_betas),
                "eps": adam_eps,
                "weight_decay": weight_decay,
            },
        ]
        super().__init__(groups, {})
        self.orthogonalizer = orthogonalizer or NewtonSchulz()

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                if group["use_muon"]:
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(g)
                    buf = st["buf"]
                    buf.lerp_(g, 1 - group["momentum"])
                    upd = g.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                    m2d = upd.reshape(upd.shape[0], -1)
                    o = self.orthogonalizer(m2d).reshape(p.shape)
                    scale = max(1.0, m2d.shape[0] / m2d.shape[1]) ** 0.5
                    p.add_(o.to(p.dtype), alpha=-group["lr"] * scale)
                else:
                    if "step" not in st:
                        st["step"], st["m"], st["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                    st["step"] += 1
                    b1, b2 = group["betas"]
                    st["m"].mul_(b1).add_(g, alpha=1 - b1)
                    st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                    m_hat = st["m"] / (1 - b1 ** st["step"])
                    v_hat = st["v"] / (1 - b2 ** st["step"])
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(group["eps"]), value=-group["lr"])
        return None
