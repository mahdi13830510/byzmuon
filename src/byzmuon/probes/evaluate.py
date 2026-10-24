import torch
from torch import nn

from byzmuon.probes.base import Probe
from byzmuon.registry import register


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total, correct, loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss += float(loss_fn(out, y))
        correct += int((out.argmax(1) == y).sum())
        total += y.numel()
    return correct / max(1, total), loss / max(1, total)


@register("probe", "evaluate")
class Evaluate(Probe):
    """Test accuracy and loss of the global model."""

    def on_round_end(self, server, r: int, metrics: dict) -> None:
        if not self.due(r) or server.eval_loader is None:
            return
        server.model.load_state_dict(server.global_state())
        acc, loss = evaluate(server.model, server.eval_loader, server.device)
        metrics["test_acc"], metrics["test_loss"] = acc, loss
