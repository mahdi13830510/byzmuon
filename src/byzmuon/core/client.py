from collections.abc import Callable, Iterator

import torch
from torch import Tensor, nn

from byzmuon.core.update import Update
from byzmuon.models.params import muon_excluded, trainable_names


class Client:
    """Trains the shared model on local data for local_steps and returns new − old as an Update.

    label_map, if given, rewrites the labels before training (used by data-poisoning attacks).
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        batches: Callable[[], Iterator[tuple[Tensor, Tensor]]],
        make_optimizer: Callable[[nn.Module], torch.optim.Optimizer],
        local_steps: int,
        device: torch.device,
        loss_fn: nn.Module | None = None,
        label_map: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.id = client_id
        self.model = model
        self.batches = batches
        self.make_optimizer = make_optimizer
        self.local_steps = local_steps
        self.device = device
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.label_map = label_map
        self.excluded = muon_excluded(model)
        self.param_names = trainable_names(model)
        self.last_loss = float("nan")

    def local_train(self, global_state: dict[str, Tensor]) -> Update:
        self.model.load_state_dict(global_state)
        self.model.train()
        opt = self.make_optimizer(self.model)
        it = self.batches()
        losses = []
        for _ in range(self.local_steps):
            x, y = next(it)
            x, y = x.to(self.device), y.to(self.device)
            if self.label_map is not None:
                y = self.label_map(y)
            opt.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.model(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        self.last_loss = sum(losses) / len(losses) if losses else float("nan")
        new_state = self.model.state_dict()
        delta = {n: (new_state[n] - global_state[n]).detach().clone() for n in self.param_names}
        return Update(
            tensors=delta, meta={"client_id": self.id, "exclude_from_muon": self.excluded, "train_loss": self.last_loss}
        )
