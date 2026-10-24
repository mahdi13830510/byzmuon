"""The federated round loop. Template method with observer hooks; everything else is injected."""

import random

import torch
from torch import Tensor, nn

from byzmuon.aggregators.base import Aggregator
from byzmuon.attacks.base import Attack, AttackContext, Knowledge
from byzmuon.core.client import Client
from byzmuon.core.update import Update
from byzmuon.models.params import muon_excluded, trainable_names
from byzmuon.optim.server_opt import ServerOptimizer, ServerState
from byzmuon.probes.base import Probe


class Server:
    def __init__(
        self,
        model: nn.Module,
        clients: list[Client],
        byz_ids: set[int],
        aggregator: Aggregator,
        attack: Attack,
        server_opt: ServerOptimizer,
        probes: list[Probe] | None = None,
        participation: float = 1.0,
        seed: int = 0,
        eval_loader=None,
        device: torch.device | None = None,
    ) -> None:
        buffers = [n for n, _ in model.named_buffers()]
        if buffers:
            raise ValueError(f"models with buffers are not supported (use GroupNorm/LayerNorm): {buffers[:3]}")
        self.model, self.clients, self.byz_ids = model, clients, set(byz_ids)
        self.aggregator, self.attack, self.server_opt = aggregator, attack, server_opt
        self.probes = probes or []
        self.participation, self.eval_loader = participation, eval_loader
        self.device = device or next(model.parameters()).device
        self.rng = random.Random(seed)
        names = trainable_names(model)
        sd = model.state_dict()
        self.state = ServerState(
            params={n: sd[n].detach().clone() for n in names}, exclude_from_muon=muon_excluded(model)
        )
        # Research-only bookkeeping for probes. Aggregators and server optimizers must never read these.
        self.last_honest: list[Update] = []
        self.last_malicious: list[Update] = []

    def global_state(self) -> dict[str, Tensor]:
        return self.state.params

    def sample_clients(self) -> list[Client]:
        k = max(1, round(self.participation * len(self.clients)))
        return self.rng.sample(self.clients, k) if k < len(self.clients) else list(self.clients)

    def run_round(self, r: int) -> dict:
        for p in self.probes:
            p.on_round_start(self, r)
        sampled = self.sample_clients()
        honest_clients = [c for c in sampled if c.id not in self.byz_ids]
        byz_clients = [c for c in sampled if c.id in self.byz_ids]
        gs = self.global_state()
        honest = [c.local_train(gs) for c in honest_clients]
        byz_own = [c.local_train(gs) for c in byz_clients]
        malicious: list[Update] = []
        if byz_clients and honest:
            ctx = AttackContext(
                round=r,
                knowledge=self.attack.knowledge,
                byz_own_updates=byz_own,
                benign_updates=honest if self.attack.knowledge is Knowledge.OMNISCIENT else None,
                aggregator=self.aggregator,
                server_state=self.state,
                budget=self.attack.budget,
                n_mal=len(byz_clients),
            )
            malicious = self.attack.craft(ctx)
            assert len(malicious) == len(byz_clients), "attack returned the wrong number of updates"
        self.last_honest, self.last_malicious = honest, malicious
        # The server sees anonymous updates: identity and training metadata are stripped, order shuffled.
        updates = [
            Update(
                tensors=u.tensors, extras=u.extras, meta={"exclude_from_muon": u.meta.get("exclude_from_muon", set())}
            )
            for u in honest + malicious
        ]
        self.rng.shuffle(updates)
        for p in self.probes:
            p.on_updates(self, r, updates)
        agg = self.aggregator.aggregate(updates)
        for p in self.probes:
            p.on_aggregated(self, r, agg)
        self.state = self.server_opt.step(self.state, agg)
        metrics = {
            "round": r,
            "n_honest": len(honest),
            "n_byz": len(malicious),
            "train_loss": sum(c.last_loss for c in honest_clients) / max(1, len(honest_clients)),
            "agg_fro": agg.fro_norm(),
        }
        for p in self.probes:
            p.on_round_end(self, r, metrics)
        self.state.history.append({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        return metrics

    def run(self, rounds: int) -> list[dict]:
        out = [self.run_round(r) for r in range(rounds)]
        for p in self.probes:
            p.close()
        return out
