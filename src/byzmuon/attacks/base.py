from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from torch import Tensor

from byzmuon.aggregators.mean import Mean
from byzmuon.core.update import Update


class Knowledge(Enum):
    OMNISCIENT = "omniscient"  # sees all honest updates this round
    PARTIAL = "partial"  # sees only its own clients' honest updates
    AGNOSTIC = "agnostic"  # sees nothing from this round


@dataclass
class BudgetSpec:
    """Norm limits on each malicious update. None means unlimited."""

    fro_rel: float | None = None  # max Frobenius norm relative to the reference update's norm
    fro_abs: float | None = None  # max Frobenius norm, absolute


@dataclass
class AttackContext:
    round: int
    knowledge: Knowledge
    byz_own_updates: list[Update]  # what the Byzantine clients computed on their own data
    benign_updates: list[Update] | None = None  # only filled for OMNISCIENT
    aggregator: Any = None  # for adaptive attacks
    server_state: Any = None  # global params, server momentum, history
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    n_mal: int | None = None  # how many malicious updates to produce this round (partial participation)

    def reference(self) -> Update:
        """Best available estimate of the honest mean at this knowledge level."""
        if self.knowledge is Knowledge.OMNISCIENT:
            assert self.benign_updates, "omniscient context without benign updates"
            return Mean().aggregate(self.benign_updates)
        if self.knowledge is Knowledge.PARTIAL:
            return Mean().aggregate(self.byz_own_updates)
        raise PermissionError("agnostic attacker has no reference update")

    def reference_set(self) -> list[Update]:
        """The individual updates the attacker may look at (for std / distance based attacks)."""
        if self.knowledge is Knowledge.OMNISCIENT:
            assert self.benign_updates, "omniscient context without benign updates"
            return self.benign_updates
        if self.knowledge is Knowledge.PARTIAL:
            return self.byz_own_updates
        raise PermissionError("agnostic attacker has no reference updates")


class Attack(ABC):
    poisons_data: bool = False  # True for attacks that corrupt the Byzantine clients' training data

    def __init__(
        self, num_byz: int, knowledge: Knowledge | str = Knowledge.OMNISCIENT, budget: BudgetSpec | dict | None = None
    ) -> None:
        self.num_byz = num_byz
        self.knowledge = Knowledge(knowledge) if isinstance(knowledge, str) else knowledge
        self.budget = BudgetSpec(**budget) if isinstance(budget, dict) else (budget or BudgetSpec())

    @abstractmethod
    def craft(self, ctx: AttackContext) -> list[Update]:
        """Return exactly self.count(ctx) malicious updates."""

    def count(self, ctx: AttackContext) -> int:
        return ctx.n_mal if ctx.n_mal is not None else self.num_byz

    def label_map(self, num_classes: int) -> Callable[[Tensor], Tensor] | None:
        return None

    def apply_budget(self, u: Update, ctx: AttackContext) -> Update:
        """Scale u down so it respects ctx.budget. Attacks call this on every update they return."""
        limit = None
        if ctx.budget.fro_abs is not None:
            limit = ctx.budget.fro_abs
        if ctx.budget.fro_rel is not None:
            rel = ctx.budget.fro_rel * ctx.reference().fro_norm()
            limit = rel if limit is None else min(limit, rel)
        if limit is None:
            return u
        norm = u.fro_norm()
        return u if norm <= limit else u * (limit / norm)
