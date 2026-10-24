from byzmuon.attacks import baselines, spectral  # noqa: F401  (registers)
from byzmuon.attacks.base import Attack, AttackContext, BudgetSpec, Knowledge
from byzmuon.registry import REGISTRY


def build_attack(spec: dict, num_byz: int) -> Attack:
    spec = dict(spec)
    return REGISTRY.build("attack", spec.pop("name"), num_byz=num_byz, **spec)


__all__ = ["Attack", "AttackContext", "BudgetSpec", "Knowledge", "build_attack"]
