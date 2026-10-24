from byzmuon.probes import evaluate, logging, spectrum  # noqa: F401  (registers)
from byzmuon.probes.base import Probe
from byzmuon.registry import REGISTRY


def build_probes(specs: list[dict]) -> list[Probe]:
    out = []
    for s in specs or []:
        s = dict(s)
        out.append(REGISTRY.build("probe", s.pop("name"), **s))
    return out


__all__ = ["Probe", "build_probes"]
