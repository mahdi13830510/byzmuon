"""Run configuration: dataclass with nested dicts for registry-built parts, YAML loading, dotted overrides,
and a content hash that names the results directory."""

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunConfig:
    seed: int = 0
    device: str = "auto"
    rounds: int = 10
    local_steps: int = 5
    batch_size: int = 32
    participation: float = 1.0
    n_clients: int = 10
    num_byz: int = 0
    num_byz_assumed: int | None = None  # what the aggregator is told; defaults to num_byz
    data: dict = field(default_factory=lambda: {"name": "synthetic"})
    partition: dict = field(default_factory=lambda: {"name": "iid"})
    model: dict = field(default_factory=lambda: {"name": "mlp"})
    client_opt: dict = field(default_factory=lambda: {"name": "sgd", "lr": 0.05})
    server_opt: dict = field(default_factory=lambda: {"name": "sgd", "lr": 1.0})
    aggregator: dict = field(default_factory=lambda: {"base": {"name": "mean"}, "pre": []})
    attack: dict = field(default_factory=lambda: {"name": "none"})
    probes: list = field(default_factory=lambda: [{"name": "evaluate"}])
    out_dir: str = "results"
    tag: str = ""

    def describe(self) -> str:
        """Short human label for logs: 'FedMuon | lie z=1.0 | median | seed 0'."""
        opt = self.server_opt.get("name", "?")
        if opt == "sgd" and float(self.client_opt.get("local_momentum", 0) or 0) > 0:
            opt = "momentum SGD"
        elif opt == "muon":
            opt = "FedMuon"
        atk = dict(self.attack)
        atk_name = atk.pop("name", "none")
        atk.pop("knowledge", None)
        atk_s = atk_name + "".join(f" {k}={v}" for k, v in sorted(atk.items()))
        agg = self.aggregator.get("base", {}).get("name", "?")
        return f"{opt} | {atk_s} | {agg} | seed {self.seed}"

    def hash(self) -> str:
        d = asdict(self)
        d.pop("out_dir")
        d.pop("device")
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.hash()


def _parse_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        if p.isdigit() and isinstance(cur, list):
            cur = cur[int(p)]
        else:
            cur = cur.setdefault(p, {})
    last = parts[-1]
    if last.isdigit() and isinstance(cur, list):
        cur[int(last)] = value
    elif last == "name" and isinstance(cur, dict) and cur.get("name") not in (None, value):
        # Changing a component's name discards the old component's keyword arguments,
        # so `attack.name=none` on top of {name: subgap, eps: 0.05} does not pass eps to NoAttack.
        cur.clear()
        cur["name"] = value
    else:
        cur[last] = value


def from_dict(d: dict) -> RunConfig:
    known = {f.name for f in fields(RunConfig)}
    unknown = set(d) - known
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    return RunConfig(**copy.deepcopy(d))


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None, base: dict | None = None
) -> RunConfig:
    d: dict = copy.deepcopy(base) if base else {}
    if path is not None:
        with open(path) as f:
            d.update(yaml.safe_load(f) or {})
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key=value, got '{ov}'")
        k, v = ov.split("=", 1)
        _set_dotted(d, k.strip(), _parse_value(v.strip()))
    return from_dict(d)


def to_dict(cfg: RunConfig) -> dict:
    return asdict(cfg)


def save_config(cfg: RunConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)
