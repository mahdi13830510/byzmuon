"""Expand a sweep file into RunConfigs and run them, skipping finished ones.

Sweep YAML:
  base: path/to/config.yaml      # or an inline dict under `config:`
  grid:
    attack.name: [none, signflip, subgap]
    aggregator.base.name: [mean, median, krum]
  seeds: [0, 1, 2]
"""

import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from byzmuon.config import RunConfig, load_config
from byzmuon.runner.run import run


def expand(sweep_path: str | Path) -> list[RunConfig]:
    with open(sweep_path) as f:
        sw = yaml.safe_load(f)
    base_path = sw.get("base")
    base_dict = sw.get("config") or {}
    grid: dict = sw.get("grid") or {}
    seeds = sw.get("seeds") or [None]
    keys = list(grid)
    cfgs = []
    for values in itertools.product(*[grid[k] for k in keys]):
        for seed in seeds:
            ov = [
                f"{k}={yaml.safe_dump(v, default_flow_style=True).strip()}" for k, v in zip(keys, values, strict=True)
            ]
            if seed is not None:
                ov.append(f"seed={seed}")
            cfgs.append(load_config(base_path, ov, base=base_dict))
    return cfgs


def _run_one(args):
    cfg, force = args
    return run(cfg, force=force)


def run_sweep(sweep_path: str | Path, workers: int = 1, force: bool = False) -> list[dict]:
    cfgs = expand(sweep_path)
    if workers <= 1:
        return [run(c, force=force) for c in cfgs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_run_one, [(c, force) for c in cfgs]))
