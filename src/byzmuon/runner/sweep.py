"""Expand a sweep file into RunConfigs and run them, skipping finished ones.

Sweep YAML:
  base: path/to/config.yaml      # or an inline dict under `config:`
  grid:
    attack.name: [none, signflip, subgap]
    aggregator.base.name: [mean, median, krum]
  seeds: [0, 1, 2]

With workers > 1, each worker process is pinned to one GPU (worker i -> GPU i mod n_gpus) before torch is
imported, so a two-GPU machine runs two jobs at once. Results are cached by config hash, so a sweep can be
restarted at any time and only unfinished runs execute.
"""

import itertools
import multiprocessing as mp
import os
from pathlib import Path

import yaml

from byzmuon.config import RunConfig, load_config


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


def gpu_for_worker(worker: int, n_gpus: int) -> str | None:
    """Value for CUDA_VISIBLE_DEVICES, or None when there is no GPU to pin."""
    return str(worker % n_gpus) if n_gpus > 0 else None


def _count_gpus() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([v for v in visible.split(",") if v.strip()])
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def _worker(worker: int, n_gpus: int, queue, results, force: bool) -> None:
    gpu = gpu_for_worker(worker, n_gpus)
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # must happen before torch is imported in this process
    from byzmuon.runner.run import run

    while True:
        item = queue.get()
        if item is None:
            return
        idx, cfg = item
        try:
            results.put((idx, run(cfg, force=force), None))
        except Exception as e:  # keep the other runs going; report at the end
            results.put((idx, None, f"{type(e).__name__}: {e}"))


def run_sweep(sweep_path: str | Path, workers: int = 1, force: bool = False) -> list[dict]:
    cfgs = expand(sweep_path)
    if workers <= 1:
        from byzmuon.runner.run import run

        return [run(c, force=force) for c in cfgs]
    n_gpus = _count_gpus()
    ctx = mp.get_context("spawn")
    queue, results = ctx.Queue(), ctx.Queue()
    for i, c in enumerate(cfgs):
        queue.put((i, c))
    for _ in range(workers):
        queue.put(None)
    procs = [ctx.Process(target=_worker, args=(w, n_gpus, queue, results, force), daemon=False) for w in range(workers)]
    for p in procs:
        p.start()
    out: list = [None] * len(cfgs)
    errors = []
    for _ in cfgs:
        idx, res, err = results.get()
        out[idx] = res
        if err:
            errors.append((cfgs[idx].hash(), err))
    for p in procs:
        p.join()
    if errors:
        raise RuntimeError("some runs failed:\n" + "\n".join(f"  {h}: {e}" for h, e in errors))
    return out
