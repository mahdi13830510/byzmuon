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


def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def expand(sweep_path: str | Path, seeds: list[int] | None = None) -> list[RunConfig]:
    """Precedence, lowest to highest: `base` file < inline `config` < grid values < seed.
    `seeds` overrides the file's seed list (used to add seeds to a finished sweep without editing it)."""
    with open(sweep_path) as f:
        sw = yaml.safe_load(f)
    base_dict: dict = {}
    if sw.get("base"):
        with open(sw["base"]) as f:
            base_dict = yaml.safe_load(f) or {}
    _deep_merge(base_dict, sw.get("config") or {})
    base_path = None
    grid: dict = sw.get("grid") or {}
    seeds = seeds or sw.get("seeds") or [None]
    keys = list(grid)
    cfgs = []
    for values in itertools.product(*[grid[k] for k in keys]):
        for seed in seeds:
            ov = []
            for k, v in zip(keys, values, strict=True):
                # "a,b": [[va, vb], ...] pairs several keys so they vary together instead of as a product
                names = [n.strip() for n in k.split(",")]
                vals = v if len(names) > 1 else [v]
                ov += [
                    f"{n}={yaml.safe_dump(x, default_flow_style=True).strip()}"
                    for n, x in zip(names, vals, strict=True)
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


def run_sweep(
    sweep_path: str | Path, workers: int = 1, force: bool = False, seeds: list[int] | None = None
) -> list[dict]:
    cfgs = expand(sweep_path, seeds=seeds)
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
