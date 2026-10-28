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
import time
from pathlib import Path

import yaml

from byzmuon.config import RunConfig, load_config


def _deep_merge(dst: dict, src: dict) -> dict:
    """Merge src into dst. A component spec (a dict with a "name") replaces the old one wholesale, so that
    `attack: {name: none}` does not inherit the `eps` of a base `attack: {name: subgap, eps: 0.05}`."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict) and "name" not in v:
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


def _worker(worker: int, n_gpus: int, queue, results, force: bool, deadline: float | None) -> None:
    gpu = gpu_for_worker(worker, n_gpus)
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # must happen before torch is imported in this process
    from byzmuon.runner.run import run

    while True:
        item = queue.get()
        if item is None:
            return
        idx, cfg = item
        if _is_cached(cfg) and not force:
            results.put((idx, run(cfg), "cached"))
            continue
        if deadline is not None and time.time() > deadline:
            results.put((idx, None, "skipped: time budget exhausted"))
            continue
        print(
            f"[start] {cfg.describe()} ({cfg.hash()}) on worker {worker}" + (f", GPU {gpu}" if gpu else ""), flush=True
        )
        try:
            results.put((idx, run(cfg, force=force), None))
        except Exception as e:  # keep the other runs going; report at the end
            results.put((idx, None, f"{type(e).__name__}: {e}"))


def _is_cached(cfg: RunConfig) -> bool:
    return (cfg.run_dir() / "final.json").exists()


def run_sweep(
    sweep_path: str | Path,
    workers: int = 1,
    force: bool = False,
    seeds: list[int] | None = None,
    hours: float | None = None,
) -> list[dict]:
    """Run every config of the sweep. `hours` is a wall-clock budget: once it is used up, no NEW run is started
    (runs already going finish), so a time-limited session ends cleanly with its finished runs on disk. Skipped
    runs are reported, not raised; rerun the same sweep later to finish them (finished runs are cached)."""
    cfgs = expand(sweep_path, seeds=seeds)
    deadline = time.time() + hours * 3600 if hours else None
    t_start = time.time()
    n_cached = sum(_is_cached(c) and not force for c in cfgs)
    print(
        f"sweep {Path(sweep_path).name}: {len(cfgs)} runs, {n_cached} already finished, {len(cfgs) - n_cached} to run, "
        f"{workers} worker(s)" + (f", budget {hours:g} h" if hours else ""),
        flush=True,
    )
    for c in cfgs:
        print(f"  {'done' if _is_cached(c) and not force else 'todo'}  {c.describe()}  ({c.hash()})", flush=True)
    if workers <= 1:
        from byzmuon.runner.run import run

        out = []
        for c in cfgs:
            if deadline is not None and time.time() > deadline and not _is_cached(c):
                print(f"[skipped] {c.describe()}: time budget exhausted", flush=True)
                continue
            out.append(run(c, force=force))
        return out
    n_gpus = _count_gpus()
    ctx = mp.get_context("spawn")
    queue, results = ctx.Queue(), ctx.Queue()
    for i, c in enumerate(cfgs):
        queue.put((i, c))
    for _ in range(workers):
        queue.put(None)
    procs = [
        ctx.Process(target=_worker, args=(w, n_gpus, queue, results, force, deadline), daemon=False)
        for w in range(workers)
    ]
    for p in procs:
        p.start()
    out: list = [None] * len(cfgs)
    errors, skipped, durations = [], [], []
    n_done = 0
    for _ in cfgs:
        idx, res, err = results.get()
        out[idx] = res
        c = cfgs[idx]
        if err == "cached":
            continue
        n_done += 1
        if err and err.startswith("skipped"):
            skipped.append(c.hash())
            print(f"[skipped] {c.describe()}: time budget exhausted", flush=True)
        elif err:
            errors.append((c.hash(), err))
            print(f"[failed] {c.describe()}: {err}", flush=True)
        else:
            dur = float(res.get("elapsed_s", 0.0)) if isinstance(res, dict) else 0.0
            durations.append(dur)
            acc = res.get("final_test_acc") if isinstance(res, dict) else None
            left = len(cfgs) - n_cached - n_done
            eta = f", ~{max(durations) * left / workers / 3600:.1f} h for the remaining {left}" if left else ""
            print(
                f"[done {n_done}/{len(cfgs) - n_cached}] {c.describe()}: final acc "
                f"{(acc if acc is not None else float('nan')):.4f} in {dur / 60:.0f} min; "
                f"{(time.time() - t_start) / 3600:.1f} h elapsed{eta}",
                flush=True,
            )
    for p in procs:
        p.join()
    if skipped:
        print(
            f"time budget exhausted: {len(skipped)} of {len(cfgs)} runs not started; rerun the sweep to finish them",
            flush=True,
        )
    if errors:
        raise RuntimeError("some runs failed:\n" + "\n".join(f"  {h}: {e}" for h, e in errors))
    return [o for o in out if o is not None]
