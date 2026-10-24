"""Build a Server from a RunConfig and run it, caching results by config hash."""

import json
import time
from pathlib import Path

import numpy as np
import torch

import byzmuon.aggregators  # noqa: F401
import byzmuon.attacks  # noqa: F401
import byzmuon.models  # noqa: F401
import byzmuon.probes  # noqa: F401
from byzmuon.aggregators import build_aggregator
from byzmuon.attacks import build_attack
from byzmuon.config import RunConfig, save_config, to_dict
from byzmuon.core.client import Client
from byzmuon.core.device import pick_device
from byzmuon.core.server import Server
from byzmuon.data import cycling_batches, eval_loader, labels_of, load_dataset, make_partition
from byzmuon.optim import build_server_opt, make_local_optimizer
from byzmuon.probes import build_probes
from byzmuon.probes.logging import JsonlLogger
from byzmuon.registry import REGISTRY


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_server(cfg: RunConfig, log_path: Path | None = None) -> Server:
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)

    data_spec = dict(cfg.data)
    data_name = data_spec.pop("name")
    train_ds = load_dataset(data_name, train=True, **data_spec)
    test_ds = load_dataset(data_name, train=False, **{k: v for k, v in data_spec.items() if k != "augment"})
    labels = labels_of(train_ds)
    num_classes = int(labels.max()) + 1
    part_spec = dict(cfg.partition)
    parts = make_partition(part_spec.pop("name"), labels, cfg.n_clients, seed=cfg.seed, **part_spec)

    model_spec = dict(cfg.model)
    model = REGISTRY.build("model", model_spec.pop("name"), **model_spec).to(device)

    attack = build_attack(cfg.attack, num_byz=cfg.num_byz)
    rng = np.random.default_rng(cfg.seed)
    byz_ids = set(rng.choice(cfg.n_clients, size=cfg.num_byz, replace=False).tolist()) if cfg.num_byz else set()
    label_map = attack.label_map(num_classes) if attack.poisons_data else None

    make_opt = make_local_optimizer(cfg.client_opt)
    clients = [
        Client(
            i,
            model,
            cycling_batches(train_ds, parts[i], cfg.batch_size, seed=cfg.seed * 1000 + i),
            make_opt,
            cfg.local_steps,
            device,
            label_map=label_map if i in byz_ids else None,
        )
        for i in range(cfg.n_clients)
    ]

    f_assumed = cfg.num_byz if cfg.num_byz_assumed is None else cfg.num_byz_assumed
    aggregator = build_aggregator(cfg.aggregator, num_byz=f_assumed)
    server_opt = build_server_opt(cfg.server_opt)
    probes = build_probes(cfg.probes)
    if log_path is not None:
        probes.append(JsonlLogger(log_path))
    return Server(
        model,
        clients,
        byz_ids,
        aggregator,
        attack,
        server_opt,
        probes,
        participation=cfg.participation,
        seed=cfg.seed,
        eval_loader=eval_loader(test_ds),
        device=device,
    )


def run(cfg: RunConfig, force: bool = False) -> dict:
    run_dir = cfg.run_dir()
    final_path = run_dir / "final.json"
    if final_path.exists() and not force:
        return json.loads(final_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.jsonl").unlink(missing_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    t0 = time.time()
    server = build_server(cfg, log_path=run_dir / "metrics.jsonl")
    history = server.run(cfg.rounds)
    accs = [h["test_acc"] for h in history if "test_acc" in h]
    final = {
        "hash": cfg.hash(),
        "config": to_dict(cfg),
        "elapsed_s": time.time() - t0,
        "final": {k: v for k, v in history[-1].items() if isinstance(v, (int, float))},
        "best_test_acc": max(accs) if accs else None,
        "final_test_acc": accs[-1] if accs else None,
    }
    final_path.write_text(json.dumps(final, indent=1, default=float))
    return final
