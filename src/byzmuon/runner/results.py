import json
from pathlib import Path

import pandas as pd


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v if not isinstance(v, list) else json.dumps(v)
    return out


def load_results(results_dir: str | Path = "results") -> pd.DataFrame:
    rows = []
    for p in Path(results_dir).glob("*/final.json"):
        d = json.loads(p.read_text())
        row = {
            "hash": d["hash"],
            "elapsed_s": d["elapsed_s"],
            "best_test_acc": d["best_test_acc"],
            "final_test_acc": d["final_test_acc"],
        }
        row.update(_flatten(d["config"], "cfg."))
        row.update(_flatten(d["final"], "final."))
        rows.append(row)
    return pd.DataFrame(rows)


def load_metrics(run_dir: str | Path) -> pd.DataFrame:
    lines = Path(run_dir, "metrics.jsonl").read_text().splitlines()
    return pd.DataFrame([json.loads(line) for line in lines if line.strip()])
