import json

import yaml

from byzmuon.analysis import make_figures, summarize
from byzmuon.config import load_config, to_dict
from byzmuon.runner.results import load_results


def fake_run(root, opt, attack, agg, seed, accs, extra_metrics=None):
    cfg = load_config(
        base={
            "server_opt": {"name": opt},
            "attack": {"name": attack},
            "seed": seed,
            "aggregator": {"base": {"name": agg}, "pre": []},
            "out_dir": str(root),
        }
    )
    d = cfg.run_dir()
    d.mkdir(parents=True)
    yaml.safe_dump(to_dict(cfg), (d / "config.yaml").open("w"))
    with (d / "metrics.jsonl").open("w") as f:
        for r, a in enumerate(accs):
            m = {"round": r, "test_acc": a, "train_loss": 1.0 - a}
            m.update(extra_metrics or {})
            f.write(json.dumps(m) + "\n")
    (d / "final.json").write_text(
        json.dumps(
            {
                "hash": cfg.hash(),
                "config": to_dict(cfg),
                "elapsed_s": 1.0,
                "final": {"round": len(accs) - 1, "test_acc": accs[-1]},
                "best_test_acc": max(accs),
                "final_test_acc": accs[-1],
            }
        )
    )


def test_summary_and_figures(tmp_path):
    amp = {
        "amp/w/tail/0.01": 20.0,
        "amp/w/random/0.01": 15.0,
        "amp/mean/tail/0.01": 20.0,
        "amp/mean/random/0.01": 15.0,
        "hspec/w/erank": 8.0,
        "hspec/w/tail_frac": 0.4,
    }
    for seed in (0, 1):
        for agg in ("mean", "median"):
            fake_run(tmp_path, "muon", "none", agg, seed, [0.2, 0.5, 0.8 - 0.01 * seed], amp)
            fake_run(tmp_path, "muon", "lie", agg, seed, [0.2, 0.4, 0.5])
            fake_run(tmp_path, "sgd", "none", agg, seed, [0.2, 0.5, 0.7])
            fake_run(tmp_path, "sgd", "lie", agg, seed, [0.2, 0.45, 0.6])
    df = load_results(tmp_path)
    assert len(df) == 16
    s = summarize(
        df.assign(
            opt=df["cfg.server_opt.name"],
            agg=df["cfg.aggregator.base.name"],
            attack=df["cfg.attack.name"],
            seed=df["cfg.seed"],
        )
    )
    muon_lie = s[(s["opt"] == "muon") & (s["attack"] == "lie") & (s["agg"] == "mean")].iloc[0]
    assert abs(muon_lie["drop"] - (0.795 - 0.5)) < 1e-9 and muon_lie["count"] == 2
    written = make_figures(tmp_path)
    names = {p.name for p in written}
    assert {
        "summary.csv",
        "summary.md",
        "curves_mlp.png",
        "curves_mlp.pdf",
        "drop.png",
        "amplification.png",
        "spectrum.png",
    } <= names
    assert all(p.stat().st_size > 1000 for p in written if p.suffix in (".png", ".pdf"))
    md = (tmp_path / "figures" / "summary.md").read_text()
    assert "| mlp | mean | lie | FedMuon |" in md and "+29.5 pp" in md
