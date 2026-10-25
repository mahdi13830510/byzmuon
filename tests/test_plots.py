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


def test_lie_figures(tmp_path):
    from byzmuon.analysis import make_lie_figures
    from byzmuon.analysis.lie_report import lie_formula_z, simulate_shift

    assert abs(lie_formula_z(20, 4) - 0.157) < 0.01
    sim = simulate_shift(20, 4, [0.0, 1.0], ["mean", "median", "trimmed_mean"], trials=2000)
    assert abs(sim["mean"][1] - 0.2) < 0.03 and sim["median"][1] > 0 and all(abs(v) < 0.02 for v in sim["mean"][:1])
    extra = {"agg_fro": 1.0, "spec/w/erank": 5.0, "test_loss": 1.0}
    for agg in ("mean", "median"):
        fake_run(tmp_path, "muon", "none", agg, 0, [0.2, 0.5, 0.8], extra)
        for z in (None, 1.0, 2.0):
            cfg_extra = {} if z is None else {"z": z}
            fake_run_attack(tmp_path, agg, {"name": "lie", **cfg_extra}, [0.2, 0.4, 0.7 - 0.05 * (z or 0)], extra)
    written = make_lie_figures(tmp_path)
    names = {p.name for p in written}
    assert {
        "lie_final.png",
        "lie_timeline.png",
        "lie_mechanism.png",
        "lie_geometry.png",
        "lie_shift.png",
        "lie_curves.png",
    } <= names
    assert all(p.stat().st_size > 1000 for p in written)


def fake_run_attack(root, agg, attack, accs, extra_metrics=None):
    cfg = load_config(
        base={
            "server_opt": {"name": "muon"},
            "attack": attack,
            "seed": 0,
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
