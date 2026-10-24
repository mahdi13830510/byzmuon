import pytest
import yaml

from byzmuon.config import RunConfig, load_config, to_dict
from byzmuon.runner.results import load_metrics, load_results
from byzmuon.runner.run import run
from byzmuon.runner.sweep import expand, run_sweep


def test_defaults_and_hash_stability():
    a, b = RunConfig(), RunConfig()
    assert a.hash() == b.hash() and len(a.hash()) == 12
    assert RunConfig(seed=1).hash() != a.hash()
    assert RunConfig(out_dir="x").hash() == a.hash()  # output location does not change identity


def test_overrides_dotted_and_typed(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"rounds": 3, "aggregator": {"base": {"name": "mean"}, "pre": []}}))
    cfg = load_config(
        p,
        ["rounds=7", "aggregator.base.name=krum", "attack.name=subgap", "attack.eps=0.1", "probes=[{name: evaluate}]"],
    )
    assert cfg.rounds == 7 and cfg.aggregator["base"]["name"] == "krum"
    assert cfg.attack == {"name": "subgap", "eps": 0.1} and cfg.probes == [{"name": "evaluate"}]
    swapped = load_config(p, ["attack.name=subgap", "attack.eps=0.1", "attack.name=none"])
    assert swapped.attack == {"name": "none"}  # new name drops old kwargs
    same = load_config(p, ["attack.name=subgap", "attack.eps=0.1", "attack.name=subgap"])
    assert same.attack == {"name": "subgap", "eps": 0.1}  # same name keeps them
    with pytest.raises(KeyError):
        load_config(p, ["nonsense=1"])
    with pytest.raises(ValueError):
        load_config(p, ["rounds"])


def smoke_cfg(tmp_path, **kw):
    base = {
        "device": "cpu",
        "rounds": 3,
        "local_steps": 3,
        "n_clients": 4,
        "num_byz": 1,
        "data": {"name": "synthetic", "n": 400},
        "model": {"name": "mlp", "hidden": 8},
        "server_opt": {"name": "muon", "lr": 0.02},
        "attack": {"name": "subgap"},
        "probes": [{"name": "evaluate"}, {"name": "amplification", "every": 1}],
        "out_dir": str(tmp_path / "res"),
    }
    base.update(kw)
    return load_config(base=base)


def test_run_writes_results_and_caches(tmp_path):
    cfg = smoke_cfg(tmp_path)
    out = run(cfg)
    d = cfg.run_dir()
    assert (d / "final.json").exists() and (d / "config.yaml").exists()
    metrics = load_metrics(d)
    assert len(metrics) == 3 and "amp/mean/tail/0.01" in metrics.columns
    assert out["final_test_acc"] is not None and out["config"] == to_dict(cfg)
    cached = run(cfg)
    assert cached["elapsed_s"] == out["elapsed_s"]  # served from disk
    assert run(cfg, force=True)["elapsed_s"] != out["elapsed_s"]
    df = load_results(tmp_path / "res")
    assert len(df) == 1 and df.loc[0, "cfg.attack.name"] == "subgap"


def test_sweep_expands_and_runs(tmp_path):
    base = tmp_path / "base.yaml"
    yaml.safe_dump(to_dict(smoke_cfg(tmp_path)), base.open("w"))
    sweep = tmp_path / "sweep.yaml"
    yaml.safe_dump(
        {
            "base": str(base),
            "grid": {"attack.name": ["none", "signflip"], "aggregator.base.name": ["mean", "median"]},
            "seeds": [0, 1],
        },
        sweep.open("w"),
    )
    cfgs = expand(sweep)
    assert len(cfgs) == 8 and len({c.hash() for c in cfgs}) == 8
    outs = run_sweep(sweep)
    assert len(outs) == 8 and len(load_results(tmp_path / "res")) == 8


def test_cli_run(tmp_path, capsys):
    from byzmuon.cli import main

    base = tmp_path / "base.yaml"
    yaml.safe_dump(to_dict(smoke_cfg(tmp_path)), base.open("w"))
    main(["run", str(base), "rounds=2"])
    assert "final_test_acc" in capsys.readouterr().out
    main(["results", str(tmp_path / "res")])
    assert "subgap" in capsys.readouterr().out


def test_sweep_two_workers_on_cpu(tmp_path):
    from byzmuon.runner.sweep import gpu_for_worker

    assert [gpu_for_worker(i, 2) for i in range(4)] == ["0", "1", "0", "1"]
    assert gpu_for_worker(0, 0) is None
    base = tmp_path / "base.yaml"
    yaml.safe_dump(to_dict(smoke_cfg(tmp_path, rounds=2)), base.open("w"))
    sweep = tmp_path / "sweep.yaml"
    yaml.safe_dump({"base": str(base), "grid": {"attack.name": ["none", "signflip", "lie"]}}, sweep.open("w"))
    outs = run_sweep(sweep, workers=2)
    assert len(outs) == 3 and all(o["final_test_acc"] is not None for o in outs)
    assert len(load_results(tmp_path / "res")) == 3
