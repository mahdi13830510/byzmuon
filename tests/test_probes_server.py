import json

import pytest
import torch
from conftest import low_rank_update

from byzmuon.aggregators import build_aggregator
from byzmuon.aggregators.mean import Mean
from byzmuon.attacks import build_attack
from byzmuon.core.client import Client
from byzmuon.core.server import Server
from byzmuon.data.datasets import SyntheticClassification, cycling_batches, eval_loader
from byzmuon.models.mlp import MLP
from byzmuon.optim.orthogonalize import ExactPolar, RegularizedPolar
from byzmuon.optim.server_opt import build_server_opt
from byzmuon.probes.evaluate import Evaluate
from byzmuon.probes.logging import JsonlLogger
from byzmuon.probes.spectrum import Amplification, Spectrum, amplification_gain, spectrum_stats


def make_server(server_opt, aggregator, attack, n_clients=6, num_byz=1, rounds_seed=0, probes=None, participation=1.0):
    ds = SyntheticClassification(n=600, in_dim=20, num_classes=4, seed=0)
    te = SyntheticClassification(n=200, in_dim=20, num_classes=4, seed=0, train=False)
    model = MLP(in_dim=20, hidden=16, num_classes=4)
    per = len(ds) // n_clients
    atk = build_attack(attack, num_byz=num_byz)
    byz = set(range(num_byz))
    lm = atk.label_map(4) if atk.poisons_data else None
    clients = [
        Client(
            i,
            model,
            cycling_batches(ds, list(range(i * per, (i + 1) * per)), 32, seed=i),
            lambda m: torch.optim.SGD(m.parameters(), lr=0.05),
            5,
            torch.device("cpu"),
            label_map=lm if i in byz else None,
        )
        for i in range(n_clients)
    ]
    return Server(
        model,
        clients,
        byz,
        build_aggregator(aggregator, num_byz),
        atk,
        build_server_opt(server_opt),
        probes or [Evaluate()],
        participation=participation,
        seed=rounds_seed,
        eval_loader=eval_loader(te),
    )


def test_spectrum_stats_and_gain_on_low_rank():
    m = low_rank_update(rank=4, noise=1e-3).tensors["w"]
    st = spectrum_stats(m)
    assert 3 < st["erank"] < 6 and st["s1"] > 0.9 and st["tail_frac"] > 0.5
    gain_tail = amplification_gain(m, ExactPolar(), 0.01, "tail", gen=torch.Generator().manual_seed(0))
    gain_rand = amplification_gain(m, ExactPolar(), 0.01, "random", gen=torch.Generator().manual_seed(0))
    assert gain_tail > 5 and gain_rand > 3
    assert amplification_gain(m, RegularizedPolar(0.1), 0.01, "tail", gen=torch.Generator().manual_seed(0)) < gain_tail


def test_clean_training_improves_accuracy_for_all_server_opts():
    for spec in [
        {"name": "sgd", "lr": 1.0},
        {"name": "adam", "lr": 0.05},
        {"name": "muon", "lr": 0.05, "orthogonalizer": {"name": "ns5"}},
    ]:
        s = make_server(spec, {"name": "mean"}, {"name": "none"}, num_byz=0)
        hist = s.run(15)
        assert hist[-1]["test_acc"] > 0.6, (spec, hist[-1])
        assert hist[-1]["n_byz"] == 0 and hist[-1]["n_honest"] == 6


@pytest.mark.parametrize(
    "attack", ["gaussian", "signflip", "lie", "ipm", "minmax", "labelflip", "subgap", "spike", "intermittent"]
)
@pytest.mark.parametrize("aggregator", ["mean", "median", "krum", "centered_clip"])
def test_every_attack_runs_with_every_aggregator(attack, aggregator):
    spec = {"name": attack}
    if attack == "intermittent":
        spec |= {"inner": {"name": "signflip"}, "every": 2}
    s = make_server({"name": "muon", "lr": 0.02}, {"name": aggregator}, spec, num_byz=2)
    hist = s.run(3)
    assert hist[-1]["n_byz"] == 2 and hist[-1]["n_honest"] == 4
    assert all(torch.isfinite(p).all() for p in s.state.params.values())


def test_server_is_blind_to_byzantine_identity():
    s = make_server({"name": "sgd", "lr": 1.0}, {"name": "mean"}, {"name": "signflip"}, num_byz=2)
    seen = []

    class Spy(Mean):
        def aggregate(self, updates):
            seen.extend(set(u.meta) for u in updates)
            return super().aggregate(updates)

    s.aggregator = Spy()
    s.run_round(0)
    assert len(seen) == 6 and all(keys == {"exclude_from_muon"} for keys in seen)  # no ids, no losses
    assert len(s.last_honest) == 4 and len(s.last_malicious) == 2  # probes still see both


def test_partial_participation_and_pipeline_defense():
    s = make_server(
        {"name": "muon", "lr": 0.02, "orthogonalizer": {"name": "regularized", "delta": 0.1}},
        {"base": {"name": "trimmed_mean"}, "pre": [{"name": "spectral_clip"}, {"name": "nnm"}]},
        {"name": "spike"},
        num_byz=1,
        participation=0.5,
    )
    hist = s.run(3)
    assert hist[-1]["n_honest"] + hist[-1]["n_byz"] == 3


def test_probes_write_metrics(tmp_path):
    log = tmp_path / "m.jsonl"
    s = make_server(
        {"name": "muon", "lr": 0.02},
        {"name": "mean"},
        {"name": "subgap"},
        num_byz=1,
        probes=[Evaluate(), Spectrum(every=1), Amplification(every=1, eps=(0.01,)), JsonlLogger(log)],
    )
    hist = s.run(2)
    keys = hist[-1].keys()
    assert any(k.startswith("spec/") for k in keys) and any(k.startswith("hspec/") for k in keys)
    assert "amp/mean/tail/0.01" in keys and "amp/mean/random/0.01" in keys
    s.probes[-1].close()
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(lines) == 2 and lines[1]["round"] == 1 and "test_acc" in lines[1]
