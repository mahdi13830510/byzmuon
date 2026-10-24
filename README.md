# byzmuon

Byzantine attacks and defenses for Muon-style (orthogonalized-momentum) optimizers in federated learning.

## Setup

```bash
uv sync                      # installs torch (XPU build on this machine) and dev tools
uv run pytest -q             # ~115 tests, about 15 s on CPU
uv run byzmuon fetch cifar10 # CIFAR-10 from the Hugging Face mirror (~45 s); the torchvision server is very slow
uv run byzmuon run configs/smoke.yaml                     # 5-round synthetic sanity run
uv run byzmuon sweep configs/sweeps/smoke_grid.yaml       # 9-run grid, cached by config hash
uv run byzmuon results results/smoke                      # table of finished runs
```

Any config key can be overridden on the command line with dotted keys:

```bash
uv run byzmuon run configs/cifar10_fedmuon_a.yaml attack.name=subgap attack.eps=0.05 aggregator.base.name=krum
```

Changing a component's `name` resets that component's other keys, so `attack.name=none` never
passes `eps` to the no-attack class.

## Layout

```
src/byzmuon/
  config.py             RunConfig dataclass, YAML loading, dotted overrides, content hash
  registry.py           name -> class registry; every component registers with @register(kind, name)
  core/update.py        Update: per-layer dict of tensors (+ flat/unflat, stack, norms, matrix view)
  core/client.py        local training, returns new − old
  core/server.py        round loop (template method) with probe hooks; anonymizes updates
  core/device.py        cuda > xpu > cpu
  optim/orthogonalize.py  ExactPolar, NewtonSchulz (Muon's), TruncatedPolar, RegularizedPolar (defenses D2)
  optim/server_opt.py   ServerSGD (FedAvg), ServerAdam (FedAdam), ServerMuon (mode A)
  optim/muon_local.py   client-side Muon (mode B)
  aggregators/          mean, median, trimmed_mean, krum, multikrum, bulyan, geomed, centered_clip
  aggregators/preagg.py nnm, fro_clip, spectral_clip (defense D1); Pipeline chains them
  attacks/baselines.py  none, gaussian, signflip, ipm, lie, minmax, minsum, labelflip
  attacks/spectral.py   subgap (A1), spike (A2), intermittent (A3 wrapper)
  probes/               evaluate, spectrum, amplification (the go/no-go measurement), jsonl, console
  runner/               run (cached by hash), sweep (grid expansion), results (pandas loader)
configs/                example runs and sweeps
tests/                  one file per module
```

## Notes for the Intel GPU (XPU)

The first run on a fresh machine takes minutes: the Intel driver compiles every kernel once and caches it in
`~/.cache/neo_compiler_cache`. Later runs of the same model are fast. SVD-based probes run on CPU by design.

## Deployment modes

* Mode A, aggregate-then-polarize: `client_opt: {name: sgd}` and `server_opt: {name: muon}`.
* Mode B, polarize-then-aggregate: `client_opt: {name: muon}` and `server_opt: {name: sgd, lr: 1.0}`.

## Results

Each run writes `results/<out_dir>/<hash>/{config.yaml, metrics.jsonl, final.json}`. A run whose
`final.json` exists is skipped unless `--force` is given. `byzmuon.runner.results.load_results(dir)`
returns one pandas row per run with the config flattened into `cfg.*` columns.
