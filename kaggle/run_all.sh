#!/usr/bin/env bash
# One-session pipeline: (1) tune the momentum-SGD baseline, (2) pick the best setting, (3) run the full
# FedMuon-vs-momentum-SGD comparison with that setting, (4) figures + results.zip.
#   !git clone --depth 1 https://github.com/mahdi13830510/byzmuon.git /kaggle/working/byzmuon 2>/dev/null; bash /kaggle/working/byzmuon/kaggle/run_all.sh
# Attach the previous session's output (or a results.zip dataset) as an input: finished runs are restored from it,
# and the whole pipeline then fits in one 12 h session (~40 min tuning + ~8 h for the 24 momentum-SGD runs).
# Without that input it also trains the 24 FedMuon runs and stops starting runs at the HOURS budget (default 10.5),
# so a second session with this one's output attached finishes the rest.
# Every path/command can be overridden through the environment so the same script can be dry-run locally
# (see tests/test_run_all.sh).
set -euo pipefail
REPO=${REPO:-https://github.com/mahdi13830510/byzmuon.git}
WORK=${WORK:-/kaggle/working}
INPUT=${INPUT:-/kaggle/input}
HOURS=${HOURS:-10.5}
TUNE=${TUNE:-configs/sweeps/kaggle_tune_momsgd.yaml}
TEMPLATE=${TEMPLATE:-configs/sweeps/kaggle_muon_vs_momentum.yaml}
SWEEP=${SWEEP:-configs/sweeps/generated_muon_vs_momentum.yaml}
PY=${PY:-python}
BYZ=${BYZ:-byzmuon}
PIP_INSTALL=${PIP_INSTALL:-1}
FETCH_DATA=${FETCH_DATA:-1}
START=$(date +%s)
left_hours() { $PY -c "import time; print(max(0.1, $HOURS - (time.time() - $START) / 3600))"; }

cd "$WORK"
[ -d byzmuon ] || git clone --depth 1 "$REPO" byzmuon
cd byzmuon
mkdir -p "$WORK/results"
[ -L results ] || { rm -rf results; ln -s "$WORK/results" results; }
[ "$PIP_INSTALL" = 1 ] && pip install -q -e . --no-deps --ignore-requires-python
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'gpus')"
[ "$FETCH_DATA" = 1 ] && { [ -f data/cifar10/train.npz ] || $BYZ fetch cifar10 --root data; }
$PY kaggle/restore_cache.py "$INPUT"
NGPU=$($PY -c "import torch;print(max(1,torch.cuda.device_count()))")
TUNE_OUT=$($PY -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['config']['out_dir'])" "$TUNE")
OUT=$($PY -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['config']['out_dir'])" "$TEMPLATE")

echo "=== 1/3 tuning the momentum-SGD baseline ($(left_hours) h left)"
$BYZ sweep "$TUNE" --workers "$NGPU" --hours "$(left_hours)"
$BYZ results "$TUNE_OUT" || true
$BYZ plot "$TUNE_OUT" || true

echo "=== 2/3 choosing the best momentum-SGD setting"
$PY - "$TEMPLATE" "$SWEEP" "$TUNE_OUT" <<'PY'
import glob, json, sys, yaml
template, out, tune_out = sys.argv[1], sys.argv[2], sys.argv[3]
best = None
for f in glob.glob(f"{tune_out}/*/final.json"):
    d = json.load(open(f)); c = d["config"]
    if c["server_opt"]["name"] == "muon" or d.get("final_test_acc") is None:
        continue
    acc = float(d["final_test_acc"])
    print(f"  {100 * acc:5.1f}%  server {c['server_opt']}  client {c['client_opt']}")
    if acc == acc and (best is None or acc > best[0]):  # skip NaN (diverged)
        best = (acc, c["server_opt"], c["client_opt"])
if best is None:
    sys.exit("no finished momentum-SGD tuning run")
acc, server_opt, client_opt = best
print(f"best momentum-SGD setting: server {server_opt}, client {client_opt} -> {100 * acc:.1f}%")
sw = yaml.safe_load(open(template))
pairs = sw["grid"]["server_opt,client_opt"]
pairs[:] = [[server_opt, client_opt]] + [p for p in pairs if p[0]["name"] == "muon"]  # new runs first
yaml.safe_dump(sw, open(out, "w"), sort_keys=False)
print("wrote", out)
PY

echo "=== 3/3 FedMuon vs momentum SGD under LIE / Min-Max / Min-Sum ($(left_hours) h left)"
$BYZ sweep "$SWEEP" --workers "$NGPU" --hours "$(left_hours)" ${SEEDS:+--seeds $SEEDS}
# Runs restored from an earlier session that used a different momentum-SGD setting would be averaged into the
# figures as if they were extra seeds; move everything that is not part of this sweep out of the way.
$PY - "$SWEEP" "$OUT" <<'PY'
import shutil, sys
from pathlib import Path
from byzmuon.runner.sweep import expand
sweep, out = sys.argv[1], Path(sys.argv[2])
keep = {c.hash() for c in expand(sweep)}
stale = out.parent / (out.name + "_stale")
for d in out.iterdir():
    if d.is_dir() and d.name != "figures" and d.name not in keep:
        stale.mkdir(exist_ok=True)
        shutil.move(str(d), str(stale / d.name))
        print("moved stale run", d.name, "to", stale)
PY
$BYZ results "$OUT"
$BYZ plot "$OUT" --lie
cp "$SWEEP" "$OUT/sweep_used.yaml"
rm -f "$WORK/results.zip"
(cd "$WORK" && zip -rq results.zip results) && ls -lh "$WORK/results.zip"
if [ -n "${GITHUB_TOKEN:-}" ]; then bash kaggle/push_results.sh "$WORK/results"; fi
echo "=== done in $(( ($(date +%s) - START) / 60 )) min"
