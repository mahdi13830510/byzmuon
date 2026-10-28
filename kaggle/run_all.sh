#!/usr/bin/env bash
# One-session pipeline: (1) tune the momentum-SGD baseline, (2) pick the best setting, (3) run the full
# FedMuon-vs-momentum-SGD comparison with that setting, (4) figures + results.zip.
#   !git clone --depth 1 https://github.com/mahdi13830510/byzmuon.git /kaggle/working/byzmuon 2>/dev/null; bash /kaggle/working/byzmuon/kaggle/run_all.sh
# Attach the previous session's output (or a results.zip dataset) as an input: finished runs are restored from it,
# and the whole pipeline then fits in one 12 h session (~45 min tuning + ~8 h for the 24 momentum-SGD runs).
# Without that input it also trains the 24 FedMuon runs and stops starting runs at the HOURS budget (default 10.5),
# so a second session with this one's output attached finishes the rest.
set -euo pipefail
REPO=${REPO:-https://github.com/mahdi13830510/byzmuon.git}
HOURS=${HOURS:-10.5}
TUNE=configs/sweeps/kaggle_tune_momsgd.yaml
TEMPLATE=configs/sweeps/kaggle_muon_vs_momentum.yaml
SWEEP=configs/sweeps/generated_muon_vs_momentum.yaml
START=$(date +%s)
left_hours() { python -c "import time; print(max(0.1, $HOURS - (time.time() - $START) / 3600))"; }

cd /kaggle/working
[ -d byzmuon ] || git clone --depth 1 "$REPO" byzmuon
cd byzmuon
mkdir -p /kaggle/working/results
[ -L results ] || { rm -rf results; ln -s /kaggle/working/results results; }
pip install -q -e . --no-deps --ignore-requires-python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'gpus')"
[ -f data/cifar10/train.npz ] || byzmuon fetch cifar10 --root data
python kaggle/restore_cache.py /kaggle/input
NGPU=$(python -c "import torch;print(max(1,torch.cuda.device_count()))")

echo "=== 1/3 tuning the momentum-SGD baseline"
byzmuon sweep "$TUNE" --workers "$NGPU" --hours "$(left_hours)"
byzmuon results results/tune_momsgd2 || true
byzmuon plot results/tune_momsgd2 || true

echo "=== 2/3 choosing the best momentum-SGD setting"
python - "$TEMPLATE" "$SWEEP" <<'PY'
import glob, json, sys, yaml
template, out = sys.argv[1], sys.argv[2]
best = None
for f in glob.glob("results/tune_momsgd2/*/final.json"):
    d = json.load(open(f)); c = d["config"]
    if c["server_opt"]["name"] != "sgd" or d.get("final_test_acc") is None:
        continue
    acc = float(d["final_test_acc"])
    if acc == acc and (best is None or acc > best[0]):  # skip NaN (diverged)
        best = (acc, c["server_opt"], c["client_opt"])
if best is None:
    sys.exit("no finished momentum-SGD tuning run")
acc, server_opt, client_opt = best
print(f"best momentum-SGD setting: server {server_opt}, client {client_opt} -> {100 * acc:.1f}% at round 50")
sw = yaml.safe_load(open(template))
pairs = sw["grid"]["server_opt,client_opt"]
pairs[:] = [[server_opt, client_opt]] + [p for p in pairs if p[0]["name"] == "muon"]  # new runs first
yaml.safe_dump(sw, open(out, "w"), sort_keys=False)
print("wrote", out)
PY

echo "=== 3/3 FedMuon vs momentum SGD under LIE / Min-Max / Min-Sum"
byzmuon sweep "$SWEEP" --workers "$NGPU" --hours "$(left_hours)" ${SEEDS:+--seeds $SEEDS}
OUT=results/muon_vs_momentum
# Runs restored from an earlier session that used a different momentum-SGD setting would be averaged into the
# figures as if they were extra seeds; move everything that is not part of this sweep out of the way.
python - "$SWEEP" "$OUT" <<'PY'
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
byzmuon results "$OUT" || true
byzmuon plot "$OUT" --lie || true
cp "$SWEEP" "$OUT/sweep_used.yaml"
(cd /kaggle/working && zip -rq results.zip results) && ls -lh /kaggle/working/results.zip
if [ -n "${GITHUB_TOKEN:-}" ]; then bash kaggle/push_results.sh /kaggle/working/results; fi
