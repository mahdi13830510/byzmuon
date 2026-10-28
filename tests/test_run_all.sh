#!/usr/bin/env bash
# Dry-run of kaggle/run_all.sh on CPU with synthetic data: runs it twice, the second time with the first run's
# results.zip attached as an "input", and checks that the cache restore, tuning, selection, stale-run pruning,
# figures and zip all work. Usage (from the repo root): bash tests/test_run_all.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
[ -n "${KEEP:-}" ] && echo "keeping $TMP" || trap 'rm -rf "$TMP"' EXIT
export WORK="$TMP/work" INPUT="$TMP/input" REPO="$ROOT" PIP_INSTALL=0 FETCH_DATA=0
export TUNE=configs/sweeps/smoke_tune_momsgd.yaml TEMPLATE=configs/sweeps/smoke_muon_vs_momentum.yaml
export SWEEP=configs/sweeps/generated_smoke.yaml PY="$ROOT/.venv/bin/python" BYZ="$ROOT/.venv/bin/byzmuon"
mkdir -p "$WORK" "$INPUT"
# use the working tree (not a clone) so uncommitted changes are exercised
rsync -a --exclude .venv --exclude results --exclude data --exclude .git "$ROOT/" "$WORK/byzmuon/"

echo "### first session (nothing attached)"
bash "$ROOT/kaggle/run_all.sh" > "$TMP/log1" 2>&1 || { tail -30 "$TMP/log1"; exit 1; }
grep -q "restored 0 cached runs" "$TMP/log1"
OUT="$WORK/results/smoke_muon_vs_momentum"
[ "$(ls -d "$OUT"/*/ | grep -vc figures)" = 30 ] || { echo "expected 30 runs"; ls "$OUT"; exit 1; }
for f in lie_compare.png attack_compare.png drop.png muon/lie_final.png momsgd/lie_final.png momsgd/lie_timeline.png summary.md; do
  [ -s "$OUT/figures/$f" ] || { echo "missing figure $f"; exit 1; }
done
grep -q "Momentum SGD" "$OUT/figures/summary.md"
[ -s "$WORK/results.zip" ]

echo "### second session (first session's zip attached, different baseline chosen)"
mkdir -p "$INPUT/prev"; cp "$WORK/results.zip" "$INPUT/prev/results (1).zip"
# add a run from an "older baseline" (different hash, same out_dir) to the attached results: it must be pruned
STALE="$WORK/results/smoke_muon_vs_momentum/deadbeef0000"
cp -r "$(ls -d "$OUT"/*/ | grep -v figures | head -1)" "$STALE"
"$PY" - "$STALE/final.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d["hash"] = "deadbeef0000"; d["config"]["server_opt"]["lr"] = 99.0
json.dump(d, open(sys.argv[1], "w"))
PY
(cd "$WORK" && rm -f results.zip && zip -rq results.zip results)
cp "$WORK/results.zip" "$INPUT/prev/results (1).zip"
rm -rf "$WORK/results" "$WORK/results.zip" "$WORK/byzmuon/results"
bash "$ROOT/kaggle/run_all.sh" > "$TMP/log2" 2>&1 || { tail -30 "$TMP/log2"; exit 1; }
grep -q "restored 33 cached runs" "$TMP/log2"
grep -q "moved stale run deadbeef0000" "$TMP/log2"
[ -d "$WORK/results/smoke_muon_vs_momentum_stale/deadbeef0000" ]
! grep -q "99.0" "$OUT/figures/summary.md"
[ "$(ls -d "$OUT"/*/ | grep -vc figures)" = 30 ]
[ -s "$OUT/figures/lie_compare.png" ]
echo "### run_all.sh dry-run OK"; grep -E "^=== |restored|best momentum" "$TMP/log1" "$TMP/log2"
