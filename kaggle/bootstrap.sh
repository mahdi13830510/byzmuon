#!/usr/bin/env bash
# Kaggle entry point. In a notebook cell:
#   !git clone --depth 1 https://github.com/mahdi13830510/byzmuon.git /kaggle/working/byzmuon 2>/dev/null; bash /kaggle/working/byzmuon/kaggle/bootstrap.sh configs/sweeps/kaggle_lie_resnet18.yaml
# Run the notebook with "Save Version -> Save & Run All (Commit)" so it runs in the background and the
# output is kept even if you close the browser. Interactive sessions are deleted after ~20 min idle.
# Results are written directly to /kaggle/working/results (symlinked), so every finished run is already
# in the notebook output; a killed session loses at most the runs still in progress.
# Optional: set a Kaggle secret GITHUB_TOKEN (repo scope) and export it in the cell to also push results
# to the `results` branch of the repo after each sweep.
set -euo pipefail
SWEEP=${1:-configs/sweeps/kaggle_lie_resnet18.yaml}
REPO=${REPO:-https://github.com/mahdi13830510/byzmuon.git}
cd /kaggle/working
[ -d byzmuon ] || git clone --depth 1 "$REPO" byzmuon
cd byzmuon
mkdir -p /kaggle/working/results
[ -L results ] || { rm -rf results; ln -s /kaggle/working/results results; }
pip install -q -e . --no-deps --ignore-requires-python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'gpus')"
[ -f data/cifar10/train.npz ] || byzmuon fetch cifar10 --root data
# results from a previous session, if its output was attached as an input dataset
for prev in /kaggle/input/*/results; do [ -d "$prev" ] && cp -rn "$prev"/. results/ 2>/dev/null || true; done
NGPU=$(python -c "import torch;print(max(1,torch.cuda.device_count()))")
byzmuon sweep "$SWEEP" --workers "$NGPU"   # one run per GPU; finished runs are cached by config hash
OUT="$(python -c "import yaml;print(yaml.safe_load(open('$SWEEP'))['config']['out_dir'])")"
byzmuon results "$OUT"
byzmuon plot "$OUT"          # figures + summary.md land in $OUT/figures
(cd /kaggle/working && zip -rq results.zip results) && ls -lh /kaggle/working/results.zip
if [ -n "${GITHUB_TOKEN:-}" ]; then
  bash kaggle/push_results.sh /kaggle/working/results
fi
