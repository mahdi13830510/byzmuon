#!/usr/bin/env bash
# Paste into a Kaggle notebook cell as:  !bash <(curl -sL RAW_URL_OF_THIS_FILE) <git-repo-url> <sweep-file>
# Uses Kaggle's preinstalled CUDA torch; our pyproject's XPU pins are for the laptop only, so install without deps.
set -euo pipefail
REPO=${1:?git repo url}
SWEEP=${2:-configs/sweeps/kaggle_lie_resnet18.yaml}
cd /kaggle/working
[ -d byzmuon ] || git clone --depth 1 "$REPO" byzmuon
cd byzmuon
pip install -q -e . --no-deps
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'gpus')"
[ -f data/cifar10/train.npz ] || byzmuon fetch cifar10 --root data
# results from a previous session, if its output was attached as an input dataset
for prev in /kaggle/input/*/results; do [ -d "$prev" ] && cp -rn "$prev"/. results/ 2>/dev/null || true; done
byzmuon sweep "$SWEEP"
OUT="$(python -c "import yaml;print(yaml.safe_load(open('$SWEEP'))['config']['out_dir'])")"
byzmuon results "$OUT"
byzmuon plot "$OUT"          # figures + summary.md land in $OUT/figures
cp -r results /kaggle/working/results   # survives the session as notebook output
