#!/usr/bin/env bash
# Push a results directory to the `results` branch of the repo (needs GITHUB_TOKEN with repo scope).
set -euo pipefail
SRC=${1:-/kaggle/working/results}
REPO=${REPO:-github.com/mahdi13830510/byzmuon.git}
TMP=$(mktemp -d)
git clone -q --depth 1 --branch results "https://x-access-token:${GITHUB_TOKEN}@${REPO}" "$TMP" 2>/dev/null \
  || { git init -q "$TMP"; git -C "$TMP" checkout -q -b results; git -C "$TMP" remote add origin "https://x-access-token:${GITHUB_TOKEN}@${REPO}"; }
rsync -a "$SRC"/ "$TMP"/results/
cd "$TMP"
git -c user.name=kaggle -c user.email=kaggle@localhost add -A
git -c user.name=kaggle -c user.email=kaggle@localhost commit -qm "results: $(date -u +%Y-%m-%dT%H:%MZ)" || { echo "nothing new"; exit 0; }
git push -q origin results
echo "pushed to branch results"
