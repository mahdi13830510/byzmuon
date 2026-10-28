"""Restore finished runs from attached Kaggle inputs so the sweep cache can skip them.

Looks for every final.json under /kaggle/input (inside directories or results*.zip archives), reads the run's
out_dir and hash from it, and copies the run folder to <repo>/<out_dir>/<hash> unless it is already there.
Also restores from the `results` branch of the GitHub repo (where kaggle/push_results.sh backs runs up during a
session), so a session that was killed by the quota or the 12 h cap loses nothing that was pushed.
Usage: python kaggle/restore_cache.py [input_root]   (env RESULTS_REPO overrides the repo URL, empty disables)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def restore_tree(root: Path, repo: Path) -> int:
    n = 0
    for fj in root.rglob("final.json"):
        try:
            d = json.loads(fj.read_text())
            out_dir, h = d["config"]["out_dir"], d["hash"]
        except (KeyError, ValueError, OSError):
            continue
        dst = repo / out_dir / h
        if (dst / "final.json").exists():
            continue
        shutil.copytree(fj.parent, dst, dirs_exist_ok=True)
        n += 1
    return n


def restore_branch(repo: Path, url: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(
            ["git", "clone", "-q", "--depth", "1", "--branch", "results", url, tmp], capture_output=True, text=True
        )
        if r.returncode != 0:
            print("no results branch to restore from")
            return 0
        return restore_tree(Path(tmp), repo)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/kaggle/input")
    repo = Path.cwd()
    n = 0
    if root.exists():
        n = restore_tree(root, repo)
        for z in root.rglob("results*.zip"):
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    zipfile.ZipFile(z).extractall(tmp)
                except zipfile.BadZipFile:
                    continue
                n += restore_tree(Path(tmp), repo)
        print(f"restored {n} cached runs from {root}")
    else:
        print("no inputs attached")
    url = os.environ.get("RESULTS_REPO", "https://github.com/mahdi13830510/byzmuon.git")
    if url:
        m = restore_branch(repo, url)
        print(f"restored {m} more cached runs from the results branch of {url}")


if __name__ == "__main__":
    main()
