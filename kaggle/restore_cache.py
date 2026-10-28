"""Restore finished runs from attached Kaggle inputs so the sweep cache can skip them.

Looks for every final.json under /kaggle/input (inside directories or results*.zip archives), reads the run's
out_dir and hash from it, and copies the run folder to <repo>/<out_dir>/<hash> unless it is already there.
Usage: python kaggle/restore_cache.py [input_root]
"""

import json
import shutil
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


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/kaggle/input")
    repo = Path.cwd()
    if not root.exists():
        print("no inputs attached")
        return
    n = restore_tree(root, repo)
    for z in root.rglob("results*.zip"):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                zipfile.ZipFile(z).extractall(tmp)
            except zipfile.BadZipFile:
                continue
            n += restore_tree(Path(tmp), repo)
    print(f"restored {n} cached runs from {root}")


if __name__ == "__main__":
    main()
