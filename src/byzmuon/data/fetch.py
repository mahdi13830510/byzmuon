"""Fast dataset fetching from the Hugging Face mirror (the canonical CIFAR server is very slow).

Writes data/<name>/{train,test}.npz with uint8 images [N, H, W, C] and int64 labels. The dataset
loaders in datasets.py use these files when present and fall back to torchvision otherwise.
"""

import io
import sys
import urllib.request
from pathlib import Path

import numpy as np

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
SOURCES = {
    "cifar10": {
        "repo": "uoft-cs/cifar10",
        "train": "plain_text/train-00000-of-00001.parquet",
        "test": "plain_text/test-00000-of-00001.parquet",
        "img": "img",
        "label": "label",
    },
    "cifar100": {
        "repo": "uoft-cs/cifar100",
        "train": "cifar100/train-00000-of-00001.parquet",
        "test": "cifar100/test-00000-of-00001.parquet",
        "img": "img",
        "label": "fine_label",
    },
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:  # noqa: S310
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", file=sys.stderr)
        print(file=sys.stderr)
    tmp.replace(dest)


def _parquet_to_npz(parquet: Path, npz: Path, img_col: str, label_col: str) -> None:
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(parquet)
    imgs = table.column(img_col).to_pylist()
    labels = np.asarray(table.column(label_col).to_pylist(), dtype=np.int64)
    arr = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in imgs])
    np.savez_compressed(npz, x=arr, y=labels)


def fetch(name: str, root: str | Path = "data", force: bool = False) -> Path:
    src = SOURCES[name]
    out = Path(root) / name
    for split in ("train", "test"):
        npz = out / f"{split}.npz"
        if npz.exists() and not force:
            print(f"  {npz} exists, skipping", file=sys.stderr)
            continue
        parquet = out / f"{split}.parquet"
        if not parquet.exists() or force:
            _download(HF.format(repo=src["repo"], path=src[split]), parquet)
        _parquet_to_npz(parquet, npz, src["img"], src["label"])
        parquet.unlink()
        print(f"  wrote {npz}", file=sys.stderr)
    return out
