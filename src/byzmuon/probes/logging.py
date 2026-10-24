import json
import time
from pathlib import Path

from byzmuon.probes.base import Probe
from byzmuon.registry import register


@register("probe", "jsonl")
class JsonlLogger(Probe):
    """Appends one JSON line of metrics per round."""

    def __init__(self, path: str | Path, every: int = 1) -> None:
        super().__init__(every)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a")
        self._t0 = time.time()

    def on_round_end(self, server, r: int, metrics: dict) -> None:
        metrics["elapsed_s"] = time.time() - self._t0
        self._f.write(json.dumps(metrics, default=float) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


@register("probe", "console")
class Console(Probe):
    def __init__(self, every: int = 1, keys: tuple[str, ...] = ("test_acc", "test_loss", "train_loss")) -> None:
        super().__init__(every)
        self.keys = keys

    def on_round_end(self, server, r: int, metrics: dict) -> None:
        if not self.due(r):
            return
        parts = [f"{k}={metrics[k]:.4f}" for k in self.keys if k in metrics]
        print(f"round {r:4d} | " + " ".join(parts), flush=True)
