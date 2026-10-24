from __future__ import annotations

from typing import TYPE_CHECKING

from byzmuon.core.update import Update

if TYPE_CHECKING:
    from byzmuon.core.server import Server


class Probe:
    """Observer hooks on the server round. Override what you need; add numbers to `metrics`."""

    def __init__(self, every: int = 1) -> None:
        self.every = every

    def due(self, r: int) -> bool:
        return r % self.every == 0

    def on_round_start(self, server: Server, r: int) -> None: ...

    def on_updates(self, server: Server, r: int, updates: list[Update]) -> None: ...

    def on_aggregated(self, server: Server, r: int, agg: Update) -> None: ...

    def on_round_end(self, server: Server, r: int, metrics: dict) -> None: ...

    def close(self) -> None: ...
