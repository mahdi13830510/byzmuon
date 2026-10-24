from collections.abc import Callable
from typing import Any


class Registry:
    """Maps (kind, name) to a class. Kinds: 'aggregator', 'attack', 'orthogonalizer', ..."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, type]] = {}

    def register(self, kind: str, name: str) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            table = self._items.setdefault(kind, {})
            if name in table:
                raise KeyError(f"{kind} '{name}' registered twice")
            table[name] = cls
            return cls

        return decorator

    def get(self, kind: str, name: str) -> type:
        try:
            return self._items[kind][name]
        except KeyError:
            known = ", ".join(sorted(self._items.get(kind, {}))) or "(none)"
            raise KeyError(f"unknown {kind} '{name}'. Known: {known}") from None

    def build(self, kind: str, name: str, **kwargs: Any) -> Any:
        return self.get(kind, name)(**kwargs)

    def names(self, kind: str) -> list[str]:
        return sorted(self._items.get(kind, {}))


REGISTRY = Registry()
register = REGISTRY.register
