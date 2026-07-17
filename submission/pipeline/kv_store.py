"""In-memory key-value store.

Python 3.10+; dependencies: standard library.
"""

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any


class InMemoryKVStore:
    """Thread-safe in-memory store with explicit namespaced keys."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._lock = RLock()

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._values[key] = deepcopy(value)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return deepcopy(self._values.get(key, default))

    def keys(self, prefix: str = "") -> list[str]:
        with self._lock:
            return sorted(key for key in self._values if key.startswith(prefix))

    def dump(self, prefix: str = "") -> dict[str, Any]:
        with self._lock:
            return {key: deepcopy(self._values[key]) for key in self.keys(prefix)}
