# 进程内键值存储。

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any


# 提供带命名空间的线程安全键值存储。
class InMemoryKVStore:
    # 初始化存储和互斥锁。
    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._lock = RLock()

    # 写入键值。
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._values[key] = deepcopy(value)

    # 读取键值。
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return deepcopy(self._values.get(key, default))

    # 返回指定前缀的键。
    def keys(self, prefix: str = "") -> list[str]:
        with self._lock:
            return sorted(key for key in self._values if key.startswith(prefix))

    # 返回指定前缀的键值副本。
    def dump(self, prefix: str = "") -> dict[str, Any]:
        with self._lock:
            return {key: deepcopy(self._values[key]) for key in self.keys(prefix)}
