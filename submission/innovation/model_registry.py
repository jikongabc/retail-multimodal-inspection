# 支持门禁激活和显式回滚的本地模型注册表。

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


# 生成带时区的 UTC 时间戳。
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 维护候选模型谱系并确保只有一个激活版本。
class ModelRegistry:
    # 初始化注册表文件位置并确保目录存在。
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # 读取当前模型注册记录。
    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # 覆盖写回模型注册记录。
    def _write(self, records: list[dict]) -> None:
        content = "\n".join(json.dumps(item, ensure_ascii=False) for item in records)
        self.path.write_text(content + ("\n" if content else ""), encoding="utf-8")

    # 将 checkpoint 保存为相对注册表的可移植路径。
    def _store_checkpoint(self, checkpoint: str | Path) -> str:
        path = Path(checkpoint)
        if not path.is_absolute():
            return path.as_posix()
        return Path(os.path.relpath(path, self.path.parent.resolve())).as_posix()

    # 登记一个候选模型及其门禁结果。
    def register(
        self,
        version: str,
        checkpoint: str | Path,
        metrics: dict,
        gate_passed: bool,
        parent: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        records = self._read()
        if any(item.get("version") == version for item in records):
            raise ValueError(f"model version already exists: {version}")
        record = {
            "version": version,
            "checkpoint": self._store_checkpoint(checkpoint),
            "metrics": metrics,
            "gate_passed": bool(gate_passed),
            "active": False,
            "parent": parent,
            "metadata": metadata or {},
            "created_at": _now(),
        }
        records.append(record)
        self._write(records)
        return record

    # 激活一个已通过门禁的模型版本。
    def activate(self, version: str, *, reason: str = "regression_gate_passed") -> dict:
        records = self._read()
        target = next((item for item in records if item["version"] == version), None)
        if target is None:
            raise KeyError(f"unknown model version: {version}")
        if not target.get("gate_passed"):
            raise ValueError(f"model {version} did not pass regression gate")
        activated_at = _now()
        for item in records:
            item["active"] = item["version"] == version
            if item["active"]:
                item["activated_at"] = activated_at
                item["activation_reason"] = reason
        self._write(records)
        return target

    # 回滚到指定版本或当前激活模型的父版本。
    def rollback(self, version: str | None = None, *, reason: str) -> dict:
        records = self._read()
        active = next((item for item in reversed(records) if item.get("active")), None)
        target_version = version or (active.get("parent") if active else None)
        if not target_version:
            raise ValueError("no rollback target is available")
        target = next(
            (item for item in records if item.get("version") == target_version), None
        )
        if target is None:
            raise KeyError(f"unknown rollback target: {target_version}")
        if not target.get("gate_passed"):
            raise ValueError(f"rollback target {target_version} is not eligible")
        rolled_back_at = _now()
        for item in records:
            item["active"] = item.get("version") == target_version
            if item["active"]:
                item["activated_at"] = rolled_back_at
                item["activation_reason"] = f"rollback: {reason}"
            elif active and item.get("version") == active.get("version"):
                item["rolled_back_at"] = rolled_back_at
                item["rollback_reason"] = reason
        self._write(records)
        return target

    # 返回当前激活的模型记录。
    def active(self) -> dict | None:
        return next(
            (item for item in reversed(self._read()) if item.get("active")), None
        )

    # 返回所有模型注册记录。
    def all(self) -> list[dict]:
        return self._read()

    # 将注册表里的 checkpoint 路径解析到当前机器。
    def resolve_checkpoint(self, version: str) -> Path:
        record = next(
            (item for item in self._read() if item.get("version") == version), None
        )
        if record is None:
            raise KeyError(f"unknown model version: {version}")
        checkpoint = Path(str(record["checkpoint"]))
        if checkpoint.is_absolute():
            return checkpoint
        return (self.path.parent / checkpoint).resolve()
