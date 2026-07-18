"""Small local model registry with explicit activation gates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class ModelRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def register(self, version: str, checkpoint: str, metrics: dict, gate_passed: bool, parent: str | None = None) -> dict:
        record = {
            "version": version,
            "checkpoint": str(checkpoint),
            "metrics": metrics,
            "gate_passed": bool(gate_passed),
            "active": False,
            "parent": parent,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def activate(self, version: str) -> dict:
        records = self._read()
        target = next((item for item in records if item["version"] == version), None)
        if target is None:
            raise KeyError(f"unknown model version: {version}")
        if not target.get("gate_passed"):
            raise ValueError(f"model {version} did not pass regression gate")
        for item in records:
            item["active"] = item["version"] == version
        self.path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
        return target

    def active(self) -> dict | None:
        return next((item for item in reversed(self._read()) if item.get("active")), None)

