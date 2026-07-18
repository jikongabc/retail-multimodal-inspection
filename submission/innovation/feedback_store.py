# Task4 数据飞轮的可审核反馈存储。

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from submission.router.mm_router import WORKERS


TRUSTED_ENVIRONMENT_SOURCES = {"environment_gold", "verified_evaluator"}
TRAINABLE_STATUS = "approved"


# 规范化文本以稳定计算反馈指纹。
def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().lower().split())


# 计算多个字段组合后的稳定哈希。
def _sha256(*values: str) -> str:
    payload = "\x1f".join(values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# 生成不依赖本机绝对路径的文件身份。
def portable_path_identity(value: str) -> str:
    path = Path(value)
    parts = path.parts
    if "submission" in parts:
        index = len(parts) - 1 - tuple(reversed(parts)).index("submission")
        return Path(*parts[index:]).as_posix()
    return path.as_posix()


@dataclass(frozen=True)
# 描述一条来自人工或环境信号的路由纠错。
class Feedback:
    request_id: str
    image_path: str
    query: str
    original_worker: str
    correct_worker: str
    reason: str
    source_split: str = "production"
    signal_source: str = "human_correction"
    confidence: float = 1.0
    environment_reward: float | None = None
    auto_approve: bool = False

    @property
    # 标识同一个环境样本以发现互斥标签。
    def case_fingerprint(self) -> str:
        return _sha256(
            _normalize(portable_path_identity(self.image_path)),
            _normalize(self.query),
        )

    @property
    # 标识一条完整纠错以执行精确去重。
    def fingerprint(self) -> str:
        return _sha256(self.case_fingerprint, self.correct_worker)


# 管理反馈的写入、审核、隔离和训练导出。
class FeedbackStore:
    # 初始化反馈文件位置并确保目录存在。
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # 读取当前 JSONL 反馈快照。
    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # 覆盖写回小规模 JSONL 反馈快照。
    def _write(self, records: list[dict]) -> None:
        content = "\n".join(
            json.dumps(record, ensure_ascii=False) for record in records
        )
        self.path.write_text(content + ("\n" if content else ""), encoding="utf-8")

    # 将图片路径保存为相对反馈文件的可移植路径。
    def _store_path(self, value: str) -> str:
        path = Path(value)
        absolute = path if path.is_absolute() else path.resolve()
        return Path(os.path.relpath(absolute, self.path.parent.resolve())).as_posix()

    # 将存储路径解析为运行时可读取的绝对路径。
    def _runtime_path(self, value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str((self.path.parent / path).resolve())

    @staticmethod
    # 校验反馈字段和训练安全约束。
    def _validate(feedback: Feedback) -> None:
        if not feedback.request_id or not feedback.image_path or not feedback.query:
            raise ValueError("request_id, image_path and query are required")
        if (
            feedback.original_worker not in WORKERS
            or feedback.correct_worker not in WORKERS
        ):
            raise ValueError(f"workers must be one of {WORKERS}")
        if not feedback.reason.strip():
            raise ValueError("feedback reason is required")
        if feedback.source_split == "test":
            raise ValueError("test-set feedback cannot enter incremental training")
        if not 0.0 <= feedback.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if feedback.environment_reward is not None and not (
            -1.0 <= feedback.environment_reward <= 1.0
        ):
            raise ValueError("environment_reward must be between -1 and 1")

    # 写入一条反馈并处理去重、可信自动审核和冲突隔离。
    def add(self, feedback: Feedback) -> bool:
        self._validate(feedback)
        records = self._read()
        if any(item.get("fingerprint") == feedback.fingerprint for item in records):
            return False

        conflicts = [
            item
            for item in records
            if item.get("case_fingerprint") == feedback.case_fingerprint
            and item.get("correct_worker") != feedback.correct_worker
        ]
        if conflicts:
            for item in conflicts:
                item["status"] = "quarantined_conflict"
                item["review_note"] = "同一环境样本出现互斥 Worker 标签。"
            status = "quarantined_conflict"
        elif (
            feedback.auto_approve
            and feedback.signal_source in TRUSTED_ENVIRONMENT_SOURCES
            and feedback.confidence >= 0.9
        ):
            status = TRAINABLE_STATUS
        else:
            status = "pending_review"

        record = asdict(feedback)
        record.update(
            {
                "image_path": self._store_path(feedback.image_path),
                "case_fingerprint": feedback.case_fingerprint,
                "fingerprint": feedback.fingerprint,
                "status": status,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        records.append(record)
        self._write(records)
        return True

    # 手动审批或拒绝一条反馈并记录审核信息。
    def set_status(
        self,
        fingerprint: str,
        status: str,
        *,
        reviewer: str,
        note: str = "",
    ) -> dict:
        if status not in {"approved", "rejected"}:
            raise ValueError("status must be approved or rejected")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        records = self._read()
        target = next(
            (item for item in records if item.get("fingerprint") == fingerprint), None
        )
        if target is None:
            raise KeyError(f"unknown feedback fingerprint: {fingerprint}")
        target.update(
            {
                "status": status,
                "reviewer": reviewer,
                "review_note": note,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._write(records)
        return target

    # 返回所有反馈记录。
    def all(self) -> list[dict]:
        return self._read()

    # 返回待人工处理或冲突隔离的反馈记录。
    def pending(self) -> list[dict]:
        return [
            item
            for item in self._read()
            if item.get("status") in {"pending_review", "quarantined_conflict"}
        ]

    # 按审核状态统计反馈数量。
    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for item in self._read():
            status = str(item.get("status") or "unknown")
            result[status] = result.get(status, 0) + 1
        return result

    # 判断已批准反馈是否达到训练触发阈值。
    def ready(self, minimum: int = 5) -> bool:
        if minimum <= 0:
            raise ValueError("minimum must be positive")
        return len(self.training_records()) >= minimum

    # 将已批准反馈转换为路由器训练样本。
    def training_records(self) -> list[dict]:
        output = []
        for item in self._read():
            if item.get("status") != TRAINABLE_STATUS:
                continue
            output.append(
                {
                    "id": f"feedback-{item['fingerprint'][:12]}",
                    "image_path": self._runtime_path(item["image_path"]),
                    "query": item["query"],
                    "label": item["correct_worker"],
                    "split": "train",
                    "label_reason": item["reason"],
                    "risk_level": (
                        "high" if item["correct_worker"] == "Worker-D" else "medium"
                    ),
                    "template_group": f"feedback_{item['case_fingerprint'][:8]}",
                    "source": item.get("signal_source", "user_feedback"),
                    "feedback_fingerprint": item["fingerprint"],
                    "environment_reward": item.get("environment_reward"),
                    "feedback_confidence": item.get("confidence", 1.0),
                }
            )
        return output

    # 导出 Task4 要求的增量训练 JSONL。
    def export_training_jsonl(self, path: str | Path) -> int:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for item in self.training_records():
            record = dict(item)
            image_path = Path(str(record.get("image_path") or ""))
            if image_path.is_absolute():
                record["image_path"] = Path(
                    os.path.relpath(image_path, target.parent.resolve())
                ).as_posix()
            records.append(record)
        content = "\n".join(
            json.dumps(record, ensure_ascii=False) for record in records
        )
        target.write_text(content + ("\n" if content else ""), encoding="utf-8")
        return len(records)
