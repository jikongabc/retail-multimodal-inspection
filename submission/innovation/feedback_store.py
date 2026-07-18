"""Validated, deduplicated user feedback storage for Task 4."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from submission.router.mm_router import WORKERS


@dataclass(frozen=True)
class Feedback:
    request_id: str
    image_path: str
    query: str
    original_worker: str
    correct_worker: str
    reason: str
    source_split: str = "production"

    @property
    def fingerprint(self) -> str:
        value = "\x1f".join((self.image_path, self.query, self.correct_worker))
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


class FeedbackStore:
    """Append-only JSONL store; test-set feedback is rejected by design."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def add(self, feedback: Feedback) -> bool:
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
        records = self._read()
        if any(item.get("fingerprint") == feedback.fingerprint for item in records):
            return False
        record = asdict(feedback)
        record["fingerprint"] = feedback.fingerprint
        record["status"] = "pending_review"
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True

    def pending(self) -> list[dict]:
        return [item for item in self._read() if item.get("status") == "pending_review"]

    def ready(self, minimum: int = 5) -> bool:
        """Return whether enough deduplicated feedback exists to trigger training."""
        if minimum <= 0:
            raise ValueError("minimum must be positive")
        return len(self.training_records()) >= minimum

    def training_records(
        self, include_status: tuple[str, ...] = ("approved", "pending_review")
    ) -> list[dict]:
        """Convert feedback into router records without changing the fixed test set."""
        output = []
        for item in self._read():
            if item.get("status") not in include_status:
                continue
            output.append(
                {
                    "id": f"feedback-{item['fingerprint'][:12]}",
                    "image_path": item["image_path"],
                    "query": item["query"],
                    "label": item["correct_worker"],
                    "split": "train",
                    "label_reason": item["reason"],
                    "risk_level": "high"
                    if item["correct_worker"] == "Worker-D"
                    else "medium",
                    "template_group": f"feedback_{item['fingerprint'][:8]}",
                    "source": "user_feedback",
                    "feedback_fingerprint": item["fingerprint"],
                }
            )
        return output
