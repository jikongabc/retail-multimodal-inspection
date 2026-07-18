"""Replay-based incremental sep-CMA-ES training and regression gate."""

from __future__ import annotations

import json
from pathlib import Path

from submission.router.mm_router import MultimodalRouter, RouterConfig, load_jsonl

from .feedback_store import FeedbackStore
from .model_registry import ModelRegistry


class IncrementalTrainer:
    def __init__(self, data_path: str | Path, feedback_store: FeedbackStore, registry: ModelRegistry):
        self.data_path = Path(data_path)
        self.feedback_store = feedback_store
        self.registry = registry

    @staticmethod
    def _gate_passed(before: dict, after: dict) -> bool:
        # Protect both the learnable router and the deployed gated policy.
        for metric_name in ("raw_metrics", "gated_metrics"):
            before_metrics = before["test"][metric_name]
            after_metrics = after["test"][metric_name]
            if after_metrics["macro_f1"] + 0.01 < before_metrics["macro_f1"]:
                return False
            for worker, metrics in before_metrics["per_class"].items():
                if after_metrics["per_class"][worker]["recall"] + 0.05 < metrics["recall"]:
                    return False
        return True

    def train(self, version: str, checkpoint: str | Path, seed: int = 7, generations: int = 90) -> dict:
        records = load_jsonl(self.data_path)
        base_train = [item for item in records if item.get("split") == "train"]
        fixed_test = [item for item in records if item.get("split") == "test"]
        feedback = self.feedback_store.training_records()
        if not feedback:
            raise ValueError("no feedback records available")
        baseline = MultimodalRouter(config=RouterConfig(seed=seed, generations=generations))
        baseline.fit(base_train)
        before = {"test": baseline.evaluate(fixed_test)}
        candidate = MultimodalRouter(config=RouterConfig(seed=seed, generations=generations))
        candidate.fit(base_train + feedback)
        after = {"test": candidate.evaluate(fixed_test)}
        gate_passed = self._gate_passed(before, after)
        if gate_passed:
            candidate.save(checkpoint)
        record = self.registry.register(version, checkpoint, {"before": before, "after": after, "feedback_count": len(feedback)}, gate_passed, self.registry.active().get("version") if self.registry.active() else None)
        if gate_passed:
            self.registry.activate(version)
        result = {"version": version, "gate_passed": gate_passed, "feedback_count": len(feedback), "before": before, "after": after, "registry": record}
        return result

    def train_if_ready(self, version: str, checkpoint: str | Path, minimum_feedback: int = 5, seed: int = 7, generations: int = 90) -> dict:
        """Trigger training only after the configured feedback threshold is reached."""
        if not self.feedback_store.ready(minimum_feedback):
            return {"status": "waiting", "feedback_count": len(self.feedback_store.training_records()), "minimum_feedback": minimum_feedback}
        result = self.train(version, checkpoint, seed=seed, generations=generations)
        result["status"] = "trained"
        return result
