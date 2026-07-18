# 基于回放训练和回归门禁的增量训练器。

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from submission.router.mm_router import MultimodalRouter, RouterConfig, load_jsonl

from .feedback_store import FeedbackStore, portable_path_identity
from .model_registry import ModelRegistry


# 训练并发布通过门禁的路由器候选版本。
class IncrementalTrainer:
    # 保存固定数据集、反馈仓库和模型注册表依赖。
    def __init__(
        self,
        data_path: str | Path,
        feedback_store: FeedbackStore,
        registry: ModelRegistry,
    ):
        self.data_path = Path(data_path)
        self.feedback_store = feedback_store
        self.registry = registry

    @staticmethod
    # 生成固定测试集上的宏平均和类别召回回归检查。
    def _fixed_regression_checks(before: dict, after: dict) -> list[dict]:
        checks = []
        for metric_name in ("raw_metrics", "gated_metrics"):
            old = before[metric_name]
            new = after[metric_name]
            checks.append(
                {
                    "name": f"fixed_{metric_name}_macro_f1",
                    "passed": new["macro_f1"] + 0.01 >= old["macro_f1"],
                    "before": old["macro_f1"],
                    "after": new["macro_f1"],
                    "tolerance": -0.01,
                }
            )
            for worker, old_class in old["per_class"].items():
                new_recall = new["per_class"][worker]["recall"]
                checks.append(
                    {
                        "name": f"fixed_{metric_name}_{worker}_recall",
                        "passed": new_recall + 0.05 >= old_class["recall"],
                        "before": old_class["recall"],
                        "after": new_recall,
                        "tolerance": -0.05,
                    }
                )
        return checks

    @classmethod
    # 汇总固定测试集和环境挑战集的发布门禁结果。
    def _gate_report(
        cls,
        before_test: dict,
        after_test: dict,
        before_challenge: dict,
        after_challenge: dict,
    ) -> dict:
        checks = cls._fixed_regression_checks(before_test, after_test)
        checks.extend(
            [
                {
                    "name": "challenge_gated_accuracy_non_regression",
                    "passed": (
                        after_challenge["gated_metrics"]["accuracy"] + 0.01
                        >= before_challenge["gated_metrics"]["accuracy"]
                    ),
                    "before": before_challenge["gated_metrics"]["accuracy"],
                    "after": after_challenge["gated_metrics"]["accuracy"],
                    "tolerance": -0.01,
                },
                {
                    "name": "challenge_learning_gain",
                    "passed": (
                        after_challenge["raw_metrics"]["accuracy"]
                        > before_challenge["raw_metrics"]["accuracy"]
                        or after_challenge["gated_metrics"]["accuracy"]
                        > before_challenge["gated_metrics"]["accuracy"]
                    ),
                    "before": {
                        "raw_accuracy": before_challenge["raw_metrics"]["accuracy"],
                        "gated_accuracy": before_challenge["gated_metrics"]["accuracy"],
                    },
                    "after": {
                        "raw_accuracy": after_challenge["raw_metrics"]["accuracy"],
                        "gated_accuracy": after_challenge["gated_metrics"]["accuracy"],
                    },
                    "tolerance": 0.0,
                },
            ]
        )
        failed = [item["name"] for item in checks if not item["passed"]]
        return {"passed": not failed, "checks": checks, "failed_checks": failed}

    @staticmethod
    # 创建带固定随机种子的路由器实例。
    def _router(seed: int, generations: int) -> MultimodalRouter:
        return MultimodalRouter(config=RouterConfig(seed=seed, generations=generations))

    @staticmethod
    # 计算训练数据的可移植内容哈希。
    def _data_hash(records: list[dict]) -> str:
        portable_records = []
        for item in records:
            record = dict(item)
            if record.get("image_path"):
                record["image_path"] = portable_path_identity(str(record["image_path"]))
            portable_records.append(record)
        stable = json.dumps(
            portable_records, ensure_ascii=False, sort_keys=True, default=str
        )
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()

    # 训练反馈候选模型并按门禁决定发布或回滚。
    def train(
        self,
        version: str,
        checkpoint: str | Path,
        *,
        seed: int = 7,
        generations: int = 90,
        baseline_checkpoint: str | Path | None = None,
        challenge_records: list[dict] | None = None,
        replay_weight: int = 1,
    ) -> dict:
        if replay_weight <= 0:
            raise ValueError("replay_weight must be positive")
        records = load_jsonl(self.data_path)
        base_train = [item for item in records if item.get("split") == "train"]
        fixed_test = [item for item in records if item.get("split") == "test"]
        feedback = self.feedback_store.training_records()
        if not feedback:
            raise ValueError("no approved feedback records available")
        challenge = challenge_records or feedback

        baseline = self._router(seed, generations)
        if baseline_checkpoint and Path(baseline_checkpoint).exists():
            baseline.load(baseline_checkpoint)
        else:
            baseline.fit(base_train)
        before = {
            "fixed_test": baseline.evaluate(fixed_test),
            "environment_challenge": baseline.evaluate(challenge),
        }

        started = time.perf_counter()
        feedback_only = self._router(seed, generations).fit(feedback)
        replay_records = base_train + feedback * replay_weight
        replay = self._router(seed, generations).fit(replay_records)
        duration_ms = round((time.perf_counter() - started) * 1000, 3)

        strategy_results = {
            "feedback_only": {
                "train_count": len(feedback),
                "fixed_test": feedback_only.evaluate(fixed_test),
                "environment_challenge": feedback_only.evaluate(challenge),
            },
            "replay_plus_feedback": {
                "base_replay_count": len(base_train),
                "feedback_count": len(feedback),
                "feedback_replay_weight": replay_weight,
                "fixed_test": replay.evaluate(fixed_test),
                "environment_challenge": replay.evaluate(challenge),
            },
        }
        selected = strategy_results["replay_plus_feedback"]
        gate = self._gate_report(
            before["fixed_test"],
            selected["fixed_test"],
            before["environment_challenge"],
            selected["environment_challenge"],
        )
        checkpoint = Path(checkpoint)
        if gate["passed"]:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            replay.save(checkpoint)

        active = self.registry.active()
        if active is None:
            baseline_version = "router-baseline-v1"
            known_versions = {item["version"] for item in self.registry.all()}
            if baseline_version not in known_versions:
                self.registry.register(
                    baseline_version,
                    baseline_checkpoint or "in_memory_baseline",
                    {"before": before},
                    True,
                    metadata={
                        "baseline": True,
                        "training_data_hash": self._data_hash(base_train),
                    },
                )
            self.registry.activate(baseline_version, reason="baseline_registration")
            active = self.registry.active()
        metrics = {
            "before": before,
            "strategies": strategy_results,
            "selected_strategy": "replay_plus_feedback",
            "gate": gate,
        }
        record = self.registry.register(
            version,
            checkpoint,
            metrics,
            gate["passed"],
            active.get("version") if active else None,
            metadata={
                "seed": seed,
                "generations": generations,
                "feedback_count": len(feedback),
                "feedback_fingerprints": sorted(
                    item["feedback_fingerprint"] for item in feedback
                ),
                "training_data_hash": self._data_hash(base_train),
                "feedback_data_hash": self._data_hash(feedback),
                "training_duration_ms": duration_ms,
                "artifact_saved": gate["passed"],
            },
        )
        action = "rollback_candidate"
        if gate["passed"]:
            self.registry.activate(version)
            action = "publish_candidate"
            record = self.registry.active() or record
        return {
            "version": version,
            "status": "trained",
            "action": action,
            "gate_passed": gate["passed"],
            "feedback_count": len(feedback),
            "before": before,
            "strategies": strategy_results,
            "selected_strategy": "replay_plus_feedback",
            "gate": gate,
            "registry": record,
        }

    # 仅在新增已批准反馈达到阈值时触发训练。
    def train_if_ready(
        self,
        version: str,
        checkpoint: str | Path,
        minimum_feedback: int = 5,
        **kwargs,
    ) -> dict:
        records = self.feedback_store.training_records()
        if minimum_feedback <= 0:
            raise ValueError("minimum_feedback must be positive")
        active = self.registry.active()
        consumed = set(
            (active or {}).get("metadata", {}).get("feedback_fingerprints", [])
        )
        new_records = [
            item for item in records if item.get("feedback_fingerprint") not in consumed
        ]
        if len(new_records) < minimum_feedback:
            return {
                "status": "waiting",
                "feedback_count": len(records),
                "new_feedback_count": len(new_records),
                "minimum_feedback": minimum_feedback,
            }
        return self.train(version, checkpoint, **kwargs)
