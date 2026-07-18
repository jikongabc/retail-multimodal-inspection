# 反馈审核、ReAct 行动和模型回滚的回归测试。

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from submission.innovation.feedback_store import Feedback, FeedbackStore
from submission.innovation.incremental_trainer import IncrementalTrainer
from submission.innovation.model_registry import ModelRegistry
from submission.innovation.react_flywheel import ReActFlywheel


# 模拟路由器以隔离反馈闭环逻辑。
class _FakeRouter:
    # 初始化被加载 checkpoint 的记录字段。
    def __init__(self):
        self.loaded = None

    # 返回可控的裸模型与最终路由结果。
    def route(self, image_path, query):
        raw = "Worker-B" if "near" in query else "Worker-A"
        final = "Worker-A"
        return {
            "raw_worker": raw,
            "worker": final,
            "gate_reasons": ["intent_override"] if raw != final else [],
        }

    # 记录热加载候选模型的路径。
    def load(self, checkpoint):
        self.loaded = str(checkpoint)
        return self


# 模拟训练器以验证飞轮传参和发布动作。
class _FakeTrainer:
    # 初始化训练参数记录字段。
    def __init__(self):
        self.kwargs = None

    # 返回通过门禁的训练结果。
    def train_if_ready(self, version, checkpoint, minimum_feedback=5, **kwargs):
        self.kwargs = {
            "version": version,
            "checkpoint": str(checkpoint),
            "minimum_feedback": minimum_feedback,
            **kwargs,
        }
        return {"status": "trained", "gate_passed": True}


# 覆盖反馈飞轮的安全约束和生命周期行为。
class FeedbackLoopTests(unittest.TestCase):
    # 构造测试用反馈对象。
    def _feedback(self, label="Worker-A", **kwargs):
        values = {
            "request_id": "req-1",
            "image_path": "fixture.jpg",
            "query": "检查货架",
            "original_worker": "Worker-B",
            "correct_worker": label,
            "reason": "环境金标",
            "source_split": "production",
        }
        values.update(kwargs)
        return Feedback(**values)

    # 验证只有可信高置信反馈会自动进入训练。
    def test_only_trusted_high_confidence_feedback_is_auto_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback.jsonl")
            store.add(
                self._feedback(
                    signal_source="environment_gold",
                    confidence=0.95,
                    auto_approve=True,
                )
            )
            self.assertEqual(store.counts(), {"approved": 1})
            self.assertEqual(len(store.training_records()), 1)
            self.assertFalse(Path(store.all()[0]["image_path"]).is_absolute())
            self.assertTrue(
                Path(store.training_records()[0]["image_path"]).is_absolute()
            )

            store.add(
                self._feedback(
                    label="Worker-B",
                    query="另一条反馈",
                    signal_source="model_self_label",
                    confidence=1.0,
                    auto_approve=True,
                )
            )
            self.assertEqual(store.counts()["pending_review"], 1)

    # 验证同一环境样本的互斥标签会被隔离。
    def test_conflicting_labels_are_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback.jsonl")
            store.add(
                self._feedback(
                    signal_source="environment_gold",
                    auto_approve=True,
                )
            )
            store.add(
                self._feedback(
                    label="Worker-C",
                    signal_source="environment_gold",
                    auto_approve=True,
                )
            )
            self.assertEqual(store.counts(), {"quarantined_conflict": 2})
            self.assertEqual(store.training_records(), [])

    # 验证门控救回的 near-miss 会生成训练信号并触发热加载。
    def test_react_loop_learns_from_guardrail_near_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = FeedbackStore(root / "feedback.jsonl")
            router = _FakeRouter()
            trainer = _FakeTrainer()
            loop = ReActFlywheel(router, store, trainer, root / "trace.jsonl")
            result = loop.run(
                [
                    {
                        "id": "env-1",
                        "image_path": "fixture.jpg",
                        "query": "near miss",
                        "label": "Worker-A",
                        "split": "production",
                    }
                ],
                version="v2",
                checkpoint=root / "candidate.npy",
                baseline_checkpoint=root / "baseline.npy",
                minimum_feedback=1,
                generations=1,
            )
            self.assertEqual(result["action_counts"], {"approve_training_signal": 1})
            self.assertEqual(result["lifecycle_action"], "publish_and_hot_load")
            self.assertEqual(store.counts(), {"approved": 1})
            self.assertEqual(router.loaded, str(root / "candidate.npy"))
            self.assertIsNotNone(trainer.kwargs)

    # 验证模型注册表可回滚到父版本。
    def test_registry_can_rollback_to_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ModelRegistry(root / "registry.jsonl")
            registry.register("v1", "v1.npy", {}, True)
            registry.activate("v1")
            registry.register("v2", "v2.npy", {}, True, parent="v1")
            registry.activate("v2")

            target = registry.rollback(reason="online regression")

            self.assertEqual(target["version"], "v1")
            self.assertEqual(registry.active()["version"], "v1")
            self.assertEqual(registry.resolve_checkpoint("v1"), root / "v1.npy")

    # 验证训练触发只统计当前激活模型未消费的反馈。
    def test_trigger_counts_only_feedback_not_consumed_by_active_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = FeedbackStore(root / "feedback.jsonl")
            feedback = self._feedback(
                signal_source="environment_gold",
                auto_approve=True,
            )
            store.add(feedback)
            registry = ModelRegistry(root / "registry.jsonl")
            registry.register(
                "v1",
                "v1.npy",
                {},
                True,
                metadata={"feedback_fingerprints": [feedback.fingerprint]},
            )
            registry.activate("v1")
            trainer = IncrementalTrainer(root / "unused.jsonl", store, registry)

            result = trainer.train_if_ready("v2", root / "v2.npy", minimum_feedback=1)

            self.assertEqual(result["status"], "waiting")
            self.assertEqual(result["new_feedback_count"], 0)

    # 验证提交产物不包含本机 checkout 绝对路径。
    def test_committed_experiment_artifacts_do_not_embed_checkout_path(self):
        root = Path(__file__).resolve().parents[3]
        for name in (
            "feedback.jsonl",
            "model_registry.jsonl",
            "experiment_results.json",
        ):
            content = (root / "submission/innovation" / name).read_text(
                encoding="utf-8"
            )
            self.assertNotIn(str(root), content, name)


if __name__ == "__main__":
    unittest.main()
