from __future__ import annotations

import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

import numpy as np

from submission.pipeline import InspectionPipeline
from submission.pipeline.worker_pool import WorkerPool
from submission.router.data_validation import validate_file
from submission.router.mm_router import MultimodalRouter, load_jsonl
from submission.innovation.feedback_store import Feedback, FeedbackStore


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "submission/router/fixtures/a_00.png"


class SmokeTests(unittest.TestCase):
    def test_router_dataset_is_valid(self):
        report = validate_file(ROOT / "submission/router/training_data.jsonl")
        self.assertEqual(report["total"], 91)
        self.assertEqual(
            report["split_counts"], {"train": 60, "validation": 16, "test": 15}
        )
        self.assertFalse(report["errors"])

    def test_pipeline_report_is_schema_valid(self):
        report = InspectionPipeline(worker_pool=WorkerPool("mock")).inspect(
            [FIXTURE], "商品盘点"
        )
        self.assertTrue(report["mock_mode"])
        self.assertEqual(report["routing_log"][0]["worker"], "Worker-A")
        self.assertIn("model_revision", report["routing_log"][0])
        self.assertTrue(report["routing_log"][0]["mock"])

    def test_router_reports_rule_baseline_and_cost(self):
        records = load_jsonl(ROOT / "submission/router/training_data.jsonl")
        router = MultimodalRouter().load(ROOT / "submission/router/router_weights.npy")
        result = router.evaluate([row for row in records if row["split"] == "test"])
        self.assertIn("rule_only_metrics", result)
        self.assertIn("estimated_cost", result)
        self.assertIn("gate_change_rate", result)
        self.assertIn("gate_net_benefit", result)
        self.assertNotIn("gate_upgrade_rate", result)

    def test_rule_only_reuses_gate_text_only_and_image_policy(self):
        router = MultimodalRouter()
        text_only = "已有巡检结果，请仅根据上游文字结果生成报告，不要做视觉判断"
        self.assertEqual(router._rule_only_worker(text_only, FIXTURE), 2)
        decision = router._gate(np.zeros(4, dtype=float), text_only, FIXTURE)
        self.assertEqual(decision["final_idx"], 2)
        self.assertIn("explicit_text_only", decision["gate_reasons"])

    def test_real_mode_can_select_optional_b_and_c_adapters(self):
        with patch.dict(
            "os.environ",
            {
                "OSTRAKON_BASE_URL": "http://a.example/v1",
                "WORKER_B_BASE_URL": "http://b.example/v1",
                "WORKER_C_BASE_URL": "http://c.example/v1",
            },
            clear=False,
        ):
            pool = WorkerPool("real")
            self.assertEqual(pool.workers["Worker-A"].worker_id, "Worker-A")
            self.assertEqual(pool.workers["Worker-B"].worker_id, "Worker-B")
            self.assertEqual(pool.workers["Worker-C"].worker_id, "Worker-C")

    def test_real_worker_json_parse_failure_is_explicit(self):
        from submission.pipeline.worker_pool import _extract_json

        self.assertIn("_parse_error", _extract_json("not json"))

    def test_feedback_is_deduplicated_and_test_feedback_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback.jsonl")
            feedback = Feedback(
                "req-1",
                str(FIXTURE),
                "请盘点",
                "Worker-B",
                "Worker-A",
                "人工复核",
                "production",
            )
            self.assertTrue(store.add(feedback))
            self.assertFalse(store.add(feedback))
            self.assertFalse(store.ready(2))
            self.assertTrue(store.ready(1))
            with self.assertRaises(ValueError):
                store.add(
                    Feedback(
                        "req-2",
                        str(FIXTURE),
                        "请盘点",
                        "Worker-B",
                        "Worker-A",
                        "测试",
                        "test",
                    )
                )


if __name__ == "__main__":
    unittest.main()
