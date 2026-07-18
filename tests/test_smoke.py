from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from submission.pipeline import InspectionPipeline
from submission.pipeline.worker_pool import WorkerPool
from submission.router.data_validation import validate_file
from submission.innovation.feedback_store import Feedback, FeedbackStore


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "submission/router/fixtures/a_00.png"


class SmokeTests(unittest.TestCase):
    def test_router_dataset_is_valid(self):
        report = validate_file(ROOT / "submission/router/training_data.jsonl")
        self.assertEqual(report["total"], 91)
        self.assertEqual(report["split_counts"], {"train": 60, "validation": 16, "test": 15})
        self.assertFalse(report["errors"])

    def test_pipeline_report_is_schema_valid(self):
        report = InspectionPipeline(worker_pool=WorkerPool("mock")).inspect([FIXTURE], "商品盘点")
        self.assertTrue(report["mock_mode"])
        self.assertEqual(report["routing_log"][0]["worker"], "Worker-A")

    def test_real_worker_json_parse_failure_is_explicit(self):
        from submission.pipeline.worker_pool import _extract_json
        self.assertIn("_parse_error", _extract_json("not json"))

    def test_feedback_is_deduplicated_and_test_feedback_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback.jsonl")
            feedback = Feedback("req-1", str(FIXTURE), "请盘点", "Worker-B", "Worker-A", "人工复核", "production")
            self.assertTrue(store.add(feedback))
            self.assertFalse(store.add(feedback))
            self.assertFalse(store.ready(2))
            self.assertTrue(store.ready(1))
            with self.assertRaises(ValueError):
                store.add(Feedback("req-2", str(FIXTURE), "请盘点", "Worker-B", "Worker-A", "测试", "test"))


if __name__ == "__main__":
    unittest.main()
