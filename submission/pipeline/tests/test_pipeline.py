# 零售巡检 Pipeline 测试。

from __future__ import annotations

import unittest
from pathlib import Path

from submission.pipeline import InspectionPipeline
from submission.pipeline.kv_store import InMemoryKVStore
from submission.pipeline.schemas import ReportValidationError, is_valid_report
from submission.pipeline.synthesizer import EvidenceSynthesizer
from submission.pipeline.worker_pool import WorkerPool, _profile_for


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "submission" / "router" / "fixtures"


# 验证 Pipeline、Worker、Schema 和临时存储。
class PipelineTests(unittest.TestCase):
    # 创建测试用的 Mock Pipeline。
    def make_pipeline(self, profiles=None):
        return InspectionPipeline(
            worker_pool=WorkerPool("mock", profiles or {}),
            store=InMemoryKVStore(),
        )

    # 验证单图和五图输入。
    def test_one_and_five_images_are_supported(self):
        images = [FIXTURES / f"a_{i:02d}.png" for i in range(5)]
        pipeline = self.make_pipeline()
        one = pipeline.inspect(images[:1], "商品盘点", request_id="one")
        five = pipeline.inspect(images, "商品盘点", request_id="five")
        self.assertTrue(is_valid_report(one))
        self.assertTrue(is_valid_report(five))
        self.assertEqual(len(one["routing_log"]), 1)
        self.assertEqual(len(five["routing_log"]), 5)
        self.assertIn("inspection:five:report", pipeline.store.keys("inspection:five"))

    # 验证 D 的并行冲突输出为 unclear。
    def test_d_parallel_path_preserves_conflict_as_unclear(self):
        image = FIXTURES / "d_02.png"
        pipeline = self.make_pipeline({str(image.resolve()): "conflict"})
        report = pipeline.inspect([image], "综合评估", request_id="conflict")
        self.assertEqual(report["routing_log"][0]["worker"], "Worker-D")
        self.assertEqual(report["compliance_items"][0]["status"], "unclear")
        self.assertIn("冲突", report["compliance_items"][0]["evidence"])

    # 验证非法输入和报告结构会被拒绝。
    def test_invalid_input_and_schema_are_rejected(self):
        pipeline = self.make_pipeline()
        with self.assertRaises(ValueError):
            pipeline.inspect([], "商品盘点")
        with self.assertRaises(ValueError):
            pipeline.inspect([FIXTURES / "a_00.png"], "未知类型")
        with self.assertRaises(ReportValidationError):
            from submission.pipeline.schemas import validate_report

            validate_report({"store_id": "x"})

    # 验证四种 Worker 策略均可调度。
    def test_all_worker_strategies_are_dispatchable(self):
        pipeline = self.make_pipeline()
        image = str((FIXTURES / "a_00.png").resolve())
        for worker in ("Worker-A", "Worker-B", "Worker-C", "Worker-D"):
            result = pipeline.worker_pool.analyze(
                worker, image, "请分析", "inventory", "test"
            )
            self.assertEqual(result.worker_id, worker)
            self.assertGreaterEqual(result.latency_ms, 0)

    # 验证合规状态参与总分计算。
    def test_compliance_penalties_affect_overall_score(self):
        synthesizer = EvidenceSynthesizer()
        base = {
            "worker_id": "Worker-A",
            "image_path": "exit.png",
            "raw_text": "检查完成",
            "findings": [],
            "confidence": 0.9,
            "mock": True,
            "metadata": {},
        }
        passing = dict(
            base,
            compliance_items=[
                {"item": "消防通道", "status": "pass", "evidence": "无遮挡"}
            ],
        )
        failing = dict(
            base,
            compliance_items=[
                {"item": "消防通道", "status": "fail", "evidence": "有障碍物"}
            ],
        )
        pass_report = synthesizer.synthesize(
            "store", "2026-07-18T00:00:00Z", [passing], [], "合规检查", "pass", True
        )
        fail_report = synthesizer.synthesize(
            "store", "2026-07-18T00:00:00Z", [failing], [], "合规检查", "fail", True
        )
        self.assertEqual(pass_report["overall_score"], 100)
        self.assertEqual(fail_report["overall_score"], 85)

    # 验证非安全类失败不会生成消防整改建议。
    def test_non_safety_failure_uses_generic_remediation(self):
        report = EvidenceSynthesizer().synthesize(
            "store",
            "2026-07-18T00:00:00Z",
            [
                {
                    "worker_id": "Worker-A",
                    "image_path": "shelf.jpg",
                    "raw_text": "发现空位",
                    "findings": [],
                    "compliance_items": [
                        {"item": "货架空位", "status": "fail", "evidence": "可见空位"}
                    ],
                    "confidence": 0.9,
                    "mock": False,
                    "metadata": {},
                }
            ],
            [],
            "商品盘点",
            "inventory-fail",
            False,
        )
        self.assertIn("复核未通过的合规项", report["recommendations"][0])
        self.assertNotIn("消防", "".join(report["recommendations"]))

    # 验证语义文件名可以解析场景。
    def test_profile_keywords_support_semantic_demo_names(self):
        self.assertEqual(_profile_for("/tmp/shelf_front.png", {}), "shelf")
        self.assertEqual(_profile_for("/tmp/exit_obstructed.png", {}), "safety")
        self.assertEqual(_profile_for("/tmp/open_domain.png", {}), "open")
        self.assertEqual(_profile_for("/tmp/conflict_exit.png", {}), "conflict")

    # 验证键值存储的读写和前缀查询。
    def test_kv_store_round_trip_and_prefix_listing(self):
        store = InMemoryKVStore()
        store.set("inspection:req:route", {"worker": "Worker-A"})
        store.set("inspection:req:result", {"ok": True})
        self.assertEqual(store.get("inspection:req:route")["worker"], "Worker-A")
        self.assertEqual(
            store.keys("inspection:req"),
            ["inspection:req:result", "inspection:req:route"],
        )
        self.assertEqual(
            store.dump("inspection:req")["inspection:req:result"], {"ok": True}
        )

    # 验证 D 缺少来源 Worker 时返回明确错误。
    def test_worker_d_reports_missing_source_workers(self):
        pool = WorkerPool("mock")
        del pool.workers["Worker-B"]
        with self.assertRaisesRegex(
            RuntimeError, "requires both Worker-A and Worker-B"
        ):
            pool.analyze(
                "Worker-D",
                str((FIXTURES / "a_00.png").resolve()),
                "请分析",
                "inventory",
                "missing",
            )

    # 验证时间字段必须为带时区的 ISO-8601 格式。
    def test_schema_rejects_invalid_or_timezone_free_timestamps(self):
        from submission.pipeline.schemas import validate_report

        report = {
            "store_id": "store",
            "inspection_time": "helloTworld",
            "overall_score": 100,
            "routing_log": [],
            "findings": [],
            "compliance_items": [],
            "recommendations": [],
        }
        with self.assertRaises(ReportValidationError):
            validate_report(report)


if __name__ == "__main__":
    unittest.main()
