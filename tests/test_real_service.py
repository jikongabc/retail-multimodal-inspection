# 真实模型服务协议与验收逻辑测试。

from __future__ import annotations

import base64
import io
import unittest

from PIL import Image

from scripts.run_real_e2e import verify_report
from submission.pipeline.worker_pool import (
    _normalize_real_compliance,
    _normalize_real_findings,
)
from submission.services.ostrakon_server import decode_data_image, normalize_messages


# 验证真实服务输入协议和报告门禁。
class RealServiceTests(unittest.TestCase):
    # 构造测试用图片数据 URL。
    @staticmethod
    def image_url() -> str:
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    # 验证 OpenAI 图片消息可转换为 Qwen3-VL 格式。
    def test_normalize_openai_multimodal_message(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "检查货架"},
                    {"type": "image_url", "image_url": {"url": self.image_url()}},
                ],
            }
        ]
        normalized = normalize_messages(messages)
        self.assertEqual(normalized[0]["content"][0]["text"], "检查货架")
        self.assertEqual(normalized[0]["content"][1]["type"], "image")
        self.assertEqual(normalized[0]["content"][1]["image"].mode, "RGB")

    # 验证非法图片数据被拒绝。
    def test_invalid_image_data_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_data_image("https://example.com/image.jpg")

    # 验证真实报告必须通过全部工程门禁。
    def test_real_report_verification(self):
        report = {
            "request_id": "req",
            "mock_mode": False,
            "findings": [
                {
                    "category": "inventory",
                    "severity": "low",
                    "description": "存在空位",
                    "image_ref": "shelf.jpg",
                }
            ],
            "compliance_items": [],
            "routing_log": [
                {
                    "worker": "Worker-A",
                    "route_latency_ms": 10,
                    "execution_error": None,
                    "model_revision": "Ostrakon/Ostrakon-VL-8B",
                }
            ],
        }
        self.assertTrue(verify_report(report)["passed"])

    # 验证模型自由字段可归一化为巡检契约。
    def test_real_payload_normalization(self):
        parsed = {
            "findings": [
                {
                    "location": "左侧货架",
                    "item": "锅具",
                    "count": 6,
                    "evidence": "可见锅具",
                }
            ],
            "compliance_items": [
                {"item": "货架空位", "status": "存在", "evidence": "可见空位"}
            ],
        }
        findings = _normalize_real_findings(parsed, "/tmp/shelf.jpg")
        compliance = _normalize_real_compliance(parsed)
        self.assertEqual(findings[0]["category"], "锅具")
        self.assertEqual(findings[0]["image_ref"], "shelf.jpg")
        self.assertIn("数量：6", findings[0]["description"])
        self.assertEqual(compliance[0]["status"], "fail")


if __name__ == "__main__":
    unittest.main()
