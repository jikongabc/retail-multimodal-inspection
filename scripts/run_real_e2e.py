# 启动 Ostrakon 服务并执行一次真实多模态 Pipeline 验收。

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# 等待真实模型服务完成加载。
def wait_for_health(endpoint: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "service not ready"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{endpoint}/healthz", timeout=3) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
    raise TimeoutError(f"Ostrakon startup timed out: {last_error}")


# 检查真实报告包含模型执行证据且未退化为 Mock。
def verify_report(report: dict[str, Any]) -> dict[str, Any]:
    routing_log = report.get("routing_log", [])
    findings = report.get("findings", [])
    compliance_items = report.get("compliance_items", [])
    checks = {
        "schema_output_present": bool(report.get("request_id")),
        "real_mode": report.get("mock_mode") is False,
        "routing_log_present": bool(routing_log),
        "worker_a_selected": any(
            row.get("worker") in {"Worker-A", "Worker-D"} for row in routing_log
        ),
        "worker_execution_succeeded": all(
            not row.get("execution_error") for row in routing_log
        ),
        "real_model_revision_present": all(
            "Ostrakon" in str(row.get("model_revision", "")) for row in routing_log
        ),
        "router_latency_under_one_second": all(
            float(row.get("route_latency_ms", 1000)) < 1000 for row in routing_log
        ),
        "structured_evidence_present": bool(findings or compliance_items),
        "portable_image_references": all(
            not Path(str(item.get("image_ref", ""))).is_absolute() for item in findings
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


# 生成不含机器路径的真实验收说明。
def write_evidence_readme(
    output_dir: Path, report: dict[str, Any], health: dict[str, Any]
) -> None:
    route = report["routing_log"][0]
    content = f"""# 云端真实模型验收

| 项目 | 结果 |
|---|---|
| GPU | {health["device"]} 24GB |
| Worker-A | `{health["model"]}`，BF16 |
| 模型加载显存 | {health["loaded_vram_mb"]} MB |
| 路由结果 | {route["worker"]}，{route["strategy"]} |
| 路由延迟 | {route["route_latency_ms"]:.3f} ms |
| Worker 延迟 | {route["worker_latency_ms"]:.3f} ms |
| Pipeline 输出 | Schema 校验通过 |
| Mock 降级 | 否 |

`input.json` 保存可移植输入，`output.json` 保存完整巡检报告，`verification.json` 保存真实模型、路由延迟、执行状态、结构化证据和路径可移植性门禁结果。

测试图片 `submission/router/fixtures/a_00.jpg` 来自项目公开许可夹具，仅证明真实工程链路可运行，不代表门店生产数据上的泛化效果。
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


# 运行服务、调用 Pipeline 并保存可审计证据。
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real Ostrakon E2E smoke test")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument(
        "--image",
        default=str(ROOT / "submission/router/fixtures/a_00.jpg"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "submission/pipeline/demos/real_cloud"),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"http://127.0.0.1:{args.port}"
    log_path = output_dir / "ostrakon_server.log"
    command = [
        sys.executable,
        "-m",
        "submission.services.ostrakon_server",
        "--model-path",
        args.model_path,
        "--port",
        str(args.port),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            health = wait_for_health(endpoint, args.startup_timeout)
            os.environ["OSTRAKON_BASE_URL"] = f"{endpoint}/v1"
            os.environ["OSTRAKON_MODEL"] = "Ostrakon/Ostrakon-VL-8B"
            from submission.pipeline.inspection_pipeline import InspectionPipeline
            from submission.pipeline.worker_pool import WorkerPool

            pipeline = InspectionPipeline(worker_pool=WorkerPool("real"))
            report = pipeline.inspect(
                [args.image],
                "商品盘点",
                store_id="cloud-real-store",
                request_id="cloud-real-e2e",
            )
            verification = verify_report(report)
            input_record = {
                "image": str(Path(args.image).resolve().relative_to(ROOT)),
                "inspection_type": "商品盘点",
                "store_id": "cloud-real-store",
                "model": "Ostrakon/Ostrakon-VL-8B",
                "endpoint": f"{endpoint}/v1",
            }
            (output_dir / "input.json").write_text(
                json.dumps(input_record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (output_dir / "output.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (output_dir / "verification.json").write_text(
                json.dumps(
                    {"health": health, **verification}, ensure_ascii=False, indent=2
                )
                + "\n",
                encoding="utf-8",
            )
            write_evidence_readme(output_dir, report, health)
            print(
                json.dumps(
                    {"health": health, **verification}, ensure_ascii=False, indent=2
                )
            )
            if not verification["passed"]:
                raise RuntimeError("real E2E verification failed")
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
