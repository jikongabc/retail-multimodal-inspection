# 巡检执行层 Worker 适配器。

from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


WORKER_IDS = ("Worker-A", "Worker-B", "Worker-C", "Worker-D")


# 定义 Worker 与合成器之间的统一结果。
@dataclass
class WorkerResult:
    worker_id: str
    image_path: str
    raw_text: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    compliance_items: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None
    model_revision: str = "unknown"
    mock: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # 序列化 Worker 结果。
    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "image_path": self.image_path,
            "raw_text": self.raw_text,
            "findings": self.findings,
            "compliance_items": self.compliance_items,
            "confidence": round(float(self.confidence), 4),
            "latency_ms": round(float(self.latency_ms), 3),
            "error": self.error,
            "model_revision": self.model_revision,
            "mock": self.mock,
            "metadata": self.metadata,
        }


# 定义 Worker 适配器接口。
class Worker(Protocol):
    worker_id: str

    # 执行 Worker 分析。
    def analyze(
        self, image_path: str, query: str, inspection_type: str, request_id: str
    ) -> WorkerResult: ...


# 根据显式配置或文件名解析测试场景。
def _profile_for(image_path: str, profiles: dict[str, str]) -> str:
    resolved = str(Path(image_path).resolve())
    if resolved in profiles:
        return profiles[resolved]
    name = Path(image_path).stem.lower()
    if name.startswith("a_"):
        return "shelf"
    if name.startswith("b_"):
        return "open"
    if name.startswith("c_"):
        return "report"
    if name.startswith("d_"):
        return "safety"
    if "conflict" in name:
        return "conflict"
    if any(token in name for token in ("shelf", "inventory", "stock", "hybrid")):
        return "shelf"
    if any(token in name for token in ("exit", "fire", "safety", "obstruct")):
        return "safety"
    if any(token in name for token in ("open", "domain")):
        return "open"
    if any(token in name for token in ("report", "document", "receipt")):
        return "report"
    return "generic"


# 提供确定性的本地测试 Worker。
class MockWorker:
    # 初始化 Mock Worker。
    def __init__(self, worker_id: str, profiles: dict[str, str] | None = None) -> None:
        self.worker_id = worker_id
        self.profiles = profiles or {}

    # 生成 Mock Worker 结果。
    def analyze(
        self, image_path: str, query: str, inspection_type: str, request_id: str
    ) -> WorkerResult:
        started = time.perf_counter()
        profile = _profile_for(image_path, self.profiles)
        result = self._respond(profile, image_path, query, inspection_type)
        result.latency_ms = (time.perf_counter() - started) * 1000
        result.mock = True
        result.model_revision = f"mock-{self.worker_id.lower().replace('-', '_')}-v1"
        result.metadata.update({"request_id": request_id, "profile": profile})
        return result

    # 根据场景生成 Worker 响应。
    def _respond(
        self, profile: str, image_path: str, query: str, inspection_type: str
    ) -> WorkerResult:
        ref = Path(image_path).name
        if self.worker_id == "Worker-A":
            if profile in ("shelf", "hybrid"):
                return WorkerResult(
                    self.worker_id,
                    image_path,
                    "识别到零售货架，已按层记录商品和空货位；结果来自 Mock 视觉适配器。",
                    findings=[
                        {
                            "category": "inventory",
                            "severity": "low",
                            "description": "货架存在商品陈列，发现少量空货位，建议补货复核。",
                            "image_ref": ref,
                        }
                    ],
                    compliance_items=[
                        {
                            "item": "货架陈列可见性",
                            "status": "pass",
                            "evidence": f"{ref} 中可见货架层和商品面。",
                        }
                    ],
                    confidence=0.88,
                )
            if profile in ("safety", "conflict"):
                return WorkerResult(
                    self.worker_id,
                    image_path,
                    "检测到消防出口附近存在障碍物，建议立即清理。",
                    findings=[
                        {
                            "category": "safety_obstruction",
                            "severity": "high",
                            "description": "消防出口/通道附近出现疑似障碍物。",
                            "image_ref": ref,
                        }
                    ],
                    compliance_items=[
                        {
                            "item": "消防通道无遮挡",
                            "status": "fail",
                            "evidence": f"{ref} 中出口前方可见障碍物或警示区域被占用。",
                        }
                    ],
                    confidence=0.84,
                )
            return WorkerResult(
                self.worker_id,
                image_path,
                "未发现足够零售证据，无法可靠盘点。",
                confidence=0.42,
            )

        if self.worker_id == "Worker-B":
            if profile == "open":
                return WorkerResult(
                    self.worker_id,
                    image_path,
                    "画面包含开放域场景和多个物体；不套用零售先验。",
                    findings=[
                        {
                            "category": "open_scene",
                            "severity": "low",
                            "description": "图片主要呈现开放域场景，不足以支持零售库存结论。",
                            "image_ref": ref,
                        }
                    ],
                    confidence=0.91,
                )
            if profile == "conflict":
                return WorkerResult(
                    self.worker_id,
                    image_path,
                    "通用视觉适配器未确认消防通道被完全阻塞，建议人工复核。",
                    findings=[
                        {
                            "category": "safety_obstruction",
                            "severity": "med",
                            "description": "可见警示区域，但障碍物是否侵入有效通道存在不确定性。",
                            "image_ref": ref,
                        }
                    ],
                    compliance_items=[
                        {
                            "item": "消防通道无遮挡",
                            "status": "pass",
                            "evidence": f"{ref} 中未能确认有效通道被完全封堵。",
                        }
                    ],
                    confidence=0.58,
                )
            return WorkerResult(
                self.worker_id,
                image_path,
                "通用视觉描述完成，未输出领域专项结论。",
                confidence=0.63,
            )

        if self.worker_id == "Worker-C":
            return WorkerResult(
                self.worker_id,
                image_path,
                "文本汇总适配器只基于已有上下文生成报告，不重复执行视觉识别。",
                findings=[
                    {
                        "category": "report_context",
                        "severity": "low",
                        "description": "该结果可作为巡检报告汇总上下文，不能替代视觉证据。",
                        "image_ref": ref,
                    }
                ],
                confidence=0.77,
            )

        raise ValueError(f"MockWorker does not handle {self.worker_id}")


# 提供 OpenAI 兼容接口的 Ostrakon Worker。
class OpenAICompatibleOstrakonWorker:
    # 初始化 OpenAI 兼容接口参数。
    def __init__(self, endpoint: str, model: str, api_key: str = "") -> None:
        self.worker_id = "Worker-A"
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key

    # 调用 OpenAI 兼容视觉接口。
    def analyze(
        self, image_path: str, query: str, inspection_type: str, request_id: str
    ) -> WorkerResult:
        started = time.perf_counter()
        try:
            encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
            prompt = (
                f"你是零售巡检视觉 Worker。巡检类型：{inspection_type}。\n{query}\n"
                "请只返回 JSON：{findings:[], compliance_items:[], summary:string, confidence:number}。"
            )
            payload = {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded}"
                                },
                            },
                        ],
                    }
                ],
            }
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    **(
                        {"Authorization": f"Bearer {self.api_key}"}
                        if self.api_key
                        else {}
                    ),
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            text = body["choices"][0]["message"]["content"]
            parsed = _extract_json(text)
            return WorkerResult(
                self.worker_id,
                image_path,
                text,
                findings=parsed.get("findings", []),
                compliance_items=parsed.get("compliance_items", []),
                confidence=float(parsed.get("confidence", 0.5)),
                latency_ms=(time.perf_counter() - started) * 1000,
                model_revision=self.model,
                metadata={"request_id": request_id, "mock": False},
            )
        except Exception as exc:
            return WorkerResult(
                self.worker_id,
                image_path,
                "",
                confidence=0.0,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"real_worker_error: {type(exc).__name__}: {exc}",
                model_revision=self.model,
                metadata={"request_id": request_id, "mock": False},
            )


# 从模型文本中解析 JSON 对象。
def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            return {"findings": [], "compliance_items": [], "summary": stripped}
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return {"findings": [], "compliance_items": [], "summary": stripped}
    return value if isinstance(value, dict) else {"summary": stripped}


# 调度 A、B、C 和并行执行 A+B 的 D。
class WorkerPool:
    # 初始化 Worker 池。
    def __init__(
        self, mode: str = "mock", mock_profiles: dict[str, str] | None = None
    ) -> None:
        if mode not in ("mock", "real"):
            raise ValueError("worker mode must be 'mock' or 'real'")
        self.mode = mode
        profiles = {
            str(Path(key).resolve()): value
            for key, value in (mock_profiles or {}).items()
        }
        if mode == "real":
            endpoint = os.getenv("OSTRAKON_BASE_URL")
            model = os.getenv("OSTRAKON_MODEL", "Ostrakon/Ostrakon-VL-8B")
            if not endpoint:
                raise ValueError("real mode requires OSTRAKON_BASE_URL")
            worker_a: Worker = OpenAICompatibleOstrakonWorker(
                endpoint, model, os.getenv("OSTRAKON_API_KEY", "")
            )
        else:
            worker_a = MockWorker("Worker-A", profiles)
        self.workers: dict[str, Worker] = {
            "Worker-A": worker_a,
            "Worker-B": MockWorker("Worker-B", profiles),
            "Worker-C": MockWorker("Worker-C", profiles),
        }

    # 执行指定 Worker 或 D 的并行策略。
    def analyze(
        self,
        worker_id: str,
        image_path: str,
        query: str,
        inspection_type: str,
        request_id: str,
    ) -> WorkerResult:
        if worker_id in self.workers:
            return self.workers[worker_id].analyze(
                image_path, query, inspection_type, request_id
            )
        if worker_id != "Worker-D":
            raise ValueError(f"unknown worker: {worker_id}")
        if "Worker-A" not in self.workers or "Worker-B" not in self.workers:
            raise RuntimeError("Worker-D requires both Worker-A and Worker-B")
        started = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="worker-d"
        ) as executor:
            future_a = executor.submit(
                self.workers["Worker-A"].analyze,
                image_path,
                query,
                inspection_type,
                request_id,
            )
            future_b = executor.submit(
                self.workers["Worker-B"].analyze,
                image_path,
                query,
                inspection_type,
                request_id,
            )
            result_a, result_b = future_a.result(), future_b.result()
        all_findings = result_a.findings + result_b.findings
        all_compliance = result_a.compliance_items + result_b.compliance_items
        errors = [item.error for item in (result_a, result_b) if item.error]
        return WorkerResult(
            "Worker-D",
            image_path,
            f"[A] {result_a.raw_text}\n[B] {result_b.raw_text}",
            findings=all_findings,
            compliance_items=all_compliance,
            confidence=min(result_a.confidence, result_b.confidence),
            latency_ms=(time.perf_counter() - started) * 1000,
            error="; ".join(errors) if errors else None,
            model_revision=f"parallel({result_a.model_revision},{result_b.model_revision})",
            mock=result_a.mock and result_b.mock,
            metadata={
                "request_id": request_id,
                "source_results": [result_a.to_dict(), result_b.to_dict()],
            },
        )
