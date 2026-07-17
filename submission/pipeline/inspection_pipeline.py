# 零售巡检 Pipeline，依赖 numpy 和 Pillow。

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from submission.pipeline.kv_store import InMemoryKVStore
    from submission.pipeline.synthesizer import EvidenceSynthesizer, utc_now_iso
    from submission.pipeline.worker_pool import WorkerPool
    from submission.router.mm_router import MultimodalRouter
else:
    from .kv_store import InMemoryKVStore
    from .synthesizer import EvidenceSynthesizer, utc_now_iso
    from .worker_pool import WorkerPool
    from ..router.mm_router import MultimodalRouter


INSPECTION_TYPES = {
    "商品盘点": "inventory",
    "货架盘点": "inventory",
    "合规检查": "compliance",
    "综合评估": "comprehensive",
}


# 编排巡检请求的完整处理流程。
class InspectionPipeline:
    # 初始化 Pipeline 依赖。
    def __init__(
        self,
        router: MultimodalRouter | None = None,
        worker_pool: WorkerPool | None = None,
        store: InMemoryKVStore | None = None,
        synthesizer: EvidenceSynthesizer | None = None,
    ) -> None:
        self.router = router or self._load_router()
        self.worker_pool = worker_pool or WorkerPool(mode="mock")
        self.store = store or InMemoryKVStore()
        self.synthesizer = synthesizer or EvidenceSynthesizer()

    # 加载 Task2 路由器权重。
    @staticmethod
    def _load_router() -> MultimodalRouter:
        router_dir = Path(__file__).resolve().parents[1] / "router"
        model_path = router_dir / "router_weights.npy"
        data_path = router_dir / "training_data.jsonl"
        router = MultimodalRouter()
        if model_path.exists():
            return router.load(model_path)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Task 2 router weights/data not found under {router_dir}"
            )
        if __package__ in (None, ""):
            from submission.router.mm_router import load_jsonl
        else:
            from ..router.mm_router import load_jsonl
        records = load_jsonl(data_path)
        return router.fit([item for item in records if item.get("split") == "train"])

    # 校验图片数量、巡检类型和门店编号。
    @staticmethod
    def _validate_inputs(
        image_paths: Iterable[str | Path], inspection_type: str, store_id: str
    ) -> list[str]:
        paths = [str(Path(path).resolve()) for path in image_paths]
        if not 1 <= len(paths) <= 5:
            raise ValueError("image_paths must contain between 1 and 5 images")
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"image does not exist: {missing[0]}")
        if inspection_type not in INSPECTION_TYPES:
            raise ValueError(
                f"inspection_type must be one of {sorted(INSPECTION_TYPES)}"
            )
        if not store_id.strip():
            raise ValueError("store_id must be non-empty")
        return paths

    # 根据巡检类型生成路由查询。
    @staticmethod
    def _query_for(inspection_type: str, count: int) -> str:
        mode = INSPECTION_TYPES[inspection_type]
        if mode == "inventory":
            text = "请盘点货架上的商品数量，并指出缺货空位，给出每张图的可见证据。"
        elif mode == "compliance":
            text = "请检查消防通道、遮挡物和价格标签，输出 pass/fail/unclear 及证据；高风险问题需要二次复核。"
        else:
            text = "请综合评估商品陈列、门店环境和合规风险；证据不足或 Worker 结论冲突时保留不确定性。"
        if count > 1:
            text += f"共有 {count} 张图片，请逐张分析后再合成。"
        return text

    # 判断结果或 D 的嵌套结果是否来自 Mock Worker。
    @staticmethod
    def _result_is_mock(result: dict[str, Any]) -> bool:
        if bool(result.get("mock")):
            return True
        sources = result.get("metadata", {}).get("source_results", [])
        return any(bool(source.get("mock")) for source in sources)

    # 执行一次巡检请求并返回经过校验的报告。
    def inspect(
        self,
        image_paths: Iterable[str | Path],
        inspection_type: str,
        store_id: str = "demo-store",
        request_id: str | None = None,
    ) -> dict[str, Any]:
        paths = self._validate_inputs(image_paths, inspection_type, store_id)
        request_id = request_id or f"insp-{uuid.uuid4().hex[:12]}"
        query = self._query_for(inspection_type, len(paths))
        inspection_time = utc_now_iso()
        results: list[dict[str, Any]] = []
        routing_log: list[dict[str, Any]] = []

        for index, image_path in enumerate(paths):
            route_started = time.perf_counter()
            decision = self.router.route(image_path, query)
            route_latency = (time.perf_counter() - route_started) * 1000
            worker_result = self.worker_pool.analyze(
                decision["worker"],
                image_path,
                decision["prompt_rewrite"],
                INSPECTION_TYPES[inspection_type],
                request_id,
            )
            worker_dict = worker_result.to_dict()
            results.append(worker_dict)
            key_prefix = (
                f"inspection:{request_id}:image-{index + 1}:{decision['worker']}"
            )
            self.store.set(f"{key_prefix}:route", decision)
            self.store.set(f"{key_prefix}:result", worker_dict)
            routing_log.append(
                {
                    "image": Path(image_path).name,
                    "worker": decision["worker"],
                    "latency_ms": round(route_latency + worker_result.latency_ms, 3),
                    "route_latency_ms": round(route_latency, 3),
                    "worker_latency_ms": round(worker_result.latency_ms, 3),
                    "raw_worker": decision["raw_worker"],
                    "strategy": decision["strategy"],
                    "confidence": decision["confidence"],
                    "gate_upgraded": decision["gate_upgraded"],
                    "gate_reasons": decision["gate_reasons"],
                    "query_intent": decision["query_intent"],
                    "execution_error": worker_result.error,
                }
            )

        has_nested_mock = any(self._result_is_mock(item) for item in results)
        report = self.synthesizer.synthesize(
            store_id=store_id,
            inspection_time=inspection_time,
            results=results,
            routing_log=routing_log,
            inspection_type=inspection_type,
            request_id=request_id,
            mock_mode=self.worker_pool.mode == "mock" or has_nested_mock,
        )
        self.store.set(f"inspection:{request_id}:report", report)
        return report


# 提供命令行巡检入口。
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Task 3 inspection Pipeline")
    parser.add_argument("--images", nargs="+", required=True, help="1–5 image paths")
    parser.add_argument(
        "--inspection-type", choices=sorted(INSPECTION_TYPES), default="综合评估"
    )
    parser.add_argument("--store-id", default="demo-store")
    parser.add_argument("--request-id")
    parser.add_argument("--worker-mode", choices=("mock", "real"), default="mock")
    args = parser.parse_args()
    pipeline = InspectionPipeline(worker_pool=WorkerPool(mode=args.worker_mode))
    print(
        json.dumps(
            pipeline.inspect(
                args.images, args.inspection_type, args.store_id, args.request_id
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
