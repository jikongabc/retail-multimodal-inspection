# 生成三组巡检 Demo 报告。

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.pipeline import InspectionPipeline
from submission.pipeline.worker_pool import WorkerPool


OUT = Path(__file__).resolve().parent


DEMOS = [
    {
        "name": "scenario_1_inventory",
        "title": "场景 1：货架盘点",
        "inspection_type": "商品盘点",
        "store_id": "store-shelf-001",
        "images": [OUT / "scenario_1_inventory" / "shelf_front.png", OUT / "scenario_1_inventory" / "shelf_oblique.png", OUT / "scenario_1_inventory" / "shelf_sparse.png"],
        "profiles": {str((OUT / "scenario_1_inventory" / name).resolve()): "shelf" for name in ("shelf_front.png", "shelf_oblique.png", "shelf_sparse.png")},
        "expected_workers": {"shelf_front.png": "Worker-A", "shelf_oblique.png": "Worker-A", "shelf_sparse.png": "Worker-A"},
    },
    {
        "name": "scenario_2_compliance",
        "title": "场景 2：高风险合规巡检",
        "inspection_type": "合规检查",
        "store_id": "store-safety-002",
        "images": [OUT / "scenario_2_compliance" / "exit_obstructed.png", OUT / "scenario_2_compliance" / "fire_exit.png"],
        "profiles": {str((OUT / "scenario_2_compliance" / name).resolve()): "safety" for name in ("exit_obstructed.png", "fire_exit.png")},
        "expected_workers": {"exit_obstructed.png": "Worker-D", "fire_exit.png": "Worker-D"},
    },
    {
        "name": "scenario_3_comprehensive",
        "title": "场景 3：综合评估与冲突证据",
        "inspection_type": "综合评估",
        "store_id": "store-mixed-003",
        "images": [OUT / "scenario_3_comprehensive" / "open_domain.png", OUT / "scenario_3_comprehensive" / "shelf_hybrid.png", OUT / "scenario_3_comprehensive" / "conflict_exit.png"],
        "profiles": {str((OUT / "scenario_3_comprehensive" / "conflict_exit.png").resolve()): "conflict"},
        "expected_workers": {"open_domain.png": "Worker-B", "shelf_hybrid.png": "Worker-D", "conflict_exit.png": "Worker-D"},
        "error_case": {
            "image": "open_domain.png",
            "expected_worker": "Worker-B",
            "reason": "该图是开放域场景，未出现明确零售或高风险证据；综合评估提示使路由器保守升级到 D，增加了不必要的 A+B 成本。",
        },
    },
]


# 执行三组 Demo 并写入结果文件。
def main() -> None:
    index: list[dict[str, object]] = []
    for spec in DEMOS:
        paths = [path.resolve() for path in spec["images"]]
        pipeline = InspectionPipeline(worker_pool=WorkerPool("mock", spec["profiles"]))
        report = pipeline.inspect(paths, spec["inspection_type"], spec["store_id"], request_id=spec["name"])
        error_reason = (spec.get("error_case") or {}).get("reason", "实际 Worker 与 Demo 预期不一致。")
        for entry in report["routing_log"]:
            expected = spec["expected_workers"].get(entry["image"])
            if expected is not None:
                entry["expected_worker"] = expected
                entry["routing_error"] = entry["worker"] != expected
                if entry["routing_error"]:
                    entry["routing_error_reason"] = error_reason
        target = OUT / spec["name"]
        target.mkdir(parents=True, exist_ok=True)
        (target / "input.json").write_text(json.dumps({
            "store_id": spec["store_id"],
            "inspection_type": spec["inspection_type"],
            "images": [str(path.relative_to(ROOT)) for path in paths],
            "expected_workers": spec["expected_workers"],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (target / "output.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (target / "routing_log.json").write_text(json.dumps(report["routing_log"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary = {
            "title": spec["title"],
            "request_id": report["request_id"],
            "overall_score": report["overall_score"],
            "workers": [entry["worker"] for entry in report["routing_log"]],
            "expected_workers": spec["expected_workers"],
            "findings": report["findings"],
            "compliance_items": report["compliance_items"],
            "error_case": spec.get("error_case"),
        }
        (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index.append({"name": spec["name"], "output": str((target / "output.json").relative_to(ROOT)), "score": report["overall_score"]})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
