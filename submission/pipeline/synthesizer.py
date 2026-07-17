# 基于证据的巡检报告合成。

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import validate_report


PENALTIES = {"high": 20, "med": 10, "low": 5}
COMPLIANCE_PENALTIES = {"fail": 15, "unclear": 5, "pass": 0}
SEVERITY_ORDER = {"low": 0, "med": 1, "high": 2}


# 合并 Worker 结果并生成巡检报告。
class EvidenceSynthesizer:
    # 合成标准化巡检报告。
    def synthesize(
        self,
        store_id: str,
        inspection_time: str,
        results: list[dict[str, Any]],
        routing_log: list[dict[str, Any]],
        inspection_type: str,
        request_id: str,
        mock_mode: bool,
    ) -> dict[str, Any]:
        findings = self._merge_findings(results)
        compliance = self._merge_compliance(results)
        warnings: list[str] = []
        if mock_mode:
            warnings.append(
                "本报告由 Mock Worker 生成，仅用于 Pipeline 联调，不代表真实模型效果。"
            )
        if any(item.get("error") for item in results):
            warnings.append(
                "至少一个 Worker 执行或解析失败；相关图片已保留错误证据，需人工复核。"
            )
        finding_penalty = sum(PENALTIES.get(item["severity"], 0) for item in findings)
        compliance_penalty = sum(
            COMPLIANCE_PENALTIES.get(item["status"], 0) for item in compliance
        )
        score = max(0, 100 - finding_penalty - compliance_penalty)
        report = {
            "store_id": store_id,
            "inspection_time": inspection_time,
            "overall_score": score,
            "routing_log": routing_log,
            "findings": findings,
            "compliance_items": compliance,
            "recommendations": self._recommendations(findings, compliance),
            "request_id": request_id,
            "inspection_type": inspection_type,
            "mock_mode": mock_mode,
            "warnings": warnings,
            "model_versions": sorted(
                {item.get("model_revision", "unknown") for item in results}
            ),
        }
        validate_report(report)
        return report

    @staticmethod
    # 展开 D 的来源结果以保留冲突信息。
    def _iter_result_sources(results: list[dict[str, Any]]):
        for result in results:
            source_results = result.get("metadata", {}).get("source_results", [])
            if source_results:
                yield from source_results
            else:
                yield result

    # 合并并去重图片发现。
    def _merge_findings(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str, str], dict[str, Any]] = {}
        for result in self._iter_result_sources(results):
            for finding in result.get("findings", []):
                normalized = self._normalize_finding(finding, result)
                key = (
                    normalized["category"],
                    normalized["image_ref"],
                    normalized["description"],
                )
                old = merged.get(key)
                if (
                    old is None
                    or SEVERITY_ORDER[normalized["severity"]]
                    > SEVERITY_ORDER[old["severity"]]
                ):
                    merged[key] = normalized
        return sorted(
            merged.values(),
            key=lambda item: (
                -SEVERITY_ORDER[item["severity"]],
                item["image_ref"],
                item["category"],
            ),
        )

    @staticmethod
    # 规范化单条发现。
    def _normalize_finding(
        finding: dict[str, Any], result: dict[str, Any]
    ) -> dict[str, Any]:
        severity = finding.get("severity", "low")
        if severity not in PENALTIES:
            severity = "low"
        image_ref = finding.get("image_ref") or result.get("image_path", "unknown")
        return {
            "category": str(finding.get("category", "unclassified")),
            "severity": severity,
            "description": str(
                finding.get("description", result.get("raw_text", "无文字结果"))
            ),
            "image_ref": str(image_ref),
        }

    # 合并合规结果并处理冲突。
    def _merge_compliance(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for result in self._iter_result_sources(results):
            ref = Path(str(result.get("image_path", "unknown"))).name
            for item in result.get("compliance_items", []):
                name = str(item.get("item", "未命名合规项"))
                status = str(item.get("status", "unclear"))
                if status not in {"pass", "fail", "unclear"}:
                    status = "unclear"
                evidence = str(item.get("evidence", result.get("raw_text", "无证据")))
                grouped[name].append((status, evidence, ref))

        merged: list[dict[str, Any]] = []
        for name, observations in sorted(grouped.items()):
            statuses = {status for status, _, _ in observations}
            joined_evidence = " | ".join(text for _, text, _ in observations)
            if "fail" in statuses and "pass" in statuses:
                status = "unclear"
                evidence = "检测到 Worker 证据冲突：" + " | ".join(
                    f"{worker_ref}: {text}" for _, text, worker_ref in observations
                )
            elif "fail" in statuses:
                status = "fail"
                evidence = joined_evidence
            elif "unclear" in statuses:
                status = "unclear"
                evidence = joined_evidence
            else:
                status = "pass"
                evidence = joined_evidence
            merged.append({"item": name, "status": status, "evidence": evidence})
        return merged

    @staticmethod
    # 根据发现和合规状态生成建议。
    def _recommendations(
        findings: list[dict[str, Any]], compliance: list[dict[str, Any]]
    ) -> list[str]:
        recommendations: list[str] = []
        categories = {item["category"] for item in findings}
        if "safety_obstruction" in categories or any(
            item["status"] == "fail" for item in compliance
        ):
            recommendations.append(
                "立即清理消防出口和通道附近的障碍物，并由现场负责人复核。"
            )
        if "inventory" in categories:
            recommendations.append("对空货位进行补货复核，并将盘点结果与库存系统对账。")
        if "open_scene" in categories:
            recommendations.append(
                "该图片不属于明确零售证据，避免据此作库存或合规结论。"
            )
        if any(item["status"] == "unclear" for item in compliance):
            recommendations.append("对冲突或不清晰的合规项补拍近景，并安排人工复核。")
        if not recommendations:
            recommendations.append("保持现有巡检记录，按门店规则进行人工抽查。")
        return recommendations


# 返回带时区的 UTC ISO-8601 时间。
def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
