# 巡检报告结构校验。

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any


REQUIRED_FIELDS = (
    "store_id",
    "inspection_time",
    "overall_score",
    "routing_log",
    "findings",
    "compliance_items",
    "recommendations",
)
SEVERITIES = {"high", "med", "low"}
COMPLIANCE_STATUSES = {"pass", "fail", "unclear"}


# 表示报告不符合结构约束。
class ReportValidationError(ValueError):
    pass


# 检查对象是否包含指定字段。
def _require(mapping: Mapping[str, Any], key: str, errors: list[str]) -> Any:
    if key not in mapping:
        errors.append(f"missing field: {key}")
        return None
    return mapping[key]


# 校验报告字段、枚举值和数值范围。
def validate_report(report: Mapping[str, Any]) -> None:
    errors: list[str] = []
    if not isinstance(report, Mapping):
        raise ReportValidationError("report must be a JSON object")
    for field in REQUIRED_FIELDS:
        _require(report, field, errors)

    if not isinstance(report.get("store_id"), str) or not report.get("store_id"):
        errors.append("store_id must be a non-empty string")
    timestamp = report.get("inspection_time")
    if not isinstance(timestamp, str) or "T" not in timestamp:
        errors.append("inspection_time must be an ISO-8601-like string")
    else:
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed_timestamp.tzinfo is None:
                errors.append(
                    "inspection_time must include a timezone (Z or UTC offset)"
                )
        except ValueError:
            errors.append("inspection_time must be a valid ISO-8601 timestamp")
    score = report.get("overall_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not 0 <= score <= 100
    ):
        errors.append("overall_score must be a number in [0, 100]")

    routing_log = report.get("routing_log")
    if not isinstance(routing_log, list):
        errors.append("routing_log must be a list")
    else:
        for index, entry in enumerate(routing_log):
            if not isinstance(entry, Mapping):
                errors.append(f"routing_log[{index}] must be an object")
                continue
            if not isinstance(entry.get("image"), str) or not entry.get("image"):
                errors.append(f"routing_log[{index}].image must be a string")
            if not isinstance(entry.get("worker"), str) or not entry.get("worker"):
                errors.append(f"routing_log[{index}].worker must be a string")
            latency = entry.get("latency_ms")
            if (
                isinstance(latency, bool)
                or not isinstance(latency, (int, float))
                or latency < 0
            ):
                errors.append(f"routing_log[{index}].latency_ms must be non-negative")

    findings = report.get("findings")
    if not isinstance(findings, list):
        errors.append("findings must be a list")
    else:
        for index, finding in enumerate(findings):
            if not isinstance(finding, Mapping):
                errors.append(f"findings[{index}] must be an object")
                continue
            for field in ("category", "severity", "description", "image_ref"):
                if not isinstance(finding.get(field), str) or not finding.get(field):
                    errors.append(
                        f"findings[{index}].{field} must be a non-empty string"
                    )
            if finding.get("severity") not in SEVERITIES:
                errors.append(
                    f"findings[{index}].severity must be one of {sorted(SEVERITIES)}"
                )

    compliance = report.get("compliance_items")
    if not isinstance(compliance, list):
        errors.append("compliance_items must be a list")
    else:
        for index, item in enumerate(compliance):
            if not isinstance(item, Mapping):
                errors.append(f"compliance_items[{index}] must be an object")
                continue
            for field in ("item", "status", "evidence"):
                if not isinstance(item.get(field), str) or not item.get(field):
                    errors.append(
                        f"compliance_items[{index}].{field} must be a non-empty string"
                    )
            if item.get("status") not in COMPLIANCE_STATUSES:
                errors.append(
                    f"compliance_items[{index}].status must be one of {sorted(COMPLIANCE_STATUSES)}"
                )

    recommendations = report.get("recommendations")
    if not isinstance(recommendations, list) or not all(
        isinstance(item, str) and item for item in recommendations
    ):
        errors.append("recommendations must be a list of non-empty strings")

    if errors:
        raise ReportValidationError("; ".join(errors))


# 返回报告是否通过结构校验。
def is_valid_report(report: Mapping[str, Any]) -> bool:
    try:
        validate_report(report)
    except ReportValidationError:
        return False
    return True
