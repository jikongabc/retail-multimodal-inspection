# 校验 Task 2 JSONL 数据并检测切分泄漏。

from __future__ import annotations

from collections import Counter
import hashlib
import re
from pathlib import Path
from typing import Iterable

from .mm_router import WORKERS, load_jsonl


REQUIRED_FIELDS = {
    "id",
    "image_path",
    "query",
    "label",
    "split",
    "label_reason",
    "risk_level",
    "template_group",
    "source",
}


# 校验路由记录并返回数据质量报告。
def validate_records(
    records: Iterable[dict], data_path: str | Path | None = None
) -> dict:
    records = list(records)
    errors: list[str] = []
    seen_ids: set[str] = set()
    split_images: dict[str, set[str]] = {}
    split_groups: dict[str, set[str]] = {}
    split_queries: dict[str, set[str]] = {}
    split_hashes: dict[str, set[str]] = {}
    image_hash_to_path: dict[str, str] = {}
    for row in records:
        missing = sorted(REQUIRED_FIELDS - row.keys())
        if missing:
            errors.append(f"{row.get('id', '<unknown>')}: missing {','.join(missing)}")
        if row.get("id") in seen_ids:
            errors.append(f"duplicate id: {row.get('id')}")
        seen_ids.add(row.get("id"))
        if row.get("label") not in WORKERS:
            errors.append(f"{row.get('id')}: invalid label")
        split = row.get("split")
        if split not in {"train", "validation", "test"}:
            errors.append(f"{row.get('id')}: invalid split {split}")
        split_images.setdefault(split, set()).add(row.get("image_path"))
        split_groups.setdefault(split, set()).add(row.get("template_group"))
        normalized_query = re.sub(r"\s+", "", str(row.get("query", "")).lower())
        split_queries.setdefault(split, set()).add(normalized_query)
        if data_path and row.get("image_path"):
            base = Path(data_path).resolve().parent
            image = Path(row["image_path"])
            resolved = image if image.is_absolute() else base / image
            if not resolved.is_file():
                errors.append(f"{row.get('id')}: image missing {row.get('image_path')}")
            else:
                digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
                split_hashes.setdefault(split, set()).add(digest)
                image_hash_to_path.setdefault(digest, str(resolved))
    overlap_images = {
        f"{left}/{right}": sorted(
            split_images.get(left, set()) & split_images.get(right, set())
        )
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
        if split_images.get(left, set()) & split_images.get(right, set())
    }
    overlap_groups = {
        f"{left}/{right}": sorted(
            split_groups.get(left, set()) & split_groups.get(right, set())
        )
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
        if split_groups.get(left, set()) & split_groups.get(right, set())
    }
    overlap_queries = {
        f"{left}/{right}": sorted(
            split_queries.get(left, set()) & split_queries.get(right, set())
        )
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
        if split_queries.get(left, set()) & split_queries.get(right, set())
    }
    overlap_hashes = {
        f"{left}/{right}": sorted(
            split_hashes.get(left, set()) & split_hashes.get(right, set())
        )
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
        if split_hashes.get(left, set()) & split_hashes.get(right, set())
    }
    if overlap_images:
        errors.append(f"image leakage: {overlap_images}")
    if overlap_groups:
        errors.append(f"template leakage: {overlap_groups}")
    if overlap_queries:
        errors.append(f"query leakage: {overlap_queries}")
    if overlap_hashes:
        errors.append(f"image content leakage: {overlap_hashes}")
    report = {
        "total": len(records),
        "split_counts": dict(Counter(row.get("split") for row in records)),
        "label_counts": dict(Counter(row.get("label") for row in records)),
        "image_overlap": overlap_images,
        "template_overlap": overlap_groups,
        "query_overlap": overlap_queries,
        "image_hash_overlap": overlap_hashes,
        "errors": errors,
    }
    if errors:
        raise ValueError("invalid routing dataset: " + "; ".join(errors))
    return report


# 加载并校验路由 JSONL 文件。
def validate_file(path: str | Path) -> dict:
    records = load_jsonl(path)
    return validate_records(records, path)
