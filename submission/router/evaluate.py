"""Task 2 reproducible metrics, validation split, seed runs and ablations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data_validation import validate_records
from .feature_extractor import FeatureConfig
from .mm_router import MultimodalRouter, RouterConfig, load_jsonl


def split_records(records: list[dict], validation_per_class: int = 4):
    """Use explicit validation rows; otherwise split whole template groups."""
    explicit_validation = any(row.get("split") == "validation" for row in records)
    if explicit_validation:
        return (
            [row for row in records if row.get("split") == "train"],
            [row for row in records if row.get("split") == "validation"],
            [row for row in records if row.get("split") == "test"],
        )
    train, validation, test = [], [], []
    test.extend(row for row in records if row.get("split") == "test")
    by_label: dict[str, list[dict]] = {}
    for row in records:
        if row.get("split") == "train":
            by_label.setdefault(row["label"], []).append(row)
    for label, rows in sorted(by_label.items()):
        groups: dict[str, list[dict]] = {}
        for row in sorted(rows, key=lambda item: item["id"]):
            groups.setdefault(row["template_group"], []).append(row)
        selected = 0
        for group, group_rows in sorted(groups.items()):
            if selected < validation_per_class:
                validation.extend(group_rows)
                selected += len(group_rows)
            else:
                train.extend(group_rows)
    return train, validation, test


def latency_summary(router: MultimodalRouter, records: list[dict]) -> dict:
    values = []
    for row in records:
        started = time.perf_counter()
        router.route(row.get("image_path"), row["query"])
        values.append((time.perf_counter() - started) * 1000)
    if not values:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(values),
        "p50_ms": round(float(np.percentile(values, 50)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "max_ms": round(float(max(values)), 3),
    }


def run_once(train, validation, test, seed: int, feature_config=None, generations=90):
    config = RouterConfig(seed=seed, generations=generations)
    router = MultimodalRouter(config=config, extractor=None)
    if feature_config is not None:
        from .feature_extractor import MultimodalFeatureExtractor
        router = MultimodalRouter(MultimodalFeatureExtractor(feature_config), config)
    router.fit(train)
    return {
        "seed": seed,
        "validation": router.evaluate(validation),
        "test": router.evaluate(test),
        "latency": latency_summary(router, test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the multimodal router")
    parser.add_argument("--data", default=str(Path(__file__).with_name("training_data.jsonl")))
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--generations", type=int, default=90)
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--output", default=str(Path(__file__).with_name("eval_results.json")))
    args = parser.parse_args()
    records = load_jsonl(args.data)
    validation_report = validate_records(records, args.data)
    train, validation, test = split_records(records)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    results = [run_once(train, validation, test, seed, generations=args.generations) for seed in seeds]
    output = {
        "run_config": {"seeds": seeds, "generations": args.generations, "validation_per_class": 4},
        "dataset": validation_report,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "seeds": results,
    }
    if not args.skip_ablations:
        variants = {
            "text_only": FeatureConfig(use_image=False, use_text=True, use_signals=False),
            "image_only": FeatureConfig(use_image=True, use_text=False, use_signals=False),
            "multimodal": FeatureConfig(use_image=True, use_text=True, use_signals=True),
            "signals_only": FeatureConfig(use_image=False, use_text=False, use_signals=True),
        }
        output["ablations"] = {
            name: run_once(train, validation, test, seeds[0], config, generations=args.generations)
            for name, config in variants.items()
        }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
