"""Run a small reproducible feedback-loop experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from submission.router.mm_router import load_jsonl

from .feedback_store import Feedback, FeedbackStore
from .incremental_trainer import IncrementalTrainer
from .model_registry import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data", default=str(root / "submission/router/training_data.jsonl"))
    parser.add_argument("--feedback", default=str(Path(__file__).with_name("feedback.jsonl")))
    parser.add_argument("--registry", default=str(Path(__file__).with_name("model_registry.jsonl")))
    parser.add_argument("--checkpoint", default=str(Path(__file__).with_name("router_incremental.npy")))
    parser.add_argument("--output", default=str(Path(__file__).with_name("experiment_results.json")))
    args = parser.parse_args()
    store = FeedbackStore(args.feedback)
    hard_path = Path(args.data).with_name("hard_negative.jsonl")
    source = load_jsonl(hard_path) if hard_path.exists() else load_jsonl(args.data)
    for index, row in enumerate(source[:5]):
        store.add(Feedback(
            request_id=f"feedback-demo-{index}",
            image_path=row["image_path"],
            query=row["query"],
            original_worker="Worker-B" if row["label"] != "Worker-B" else "Worker-A",
            correct_worker=row["label"],
            reason="用户复核确认该查询的目标 Worker 与原路由不同。",
            source_split="hard_negative",
        ))
    trainer = IncrementalTrainer(args.data, store, ModelRegistry(args.registry))
    result = trainer.train_if_ready("router-feedback-v1", args.checkpoint, minimum_feedback=5, generations=30)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
