# 运行可复现的 Task4 ReAct 数据飞轮实验。

from __future__ import annotations

import argparse
import json
from pathlib import Path

from submission.router.mm_router import MultimodalRouter, load_jsonl

from .feedback_store import FeedbackStore
from .incremental_trainer import IncrementalTrainer
from .model_registry import ModelRegistry
from .react_flywheel import ReActFlywheel


# 清空演示状态文件以复现实验起点。
def _reset_demo_files(paths: list[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


# 将项目内路径序列化为相对仓库根目录。
def _display_path(value: str | Path, root: Path) -> str:
    path = Path(value)
    absolute = path if path.is_absolute() else path.resolve()
    try:
        return absolute.relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


# 解析命令行参数并执行完整反馈训练实验。
def main() -> None:
    root = Path(__file__).resolve().parents[2]
    innovation = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Environment feedback -> ReAct reflection -> self-training -> gate"
    )
    parser.add_argument(
        "--data", default=str(root / "submission/router/training_data.jsonl")
    )
    parser.add_argument(
        "--hard-negative",
        default=str(root / "submission/router/hard_negative.jsonl"),
    )
    parser.add_argument(
        "--baseline", default=str(root / "submission/router/router_weights.npy")
    )
    parser.add_argument("--feedback", default=str(innovation / "feedback.jsonl"))
    parser.add_argument("--registry", default=str(innovation / "model_registry.jsonl"))
    parser.add_argument("--trace", default=str(innovation / "react_trace.jsonl"))
    parser.add_argument(
        "--generated-data", default=str(innovation / "training_data.jsonl")
    )
    parser.add_argument(
        "--checkpoint", default=str(innovation / "router_incremental.npy")
    )
    parser.add_argument("--output", default=str(innovation / "experiment_results.json"))
    parser.add_argument("--version", default="router-react-v1")
    parser.add_argument("--minimum-feedback", type=int, default=2)
    parser.add_argument("--generations", type=int, default=90)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replay-weight", type=int, default=1)
    parser.add_argument(
        "--reset-demo-state",
        action="store_true",
        help="clear only the configured feedback/registry/trace demo artifacts",
    )
    args = parser.parse_args()

    feedback_path = Path(args.feedback)
    registry_path = Path(args.registry)
    trace_path = Path(args.trace)
    checkpoint_path = Path(args.checkpoint)
    if args.reset_demo_state:
        _reset_demo_files([feedback_path, registry_path, trace_path])
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    router = MultimodalRouter().load(args.baseline)
    store = FeedbackStore(feedback_path)
    registry = ModelRegistry(registry_path)
    trainer = IncrementalTrainer(args.data, store, registry)
    flywheel = ReActFlywheel(router, store, trainer, trace_path)
    challenge = load_jsonl(args.hard_negative)
    result = flywheel.run(
        challenge,
        version=args.version,
        checkpoint=checkpoint_path,
        baseline_checkpoint=args.baseline,
        minimum_feedback=args.minimum_feedback,
        seed=args.seed,
        generations=args.generations,
        replay_weight=args.replay_weight,
    )
    result["generated_training_samples"] = store.export_training_jsonl(
        args.generated_data
    )
    result["trace_path"] = _display_path(trace_path, root)
    result["generated_training_data"] = _display_path(args.generated_data, root)
    result["configuration"] = {
        "seed": args.seed,
        "generations": args.generations,
        "minimum_feedback": args.minimum_feedback,
        "replay_weight": args.replay_weight,
        "baseline": _display_path(args.baseline, root),
        "challenge": _display_path(args.hard_negative, root),
        "demo_state_reset": args.reset_demo_state,
    }
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
