# 面向环境反馈的 ReAct 路由器自改进控制器。

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from submission.router.mm_router import WORKERS, MultimodalRouter

from .feedback_store import Feedback, FeedbackStore
from .incremental_trainer import IncrementalTrainer


# 执行观察、反思、行动、训练、评估和发布闭环。
class ReActFlywheel:
    # 绑定路由器、反馈存储、训练器和轨迹输出。
    def __init__(
        self,
        router: MultimodalRouter,
        feedback_store: FeedbackStore,
        trainer: IncrementalTrainer,
        trace_path: str | Path,
        *,
        auto_approve_threshold: float = 0.9,
    ):
        if not 0.0 <= auto_approve_threshold <= 1.0:
            raise ValueError("auto_approve_threshold must be between 0 and 1")
        self.router = router
        self.feedback_store = feedback_store
        self.trainer = trainer
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.auto_approve_threshold = auto_approve_threshold

    # 写入一条可审计的 ReAct 事件轨迹。
    def _trace(self, event: dict) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with self.trace_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # 提取并校验环境期望的 Worker 标签。
    @staticmethod
    def _expected_worker(record: dict) -> str:
        worker = str(record.get("expected_worker") or record.get("label") or "")
        if worker not in WORKERS:
            raise ValueError(f"environment expected_worker must be one of {WORKERS}")
        return worker

    # 执行单条环境样本的观察、反思和行动。
    def step(self, record: dict, *, episode_id: str) -> dict:
        expected_worker = self._expected_worker(record)
        decision = self.router.route(record.get("image_path"), record["query"])
        raw_worker = decision["raw_worker"]
        final_worker = decision["worker"]
        raw_correct = raw_worker == expected_worker
        final_correct = final_worker == expected_worker
        reward = 1.0 if final_correct else -1.0
        if final_correct and not raw_correct:
            # 门控救回的裸模型错误仍需进入训练。
            reward = 0.25

        confidence = float(record.get("environment_confidence", 1.0))
        source_split = str(record.get("split") or "production")
        signal_source = str(record.get("signal_source") or "environment_gold")
        observation = {
            "record_id": record.get("id"),
            "expected_worker": expected_worker,
            "raw_worker": raw_worker,
            "final_worker": final_worker,
            "reward": reward,
            "confidence": confidence,
            "gate_reasons": decision.get("gate_reasons", []),
        }

        if source_split == "test":
            reflection = "固定测试集只用于回归评估，禁止回流训练。"
            action = "protect_fixed_test"
        elif raw_correct and final_correct:
            reflection = "学习路由与最终策略均命中，无需制造自训练样本。"
            action = "keep_policy"
        elif raw_correct and not final_correct:
            reflection = "裸模型正确但门控改错，属于策略缺陷，应进入规则审核。"
            action = "queue_policy_review"
        else:
            near_miss = final_correct
            reflection = (
                "门控已救回裸模型误路由，记录 near-miss 以降低兜底依赖。"
                if near_miss
                else "环境确认路由错误，生成带来源和奖励的纠错样本。"
            )
            auto_approve = confidence >= self.auto_approve_threshold
            feedback = Feedback(
                request_id=str(record.get("request_id") or record.get("id") or uuid4()),
                image_path=str(record.get("image_path") or ""),
                query=str(record["query"]),
                original_worker=raw_worker,
                correct_worker=expected_worker,
                reason=str(
                    record.get("environment_reason")
                    or f"环境标签要求 {expected_worker}，裸模型选择 {raw_worker}。"
                ),
                source_split=source_split,
                signal_source=signal_source,
                confidence=confidence,
                environment_reward=reward,
                auto_approve=auto_approve,
            )
            inserted = self.feedback_store.add(feedback)
            if not inserted:
                action = "deduplicate_feedback"
            elif auto_approve and signal_source in {
                "environment_gold",
                "verified_evaluator",
            }:
                action = "approve_training_signal"
            else:
                action = "queue_feedback_review"

        event = {
            "episode_id": episode_id,
            "phase": "observe_reflect_act",
            "observation": observation,
            "reflection": reflection,
            "action": action,
        }
        self._trace(event)
        return event

    # 执行完整反馈回合并热加载通过门禁的候选模型。
    def run(
        self,
        records: list[dict],
        *,
        version: str,
        checkpoint: str | Path,
        baseline_checkpoint: str | Path | None,
        minimum_feedback: int = 5,
        seed: int = 7,
        generations: int = 90,
        replay_weight: int = 1,
    ) -> dict:
        episode_id = uuid4().hex
        events = [self.step(record, episode_id=episode_id) for record in records]
        training = self.trainer.train_if_ready(
            version,
            checkpoint,
            minimum_feedback=minimum_feedback,
            seed=seed,
            generations=generations,
            baseline_checkpoint=baseline_checkpoint,
            challenge_records=records,
            replay_weight=replay_weight,
        )
        if training.get("gate_passed"):
            self.router.load(checkpoint)
            lifecycle_action = "publish_and_hot_load"
        elif training.get("status") == "waiting":
            lifecycle_action = "wait_for_more_feedback"
        else:
            lifecycle_action = "rollback_candidate"
        lifecycle = {
            "episode_id": episode_id,
            "phase": "train_evaluate_act",
            "reflection": (
                "候选通过挑战收益与固定回归门禁。"
                if training.get("gate_passed")
                else "候选未满足发布条件或反馈尚不足，保持当前模型。"
            ),
            "action": lifecycle_action,
            "gate_passed": training.get("gate_passed"),
        }
        self._trace(lifecycle)
        action_counts = Counter(event["action"] for event in events)
        return {
            "episode_id": episode_id,
            "environment_count": len(records),
            "environment_reward_mean": (
                round(
                    sum(event["observation"]["reward"] for event in events)
                    / len(events),
                    4,
                )
                if events
                else 0.0
            ),
            "action_counts": dict(sorted(action_counts.items())),
            "feedback_status_counts": self.feedback_store.counts(),
            "training": training,
            "lifecycle_action": lifecycle_action,
            "trace_path": str(self.trace_path),
        }
