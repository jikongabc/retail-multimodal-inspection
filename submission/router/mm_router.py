"""Task 2 多模态路由器：sep-CMA-ES + 置信度升级策略。

Python 3.10+
主要依赖：numpy>=1.24、Pillow>=10.0；不依赖 GPU 或外部 API。

实现原则：冻结多模态特征提取，仅训练一个小型线性路由头；训练目标可以
从监督标签切换为 Worker 端到端奖励，后者与 OpenFugu 的黑盒训练路径一致。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from feature_extractor import FeatureConfig, MultimodalFeatureExtractor, WORKERS


@dataclass
class RouterConfig:
    """Routing and optimizer hyperparameters."""

    population: int = 18
    generations: int = 90
    sigma: float = 0.35
    seed: int = 7
    confidence_threshold: float = 0.56


class SepCMAES:
    """Minimal separable CMA-ES for a black-box vector objective.

    The covariance is diagonal, so memory and update cost are O(n), matching
    the reason OpenFugu uses sep-CMA-ES for a high-dimensional router head.
    """

    def __init__(self, dimension: int, config: RouterConfig) -> None:
        self.n = dimension
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    def optimize(self, objective, initial: np.ndarray | None = None):
        mean = np.zeros(self.n, dtype=np.float32) if initial is None else initial.astype(np.float32).copy()
        diag = np.ones(self.n, dtype=np.float32)
        sigma = self.cfg.sigma
        lam = self.cfg.population
        mu = max(2, lam // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        best_x, best_score = mean.copy(), float(objective(mean))
        history = [best_score]
        for _ in range(self.cfg.generations):
            z = self.rng.normal(size=(lam, self.n)).astype(np.float32)
            candidates = mean[None, :] + sigma * np.sqrt(diag)[None, :] * z
            scores = np.asarray([objective(x) for x in candidates], dtype=np.float32)
            order = np.argsort(scores)[::-1]
            elite = candidates[order[:mu]]
            elite_z = z[order[:mu]]
            mean = np.sum(elite * weights[:, None], axis=0)
            # Rank-one diagonal adaptation; clipping prevents numerical blowup.
            diag = np.clip(0.85 * diag + 0.15 * np.sum((elite_z**2) * weights[:, None], axis=0), 0.20, 5.0)
            spread = float(np.std(scores))
            sigma = float(np.clip(sigma * (1.02 if spread > 0.03 else 0.985), 0.035, 1.2))
            if float(scores[order[0]]) > best_score:
                best_score = float(scores[order[0]])
                best_x = candidates[order[0]].copy()
            history.append(best_score)
        return best_x, best_score, history


class MultimodalRouter:
    """Four-worker router with optional parallel escalation to Worker-D."""

    def __init__(self, extractor: MultimodalFeatureExtractor | None = None, config: RouterConfig | None = None):
        self.extractor = extractor or MultimodalFeatureExtractor(FeatureConfig())
        self.config = config or RouterConfig()
        self.weights: np.ndarray | None = None

    @property
    def parameter_count(self) -> int:
        return len(WORKERS) * (self.extractor.config.dim + 1)

    def _matrix(self, records):
        X = np.vstack([self.extractor.extract(r.get("image_path"), r["query"]) for r in records])
        return np.hstack([X, np.ones((len(records), 1), dtype=np.float32)])

    def _logits(self, X, flat):
        return X @ flat.reshape(len(WORKERS), -1).T

    def _gate(self, logits: np.ndarray, query: str, image_path=None) -> dict:
        """Apply the same online gate used by training, evaluation and serving."""
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        raw_idx = int(probs.argmax())
        margin = float(np.sort(probs)[-1] - np.sort(probs)[-2])
        flags = self.extractor._semantic_flags(query)
        query_lower = query.lower()
        reasons = []
        if flags[7]:
            reasons.append("complex_keyword")
        if flags[8]:
            reasons.append("high_risk")
        if flags[10] > 0.70:
            reasons.append("long_query")
        if flags[14] > 0.0:
            reasons.append("multi_image")
        if flags[15] > 0.0:
            reasons.append("uncertain")
        if float(probs[raw_idx]) < self.config.confidence_threshold:
            reasons.append("low_confidence")
        if margin < 0.08:
            reasons.append("low_margin")
        # Explicitly text-only requests should go to the report worker even if
        # the image looks like a safety/shelf scene. This is the opposite of a
        # visual conflict: the caller has told us not to re-run perception.
        explicit_text_only = any(token in query_lower for token in (
            "不要重新识别", "不重新识别", "已有巡检结果", "文字结果", "仅根据上游", "不要做视觉判断",
        )) and bool(flags[4] or flags[13])
        if explicit_text_only:
            reasons.append("explicit_text_only")

        scene_hint = self.extractor.image_scene_hint(image_path)
        scene = scene_hint["scene"]
        # Query intent is intentionally coarse. It is a safety gate, not a
        # replacement for semantic routing. Negated inventory wording avoids
        # treating “不要数商品” as an inventory request.
        negated_inventory = any(token in query_lower for token in ("不要数商品", "不要执行盘点", "不要判断库存", "不做库存"))
        has_open_intent = bool(flags[6])
        has_inventory_intent = bool(flags[0]) and not negated_inventory
        fact_only = "只描述可见事实" in query_lower
        if fact_only and not any(token in query_lower for token in ("合规", "检查是否", "判断是否")):
            has_open_intent = True
            has_inventory_intent = False
        vague_request = any(token in query_lower for token in ("帮我看看这家店怎么样", "开放式意见", "怎么样")) and not any(
            token in query_lower for token in ("盘点", "库存", "数量", "缺货", "合规", "消防", "报告", "文字", "提取")
        )
        if flags[7] or flags[8]:
            query_intent = "complex"
        elif explicit_text_only:
            query_intent = "report"
        elif vague_request:
            query_intent = "unknown"
        elif has_open_intent and not has_inventory_intent:
            query_intent = "open"
        elif flags[4] and not has_inventory_intent:
            query_intent = "report"
        elif has_inventory_intent or flags[1] or flags[2] or flags[3]:
            query_intent = "retail_visual"
        else:
            query_intent = "unknown"

        conflict = (
            (scene == "shelf" and query_intent == "open")
            or (scene == "open" and query_intent == "retail_visual")
            or (scene == "report" and query_intent == "open")
            or (scene == "shelf" and query_intent == "report" and not explicit_text_only)
            or (scene in ("shelf", "open", "report") and query_intent == "unknown")
        )
        if conflict:
            reasons.append("image_query_conflict")

        # Resolve conflict according to query intent. Only an ambiguous request
        # is escalated to D; a clear “describe / count / summarize” instruction
        # should not be overridden by image appearance.
        if explicit_text_only:
            final_idx = 2
        elif conflict and query_intent == "unknown":
            reasons.append("ambiguous_conflict_escalation")
            final_idx = 3
        elif conflict and query_intent in ("open", "retail_visual", "report"):
            reasons.append("intent_override")
            final_idx = {"open": 1, "retail_visual": 0, "report": 2}[query_intent]
        elif query_intent in ("open", "retail_visual", "report") and any(
            reason in reasons for reason in ("low_confidence", "low_margin")
        ):
            # Real photographs are more heterogeneous than the synthetic
            # controls. A clear user intent takes precedence over generic
            # uncertainty; D remains reserved for ambiguous/complex intent.
            reasons.append("intent_confidence_override")
            final_idx = {"open": 1, "retail_visual": 0, "report": 2}[query_intent]
        elif query_intent in ("open", "retail_visual", "report"):
            intended_idx = {"open": 1, "retail_visual": 0, "report": 2}[query_intent]
            if raw_idx != intended_idx:
                reasons.append("intent_override")
                final_idx = intended_idx
            else:
                final_idx = raw_idx
        elif raw_idx != 3 and reasons:
            final_idx = 3
        else:
            final_idx = raw_idx
        return {
            "raw_idx": raw_idx,
            "final_idx": final_idx,
            "probs": probs,
            "margin": margin,
            "gate_reasons": reasons,
            "scene_hint": scene_hint,
            "query_intent": query_intent,
        }

    @staticmethod
    def _initial_from_centroids(X, y, classes):
        """Warm start with class centroids; CMA-ES still performs optimization."""
        W = np.zeros((len(classes), X.shape[1]), dtype=np.float32)
        for i, cls in enumerate(classes):
            pos = X[np.asarray(y) == i]
            centroid = pos.mean(axis=0) if len(pos) else np.zeros(X.shape[1])
            W[i] = centroid / (np.linalg.norm(centroid) + 1e-6)
        return W.ravel()

    def fit(self, records):
        X = self._matrix(records)
        y = np.asarray([WORKERS.index(r["label"]) for r in records])

        def objective(flat):
            logits = self._logits(X, flat)
            # Optimize the *online* decision, including the D gate. This keeps
            # the objective and deployed behavior aligned.
            shifted = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(shifted)
            probs /= probs.sum(axis=1, keepdims=True)
            loss = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-7, 1)).mean()
            gated = np.asarray([self._gate(row, r["query"], r.get("image_path"))["final_idx"] for row, r in zip(logits, records)])
            acc = (gated == y).mean()
            raw_acc = (logits.argmax(axis=1) == y).mean()
            return float(acc - 0.03 * loss - 0.005 * (1.0 - raw_acc))

        initial = self._initial_from_centroids(X, y, WORKERS)
        optimizer = SepCMAES(initial.size, self.config)
        self.weights, score, history = optimizer.optimize(objective, initial)
        self.fit_info = {"best_fitness": score, "generations": self.config.generations, "history": history}
        return self

    def evaluate(self, records):
        if self.weights is None:
            raise RuntimeError("router is not fitted")
        X = self._matrix(records)
        logits = self._logits(X, self.weights)
        y = np.asarray([WORKERS.index(r["label"]) for r in records])
        raw_pred = logits.argmax(axis=1)
        gated_info = [self._gate(row, r["query"], r.get("image_path")) for row, r in zip(logits, records)]
        gated_pred = np.asarray([x["final_idx"] for x in gated_info])

        def confusion(pred):
            cm = np.zeros((len(WORKERS), len(WORKERS)), dtype=int)
            for actual, predicted in zip(y, pred):
                cm[actual, predicted] += 1
            return cm.tolist()

        reasons = {}
        for x in gated_info:
            for reason in x["gate_reasons"]:
                reasons[reason] = reasons.get(reason, 0) + 1
        return {
            "accuracy": float((gated_pred == y).mean()),
            "raw_accuracy": float((raw_pred == y).mean()),
            "gated_accuracy": float((gated_pred == y).mean()),
            "raw_confusion_matrix": confusion(raw_pred),
            "gated_confusion_matrix": confusion(gated_pred),
            "raw_predictions": raw_pred.tolist(),
            "gated_predictions": gated_pred.tolist(),
            "gate_upgrades": int(np.sum(raw_pred != gated_pred)),
            "gate_reasons": reasons,
            "scene_hints": [x["scene_hint"] for x in gated_info],
        }

    def evaluate_gate_policy(self, records):
        """Evaluate gate behavior against separately annotated policy actions.

        This is intentionally distinct from Worker accuracy. ``expected_action``
        is authored before inference (none/follow_query/explicit_text_only/
        escalate_ambiguous/explicit_d) and is never used as ``label``.
        """
        if self.weights is None:
            raise RuntimeError("router is not fitted")
        counts = {"none": 0, "follow_query": 0, "explicit_text_only": 0, "escalate_ambiguous": 0, "explicit_d": 0}
        hits = {key: 0 for key in counts}
        false_escalations = 0
        d_expected = 0
        d_hits = 0
        for record in records:
            expected = record.get("expected_action", "none")
            counts[expected] = counts.get(expected, 0) + 1
            feature = self.extractor.extract(record.get("image_path"), record["query"])
            X = np.concatenate([feature, [1.0]]).reshape(1, -1)
            logits = self._logits(X, self.weights)[0]
            decision = self._gate(logits, record["query"], record.get("image_path"))
            actual = decision["final_idx"]
            expected_worker = {
                "none": WORKERS.index(record["label"]),
                "follow_query": WORKERS.index(record["label"]),
                "explicit_text_only": 2,
                "escalate_ambiguous": 3,
                "explicit_d": 3,
            }[expected]
            if actual == expected_worker:
                hits[expected] += 1
            if expected in ("escalate_ambiguous", "explicit_d"):
                d_expected += 1
                d_hits += int(actual == 3)
            elif actual == 3:
                false_escalations += 1
        total = len(records)
        return {
            "count": total,
            "action_counts": counts,
            "action_hits": hits,
            "policy_accuracy": float(sum(hits.values()) / total) if total else 0.0,
            "false_escalations": false_escalations,
            "false_escalation_rate": float(false_escalations / (total - d_expected)) if total > d_expected else 0.0,
            "expected_d": d_expected,
            "d_recall": float(d_hits / d_expected) if d_expected else 0.0,
        }

    def route(self, image_path: str | Path | None, query: str) -> dict:
        if self.weights is None:
            raise RuntimeError("router is not fitted")
        start = time.perf_counter()
        feature = self.extractor.extract(image_path, query)
        X = np.concatenate([feature, [1.0]]).reshape(1, -1)
        logits = self._logits(X, self.weights)[0]
        decision = self._gate(logits, query, image_path)
        probs = decision["probs"]
        raw_idx = decision["raw_idx"]
        idx = decision["final_idx"]
        upgraded = idx != raw_idx
        rewrite = self._rewrite(WORKERS[idx], query, upgraded)
        return {
            "worker": WORKERS[idx],
            "worker_id": idx,
            "raw_worker": WORKERS[raw_idx],
            "strategy": "parallel_ab_aggregate" if idx == 3 else "single_worker",
            "confidence": round(float(probs[raw_idx]), 4),
            "margin": round(decision["margin"], 4),
            "gate_upgraded": upgraded,
            "gate_reasons": decision["gate_reasons"],
            "scene_hint": decision["scene_hint"],
            "query_intent": decision["query_intent"],
            "prompt_rewrite": rewrite,
            "feature_backend": self.extractor.backend,
            "latency_ms": round((time.perf_counter() - start) * 1000, 3),
        }

    @staticmethod
    def _rewrite(worker: str, query: str, upgraded: bool) -> str:
        prefix = {
            "Worker-A": "以零售视觉专员身份，输出商品/货架位置、数量、证据框和不确定项。",
            "Worker-B": "以通用视觉理解模型身份，先描述可见事实，再回答问题，避免臆测。",
            "Worker-C": "只基于已提供的视觉结果和文字上下文，生成结构化报告、结论与建议。",
            "Worker-D": "并行调用零售专长视觉专家与通用视觉专家，比较证据后给出带置信度的结论。",
        }[worker]
        q = query.lower()
        requirements = []
        if any(x in q for x in ("盘点", "库存", "商品", "货架", "sku", "缺货", "空槽")):
            requirements.append("按货架层/商品/数量/缺货位置逐项输出")
        if any(x in q for x in ("合规", "消防", "通道", "价签", "价格标签", "安全")):
            requirements.append("对消防通道、价签或安全项给出可见证据和 pass/fail/unclear")
        if any(x in q for x in ("文字", "提取", "读取", "ocr", "标签")):
            requirements.append("文字逐项转写，模糊字符标为 uncertain，不补猜")
        if any(x in q for x in ("报告", "汇总", "总结", "摘要", "json", "结构化")):
            requirements.append("使用结构化字段输出结论、严重度、证据和建议")
        if any(x in q for x in ("开放域", "通用视觉", "人物", "动物", "场景", "物体")):
            requirements.append("先陈述可见事实，不套用零售先验")
        if upgraded:
            requirements.append("交叉比较 A/B 证据；冲突时保留不确定性并说明升级原因")
        if not requirements:
            requirements.append("先列可见事实，再给出与问题严格对应的结论")
        return f"{prefix}\n执行要求：{'；'.join(requirements)}。\n原始查询：{query}"

    def save(self, path: str | Path) -> None:
        if self.weights is None:
            raise RuntimeError("router is not fitted")
        np.save(path, self.weights)

    def load(self, path: str | Path) -> "MultimodalRouter":
        self.weights = np.load(path).astype(np.float32)
        return self


def load_jsonl(path: str | Path):
    base = Path(path).resolve().parent
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    for record in records:
        image_path = record.get("image_path")
        if image_path and not Path(image_path).is_absolute():
            record["image_path"] = str((base / image_path).resolve())
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate/predict the Task 2 multimodal router")
    parser.add_argument("--data", default=str(Path(__file__).with_name("training_data.jsonl")))
    parser.add_argument("--model", default=str(Path(__file__).with_name("router_weights.npy")))
    parser.add_argument("--mode", choices=("train", "evaluate", "predict"), default="train")
    parser.add_argument("--image")
    parser.add_argument("--query", default="请检查这张图片中的商品和缺货情况")
    args = parser.parse_args()
    records = load_jsonl(args.data)
    train = [r for r in records if r.get("split") == "train"]
    test = [r for r in records if r.get("split") == "test"]
    hard_path = Path(args.data).with_name("hard_negative.jsonl")
    hard = load_jsonl(hard_path) if hard_path.exists() else []
    router = MultimodalRouter()
    if args.mode == "train":
        router.fit(train).save(args.model)
        result = router.evaluate(test)
        hard_result = router.evaluate(hard) if hard else {}
        policy_result = {
            "hard_negative": router.evaluate_gate_policy(hard) if hard else {},
            "clean_test": router.evaluate_gate_policy(test) if test else {},
        }
        print(json.dumps({"train": len(train), "test": len(test), "hard_negative": len(hard), **result, "hard_negative_eval": hard_result, "gate_policy_eval": policy_result, "params": router.parameter_count}, ensure_ascii=False, indent=2))
    else:
        router.load(args.model)
        if args.mode == "evaluate":
            print(json.dumps({"test": router.evaluate(test), "hard_negative": router.evaluate(hard) if hard else {}, "gate_policy": {"hard_negative": router.evaluate_gate_policy(hard) if hard else {}, "clean_test": router.evaluate_gate_policy(test) if test else {}}}, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(router.route(args.image, args.query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
