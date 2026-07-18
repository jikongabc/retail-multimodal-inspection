# 多模态路由器及 sep-CMA-ES 训练流程。

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .feature_extractor import FeatureConfig, MultimodalFeatureExtractor, WORKERS
except ImportError:
    from feature_extractor import FeatureConfig, MultimodalFeatureExtractor, WORKERS


# 配置路由和优化器参数。
@dataclass
class RouterConfig:
    population: int = 18
    generations: int = 90
    sigma: float = 0.35
    seed: int = 7
    confidence_threshold: float = 0.56


# 实现对角协方差的 sep-CMA-ES 优化器。
class SepCMAES:
    # 初始化优化器。
    def __init__(self, dimension: int, config: RouterConfig) -> None:
        self.n = dimension
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    # 优化黑盒目标并返回最佳参数。
    def optimize(self, objective, initial: np.ndarray | None = None):
        mean = (
            np.zeros(self.n, dtype=np.float32)
            if initial is None
            else initial.astype(np.float32).copy()
        )
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
            diag = np.clip(
                0.85 * diag + 0.15 * np.sum((elite_z**2) * weights[:, None], axis=0),
                0.20,
                5.0,
            )
            spread = float(np.std(scores))
            sigma = float(
                np.clip(sigma * (1.02 if spread > 0.03 else 0.985), 0.035, 1.2)
            )
            if float(scores[order[0]]) > best_score:
                best_score = float(scores[order[0]])
                best_x = candidates[order[0]].copy()
            history.append(best_score)
        return best_x, best_score, history


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return accuracy and macro/per-class metrics without extra dependencies."""
    per_class = {}
    f1_values = []
    for index, worker in enumerate(WORKERS):
        tp = int(np.sum((y_true == index) & (y_pred == index)))
        fp = int(np.sum((y_true != index) & (y_pred == index)))
        fn = int(np.sum((y_true == index) & (y_pred != index)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        f1_values.append(f1)
        per_class[worker] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(y_true == index)),
        }
    return {
        "accuracy": round(float(np.mean(y_true == y_pred)), 4) if len(y_true) else 0.0,
        "macro_f1": round(float(np.mean(f1_values)), 4) if f1_values else 0.0,
        "per_class": per_class,
    }


# 路由到四个 Worker 并支持 D 的并行升级。
class MultimodalRouter:
    # 初始化路由器。
    def __init__(
        self,
        extractor: MultimodalFeatureExtractor | None = None,
        config: RouterConfig | None = None,
    ):
        self.extractor = extractor or MultimodalFeatureExtractor(FeatureConfig())
        self.config = config or RouterConfig()
        self.weights: np.ndarray | None = None

    @property
    # 返回路由头参数量。
    def parameter_count(self) -> int:
        return len(WORKERS) * (self.extractor.config.dim + 1)

    # 将数据记录转换为特征矩阵。
    def _matrix(self, records):
        X = np.vstack(
            [self.extractor.extract(r.get("image_path"), r["query"]) for r in records]
        )
        return np.hstack([X, np.ones((len(records), 1), dtype=np.float32)])

    # 计算四个 Worker 的 logits。
    def _logits(self, X, flat):
        return X @ flat.reshape(len(WORKERS), -1).T

    # 将按插入顺序返回的语义 flags 转为稳定的命名信号。
    def _semantic_signals(self, query: str) -> dict[str, float | bool]:
        values = self.extractor.semantic_flag_map(query)
        keyword_names = self.extractor.KEYWORDS.keys()
        return {
            name: bool(value) if name in keyword_names else float(value)
            for name, value in values.items()
        }

    # 提取不依赖模型 logits 的共享门控策略。
    def _rule_policy(self, query: str, image_path=None) -> dict:
        signals = self._semantic_signals(query)
        query_lower = query.lower()
        reasons = []
        if signals["complex"]:
            reasons.append("complex_keyword")
        if signals["high_risk"]:
            reasons.append("high_risk")
        if signals["query_length"] > 0.70:
            reasons.append("long_query")
        if signals["multi_image"]:
            reasons.append("multi_image")
        if signals["uncertainty"]:
            reasons.append("uncertain")

        explicit_text_only = any(
            token in query_lower
            for token in (
                "不要重新识别",
                "不重新识别",
                "已有巡检结果",
                "文字结果",
                "仅根据上游",
                "不要做视觉判断",
            )
        ) and bool(signals["report"] or signals["structured_output"])
        if explicit_text_only:
            reasons.append("explicit_text_only")

        scene_hint = self.extractor.image_scene_hint(image_path)
        scene = scene_hint["scene"]
        negated_inventory = any(
            token in query_lower
            for token in ("不要数商品", "不要执行盘点", "不要判断库存", "不做库存")
        )
        has_open_intent = bool(signals["open_domain"])
        has_inventory_intent = bool(signals["inventory"]) and not negated_inventory
        fact_only = "只描述可见事实" in query_lower
        if fact_only and not any(
            token in query_lower for token in ("合规", "检查是否", "判断是否")
        ):
            has_open_intent = True
            has_inventory_intent = False
        vague_request = any(
            token in query_lower
            for token in ("帮我看看这家店怎么样", "开放式意见", "怎么样")
        ) and not any(
            token in query_lower
            for token in (
                "盘点",
                "库存",
                "数量",
                "缺货",
                "合规",
                "消防",
                "报告",
                "文字",
                "提取",
            )
        )
        if signals["complex"] or signals["high_risk"]:
            query_intent = "complex"
        elif explicit_text_only:
            query_intent = "report"
        elif vague_request:
            query_intent = "unknown"
        elif has_open_intent and not has_inventory_intent:
            query_intent = "open"
        elif signals["report"] and not has_inventory_intent:
            query_intent = "report"
        elif (
            has_inventory_intent
            or signals["compliance"]
            or signals["ocr"]
            or signals["environment"]
        ):
            query_intent = "retail_visual"
        else:
            query_intent = "unknown"

        conflict = (
            (scene == "shelf" and query_intent == "open")
            or (scene == "open" and query_intent == "retail_visual")
            or (scene == "report" and query_intent == "open")
            or (
                scene == "shelf" and query_intent == "report" and not explicit_text_only
            )
            or (scene in ("shelf", "open", "report") and query_intent == "unknown")
        )
        if conflict:
            reasons.append("image_query_conflict")

        return {
            "signals": signals,
            "gate_reasons": reasons,
            "explicit_text_only": explicit_text_only,
            "scene_hint": scene_hint,
            "query_intent": query_intent,
            "conflict": conflict,
        }

    # 根据置信度、查询意图和场景执行路由门控。
    def _gate(self, logits: np.ndarray, query: str, image_path=None) -> dict:
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        raw_idx = int(probs.argmax())
        margin = float(np.sort(probs)[-1] - np.sort(probs)[-2])
        policy = self._rule_policy(query, image_path)
        reasons = list(policy["gate_reasons"])
        explicit_text_only = bool(policy["explicit_text_only"])
        query_intent = policy["query_intent"]
        conflict = bool(policy["conflict"])
        if float(probs[raw_idx]) < self.config.confidence_threshold:
            reasons.append("low_confidence")
        if margin < 0.08:
            reasons.append("low_margin")

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
            "scene_hint": policy["scene_hint"],
            "query_intent": query_intent,
        }

    @staticmethod
    # 使用类别中心初始化路由头。
    def _initial_from_centroids(X, y, classes):
        W = np.zeros((len(classes), X.shape[1]), dtype=np.float32)
        for i, cls in enumerate(classes):
            pos = X[np.asarray(y) == i]
            centroid = pos.mean(axis=0) if len(pos) else np.zeros(X.shape[1])
            W[i] = centroid / (np.linalg.norm(centroid) + 1e-6)
        return W.ravel()

    # 使用 sep-CMA-ES 训练路由头。
    def fit(self, records):
        X = self._matrix(records)
        y = np.asarray([WORKERS.index(r["label"]) for r in records])

        # 计算候选参数的端到端路由目标。
        def objective(flat):
            logits = self._logits(X, flat)
            shifted = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(shifted)
            probs /= probs.sum(axis=1, keepdims=True)
            loss = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-7, 1)).mean()
            gated = np.asarray(
                [
                    self._gate(row, r["query"], r.get("image_path"))["final_idx"]
                    for row, r in zip(logits, records)
                ]
            )
            acc = (gated == y).mean()
            raw_acc = (logits.argmax(axis=1) == y).mean()
            return float(acc - 0.03 * loss - 0.005 * (1.0 - raw_acc))

        initial = self._initial_from_centroids(X, y, WORKERS)
        optimizer = SepCMAES(initial.size, self.config)
        self.weights, score, history = optimizer.optimize(objective, initial)
        self.fit_info = {
            "best_fitness": score,
            "generations": self.config.generations,
            "history": history,
        }
        return self

    # 评估裸 logits 和门控后的路由结果。
    def evaluate(self, records):
        if self.weights is None:
            raise RuntimeError("router is not fitted")
        X = self._matrix(records)
        logits = self._logits(X, self.weights)
        y = np.asarray([WORKERS.index(r["label"]) for r in records])
        raw_pred = logits.argmax(axis=1)
        gated_info = [
            self._gate(row, r["query"], r.get("image_path"))
            for row, r in zip(logits, records)
        ]
        gated_pred = np.asarray([x["final_idx"] for x in gated_info])
        rule_pred = np.asarray(
            [self._rule_only_worker(r["query"], r.get("image_path")) for r in records]
        )

        # 构造混淆矩阵。
        def confusion(pred):
            cm = np.zeros((len(WORKERS), len(WORKERS)), dtype=int)
            for actual, predicted in zip(y, pred):
                cm[actual, predicted] += 1
            return cm.tolist()

        reasons = {}
        for x in gated_info:
            for reason in x["gate_reasons"]:
                reasons[reason] = reasons.get(reason, 0) + 1
        cost = np.asarray([2 if index == 3 else 1 for index in range(len(WORKERS))])
        raw_cost = float(np.mean(cost[raw_pred])) if len(raw_pred) else 0.0
        gated_cost = float(np.mean(cost[gated_pred])) if len(gated_pred) else 0.0
        rule_cost = float(np.mean(cost[rule_pred])) if len(rule_pred) else 0.0
        gate_corrections = int(np.sum((raw_pred != y) & (gated_pred == y)))
        gate_regressions = int(np.sum((raw_pred == y) & (gated_pred != y)))
        return {
            "accuracy": float((gated_pred == y).mean()),
            "raw_accuracy": float((raw_pred == y).mean()),
            "gated_accuracy": float((gated_pred == y).mean()),
            "raw_metrics": classification_metrics(y, raw_pred),
            "gated_metrics": classification_metrics(y, gated_pred),
            "rule_only_metrics": classification_metrics(y, rule_pred),
            "raw_confusion_matrix": confusion(raw_pred),
            "gated_confusion_matrix": confusion(gated_pred),
            "raw_predictions": raw_pred.tolist(),
            "gated_predictions": gated_pred.tolist(),
            "rule_only_predictions": rule_pred.tolist(),
            "gate_changes": int(np.sum(raw_pred != gated_pred)),
            "gate_change_rate": round(float(np.mean(raw_pred != gated_pred)), 4)
            if len(raw_pred)
            else 0.0,
            "gate_corrections": gate_corrections,
            "gate_regressions": gate_regressions,
            "gate_neutral_changes": int(
                np.sum(raw_pred != gated_pred) - gate_corrections - gate_regressions
            ),
            "gate_net_benefit": gate_corrections - gate_regressions,
            "estimated_cost": {
                "raw_mean_units": round(raw_cost, 4),
                "gated_mean_units": round(gated_cost, 4),
                "rule_only_mean_units": round(rule_cost, 4),
                "unit_definition": "single_worker=1, parallel_A_B=2",
            },
            "gate_reasons": reasons,
            "scene_hints": [x["scene_hint"] for x in gated_info],
        }

    # 提供不读取模型 logits 的规则基线，用于量化门控的独立贡献。
    def _rule_only_worker(self, query: str, image_path=None) -> int:
        """Apply the shared gate policy without model logits or probabilities."""
        policy = self._rule_policy(query, image_path)
        query_intent = policy["query_intent"]
        if policy["explicit_text_only"]:
            return WORKERS.index("Worker-C")
        intent_to_worker = {
            "open": WORKERS.index("Worker-B"),
            "retail_visual": WORKERS.index("Worker-A"),
            "report": WORKERS.index("Worker-C"),
        }
        if policy["conflict"] and query_intent == "unknown":
            return WORKERS.index("Worker-D")
        if query_intent in intent_to_worker:
            return intent_to_worker[query_intent]
        return WORKERS.index("Worker-D")

    # 评估独立标注的门控策略。
    def evaluate_gate_policy(self, records):
        if self.weights is None:
            raise RuntimeError("router is not fitted")
        counts = {
            "none": 0,
            "follow_query": 0,
            "explicit_text_only": 0,
            "escalate_ambiguous": 0,
            "explicit_d": 0,
        }
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
            "false_escalation_rate": float(false_escalations / (total - d_expected))
            if total > d_expected
            else 0.0,
            "expected_d": d_expected,
            "d_recall": float(d_hits / d_expected) if d_expected else 0.0,
        }

    # 执行单张图片的在线路由。
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
    # 生成 Worker 专用提示词。
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
            requirements.append(
                "对消防通道、价签或安全项给出可见证据和 pass/fail/unclear"
            )
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

    # 保存路由权重。
    def save(self, path: str | Path) -> None:
        if self.weights is None:
            raise RuntimeError("router is not fitted")
        np.save(path, self.weights)

    # 加载路由权重。
    def load(self, path: str | Path) -> "MultimodalRouter":
        self.weights = np.load(path).astype(np.float32)
        return self


# 读取 JSONL 数据并解析图片路径。
def load_jsonl(path: str | Path):
    base = Path(path).resolve().parent
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    for record in records:
        image_path = record.get("image_path")
        if image_path and not Path(image_path).is_absolute():
            record["image_path"] = str((base / image_path).resolve())
    return records


# 提供路由器训练、评估和预测入口。
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate/predict the Task 2 multimodal router"
    )
    parser.add_argument(
        "--data", default=str(Path(__file__).with_name("training_data.jsonl"))
    )
    parser.add_argument(
        "--model", default=str(Path(__file__).with_name("router_weights.npy"))
    )
    parser.add_argument(
        "--mode", choices=("train", "evaluate", "predict"), default="train"
    )
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
        print(
            json.dumps(
                {
                    "train": len(train),
                    "test": len(test),
                    "hard_negative": len(hard),
                    **result,
                    "hard_negative_eval": hard_result,
                    "gate_policy_eval": policy_result,
                    "params": router.parameter_count,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        router.load(args.model)
        if args.mode == "evaluate":
            print(
                json.dumps(
                    {
                        "test": router.evaluate(test),
                        "hard_negative": router.evaluate(hard) if hard else {},
                        "gate_policy": {
                            "hard_negative": router.evaluate_gate_policy(hard)
                            if hard
                            else {},
                            "clean_test": router.evaluate_gate_policy(test)
                            if test
                            else {},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(
                json.dumps(
                    router.route(args.image, args.query), ensure_ascii=False, indent=2
                )
            )


if __name__ == "__main__":
    main()
