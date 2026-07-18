# 提取图片、文本和任务信号的路由特征。

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


WORKERS = ("Worker-A", "Worker-B", "Worker-C", "Worker-D")


# 配置图像、文本和任务特征维度。
@dataclass
class FeatureConfig:
    image_dim: int = 64
    text_dim: int = 96
    use_clip: bool = False
    use_image: bool = True
    use_text: bool = True
    use_signals: bool = True

    @property
    # 返回拼接后的特征维度。
    def dim(self) -> int:
        return self.image_dim + self.text_dim + 16


# 将图片和查询转换为归一化定长特征。
class MultimodalFeatureExtractor:
    KEYWORDS = {
        "inventory": (
            "盘点",
            "库存",
            "商品",
            "货架",
            "陈列",
            "缺货",
            "空位",
            "空槽",
            "数量",
            "sku",
            "补货",
            "inventory",
            "stock",
        ),
        "compliance": (
            "合规",
            "消防",
            "通道",
            "价格标签",
            "价签",
            "安全",
            "违规",
            "compliance",
            "safety",
        ),
        "ocr": ("文字", "提取", "读取", "ocr", "标签", "价格", "促销", "文本", "read"),
        "environment": (
            "环境",
            "卫生",
            "湿滑",
            "照明",
            "拥挤",
            "评估",
            "environment",
            "cleanliness",
        ),
        "report": (
            "报告",
            "汇总",
            "总结",
            "摘要",
            "周报",
            "整理",
            "结果",
            "原因",
            "建议",
            "改写",
            "report",
            "summarize",
            "recommend",
        ),
        "reasoning": (
            "分析",
            "原因",
            "推理",
            "比较",
            "方案",
            "诊断",
            "reason",
            "compare",
        ),
        "open_domain": (
            "这是什么",
            "开放域",
            "开放式",
            "通用视觉",
            "识别人物",
            "人物",
            "动物",
            "自然",
            "场景",
            "物体",
            "不要套用零售",
            "open",
            "general",
        ),
        "complex": (
            "综合",
            "多角度",
            "高置信",
            "交叉验证",
            "二次复核",
            "冲突",
            "同时",
            "复杂",
            "all",
            "cross-check",
        ),
        "high_risk": ("立即", "高风险", "严重", "安全事故", "消防出口", "high risk"),
        "visual": ("图片", "图像", "照片", "画面", "image", "photo"),
    }
    SIGNAL_NAMES = (
        "query_length",
        "question_marks",
        "plus_signs",
        "structured_output",
        "multi_image",
        "uncertainty",
    )

    # 初始化特征提取器和可选编码器。
    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.backend = "gamma-offline"
        self._scene_cache: dict[str, dict] = {}
        self._clip = None
        self._text_model = None
        if self.config.use_clip or os.getenv("MM_ROUTER_USE_CLIP") == "1":
            self._try_load_optional_models()

    # 尝试加载 CLIP 和文本编码器。
    def _try_load_optional_models(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            from sentence_transformers import SentenceTransformer

            model_name = os.getenv(
                "MM_ROUTER_CLIP_MODEL", "openai/clip-vit-base-patch32"
            )
            self._clip = (
                CLIPProcessor.from_pretrained(model_name),
                CLIPModel.from_pretrained(model_name),
            )
            self._text_model = SentenceTransformer(
                os.getenv("MM_ROUTER_TEXT_MODEL", "all-MiniLM-L6-v2")
            )
            self.backend = "clip+sentence-transformers"
        except Exception:
            self._clip = None
            self._text_model = None
            self.backend = "gamma-offline"

    @staticmethod
    # 计算稳定的有符号哈希文本向量。
    def _hash_embedding(text: str, dim: int) -> np.ndarray:
        out = np.zeros(dim, dtype=np.float32)
        grams = list(re.findall(r"[\u4e00-\u9fff]", text.lower()))
        grams += re.findall(r"[a-z0-9]+", text.lower())
        grams += ["<bos>"] + [f"{a}|{b}" for a, b in zip(grams, grams[1:])]
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            out[idx] += sign
        norm = np.linalg.norm(out)
        return out / norm if norm else out

    # 提取图片颜色、纹理和边缘统计量。
    def _image_proxy(self, image_path: str | Path | None) -> np.ndarray:
        if not image_path or not Path(image_path).exists():
            return np.zeros(self.config.image_dim, dtype=np.float32)
        image = Image.open(image_path).convert("RGB").resize((128, 128))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        gray = arr.mean(axis=2)
        hist = []
        for c in range(3):
            hist.extend(
                np.histogram(arr[:, :, c], bins=16, range=(0, 1), density=True)[0]
                / 16.0
            )
        gx = np.abs(np.diff(gray, axis=1)).mean()
        gy = np.abs(np.diff(gray, axis=0)).mean()
        edge = (np.abs(np.diff(gray, axis=1)) > 0.12).mean() + (
            np.abs(np.diff(gray, axis=0)) > 0.12
        ).mean()
        hsv_like = np.max(arr, axis=2) - np.min(arr, axis=2)
        stats = [
            *arr.mean(axis=(0, 1)).tolist(),
            *arr.std(axis=(0, 1)).tolist(),
            float(gray.mean()),
            float(gray.std()),
            float(gx),
            float(gy),
            float(edge),
            float(hsv_like.mean()),
            float(hsv_like.std()),
            float(arr.shape[1] / arr.shape[0]),
        ]
        vec = np.asarray(hist + stats, dtype=np.float32)
        if vec.size < self.config.image_dim:
            vec = np.pad(vec, (0, self.config.image_dim - vec.size))
        return vec[: self.config.image_dim]

    # 根据图片统计量返回场景提示。
    def image_scene_hint(self, image_path: str | Path | None) -> dict:
        if not image_path or not Path(image_path).exists():
            return {"scene": "unknown", "confidence": 0.0, "evidence": "image_missing"}
        cache_key = str(Path(image_path).resolve())
        if cache_key in self._scene_cache:
            return self._scene_cache[cache_key]
        image = Image.open(image_path).convert("RGB").resize((128, 128))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        r, g, b = mean.tolist()
        if r - g > 0.14 and r - b > 0.14:
            result = {
                "scene": "safety",
                "confidence": round(min(1.0, (r - g + r - b) / 0.5), 3),
                "evidence": "red_warning_dominance",
            }
            self._scene_cache[cache_key] = result
            return result
        if b - r > 0.08 and b - g > 0.0:
            result = {
                "scene": "open",
                "confidence": round(min(1.0, (b - r + b - g) / 0.5), 3),
                "evidence": "blue_open_scene_dominance",
            }
            self._scene_cache[cache_key] = result
            return result
        if b > 0.70 and b - r > -0.02 and float(std.mean()) < 0.23:
            result = {
                "scene": "report",
                "confidence": 0.78,
                "evidence": "purple_document_background",
            }
            self._scene_cache[cache_key] = result
            return result
        if r - b > 0.13 and r - g < 0.14:
            result = {
                "scene": "shelf",
                "confidence": 0.72,
                "evidence": "warm_shelf_background",
            }
            self._scene_cache[cache_key] = result
            return result
        result = {
            "scene": "unknown",
            "confidence": 0.0,
            "evidence": "no_stable_scene_hint",
        }
        self._scene_cache[cache_key] = result
        return result

    # 提取查询关键词、复杂度和不确定性信号。
    def _semantic_flags(self, query: str) -> np.ndarray:
        q = query.lower()
        flags = []
        for terms in self.KEYWORDS.values():
            flags.append(float(any(term.lower() in q for term in terms)))
        flags.extend(
            [
                min(len(q) / 120.0, 1.0),
                min(q.count("?") + q.count("？"), 3) / 3.0,
                min(q.count("+"), 3) / 3.0,
                float(any(x in q for x in ("json", "结构化", "字段"))),
                float(any(x in q for x in ("两张", "多张", "多图", "逐张"))),
                float(any(x in q for x in ("不能确定", "不确定", "置信度", "核验"))),
            ]
        )
        return np.asarray(flags, dtype=np.float32)

    # 返回带名称的语义信号，避免调用方依赖数组位置。
    def semantic_flag_map(self, query: str) -> dict[str, float]:
        names = (*self.KEYWORDS.keys(), *self.SIGNAL_NAMES)
        return dict(zip(names, self._semantic_flags(query), strict=True))

    # 提取并归一化图片与查询的联合特征。
    def extract(self, image_path: str | Path | None, query: str) -> np.ndarray:
        if (
            self.backend == "clip+sentence-transformers"
            and self._clip
            and self._text_model
        ):
            import torch

            image = (
                Image.open(image_path).convert("RGB")
                if image_path
                else Image.new("RGB", (128, 128))
            )
            processor, model = self._clip
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_vec = model.get_image_features(**inputs).squeeze().cpu().numpy()
            text_vec = self._text_model.encode(query, normalize_embeddings=True)
            image_vec = MultimodalFeatureExtractor._resize_projection(
                image_vec, self.config.image_dim
            )
            text_vec = MultimodalFeatureExtractor._resize_projection(
                text_vec, self.config.text_dim
            )
        else:
            image_vec = self._image_proxy(image_path)
            text_vec = MultimodalFeatureExtractor._hash_embedding(
                query, self.config.text_dim
            )
        flags = self._semantic_flags(query)
        image_vec = self._resize_projection(image_vec, self.config.image_dim)
        text_vec = self._resize_projection(text_vec, self.config.text_dim)
        flags = self._resize_projection(flags, 16)
        if not self.config.use_image:
            image_vec = np.zeros_like(image_vec)
        if not self.config.use_text:
            text_vec = np.zeros_like(text_vec)
        if not self.config.use_signals:
            flags = np.zeros_like(flags)
        vec = np.concatenate([0.10 * image_vec, 0.70 * text_vec, 0.20 * flags]).astype(
            np.float32
        )
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    @staticmethod
    # 将输入向量投影到指定维度并归一化。
    def _resize_projection(values: Iterable[float], dim: int) -> np.ndarray:
        values = np.asarray(list(values), dtype=np.float32)
        if values.size == dim:
            out = values
        elif values.size > dim:
            chunks = np.array_split(values, dim)
            out = np.asarray([c.mean() for c in chunks], dtype=np.float32)
        else:
            out = np.pad(values, (0, dim - values.size))
        norm = np.linalg.norm(out)
        return out / norm if norm else out
