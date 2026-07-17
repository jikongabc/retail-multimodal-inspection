"""Build 75 Task 2 routing labels from licensed real-photo fixtures.

Python 3.10+
主要依赖：Pillow>=10.0。
默认使用 ``real_fixture_sources.jsonl`` 对应的 Wikimedia Commons 照片；
删除真实照片清单后才回退到可控合成夹具。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).parent
FIXTURES = ROOT / "fixtures"
REAL_SOURCES = ROOT / "real_fixture_sources.jsonl"


def make_image(path: Path, kind: str, seed: int) -> None:
    """Render a deterministic, varied scene fixture.

    These are still synthetic controls rather than collected store photos.  The
    scene grammar varies camera framing, density, occlusion, lighting and
    layout so that each record is not just a recolored copy of one template.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random((seed + 1) * 7919 + sum(ord(c) for c in kind) * 101)
    width, height = 640, 420
    base = {
        "shelf": (231, 205, 161),
        "safety": (224, 186, 181),
        "open": (143, 195, 226),
        "report": (207, 197, 235),
        "hybrid": (218, 199, 150),
    }[kind]
    image = Image.new("RGB", (width, height), base)
    draw = ImageDraw.Draw(image)
    # Low-frequency lighting plus fine sensor-like texture make the controls
    # visibly distinct without turning them into nondeterministic data.
    top = tuple(min(255, max(0, c + rng.randint(-16, 16))) for c in base)
    bottom = tuple(min(255, max(0, c - rng.randint(0, 22))) for c in base)
    for y in range(height):
        t = y / max(1, height - 1)
        color = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        draw.line((0, y, width, y), fill=color)
    # Small, soft background variation avoids a flat-color shortcut.
    for _ in range(160):
        x = rng.randrange(width)
        y = rng.randrange(height)
        radius = rng.choice((1, 1, 2, 3))
        delta = rng.choice((-10, -6, 6, 10))
        color = tuple(min(255, max(0, c + delta)) for c in base)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    def shelf_rows(x0: int, y0: int, shelf_w: int, rows: int, scale: float = 1.0) -> None:
        draw.rectangle((x0 - 14, y0 - 22, x0 + shelf_w + 14, height - 35), fill=(82, 70, 59), outline=(42, 37, 34), width=3)
        row_h = int(92 * scale)
        for row in range(rows):
            y = y0 + row * row_h
            draw.rectangle((x0, y, x0 + shelf_w, y + row_h - 18), fill=(194, 174, 147), outline=(115, 96, 78), width=2)
            count = rng.randint(4, 8)
            gap = max(6, shelf_w // (count + 2))
            for col in range(count):
                if rng.random() < 0.14:
                    continue
                px = x0 + 12 + col * gap + rng.randint(-3, 3)
                pw = rng.randint(max(18, int(22 * scale)), max(26, int(34 * scale)))
                ph = rng.randint(max(35, int(48 * scale)), max(48, int(70 * scale)))
                py = y + row_h - 25 - ph
                palette = ((176, 63, 48), (51, 112, 156), (232, 178, 49), (70, 137, 83), (132, 80, 137))
                fill = palette[(seed + row * 3 + col) % len(palette)]
                if (row + col + seed) % 4 == 0:
                    draw.rounded_rectangle((px, py, px + pw, py + ph), radius=5, fill=fill, outline=(52, 47, 43), width=2)
                    draw.ellipse((px + pw // 3, py - 12, px + (2 * pw) // 3, py + 5), fill=(205, 205, 188), outline=(52, 47, 43), width=2)
                else:
                    draw.rounded_rectangle((px, py, px + pw, py + ph), radius=3, fill=fill, outline=(52, 47, 43), width=2)
                    draw.line((px + 4, py + ph // 3, px + pw - 4, py + ph // 3), fill=(245, 235, 209), width=2)
            draw.rectangle((x0, y + row_h - 18, x0 + shelf_w, y + row_h - 7), fill=(57, 51, 46))
            if row % 2 == 0:
                draw.text((x0 + 8, y + row_h - 17), f"{rng.randint(9, 39)}.{rng.randint(0, 9)}", fill=(245, 235, 205))

    if kind == "shelf":
        # Front-facing shelf, oblique aisle and partially empty facings.
        variant = seed % 4
        if variant == 0:
            shelf_rows(46 + rng.randint(-8, 12), 52, 548, 3, 1.0)
        elif variant == 1:
            shelf_rows(32, 78, 365, 3, 0.9)
            draw.polygon([(410, 70), (622, 112), (622, 370), (430, 370)], fill=(189, 168, 139), outline=(105, 87, 72))
            shelf_rows(430, 95, 170, 2, 0.72)
        elif variant == 2:
            draw.polygon([(58, 70), (520, 46), (568, 360), (22, 360)], fill=(199, 179, 150), outline=(100, 83, 69))
            for y in (124, 216, 305):
                draw.line((24, y, 565, y - 18), fill=(67, 58, 51), width=8)
            for col in range(9):
                x = 55 + col * 54
                draw.rectangle((x, 70 + col * 2, x + 33, 116 + col * 2), fill=((170 + col * 7) % 240, 80 + col * 8, 56), outline=(54, 43, 35), width=2)
        else:
            shelf_rows(76, 40, 490, 2, 1.2)
            draw.rectangle((24, 314, 616, 398), fill=(111, 91, 75), outline=(64, 53, 44), width=3)
            for x in range(45, 600, 75):
                draw.line((x, 330, x + 45, 383), fill=(160, 133, 105), width=5)

    elif kind == "open":
        # Non-retail open-domain compositions: room, plant, parcel, or vehicle.
        variant = seed % 4
        draw.rectangle((0, 278, width, height), fill=(126, 143, 151))
        if variant == 0:
            draw.rectangle((60, 48, 310, 265), fill=(202, 225, 232), outline=(46, 76, 87), width=8)
            draw.line((185, 52, 185, 263), fill=(70, 105, 115), width=5)
            draw.line((63, 155, 307, 155), fill=(70, 105, 115), width=5)
            draw.rectangle((355, 177, 548, 291), fill=(153, 89, 57), outline=(70, 47, 37), width=4)
            draw.ellipse((389, 102, 506, 220), fill=(47, 121, 70), outline=(32, 76, 48), width=5)
            draw.ellipse((430, 64, 556, 187), fill=(60, 142, 81), outline=(32, 76, 48), width=5)
            draw.line((447, 185, 444, 286), fill=(80, 53, 39), width=8)
        elif variant == 1:
            draw.polygon([(30, 276), (190, 93), (385, 94), (602, 276)], fill=(88, 105, 119), outline=(49, 62, 70))
            draw.polygon([(62, 268), (204, 122), (356, 122), (554, 268)], fill=(169, 181, 185))
            draw.rectangle((248, 170, 391, 279), fill=(56, 100, 133), outline=(24, 49, 66), width=5)
            draw.ellipse((269, 190, 369, 266), fill=(210, 180, 73), outline=(66, 59, 38), width=4)
            draw.line((0, 320, width, 320), fill=(210, 218, 216), width=4)
        elif variant == 2:
            draw.ellipse((95, 100, 410, 330), fill=(225, 181, 64), outline=(90, 67, 33), width=7)
            draw.ellipse((145, 145, 360, 305), fill=(247, 222, 115), outline=(107, 80, 38), width=5)
            draw.rectangle((425, 135, 544, 290), fill=(198, 117, 54), outline=(79, 48, 30), width=5)
            draw.line((485, 142, 485, 283), fill=(245, 203, 117), width=4)
            draw.ellipse((35, 325, 602, 435), fill=(103, 92, 92), outline=(69, 61, 63), width=3)
        else:
            draw.polygon([(64, 280), (142, 105), (492, 105), (575, 280)], fill=(202, 206, 199), outline=(87, 94, 91), width=5)
            draw.rectangle((190, 152, 445, 281), fill=(68, 130, 163), outline=(39, 68, 83), width=5)
            draw.ellipse((235, 177, 397, 288), fill=(184, 101, 64), outline=(83, 51, 39), width=5)
            draw.ellipse((284, 119, 375, 205), fill=(52, 102, 64), outline=(31, 65, 41), width=4)
        draw.line((0, 278, width, 278), fill=(70, 83, 88), width=5)

    elif kind == "report":
        # A varied desk/document scene with tables, charts, receipts and forms.
        draw.rectangle((0, 322, width, height), fill=(126, 116, 126))
        variant = seed % 4
        if variant in (0, 1):
            left = 70 + rng.randint(-20, 30)
            top = 36 + rng.randint(-8, 24)
            right = left + rng.randint(380, 480)
            bottom = 332 + rng.randint(-5, 20)
            draw.rounded_rectangle((left, top, right, bottom), radius=8, fill=(247, 245, 237), outline=(93, 84, 103), width=5)
            draw.line((left + 25, top + 50, right - 30, top + 50), fill=(93, 84, 103), width=4)
            for row in range(4 + seed % 3):
                y = top + 84 + row * 38
                draw.line((left + 28, y, right - 28, y), fill=(171, 165, 174), width=2)
            split = left + (right - left) * 0.42
            draw.line((split, top + 70, split, bottom - 30), fill=(171, 165, 174), width=2)
            for bar in range(5):
                x = left + 35 + bar * 48
                h = 30 + ((seed * 17 + bar * 23) % 90)
                draw.rectangle((x, bottom - 37 - h, x + 25, bottom - 37), fill=((83, 120, 157), (178, 104, 76), (91, 143, 112))[bar % 3])
        elif variant == 2:
            draw.polygon([(92, 51), (526, 35), (572, 331), (123, 354)], fill=(247, 245, 237), outline=(91, 82, 100))
            for row in range(7):
                y = 88 + row * 31
                draw.line((132, y, 510, y - 16), fill=(130, 123, 143), width=3)
            for col in range(5):
                x = 175 + col * 70
                draw.line((x, 80, x + 22, 316), fill=(188, 180, 189), width=2)
            draw.ellipse((390, 106, 510, 226), fill=(113, 87, 157), outline=(65, 48, 92), width=4)
            draw.pieslice((390, 106, 510, 226), 20, 150, fill=(210, 135, 68))
        else:
            draw.rounded_rectangle((80, 47, 278, 346), radius=7, fill=(246, 243, 234), outline=(90, 80, 98), width=4)
            draw.rounded_rectangle((302, 72, 550, 319), radius=7, fill=(235, 231, 224), outline=(90, 80, 98), width=4)
            for y in range(93, 292, 34):
                draw.line((105, y, 248, y), fill=(133, 125, 143), width=3)
            draw.rectangle((334, 112, 512, 145), fill=(111, 141, 159))
            draw.rectangle((334, 165, 430, 291), outline=(111, 141, 159), width=4)
            draw.line((345, 272, 411, 190), fill=(178, 104, 76), width=5)
            draw.line((411, 190, 493, 235), fill=(178, 104, 76), width=5)

    elif kind == "safety":
        # Safety/compliance scene: exit door, extinguisher, cones and warning tape.
        draw.rectangle((0, 302, width, height), fill=(115, 111, 106))
        draw.rectangle((52, 45, 307, 307), fill=(104, 121, 126), outline=(58, 64, 66), width=7)
        draw.rectangle((92, 78, 267, 306), fill=(48, 108, 91), outline=(31, 58, 50), width=6)
        draw.rectangle((112, 130, 244, 161), fill=(238, 225, 72), outline=(69, 69, 28), width=3)
        draw.text((134, 138), "EXIT", fill=(35, 55, 39))
        draw.rectangle((376, 155, 443, 296), fill=(192, 42, 35), outline=(79, 29, 25), width=5)
        draw.rectangle((388, 174, 431, 220), fill=(226, 224, 202), outline=(80, 46, 37), width=3)
        draw.line((390, 223, 428, 223), fill=(246, 210, 120), width=5)
        draw.polygon([(512, 305), (581, 305), (549, 197)], fill=(234, 150, 38), outline=(93, 54, 23))
        draw.line((530, 270, 563, 270), fill=(250, 230, 150), width=10)
        draw.line((0, 262, width, 190 + rng.randint(-18, 20)), fill=(223, 49, 43), width=8)
        draw.line((0, 278, width, 206 + rng.randint(-18, 20)), fill=(247, 216, 73), width=6)

    elif kind == "hybrid":
        # Composite inspection frame: merchandise plus an explicit risk cue.
        shelf_rows(34 + rng.randint(-8, 18), 62, rng.randint(330, 430), 2, 0.9)
        draw.rectangle((438, 58, 602, 209), fill=(242, 239, 218), outline=(96, 80, 64), width=4)
        draw.polygon([(520, 79), (583, 174), (456, 174)], fill=(224, 57, 44), outline=(95, 30, 27))
        draw.rectangle((513, 112, 528, 147), fill=(247, 227, 143))
        draw.ellipse((512, 154, 529, 171), fill=(247, 227, 143))
        draw.rectangle((438, 246, 604, 342), fill=(186, 181, 165), outline=(82, 78, 72), width=4)
        for y in (270, 298, 326):
            draw.line((452, y, 584, y), fill=(91, 87, 80), width=3)
        for y in (270, 298, 326):
            draw.ellipse((458, y - 8, 474, y + 8), outline=(52, 74, 57), width=3)
        draw.rectangle((0, 365, width, height), fill=(103, 89, 74))

    image.save(path)


TRAIN_QUERIES = {
    "Worker-A": [
        "清点货架上的 SKU，并标注没有商品的槽位。",
        "检查零售陈列是否有库存不足的商品。",
        "按货架层次记录商品数量和摆放位置。",
        "找出照片中的空货位，给出盘点证据。",
        "核对每排商品是否完整，输出缺货清单。",
        "对这张门店图做一次商品库存核验。",
        "请识别货架陈列中的品类和数量，不写总结报告。",
        "统计各层陈列面，区分有货位和空位。",
        "检查商品是否按货架位置正常摆放。",
        "请做零售货架的数量盘点和缺货定位。",
        "列出可见商品的 SKU、数量与所在层。",
        "从视觉证据判断哪些陈列位需要补货。",
        "完成一次货架库存巡查，保留看不清的项目。",
        "检查门店货架上是否存在断货和漏摆。",
        "把商品陈列按位置编码后输出盘点结果。",
    ],
    "Worker-B": [
        "请用通用视觉能力说明画面的主体和背景。",
        "这张照片展示了什么场景？不要做库存判断。",
        "识别图中主要物体，并解释它们的空间关系。",
        "请回答这是什么，不要套用零售业务规则。",
        "从开放域角度描述人物、动物或物品。",
        "说明照片里最显著的视觉元素及其位置。",
        "请做普通图片问答，不需要生成巡检报告。",
        "判断画面属于什么环境，并给出可见依据。",
        "描述这个场景的内容，避免臆测不可见信息。",
        "请解释图中对象之间可能的关系。",
        "这是一个通用视觉识别问题，请先陈述事实。",
        "请识别照片中的主体，不要计算货架库存。",
        "从非零售角度回答图片理解请求。",
        "请描述画面结构和主要对象，不输出整改建议。",
        "判断这张图的场景类别并说明理由。",
    ],
    "Worker-C": [
        "把上游识别结果整理成巡检摘要和整改建议。",
        "汇总这批数据，生成管理层周报结论。",
        "将已有字段转成 JSON 检查报告并补充建议。",
        "比较两次巡检结果，分析差异原因。",
        "只依据视觉中间结果写一份风险总结。",
        "将多个 Worker 的输出合并为统一报告。",
        "把识别明细压缩成可读的门店经营摘要。",
        "请整理结果，不要重新识别图片中的商品。",
        "根据给定事实生成严重度、证据和行动项。",
        "完成文本数据汇总并输出结构化结论。",
        "请为巡检记录生成 JSON 字段和推荐措施。",
        "阅读已有 OCR 与盘点结果，写最终报告。",
        "解释异常原因并按优先级排列整改建议。",
        "把多张图片的分析结果汇总成一页报告。",
        "仅基于输入数据完成逻辑推理和管理摘要。",
    ],
    "Worker-D": [
        "同时核对商品数量和消防通道，结论需要二次复核。",
        "多张门店照片综合评估，输出高置信度结果。",
        "当视觉证据冲突时，让两个专家交叉判断。",
        "这是高风险巡检，不能确定的项目必须升级复核。",
        "综合盘点、合规和价格标签，保留冲突证据。",
        "逐张分析多图后汇总，并检查严重安全项。",
        "请并行调用零售专家与通用视觉专家再下结论。",
        "复杂门店场景要求高置信度和证据交叉验证。",
        "同时判断库存和环境风险，遇到模糊区域不要硬猜。",
        "综合检查图片中的商品、消防和价签问题。",
        "对高风险发现进行独立视觉复核并给出最终结论。",
        "多角度巡检中如果 A/B 意见不同，输出不确定性。",
        "请交叉核验货架盘点与安全合规结果。",
        "复杂图片需要 A+B 并行后生成统一判断。",
        "同时完成识别、合规和风险确认，要求可追溯证据。",
    ],
}

TEST_QUERIES = {
    "Worker-A": [
        "请数清每个货架层面的 SKU，并标出空槽位。",
        "这张陈列照片里各商品还剩多少？",
        "做一次库存盘点，给出缺货位置。",
        "请核对货架陈列是否有漏摆商品。",
    ],
    "Worker-B": [
        "画面主体是什么？请用通用视觉描述回答。",
        "不要套用零售规则，解释图片里出现的对象。",
        "请辨认照片中的场景和主要物体，不做库存判断。",
        "这是人物、动物还是物品？请说明理由。",
    ],
    "Worker-C": [
        "将已有识别结果整理成巡检摘要和整改建议。",
        "把这些字段汇总为管理层周报。",
        "仅根据上游结果输出 JSON 检查报告。",
        "比较两次巡检结果并给出原因分析。",
    ],
    "Worker-D": [
        "请同时核对商品数量和消防通道，结论需二次复核。",
        "多张门店照片综合评估，输出高置信结论。",
        "当图像证据冲突时，请让两个视觉专家交叉判断。",
    ],
}

HARD_NEGATIVES = [
    # label is the independently annotated query-intent Worker. expected_action
    # is a separate policy annotation; it is never used as the Worker gold.
    ("hn-01", "b_00", "Worker-B", "none", "开放域图片中出现一个商品词，请描述对象，不要判断库存。"),
    ("hn-02", "a_00", "Worker-B", "follow_query", "请描述这张货架照片的整体场景，不要数商品。"),
    ("hn-03", "c_00", "Worker-C", "none", "识别图片里的文字，并把结果整理成管理层摘要。"),
    ("hn-04", "a_01", "Worker-B", "escalate_ambiguous", "帮我看看这家店怎么样？请给开放式意见。"),
    ("hn-05", "b_01", "Worker-A", "follow_query", "请从照片判断商品数量，但不要回答开放域问题。"),
    ("hn-06", "d_00", "Worker-C", "explicit_text_only", "只输出已有巡检结果的报告摘要，不要重新识别图片。"),
    ("hn-07", "b_02", "Worker-B", "none", "请确认图中的对象是否有价格标签，只描述可见事实。"),
    ("hn-08", "c_01", "Worker-B", "follow_query", "这张图是什么场景？请做通用视觉说明。"),
    ("hn-09", "a_02", "Worker-C", "follow_query", "请把这张图片的已知信息整理成一段管理层摘要。"),
    ("hn-10", "b_03", "Worker-A", "follow_query", "请找出照片里的货架商品并列出数量。"),
    ("hn-11", "d_01", "Worker-C", "explicit_text_only", "根据已经给出的文字结果生成 JSON，不要做视觉判断。"),
    ("hn-12", "a_03", "Worker-B", "follow_query", "不要执行盘点，只说明这张图的视觉场景。"),
]


def fixture_file(stem: str) -> Path:
    """Resolve a real-photo asset first, then the deterministic fallback."""
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = FIXTURES / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return FIXTURES / f"{stem}.png"


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    real_photo_mode = REAL_SOURCES.exists()
    rows = []
    kinds = {"Worker-A": "shelf", "Worker-B": "open", "Worker-C": "report", "Worker-D": "hybrid"}
    for worker, queries in TRAIN_QUERIES.items():
        for i in range(15):
            stem = f"{worker[-1].lower()}_{i:02d}"
            image = fixture_file(stem)
            if not real_photo_mode:
                make_image(image, "safety" if worker == "Worker-D" and i % 3 == 0 else kinds[worker], i)
            rows.append({
                "id": f"{worker[-1]}-{i:02d}",
                "image_path": str(Path("fixtures") / image.name),
                "query": queries[i],
                "label": worker,
                "split": "train",
                "construction": "licensed_real_photo_train_wikimedia_commons" if real_photo_mode else "controlled_fixture_train_v3_diverse_synthetic_scene",
            })
    test = []
    for worker, queries in TEST_QUERIES.items():
        for i, query in enumerate(queries):
            stem = f"{worker[-1].lower()}_{15 + i:02d}"
            image = fixture_file(stem)
            if not real_photo_mode:
                make_image(image, "safety" if worker == "Worker-D" and i % 2 == 0 else kinds[worker], 100 + i)
            test.append({
                "id": f"{worker[-1]}-test-{i:02d}",
                "image_path": str(Path("fixtures") / image.name),
                "query": query,
                "label": worker,
                "expected_action": "explicit_d" if worker == "Worker-D" else "none",
                "split": "test",
                "construction": "licensed_real_photo_test_wikimedia_commons_unseen_query" if real_photo_mode else "controlled_fixture_test_v3_unseen_query_diverse_synthetic_scene",
            })
    with open(ROOT / "training_data.jsonl", "w", encoding="utf-8") as f:
        for record in rows + test:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    hard = []
    for rid, image_name, label, expected_action, query in HARD_NEGATIVES:
        image = fixture_file(image_name)
        hard.append({
            "id": rid,
            "image_path": str(Path("fixtures") / image.name),
            "query": query,
            "label": label,
            "expected_action": expected_action,
            "split": "hard_negative",
            "construction": "adversarial_boundary_v5_independent_worker_gold_and_gate_policy_annotation_reuses_train_real_photo" if real_photo_mode else "adversarial_boundary_v4_independent_worker_gold_and_gate_policy_annotation_reuses_train_image",
        })
    with open(ROOT / "hard_negative.jsonl", "w", encoding="utf-8") as f:
        for record in hard:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} train + {len(test)} unseen-query test + {len(hard)} hard negatives")


if __name__ == "__main__":
    main()
