# 测量 Ostrakon 五类零售场景的延迟和显存。

import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL_PATH = Path("/root/autodl-tmp/models/Ostrakon-VL-8B")
WORK_DIR = Path("/root/autodl-tmp/ostrakon_task0")
IMAGE_DIR = WORK_DIR / "images"
RESULT_JSON = WORK_DIR / "ostrakon_5scenes_results.json"
SUMMARY_TXT = WORK_DIR / "ostrakon_5scenes_summary.txt"
REPEATS = 3

SCENES = [
    {
        "id": "inventory",
        "name": "货架盘点",
        "image": "01_inventory.jpg",
        "prompt": (
            "你是零售货架盘点员。统计图中可见的商品陈列面数量，分别给出每层数量和总数。"
            "只报告可见证据、结论和置信度，不展示思维链，80字以内。"
        ),
    },
    {
        "id": "compliance",
        "name": "合规检查",
        "image": "02_compliance.jpg",
        "prompt": (
            "你是门店安全合规巡检员。识别图中的消防或通道违规，给出风险等级和立即整改动作。"
            "只报告可见证据和结论，不展示思维链，80字以内。"
        ),
    },
    {
        "id": "ocr",
        "name": "OCR识别",
        "image": "03_ocr.jpg",
        "prompt": (
            "读取图中所有商品名、价格和促销信息，逐行原样输出。"
            "不要猜测不可见文字，不展示思维链。"
        ),
    },
    {
        "id": "environment",
        "name": "环境巡检",
        "image": "04_environment.jpg",
        "prompt": (
            "检查门店环境卫生和通道安全。列出图中可见隐患、风险等级以及处理优先级。"
            "不展示思维链，80字以内。"
        ),
    },
    {
        "id": "out_of_stock",
        "name": "缺货检测",
        "image": "05_out_of_stock.jpg",
        "prompt": (
            "检测货架缺货情况：统计有货位、空货位和总货位，并计算缺货率。"
            "只报告可见证据和结论，不展示思维链，80字以内。"
        ),
    },
]


# 将显存字节数转换为 MB。
def gpu_mb(value):
    return round(value / 1024**2, 1)


# 执行单个场景并记录推理指标。
def run_scene(model, processor, scene):
    image_path = IMAGE_DIR / scene["image"]
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": scene["prompt"]},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    latency_samples_ms = []
    peak_samples_mb = []
    answer = ""
    for _ in range(REPEATS):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                use_cache=True,
            )
        torch.cuda.synchronize()
        latency_samples_ms.append((time.perf_counter() - started) * 1000)
        peak_samples_mb.append(gpu_mb(torch.cuda.max_memory_allocated()))

        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        del generated_ids, trimmed_ids

    latency_ms = sum(latency_samples_ms) / len(latency_samples_ms)
    peak_vram_mb = max(peak_samples_mb)
    del inputs, image
    torch.cuda.empty_cache()
    return {
        "id": scene["id"],
        "name": scene["name"],
        "image": str(image_path),
        "prompt": scene["prompt"],
        "answer": answer,
        "latency_ms": round(latency_ms, 1),
        "latency_samples_ms": [round(value, 1) for value in latency_samples_ms],
        "peak_vram_mb": peak_vram_mb,
    }


# 加载模型并完成五个场景的基准测试。
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    missing = [
        str(IMAGE_DIR / item["image"])
        for item in SCENES
        if not (IMAGE_DIR / item["image"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing scene images: {missing}")

    print("Loading Ostrakon-VL-8B in BF16...", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model.eval()

    load_vram_mb = gpu_mb(torch.cuda.memory_allocated())
    print(
        f"Model loaded | GPU={torch.cuda.get_device_name(0)} | load VRAM={load_vram_mb} MB"
    )

    results = []
    for index, scene in enumerate(SCENES, start=1):
        print(f"[{index}/5] {scene['name']}...", flush=True)
        result = run_scene(model, processor, scene)
        results.append(result)
        compact_answer = " ".join(result["answer"].split())
        print(
            f"  avg latency={result['latency_ms']} ms | peak VRAM={result['peak_vram_mb']} MB"
        )
        print(f"  samples={result['latency_samples_ms']} ms")
        print(f"  answer={compact_answer}", flush=True)

    average_latency_ms = round(
        sum(item["latency_ms"] for item in results) / len(results), 1
    )
    maximum_peak_vram_mb = max(item["peak_vram_mb"] for item in results)
    report = {
        "model": str(MODEL_PATH),
        "dtype": "torch.bfloat16",
        "device_map": "auto",
        "gpu": torch.cuda.get_device_name(0),
        "image_size": "512x512",
        "max_new_tokens": 160,
        "repeats_per_scene": REPEATS,
        "load_vram_mb": load_vram_mb,
        "average_latency_ms": average_latency_ms,
        "maximum_peak_vram_mb": maximum_peak_vram_mb,
        "scenes": results,
    }
    RESULT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "Ostrakon-VL-8B Task 0 — Five-Scene BF16 Benchmark",
        f"GPU: {report['gpu']}",
        "dtype: torch.bfloat16 | device_map: auto | input: 512x512",
        f"measurement: {REPEATS} timed runs per scene (model.generate only)",
        "",
    ]
    for index, item in enumerate(results, start=1):
        answer = " ".join(item["answer"].split())
        lines.extend(
            [
                f"[{index}] {item['name']}",
                f"avg latency={item['latency_ms']} ms | peak VRAM={item['peak_vram_mb']} MB",
                f"latency samples={item['latency_samples_ms']} ms",
                f"answer={answer}",
                "",
            ]
        )
    lines.extend(
        [
            f"Average latency: {average_latency_ms} ms",
            f"Maximum peak VRAM: {maximum_peak_vram_mb} MB",
            f"JSON result: {RESULT_JSON}",
        ]
    )
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 72)
    print(f"Average latency: {average_latency_ms} ms")
    print(f"Maximum peak VRAM: {maximum_peak_vram_mb} MB")
    print(f"Saved JSON: {RESULT_JSON}")
    print(f"Saved summary: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
