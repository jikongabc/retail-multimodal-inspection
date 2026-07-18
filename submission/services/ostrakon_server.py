# Ostrakon-VL-8B 的轻量 OpenAI 兼容推理服务。

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DEFAULT_MODEL = "Ostrakon/Ostrakon-VL-8B"
MAX_REQUEST_BYTES = 25 * 1024 * 1024


# 将 OpenAI 图片数据 URL 解码为 RGB 图像。
def decode_data_image(url: str):
    if not url.startswith("data:image/") or "," not in url:
        raise ValueError("only base64 image data URLs are supported")
    header, encoded = url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("image data URL must use base64 encoding")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("invalid base64 image data") from exc
    from PIL import Image

    return Image.open(io.BytesIO(raw)).convert("RGB")


# 将 OpenAI 消息转换为 Qwen3-VL 消息格式。
def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    image_count = 0
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise ValueError("message content must be a string or a list")
        parts: list[dict[str, Any]] = []
        for part in content:
            part_type = part.get("type")
            if part_type == "text":
                parts.append({"type": "text", "text": str(part.get("text", ""))})
            elif part_type == "image_url":
                image_url = part.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                parts.append({"type": "image", "image": decode_data_image(url)})
                image_count += 1
            else:
                raise ValueError(f"unsupported content type: {part_type}")
        normalized.append({"role": role, "content": parts})
    if image_count == 0:
        raise ValueError("at least one image is required")
    return normalized


# 在单卡 GPU 上加载并串行执行 Ostrakon-VL-8B。
class OstrakonRuntime:
    # 加载模型、处理器和 CUDA 运行时。
    def __init__(self, model_path: str, model_id: str, max_image_size: int) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Ostrakon real mode")
        self.torch = torch
        self.model_id = model_id
        self.max_image_size = max_image_size
        self.lock = threading.Lock()
        torch.backends.cuda.matmul.allow_tf32 = True
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()
        self.loaded_vram_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)

    # 执行一次多模态生成并返回文本和运行指标。
    def generate(
        self, messages: list[dict[str, Any]], max_new_tokens: int
    ) -> tuple[str, dict[str, float]]:
        normalized = normalize_messages(messages)
        for message in normalized:
            if not isinstance(message["content"], list):
                continue
            for part in message["content"]:
                if part["type"] == "image":
                    part["image"].thumbnail((self.max_image_size, self.max_image_size))
        with self.lock:
            inputs = self.processor.apply_chat_template(
                normalized,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            self.torch.cuda.reset_peak_memory_stats()
            self.torch.cuda.synchronize()
            started = time.perf_counter()
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            self.torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000
            trimmed = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated)
            ]
            text = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            peak_vram_mb = self.torch.cuda.max_memory_allocated() / 1024**2
        return text, {
            "latency_ms": round(latency_ms, 3),
            "peak_vram_mb": round(peak_vram_mb, 1),
        }


# 提供健康检查、模型列表和对话补全接口。
class OstrakonRequestHandler(BaseHTTPRequestHandler):
    runtime: OstrakonRuntime
    api_key = ""

    # 输出 JSON 响应。
    def _send_json(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(encoded)

    # 校验可选的 Bearer Token。
    def _is_authorized(self) -> bool:
        if not self.api_key:
            return True
        return self.headers.get("Authorization") == f"Bearer {self.api_key}"

    # 处理健康检查和模型列表请求。
    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model": self.runtime.model_id,
                    "device": self.runtime.torch.cuda.get_device_name(0),
                    "loaded_vram_mb": self.runtime.loaded_vram_mb,
                },
            )
            return
        if self.path == "/v1/models":
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.runtime.model_id,
                            "object": "model",
                            "owned_by": "Ostrakon",
                        }
                    ],
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    # 处理 OpenAI 兼容的多模态对话请求。
    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self._is_authorized():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("invalid request body size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
            max_new_tokens = int(payload.get("max_tokens", 256))
            max_new_tokens = max(1, min(max_new_tokens, 512))
            answer, metrics = self.runtime.generate(messages, max_new_tokens)
            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:20]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.runtime.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
            self._send_json(
                HTTPStatus.OK,
                response,
                {
                    "X-Inference-Latency-Ms": str(metrics["latency_ms"]),
                    "X-Peak-Vram-Mb": str(metrics["peak_vram_mb"]),
                },
            )
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(exc).__name__}: {exc}"},
            )

    # 使用标准错误输出记录访问日志。
    def log_message(self, format: str, *args: Any) -> None:
        super().log_message(format, *args)


# 启动仅监听指定地址的真实模型服务。
def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Ostrakon through an OpenAI API")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-image-size", type=int, default=768)
    args = parser.parse_args()
    runtime = OstrakonRuntime(args.model_path, args.model_id, args.max_image_size)
    OstrakonRequestHandler.runtime = runtime
    OstrakonRequestHandler.api_key = os.getenv("OSTRAKON_API_KEY", "")
    server = ThreadingHTTPServer((args.host, args.port), OstrakonRequestHandler)
    print(
        json.dumps(
            {
                "status": "ready",
                "host": args.host,
                "port": args.port,
                "model": args.model_id,
                "loaded_vram_mb": runtime.loaded_vram_mb,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
