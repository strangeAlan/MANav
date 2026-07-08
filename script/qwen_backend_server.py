#!/usr/bin/env python3
"""Local HTTP server for a Transformers Qwen VLM backend."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

from PIL import Image


RUNNER = None


def _strip_data_url(data_url: str) -> str:
    if "," in data_url:
        return data_url.split(",", 1)[1]
    return data_url


def _strip_thinking(text: str) -> str:
    if not isinstance(text, str):
        return text
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<think>", "").strip()


class LocalQwenTransformers:
    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        log_dir: str = "logs/qwen_backend_server",
    ) -> None:
        self.model_path = model_path
        self.device_map = device_map
        self.dtype = dtype
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(str(self.dtype).lower(), torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model.eval()

    def _prepare_images(self, images):
        result = []
        for item in images or []:
            if isinstance(item, Image.Image):
                result.append(item.convert("RGB"))
            elif isinstance(item, str):
                raw = base64.b64decode(_strip_data_url(item))
                result.append(Image.open(BytesIO(raw)).convert("RGB"))
            else:
                raise TypeError(f"Unsupported image type: {type(item)!r}")
        return result

    def generate(self, prompt: str, images=None, max_new_tokens=None) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        pil_images = self._prepare_images(images)
        content = [{"type": "text", "text": prompt}]
        for image in pil_images:
            content.append({"type": "image", "image": image})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        output = _strip_thinking(output)
        self._log(prompt, len(pil_images), output)
        return output

    def _log(self, prompt: str, num_images: int, output: str) -> None:
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_preview": prompt[:500],
            "num_images": num_images,
            "output": output,
        }
        with open(self.log_dir / "requests.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = payload.get("prompt", "")
            images = payload.get("images", [])
            max_new_tokens = payload.get("max_new_tokens")
            text = RUNNER.generate(prompt, images=images, max_new_tokens=max_new_tokens)
            self._send_json(200, {"text": text})
        except Exception as exc:
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))


def parse_args():
    default_model_path = Path("data/models/Qwen3-VL-8B-Instruct")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=18080, type=int)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("QWEN_MODEL_PATH", str(default_model_path)),
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", default=512, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--log-dir", default="logs/qwen_backend_server")
    return parser.parse_args()


def main():
    global RUNNER
    args = parse_args()
    RUNNER = LocalQwenTransformers(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        log_dir=args.log_dir,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Qwen backend server ready: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
