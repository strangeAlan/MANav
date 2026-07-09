#!/usr/bin/env python
import argparse
import base64
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF


def _ensure_lingbot_repo(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _preprocess_image(image: Image.Image, image_size: int, patch_size: int) -> torch.Tensor:
    image = image.convert("RGB")
    width, height = image.size
    new_width = image_size
    new_height = round(height * (new_width / width) / patch_size) * patch_size
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    tensor = TF.ToTensor()(image)
    if new_height > image_size:
        start_y = (new_height - image_size) // 2
        tensor = tensor[:, start_y:start_y + image_size, :]
    return tensor


class LingBotDepthRunner:
    def __init__(self, args):
        _ensure_lingbot_repo(args.lingbot_repo)
        from lingbot_map.models.gct_stream import GCTStream

        self.args = args
        self.device = torch.device(args.device)
        self.history = []
        self.input_hw = None
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = GCTStream(
            img_size=args.image_size,
            patch_size=args.patch_size,
            enable_3d_rope=True,
            max_frame_num=args.max_frame_num,
            kv_cache_sliding_window=args.kv_cache_sliding_window,
            kv_cache_scale_frames=args.num_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=args.use_sdpa,
            camera_num_iterations=args.camera_num_iterations,
        )
        ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"[lingbot_depth_server] checkpoint loaded missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        self.model = self.model.to(self.device).eval()

    def reset(self):
        self.history = []
        self.input_hw = None
        if hasattr(self.model, "clean_kv_cache"):
            self.model.clean_kv_cache()

    def predict(self, image: Image.Image, output_height: int, output_width: int):
        self.input_hw = (output_height, output_width)
        tensor = _preprocess_image(image, self.args.image_size, self.args.patch_size)
        self.history.append(tensor)
        if self.history and self.history[0].shape != tensor.shape:
            self.history = [tensor]
        if len(self.history) > self.args.max_history:
            self.history = self.history[-self.args.max_history:]

        images = torch.stack(self.history).to(self.device)
        scale_frames = min(max(1, self.args.num_scale_frames), images.shape[0])
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.dtype, enabled=self.device.type == "cuda"):
            pred = self.model.inference_streaming(
                images,
                num_scale_frames=scale_frames,
                keyframe_interval=self.args.keyframe_interval,
                output_device=torch.device("cpu"),
            )
        depth = pred["depth"][0, -1, ..., 0].float().numpy()
        conf = pred.get("depth_conf")
        if conf is not None:
            conf = conf[0, -1].float().numpy()

        depth = np.asarray(
            Image.fromarray(depth.astype(np.float32), mode="F").resize(
                (output_width, output_height),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )
        if conf is not None:
            conf = np.asarray(
                Image.fromarray(conf.astype(np.float32), mode="F").resize(
                    (output_width, output_height),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            )

        return {
            "depth_b64": _encode_float_array(depth),
            "depth_shape": list(depth.shape),
            "confidence_b64": None if conf is None else _encode_float_array(conf),
            "confidence_shape": None if conf is None else list(conf.shape),
            "source": "lingbot-map-sdpa" if self.args.use_sdpa else "lingbot-map-flashinfer",
            "ready": len(self.history) >= self.args.num_scale_frames,
            "history": len(self.history),
            "elapsed_sec": time.time() - t0,
        }


def _encode_float_array(array: np.ndarray) -> str:
    return base64.b64encode(np.asarray(array, dtype=np.float32).tobytes()).decode("ascii")


def make_handler(runner: LingBotDepthRunner):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code, payload):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "backend": "lingbot-map"})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                if self.path == "/reset":
                    runner.reset()
                    self._send_json(200, {"status": "reset"})
                    return
                if self.path != "/predict":
                    self._send_json(404, {"error": "not found"})
                    return
                raw = base64.b64decode(data["image"])
                image = Image.open(BytesIO(raw))
                height = int(data.get("height") or image.height)
                width = int(data.get("width") or image.width)
                result = runner.predict(image, height, width)
                self._send_json(200, result)
            except Exception as exc:
                traceback.print_exc()
                self._send_json(500, {"error": str(exc)})

        def log_message(self, fmt, *args):
            print(f"[lingbot_depth_server] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18180)
    parser.add_argument("--model-path", default="/home/hsy/model/lingbot-map/lingbot-map.pt")
    parser.add_argument("--lingbot-repo", default="/home/hsy/lingbot-map")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--num-scale-frames", type=int, default=4)
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--max-frame-num", type=int, default=1024)
    parser.add_argument("--kv-cache-sliding-window", type=int, default=64)
    parser.add_argument("--keyframe-interval", type=int, default=1)
    parser.add_argument("--camera-num-iterations", type=int, default=1)
    parser.add_argument("--use-sdpa", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    runner = LingBotDepthRunner(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runner))
    print(f"[lingbot_depth_server] ready http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
