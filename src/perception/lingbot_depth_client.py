import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import numpy as np
import requests
from PIL import Image


@dataclass
class LingBotDepthResult:
    depth_m: np.ndarray
    confidence: Optional[np.ndarray]
    source: str
    ready: bool


class LingBotDepthClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        min_depth_m: float = 0.2,
        max_depth_m: float = 30.0,
        scale: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.scale = scale
        self.step = 0

    def reset(self) -> None:
        self.step = 0
        try:
            requests.post(f"{self.base_url}/reset", timeout=min(self.timeout, 10.0))
        except requests.RequestException:
            pass

    def predict(self, rgb: np.ndarray) -> LingBotDepthResult:
        payload = {
            "image": self._encode_rgb(rgb),
            "step": self.step,
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
        }
        self.step += 1
        response = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        depth = self._decode_array(data, "depth") * float(self.scale)
        depth = np.clip(depth, self.min_depth_m, self.max_depth_m)
        confidence = self._decode_array(data, "confidence") if self._has_array(data, "confidence") else None
        return LingBotDepthResult(
            depth_m=depth,
            confidence=confidence,
            source=data.get("source", "lingbot-map"),
            ready=bool(data.get("ready", True)),
        )

    def metric_depth_to_habitat_obs(self, depth_m: np.ndarray, min_d: float, max_d: float) -> np.ndarray:
        """Encode metric depth as the normalized depth format UniGoal preprocesses."""
        depth_norm = (depth_m - float(min_d)) / max(float(max_d), 1e-6)
        depth_norm = np.clip(depth_norm, 0.0, 0.99).astype(np.float32)
        return depth_norm[..., None]

    @staticmethod
    def _encode_rgb(rgb: np.ndarray) -> str:
        image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _has_array(data, key: str) -> bool:
        return key in data or f"{key}_b64" in data

    @staticmethod
    def _decode_array(data, key: str) -> np.ndarray:
        encoded_key = f"{key}_b64"
        shape_key = f"{key}_shape"
        if encoded_key in data:
            raw = base64.b64decode(data[encoded_key])
            array = np.frombuffer(raw, dtype=np.float32)
            return array.reshape(tuple(data[shape_key])).copy()
        return np.asarray(data[key], dtype=np.float32)
