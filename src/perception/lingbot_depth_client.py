import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Optional, Tuple

import numpy as np
import requests
from PIL import Image


@dataclass
class LingBotDepthResult:
    depth_m: np.ndarray
    confidence: Optional[np.ndarray]
    stats: Dict[str, float]
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
        confidence_threshold: float = 0.0,
        invalid_fill_m: Optional[float] = None,
        max_jump_m: float = 0.0,
        temporal_alpha: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.scale = scale
        self.confidence_threshold = confidence_threshold
        self.invalid_fill_m = invalid_fill_m if invalid_fill_m is not None else max_depth_m
        self.max_jump_m = max_jump_m
        self.temporal_alpha = temporal_alpha
        self.step = 0
        self.prev_depth_m: Optional[np.ndarray] = None
        self.last_stats: Dict[str, float] = {}

    def reset(self) -> None:
        self.step = 0
        self.prev_depth_m = None
        self.last_stats = {}
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
        raw_depth = self._decode_array(data, "depth")
        confidence = self._decode_array(data, "confidence") if self._has_array(data, "confidence") else None
        depth, stats = self._postprocess_depth(raw_depth, confidence)
        return LingBotDepthResult(
            depth_m=depth,
            confidence=confidence,
            stats=stats,
            source=data.get("source", "lingbot-map"),
            ready=bool(data.get("ready", True)),
        )

    def metric_depth_to_habitat_obs(self, depth_m: np.ndarray, min_d: float, max_d: float) -> np.ndarray:
        """Encode metric depth as the normalized depth format UniGoal preprocesses."""
        depth_norm = (depth_m - float(min_d)) / max(float(max_d), 1e-6)
        depth_norm = np.clip(depth_norm, 0.0, 0.99).astype(np.float32)
        return depth_norm[..., None]

    def _postprocess_depth(
        self,
        raw_depth: np.ndarray,
        confidence: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        raw_depth = raw_depth.astype(np.float32, copy=False)
        depth = raw_depth * float(self.scale)

        valid = np.isfinite(depth)
        if confidence is not None and self.confidence_threshold > 0:
            valid &= confidence >= float(self.confidence_threshold)

        low_conf_ratio = 0.0
        if confidence is not None:
            low_conf_ratio = float((confidence < float(self.confidence_threshold)).mean()) \
                if self.confidence_threshold > 0 else 0.0

        depth = np.where(valid, depth, float(self.invalid_fill_m))
        depth = np.clip(depth, self.min_depth_m, self.max_depth_m)

        if self.prev_depth_m is not None and self.prev_depth_m.shape == depth.shape:
            if self.max_jump_m > 0:
                jump = np.abs(depth - self.prev_depth_m)
                depth = np.where(jump > float(self.max_jump_m), self.prev_depth_m, depth)
            if 0.0 < self.temporal_alpha < 1.0:
                depth = (
                    float(self.temporal_alpha) * self.prev_depth_m
                    + (1.0 - float(self.temporal_alpha)) * depth
                ).astype(np.float32)

        self.prev_depth_m = depth.copy()

        stats = {
            "raw_min_m": float(np.nanmin(raw_depth)),
            "raw_mean_m": float(np.nanmean(raw_depth)),
            "raw_max_m": float(np.nanmax(raw_depth)),
            "depth_min_m": float(np.nanmin(depth)),
            "depth_mean_m": float(np.nanmean(depth)),
            "depth_max_m": float(np.nanmax(depth)),
            "invalid_ratio": float((~valid).mean()),
            "low_conf_ratio": low_conf_ratio,
        }
        if confidence is not None:
            stats.update({
                "conf_min": float(np.nanmin(confidence)),
                "conf_mean": float(np.nanmean(confidence)),
                "conf_max": float(np.nanmax(confidence)),
            })
        self.last_stats = stats
        return depth.astype(np.float32), stats

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
