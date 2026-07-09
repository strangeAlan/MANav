from dataclasses import dataclass, field
import logging
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from configs.categories import name2index


@dataclass
class SparseObjectEntity:
    entity_id: str
    label: str
    first_seen_step: int
    last_seen_step: int
    observation_count: int = 1
    confidence: float = 0.0
    bearing_deg_history: List[float] = field(default_factory=list)
    distance_m_history: List[float] = field(default_factory=list)
    source: str = "detectron2"
    status: str = "candidate"

    @property
    def bearing_deg(self) -> Optional[float]:
        if not self.bearing_deg_history:
            return None
        return float(np.median(np.asarray(self.bearing_deg_history, dtype=np.float32)))

    @property
    def distance_m(self) -> Optional[float]:
        if not self.distance_m_history:
            return None
        return float(np.median(np.asarray(self.distance_m_history, dtype=np.float32)))

    def to_prompt_item(self) -> str:
        bearing = self.bearing_deg
        distance = self.distance_m
        bearing_text = "unknown bearing" if bearing is None else "{:.0f}deg".format(bearing)
        distance_text = "unknown distance" if distance is None else "{:.1f}m".format(distance)
        return "{} {} {}, seen {} times, status {}".format(
            self.label, bearing_text, distance_text, self.observation_count, self.status
        )


class SparseSceneMemory:
    """Object-centric side memory used to bias frontier exploration.

    This module intentionally does not build a dense occupancy map. It keeps
    stable object observations, recent positions, and a small set of labels the
    VLM/LLM considers useful for the current goal.
    """

    def __init__(self, args, llm=None):
        self.args = args
        self.llm = llm
        self.index2name = {v: k for k, v in name2index.items()}
        self.entities: Dict[str, SparseObjectEntity] = {}
        self.recent_positions: List[Tuple[int, int, int]] = []
        self.next_id = 0
        self.goal_name = ""
        self.vlm_preferred_labels: List[str] = []
        self.last_vlm_step = -1

    def reset(self, goal_name: str = ""):
        self.entities.clear()
        self.recent_positions.clear()
        self.next_id = 0
        self.goal_name = goal_name or ""
        self.vlm_preferred_labels = self._default_related_labels(self.goal_name)
        self.last_vlm_step = -1

    def update_pose(self, step: int, full_pose, map_resolution: float):
        if full_pose is None:
            return
        pose = np.asarray(full_pose, dtype=np.float32).reshape(-1)
        if len(pose) < 2:
            return
        row = int(pose[1] * 100.0 / float(map_resolution))
        col = int(pose[0] * 100.0 / float(map_resolution))
        self.recent_positions.append((step, row, col))
        max_positions = int(getattr(self.args, "sparse_memory_recent_positions", 80))
        if len(self.recent_positions) > max_positions:
            self.recent_positions = self.recent_positions[-max_positions:]

    def update_observation(
        self,
        step: int,
        rgb: np.ndarray,
        detections: Sequence,
        depth_m: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
    ):
        if rgb is None or detections is None:
            return
        h, w = rgb.shape[:2]
        for det in detections:
            label_idx = int(det[0])
            label = self.index2name.get(label_idx, str(label_idx))
            score = float(det[1]) if len(det) > 1 else 0.0
            bbox = np.asarray(det[2], dtype=np.float32)
            if bbox.size != 4:
                continue
            x1, y1, x2, y2 = bbox
            area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
            min_area = float(getattr(self.args, "sparse_memory_min_bbox_area", 600.0))
            if area < min_area:
                continue

            center_x = 0.5 * float(x1 + x2)
            bearing = (center_x - w / 2.0) / max(w / 2.0, 1.0)
            bearing *= float(getattr(self.args, "hfov", 79)) / 2.0
            distance = self._estimate_bbox_depth(depth_m, confidence, bbox, (h, w))
            self._merge_or_insert(step, label, score, bearing, distance)

    def maybe_update_vlm_hint(self, step: int):
        if self.llm is None:
            return
        if not getattr(self.args, "sparse_memory_use_vlm_hint", False):
            return
        interval = int(getattr(self.args, "sparse_memory_vlm_hint_interval", 80))
        if self.last_vlm_step >= 0 and step - self.last_vlm_step < interval:
            return
        self.last_vlm_step = step
        summary = "; ".join(self.memory_prompt_items()[:12]) or "no stable objects yet"
        known_labels = ", ".join(sorted(set(self.index2name.values())))
        prompt = (
            "We are navigating indoors to find {}. Current sparse observations: {}. "
            "From these object labels [{}], choose up to three labels that are useful "
            "context or nearby clues for exploration. Answer only comma-separated labels."
        ).format(self.goal_name, summary, known_labels)
        try:
            response = self.llm(prompt=prompt)
        except Exception as exc:
            logging.info("[SparseMemoryVLM] failed: %s", exc)
            return
        labels = []
        response_text = str(response).lower()
        for label in self.index2name.values():
            if label.lower() in response_text:
                labels.append(label)
        if labels:
            self.vlm_preferred_labels = labels[:3]
            self._log(step, "[SparseMemoryVLM] preferred_labels={}".format(self.vlm_preferred_labels))

    def choose_frontier(
        self,
        frontier_locations: Optional[np.ndarray],
        current_full_pose,
        map_resolution: float,
        default_goal: Optional[Sequence[float]] = None,
    ) -> Optional[np.ndarray]:
        if frontier_locations is None or len(frontier_locations) == 0:
            return None
        pose = np.asarray(current_full_pose, dtype=np.float32).reshape(-1)
        if len(pose) < 2:
            return None
        cur_row = pose[1] * 100.0 / float(map_resolution)
        cur_col = pose[0] * 100.0 / float(map_resolution)
        frontiers = np.asarray(frontier_locations, dtype=np.float32)
        if frontiers.ndim != 2 or frontiers.shape[1] < 2:
            return None

        scores = np.zeros((len(frontiers),), dtype=np.float32)
        drow = frontiers[:, 0] - cur_row
        dcol = frontiers[:, 1] - cur_col
        dist = np.sqrt(drow ** 2 + dcol ** 2)
        scores += np.clip(dist / 80.0, 0.0, 1.5)

        if default_goal is not None:
            default_goal = np.asarray(default_goal, dtype=np.float32).reshape(-1)
            if len(default_goal) >= 2:
                default_dist = np.sqrt(
                    (frontiers[:, 0] - default_goal[0]) ** 2
                    + (frontiers[:, 1] - default_goal[1]) ** 2
                )
                scores += 1.0 / (1.0 + default_dist / 30.0)

        for _, row, col in self.recent_positions[-30:]:
            recent_dist = np.sqrt((frontiers[:, 0] - row) ** 2 + (frontiers[:, 1] - col) ** 2)
            scores -= np.clip(1.0 - recent_dist / 35.0, 0.0, 1.0) * 0.7

        preferred_bearings = self._preferred_bearings()
        if preferred_bearings:
            frontier_bearings = np.rad2deg(np.arctan2(dcol, np.maximum(drow, 1e-3)))
            for bearing in preferred_bearings:
                diff = np.abs(((frontier_bearings - bearing + 180.0) % 360.0) - 180.0)
                scores += np.clip(1.0 - diff / 70.0, 0.0, 1.0) * 0.8

        idx = int(np.argmax(scores))
        return frontiers[idx].astype(np.int32)

    def memory_prompt_items(self) -> List[str]:
        stable = [entity for entity in self.entities.values() if entity.status == "stable"]
        stable = sorted(stable, key=lambda item: (item.label not in self.vlm_preferred_labels, -item.observation_count))
        return [entity.to_prompt_item() for entity in stable]

    def _estimate_bbox_depth(self, depth_m, confidence, bbox, image_shape) -> Optional[float]:
        if depth_m is None:
            return None
        depth = np.asarray(depth_m, dtype=np.float32)
        h, w = image_shape
        dh, dw = depth.shape[:2]
        x1, y1, x2, y2 = bbox
        yy1 = int(np.clip(y1 * dh / max(h, 1), 0, dh - 1))
        yy2 = int(np.clip(y2 * dh / max(h, 1), yy1 + 1, dh))
        xx1 = int(np.clip(x1 * dw / max(w, 1), 0, dw - 1))
        xx2 = int(np.clip(x2 * dw / max(w, 1), xx1 + 1, dw))
        crop = depth[yy1:yy2, xx1:xx2]
        if confidence is not None:
            conf = np.asarray(confidence)
            conf_crop = conf[yy1:yy2, xx1:xx2]
            threshold = float(getattr(self.args, "sparse_memory_depth_confidence_threshold", 0.0))
            if threshold > 0:
                crop = crop[conf_crop >= threshold]
        crop = crop[np.isfinite(crop)]
        crop = crop[(crop > 0.05) & (crop < float(getattr(self.args, "max_depth", 30.0)))]
        if len(crop) == 0:
            return None
        scale = float(getattr(self.args, "sparse_memory_depth_scale", 1.0))
        return float(np.median(crop)) * scale

    def _merge_or_insert(self, step, label, score, bearing, distance):
        best_entity = None
        best_cost = float("inf")
        for entity in self.entities.values():
            if entity.label != label:
                continue
            entity_bearing = entity.bearing_deg
            if entity_bearing is None:
                continue
            cost = abs(entity_bearing - bearing)
            if distance is not None and entity.distance_m is not None:
                cost += abs(entity.distance_m - distance) * 8.0
            if cost < best_cost:
                best_entity = entity
                best_cost = cost

        merge_thr = float(getattr(self.args, "sparse_memory_merge_bearing_deg", 25.0))
        if best_entity is None or best_cost > merge_thr:
            entity_id = "sparse_{:04d}".format(self.next_id)
            self.next_id += 1
            best_entity = SparseObjectEntity(
                entity_id=entity_id,
                label=label,
                first_seen_step=step,
                last_seen_step=step,
                confidence=score,
            )
            self.entities[entity_id] = best_entity
        else:
            best_entity.observation_count += 1
            best_entity.last_seen_step = step
            best_entity.confidence = max(best_entity.confidence, score)

        best_entity.bearing_deg_history.append(float(bearing))
        if distance is not None:
            best_entity.distance_m_history.append(float(distance))
        max_hist = int(getattr(self.args, "sparse_memory_entity_history", 10))
        best_entity.bearing_deg_history = best_entity.bearing_deg_history[-max_hist:]
        best_entity.distance_m_history = best_entity.distance_m_history[-max_hist:]
        stable_count = int(getattr(self.args, "sparse_memory_stable_count", 2))
        if best_entity.observation_count >= stable_count:
            best_entity.status = "stable"

    def _preferred_bearings(self) -> List[float]:
        labels = set(self.vlm_preferred_labels + self._default_related_labels(self.goal_name))
        bearings = []
        for entity in self.entities.values():
            if entity.status != "stable":
                continue
            if entity.label not in labels:
                continue
            bearing = entity.bearing_deg
            if bearing is not None:
                bearings.append(bearing)
        return bearings

    def _default_related_labels(self, goal_name: str) -> List[str]:
        goal = (goal_name or "").lower()
        related = {
            "chair": ["chair", "table", "sofa"],
            "sofa": ["sofa", "chair", "table"],
            "bed": ["bed", "chair", "table"],
            "plant": ["plant", "vase", "table"],
            "tv_monitor": ["tv_monitor", "sofa", "chair"],
            "table": ["table", "chair", "sofa"],
        }
        return related.get(goal, [goal] if goal else [])

    def _log(self, step: int, message: str):
        if getattr(self.args, "sparse_memory_log", False):
            text = "[SparseMemory] step={} {}".format(step, message)
            print(text)
            logging.info(text)
