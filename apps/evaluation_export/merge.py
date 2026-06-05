"""Postprocessing + fragment fusion for stitched detections.

Migrated from ``yolox_eval/src/{postprocessing,box_fusion}.py`` (Phase 0 map),
combined into one module because ``FusionConfig`` and ``Detection`` are tightly
coupled and there's no public API value to splitting them.

Stage 1 — per-class NMS / Soft-NMS / WBF (``apply_postprocessing``).
Stage 2 — fragment fusion for long objects that span multiple tiles
          (``merge_fragmented_boxes``). Stock NMS doesn't merge weakly-overlapping
          fragments; we use a union-find graph with IoU/IoS/gap-distance triggers
          and a shape-limits roll-back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from packages.common.bbox_ops import iou_xyxy as _iou_pair

BBox = Tuple[float, float, float, float]


# ---------------------------------------------------------------------- #
# Detection dataclass
# ---------------------------------------------------------------------- #

@dataclass
class Detection:
    """A single detection in **source-image coordinates** (post-stitch)."""

    class_id: int
    confidence: float
    bbox: BBox  # xyxy in source coords
    class_name: Optional[str] = None
    scale_level: int = -1
    crop_id: int = -1
    source: str = "tile"  # tile | merged | wbf
    original_size: Optional[Tuple[int, int]] = None
    scale_size: Optional[Tuple[int, int]] = None
    crop_offset: Optional[Tuple[int, int]] = None
    inference_time: float = 0.0
    merged_from: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": [float(v) for v in self.bbox],
            "scale_level": int(self.scale_level),
            "crop_id": int(self.crop_id),
            "source": self.source,
            "original_size": list(self.original_size) if self.original_size else None,
            "scale_size": list(self.scale_size) if self.scale_size else None,
            "crop_offset": list(self.crop_offset) if self.crop_offset else None,
            "inference_time": float(self.inference_time),
            "merged_from": list(self.merged_from),
        }


# ---------------------------------------------------------------------- #
# numpy bbox helpers (batch-friendly, kept inside the eval pipeline)
# ---------------------------------------------------------------------- #

def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pair-wise IoU for ``(N,4)`` and ``(M,4)`` xyxy arrays."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
    a = boxes_a.astype(np.float32)
    b = boxes_b.astype(np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0],   b[:, 1],   b[:, 2],   b[:, 3]
    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _box_area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_wh(box: BBox) -> Tuple[float, float]:
    return box[2] - box[0], box[3] - box[1]


def _aspect_ratio(box: BBox) -> float:
    w, h = _box_wh(box)
    if w <= 0 or h <= 0:
        return float("inf")
    return max(w / h, h / w)


def _ios(a: BBox, b: BBox) -> float:
    """Intersection over the smaller box."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    smaller = min(_box_area(a), _box_area(b))
    return float(inter / smaller) if smaller > 0 else 0.0


def _gap_distance(a: BBox, b: BBox) -> float:
    """Shortest distance between the two rectangles (0 if they touch / overlap)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return float((dx * dx + dy * dy) ** 0.5)


def _union_box(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _to_arrays(dets: List[Detection]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not dets:
        return (np.zeros((0, 4), np.float32),
                np.zeros((0,),    np.float32),
                np.zeros((0,),    np.int64))
    boxes   = np.array([d.bbox for d in dets],       dtype=np.float32)
    scores  = np.array([d.confidence for d in dets], dtype=np.float32)
    classes = np.array([d.class_id for d in dets],   dtype=np.int64)
    return boxes, scores, classes


# ---------------------------------------------------------------------- #
# Stage 1 — postprocessing (NMS / Soft-NMS / WBF)
# ---------------------------------------------------------------------- #

def nms_per_class(
    detections: List[Detection],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    max_detections: Optional[int] = None,
) -> List[Detection]:
    """Classical greedy NMS, per class."""
    if not detections:
        return []
    by_class: Dict[int, List[Detection]] = {}
    for d in detections:
        if d.confidence < score_threshold:
            continue
        by_class.setdefault(d.class_id, []).append(d)

    out: List[Detection] = []
    for dets in by_class.values():
        boxes, scores, _ = _to_arrays(dets)
        keep = _nms_indices(boxes, scores, iou_threshold)
        for i in keep:
            out.append(dets[i])
    out.sort(key=lambda d: d.confidence, reverse=True)
    if max_detections is not None and len(out) > max_detections:
        out = out[:max_detections]
    return out


def _nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    if boxes.size == 0:
        return []
    order = scores.argsort()[::-1].tolist()
    keep: List[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        if not order:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order]).ravel()
        order = [j for j, iou in zip(order, ious) if iou <= iou_threshold]
    return keep


def soft_nms_per_class(
    detections: List[Detection],
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    max_detections: Optional[int] = None,
) -> List[Detection]:
    """Gaussian Soft-NMS, per class. Decays neighbours' confidence by exp(-iou²/σ)."""
    if not detections:
        return []
    by_class: Dict[int, List[Detection]] = {}
    for d in detections:
        by_class.setdefault(d.class_id, []).append(d)

    out: List[Detection] = []
    for dets in by_class.values():
        boxes, scores, _ = _to_arrays(dets)
        scores = scores.copy()
        alive = list(range(len(dets)))
        result_idx: List[Tuple[int, float]] = []
        while alive:
            best_pos = int(np.argmax(scores[alive]))
            best = alive.pop(best_pos)
            result_idx.append((best, float(scores[best])))
            if not alive:
                break
            ious = iou_matrix(boxes[best:best + 1], boxes[alive]).ravel()
            decay = np.exp(-(ious ** 2) / max(sigma, 1e-6))
            decay = np.where(ious > iou_threshold, decay, 1.0)
            scores[alive] = scores[alive] * decay
        for i, s in result_idx:
            if s < score_threshold:
                continue
            out.append(_clone_with_score(dets[i], s))
    out.sort(key=lambda d: d.confidence, reverse=True)
    if max_detections is not None and len(out) > max_detections:
        out = out[:max_detections]
    return out


def weighted_boxes_fusion_per_class(
    detections: List[Detection],
    iou_threshold: float = 0.55,
    score_threshold: float = 0.0,
    max_detections: Optional[int] = None,
) -> List[Detection]:
    """Simplified WBF: per-class clustering by IoU, weighted-mean box, mean score."""
    if not detections:
        return []
    by_class: Dict[int, List[Detection]] = {}
    for d in detections:
        if d.confidence < score_threshold:
            continue
        by_class.setdefault(d.class_id, []).append(d)

    out: List[Detection] = []
    for cls, dets in by_class.items():
        dets_sorted = sorted(dets, key=lambda d: d.confidence, reverse=True)
        clusters: List[List[int]] = []
        cluster_boxes: List[np.ndarray] = []
        for i, d in enumerate(dets_sorted):
            box = np.array(d.bbox, dtype=np.float32)
            placed = False
            for ci, cb in enumerate(cluster_boxes):
                iou = float(iou_matrix(box[None], cb[None]).ravel()[0])
                if iou >= iou_threshold:
                    clusters[ci].append(i)
                    cluster_boxes[ci] = _wbf_weighted_box(
                        [dets_sorted[k] for k in clusters[ci]]
                    )
                    placed = True
                    break
            if not placed:
                clusters.append([i])
                cluster_boxes.append(box)

        for cl, cb in zip(clusters, cluster_boxes):
            members = [dets_sorted[k] for k in cl]
            out.append(Detection(
                class_id=cls,
                confidence=float(np.mean([m.confidence for m in members])),
                bbox=(float(cb[0]), float(cb[1]), float(cb[2]), float(cb[3])),
                class_name=members[0].class_name,
                scale_level=-1, crop_id=-1,
                source="wbf",
                merged_from=[id(m) for m in members],
            ))

    out.sort(key=lambda d: d.confidence, reverse=True)
    if max_detections is not None and len(out) > max_detections:
        out = out[:max_detections]
    return out


def _wbf_weighted_box(members: List[Detection]) -> np.ndarray:
    weights = np.array([m.confidence for m in members], dtype=np.float32)
    boxes = np.array([m.bbox for m in members], dtype=np.float32)
    w_sum = float(weights.sum())
    if w_sum <= 0:
        return boxes.mean(axis=0)
    return (boxes * weights[:, None]).sum(axis=0) / w_sum


def _clone_with_score(d: Detection, new_score: float) -> Detection:
    return Detection(
        class_id=d.class_id, confidence=new_score, bbox=tuple(d.bbox),
        class_name=d.class_name, scale_level=d.scale_level, crop_id=d.crop_id,
        source=d.source, original_size=d.original_size, scale_size=d.scale_size,
        crop_offset=d.crop_offset, inference_time=d.inference_time,
        merged_from=list(d.merged_from),
    )


def apply_postprocessing(
    detections: List[Detection],
    method: str = "nms",
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    sigma: float = 0.5,
    max_detections: Optional[int] = None,
) -> List[Detection]:
    method = method.lower()
    if method in ("nms", "class_aware_nms"):
        return nms_per_class(detections, iou_threshold, score_threshold, max_detections)
    if method == "soft_nms":
        return soft_nms_per_class(detections, iou_threshold, sigma, score_threshold, max_detections)
    if method == "wbf":
        return weighted_boxes_fusion_per_class(detections, iou_threshold, score_threshold, max_detections)
    raise ValueError(f"Unknown postprocessing method: {method!r}")


# ---------------------------------------------------------------------- #
# Stage 2 — fragment fusion for long objects
# ---------------------------------------------------------------------- #

@dataclass
class FusionConfig:
    enabled: bool = True
    per_class: bool = True
    max_distance_px: float = 50.0
    min_iou: float = 0.05
    min_ios: float = 0.3
    require_same_scale_level: bool = False
    require_adjacent_crops: bool = False
    fusion_strategy: str = "union"  # union | weighted
    max_aspect_ratio: float = 30.0
    max_area_ratio: float = 50.0
    disabled_classes: List[int] = field(default_factory=list)


@dataclass
class FusionStats:
    before_nms: int = 0
    after_nms: int = 0
    after_merge: int = 0
    merged_groups: List[List[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "before_nms": self.before_nms,
            "after_nms":  self.after_nms,
            "after_merge": self.after_merge,
            "merged_groups": self.merged_groups,
        }


def merge_fragmented_boxes(
    detections: List[Detection], config: FusionConfig,
) -> Tuple[List[Detection], FusionStats]:
    """Union-find clustering of weakly-overlapping same-class boxes.

    Each edge fires if IoU≥min_iou OR IoS≥min_ios OR gap≤max_distance_px.
    The fused box must pass shape limits (aspect ratio + area ratio) or the
    merge is rolled back for that component.
    """
    stats = FusionStats(after_nms=len(detections))
    if not config.enabled or not detections:
        stats.after_merge = len(detections)
        return list(detections), stats

    by_class: Dict[int, List[int]] = {}
    for i, d in enumerate(detections):
        by_class.setdefault(d.class_id, []).append(i)

    out: List[Detection] = []
    groups_global: List[List[int]] = []

    for cls, indices in by_class.items():
        if cls in config.disabled_classes:
            for i in indices:
                out.append(detections[i])
                groups_global.append([i])
            continue

        parent = list(range(len(indices)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                if _should_merge(detections[indices[a]], detections[indices[b]], config):
                    union(a, b)

        comps: Dict[int, List[int]] = {}
        for k in range(len(indices)):
            comps.setdefault(find(k), []).append(k)

        for comp in comps.values():
            global_idx = [indices[k] for k in comp]
            members = [detections[i] for i in global_idx]
            if len(members) == 1:
                out.append(members[0])
                groups_global.append(global_idx)
                continue
            merged = _merge_detections(members, config)
            if not _passes_shape_limits(merged, members, config):
                # roll back
                for i, m in zip(global_idx, members):
                    out.append(m)
                    groups_global.append([i])
            else:
                merged.merged_from = global_idx
                out.append(merged)
                groups_global.append(global_idx)

    stats.after_merge = len(out)
    stats.merged_groups = [g for g in groups_global if len(g) > 1]
    out.sort(key=lambda d: d.confidence, reverse=True)
    return out, stats


def _should_merge(a: Detection, b: Detection, cfg: FusionConfig) -> bool:
    if a.class_id != b.class_id:
        return False
    if cfg.require_same_scale_level and a.scale_level != b.scale_level:
        return False
    if cfg.require_adjacent_crops and abs(a.crop_id - b.crop_id) > 1:
        return False

    iou_v = _iou_pair(a.bbox, b.bbox)
    ios_v = _ios(a.bbox, b.bbox)
    gap_v = _gap_distance(a.bbox, b.bbox)

    iou_trigger   = iou_v >= cfg.min_iou
    ios_trigger   = ios_v >= cfg.min_ios
    touch_trigger = gap_v <= cfg.max_distance_px
    if not (iou_trigger or ios_trigger or touch_trigger):
        return False
    # if far apart but neither IoU nor IoS reaches threshold — refuse.
    if gap_v > cfg.max_distance_px and not (iou_trigger or ios_trigger):
        return False
    return True


def _merge_detections(members: List[Detection], cfg: FusionConfig) -> Detection:
    if cfg.fusion_strategy == "union":
        box = members[0].bbox
        for m in members[1:]:
            box = _union_box(box, m.bbox)
        score = max(m.confidence for m in members)
    elif cfg.fusion_strategy == "weighted":
        weights = np.array([m.confidence for m in members], dtype=np.float32)
        boxes = np.array([m.bbox for m in members], dtype=np.float32)
        w_sum = float(weights.sum())
        if w_sum <= 0:
            box = tuple(boxes.mean(axis=0).tolist())
        else:
            box = tuple(((boxes * weights[:, None]).sum(axis=0) / w_sum).tolist())
        score = float(np.mean(weights))
    else:
        raise ValueError(f"Unknown fusion_strategy: {cfg.fusion_strategy!r}")

    first = members[0]
    return Detection(
        class_id=first.class_id,
        confidence=float(score),
        bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
        class_name=first.class_name,
        scale_level=first.scale_level, crop_id=first.crop_id,
        source="merged",
        original_size=first.original_size, scale_size=first.scale_size,
        crop_offset=first.crop_offset,
        inference_time=sum(m.inference_time for m in members),
        merged_from=[],
    )


def _passes_shape_limits(merged: Detection, members: List[Detection], cfg: FusionConfig) -> bool:
    if _aspect_ratio(merged.bbox) > cfg.max_aspect_ratio:
        return False
    sum_area = sum(_box_area(m.bbox) for m in members)
    if sum_area <= 0:
        return False
    if _box_area(merged.bbox) / sum_area > cfg.max_area_ratio:
        return False
    return True
