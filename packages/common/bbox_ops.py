"""Single source of truth for bbox math used across enrichment, training and eval.

Two conventions are supported:

- COCO ``xywh``: ``(x, y, w, h)`` — top-left corner + width/height (non-negative).
- ``xyxy``:      ``(x1, y1, x2, y2)`` — top-left + bottom-right corners.

All functions are pure Python. No numpy dependency; OK to use inside tight inner
loops because each call is a couple of multiplications. Tests in
``tests/unit/test_bbox_ops.py`` cover boundary cases.
"""
from __future__ import annotations

from typing import Optional, Sequence

BBoxXYXY = tuple[float, float, float, float]
BBoxCOCO = tuple[float, float, float, float]


def coco_xywh_to_xyxy(box: Sequence[float]) -> BBoxXYXY:
    x, y, w, h = box
    return (float(x), float(y), float(x) + float(w), float(y) + float(h))


def xyxy_to_coco_xywh(box: Sequence[float]) -> BBoxCOCO:
    x1, y1, x2, y2 = box
    return (float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1))


def clip_xyxy(box: Sequence[float], width: float, height: float) -> BBoxXYXY:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(float(x1), float(width))),
        max(0.0, min(float(y1), float(height))),
        max(0.0, min(float(x2), float(width))),
        max(0.0, min(float(y2), float(height))),
    )


def area_xyxy(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def resize_xyxy(box: Sequence[float], scale_x: float, scale_y: float) -> BBoxXYXY:
    x1, y1, x2, y2 = box
    return (float(x1) * scale_x, float(y1) * scale_y,
            float(x2) * scale_x, float(y2) * scale_y)


def translate_xyxy(box: Sequence[float], dx: float, dy: float) -> BBoxXYXY:
    x1, y1, x2, y2 = box
    return (float(x1) + dx, float(y1) + dy, float(x2) + dx, float(y2) + dy)


def intersect_xyxy(a: Sequence[float], b: Sequence[float]) -> Optional[BBoxXYXY]:
    """Return intersection rectangle or None if boxes don't overlap (or only touch)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(float(ax1), float(bx1))
    y1 = max(float(ay1), float(by1))
    x2 = min(float(ax2), float(bx2))
    y2 = min(float(ay2), float(by2))
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    inter = intersect_xyxy(a, b)
    if inter is None:
        return 0.0
    inter_area = area_xyxy(inter)
    union = area_xyxy(a) + area_xyxy(b) - inter_area
    return inter_area / union if union > 0 else 0.0


def visible_ratio(reference_box: Sequence[float], visible_box: Sequence[float]) -> float:
    """Fraction of ``reference_box`` covered by ``visible_box`` (clipped/intersected version).

    Returns 0.0 if the reference has zero area. Clamped to [0.0, 1.0].
    """
    ref_area = area_xyxy(reference_box)
    if ref_area <= 0:
        return 0.0
    return min(1.0, area_xyxy(visible_box) / ref_area)


def is_valid_box(box: Sequence[float], min_area: float = 1.0) -> bool:
    return area_xyxy(box) >= min_area


def project_xywh_to_tile(
    source_xywh: Sequence[float],
    *,
    resize_scale: tuple[float, float],
    tile_offset: tuple[float, float],
    tile_size: int,
    visible_threshold: float = 0.80,
    min_area: float = 1.0,
) -> Optional[tuple[BBoxCOCO, float]]:
    """Project a COCO bbox from a source image onto a single crop tile.

    Order of operations:

    1. COCO ``xywh`` → ``xyxy`` in source coords.
    2. Apply ``resize_scale=(sx, sy)`` to get coords in the *scaled* image.
    3. Intersect with the tile rectangle (also in scaled image coords).
    4. ``visible_ratio = area(intersection) / area(resized_box)``.
       If below ``visible_threshold`` → drop (return None).
    5. Translate intersection into tile-local coords (subtract ``tile_offset``).
    6. Clip to ``[0, tile_size]``.
    7. Convert back to COCO ``xywh``.
    8. Drop if final area < ``min_area``.

    Returns ``(new_box_xywh, visible_ratio)`` on keep, ``None`` on drop.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    sx, sy = resize_scale
    ox, oy = tile_offset

    box_resized = resize_xyxy(coco_xywh_to_xyxy(source_xywh), sx, sy)
    tile_xyxy = (float(ox), float(oy),
                 float(ox) + float(tile_size), float(oy) + float(tile_size))
    inter = intersect_xyxy(box_resized, tile_xyxy)
    if inter is None:
        return None
    ratio = visible_ratio(box_resized, inter)
    if ratio < visible_threshold:
        return None

    in_tile = translate_xyxy(inter, -ox, -oy)
    clipped = clip_xyxy(in_tile, float(tile_size), float(tile_size))
    if not is_valid_box(clipped, min_area=min_area):
        return None

    return xyxy_to_coco_xywh(clipped), ratio


def from_yolo_normalized(
    yolo_box: Sequence[float], image_w: int, image_h: int,
) -> BBoxXYXY:
    """YOLO ``(cx, cy, w, h)`` normalized to ``[0, 1]`` → absolute ``xyxy``."""
    cxn, cyn, wn, hn = yolo_box
    w = float(wn) * image_w
    h = float(hn) * image_h
    cx = float(cxn) * image_w
    cy = float(cyn) * image_h
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def to_yolo_normalized(
    box_xyxy: Sequence[float], image_w: int, image_h: int,
) -> tuple[float, float, float, float]:
    """Absolute ``xyxy`` → YOLO ``(cx, cy, w, h)`` normalized."""
    x1, y1, x2, y2 = box_xyxy
    cx = (float(x1) + float(x2)) / 2.0
    cy = (float(y1) + float(y2)) / 2.0
    w = float(x2) - float(x1)
    h = float(y2) - float(y1)
    return cx / image_w, cy / image_h, w / image_w, h / image_h
