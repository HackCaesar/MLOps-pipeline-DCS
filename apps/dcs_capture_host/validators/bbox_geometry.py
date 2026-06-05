"""Pure 2D bbox geometry helpers used by the visible-bbox refiner.

Bbox format throughout the project: ``[x_min, y_min, x_max, y_max]`` in
image pixels (``y`` grows downward). All helpers in this module are
side-effect free; they receive plain sequences and return plain Python
values so they can be unit-tested in isolation.

``clip_bbox`` is the workhorse for ROI bookkeeping inside the refiner — it
clips a bbox to image bounds *minus* an ignored bottom strip (the DCS HUD
overlay area), and returns ``None`` when the clipped bbox would be empty.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def clip_bbox(
    bbox: Sequence[float],
    width_px: int,
    height_px: int,
    ignore_bottom_px: int,
) -> Optional[List[float]]:
    """Clip ``bbox`` to ``[0, width_px] × [0, usable_height]`` where
    ``usable_height = max(1, height_px - max(0, ignore_bottom_px))``.

    Returns ``None`` when the result has zero or negative area.
    """
    usable_height = max(1, height_px - max(0, ignore_bottom_px))
    x_min, y_min, x_max, y_max = bbox
    clipped = [
        max(0.0, min(float(width_px), float(x_min))),
        max(0.0, min(float(usable_height), float(y_min))),
        max(0.0, min(float(width_px), float(x_max))),
        max(0.0, min(float(usable_height), float(y_max))),
    ]
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        return None
    return clipped


def bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def bbox_overlap_area(a: Sequence[float], b: Sequence[float]) -> float:
    return bbox_area(
        [
            max(float(a[0]), float(b[0])),
            max(float(a[1]), float(b[1])),
            min(float(a[2]), float(b[2])),
            min(float(a[3]), float(b[3])),
        ]
    )


def bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5)
