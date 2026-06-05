"""Debug overlay rendering.

Draws geometry/visible/final bboxes and projected 3D corner points on top of
a screenshot and writes ``<frame_id>_overlay.png``. Used by the orchestrator
right after metadata writing for visual debugging.

OpenCV is optional: if ``cv2`` is not installed the call is a no-op (logs
once and returns), matching the previous behaviour.

Color contract (kept stable for the existing overlays in ``dataset/``):

- yellow  ``(0, 255, 255)``: geometry-projected bbox
- cyan    ``(255, 255, 0)``: visible-pixel refined bbox
- green   ``(0, 255, 0)``:   final export bbox when ``validation.valid``
- orange  ``(0, 165, 255)``: final/reject bbox when ``validation.valid`` is False
- red     ``(0, 0, 255)``:   projected 3D corner points
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None


def render_overlay(
    image_path: Path,
    frame_metadata: Dict[str, Any],
    overlay_path: Path,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Render the debug overlay for ``frame_metadata`` next to ``image_path``.

    Writes the result to ``overlay_path``. Silently no-ops when OpenCV is
    missing or the source image cannot be read.
    """
    logger = logger or logging.getLogger(__name__)
    if cv2 is None:
        logger.info("OpenCV not installed, skipping debug overlay")
        return

    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning("Could not read image for overlay: %s", image_path)
        return

    def draw_bbox(
        label: str,
        bbox: Optional[Sequence[float]],
        color: Tuple[int, int, int],
        thickness: int,
        y_offset: int,
    ) -> None:
        if not bbox:
            return
        x_min, y_min, x_max, y_max = [int(round(value)) for value in bbox]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        cv2.putText(
            image,
            label,
            (x_min, max(20, y_min - y_offset)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    for obj in frame_metadata.get("objects", []):
        projection = obj.get("projection", {})
        is_valid = obj.get("validation", {}).get("valid", False)
        status = obj.get("quality_tier") if is_valid else "reject"
        object_label = f"{obj.get('class', 'unknown')}:{obj.get('id', 'unknown')}:{status}"

        geometry_bbox = projection.get("bbox_xyxy_geometry_px") or projection.get("bbox_xyxy_unclipped_px")
        visible_bbox = projection.get("bbox_xyxy_visible_px")
        final_bbox = projection.get("bbox_xyxy_px")

        draw_bbox("geom", geometry_bbox, (0, 255, 255), 1, 34)
        draw_bbox("visible", visible_bbox, (255, 255, 0), 1, 21)

        final_color = (0, 255, 0) if is_valid else (0, 165, 255)
        final_label = f"visible/final {object_label}" if visible_bbox else f"final {object_label}"
        draw_bbox(final_label, final_bbox, final_color, 2 if is_valid else 1, 8)

        if not final_bbox and geometry_bbox:
            draw_bbox(f"reject {object_label}", geometry_bbox, (0, 165, 255), 1, 8)

        for point in obj.get("projection", {}).get("projected_points_px", []):
            px, py = int(round(point[0])), int(round(point[1]))
            if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                cv2.circle(image, (px, py), 2, (0, 0, 255), -1)

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), image)
