"""Visible-pixel bbox refinement — facade.

The geometry projector produces a safe proposal box, not a tight training
box. This module uses that proposal only as a search region and replaces
it with a box fitted to the visible rendered pixels in the screenshot.

Heavy lifting lives in two mixins that this class combines:

- :class:`validators.airborne_refiner.AirborneRefinerMixin` — planes,
  helicopters.
- :class:`validators.ship_refiner.ShipRefinerMixin` — ships.

The facade keeps the public entry points (``refine_frame``,
``refine_object``), shared bbox / component / debug helpers, and the
final write-back to frame metadata.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency in DCS environment
    cv2 = None

from validators.airborne_refiner import AirborneRefinerMixin
from validators.bbox_geometry import (
    bbox_area as _bbox_area,
)
from validators.bbox_geometry import (
    clip_bbox as _clip_bbox,
)
from validators.refined_bbox import RefinedBBox
from validators.ship_refiner import ShipRefinerMixin

AIRBORNE_CLASSES = {"airplanes", "helicopters"}


class VisibleBBoxRefiner(AirborneRefinerMixin, ShipRefinerMixin):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        refinement_cfg = config.get("bbox_refinement", {})
        self.enabled = bool(refinement_cfg.get("enabled", True))
        self.ignore_bottom_px = int(refinement_cfg.get("ignore_bottom_px", 45))
        self.min_refined_area_px = float(refinement_cfg.get("min_refined_area_px", 16.0))
        self.airborne_padding_px = int(refinement_cfg.get("airborne_padding_px", 2))
        self.ship_padding_px = int(refinement_cfg.get("ship_padding_px", 3))
        self.fallback_to_geometry_on_failure = bool(
            refinement_cfg.get("fallback_to_geometry_on_failure", False)
        )

    def refine_frame(self, image_path: Path, frame_metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return frame_metadata
        if cv2 is None:
            frame_metadata.setdefault("bbox_refinement", {})["status"] = "skipped_opencv_missing"
            return frame_metadata

        image = cv2.imread(str(image_path))  # type: ignore[union-attr]
        if image is None:
            frame_metadata.setdefault("bbox_refinement", {})["status"] = "skipped_image_unreadable"
            return frame_metadata

        refined_count = 0
        failed_count = 0
        fallback_count = 0
        debug_ship_dir = image_path.parent.parent / "debug_refiner" / "ships"
        debug_airborne_dir = image_path.parent.parent / "debug_refiner" / "airborne"
        frame_id = str(frame_metadata.get("frame_id") or image_path.stem)
        for obj in frame_metadata.get("objects", []):
            projection = obj.get("projection", {})
            geometry_bbox = projection.get("bbox_xyxy_geometry_px") or projection.get("bbox_xyxy_px")
            if not geometry_bbox:
                failed_count += 1
                obj["visible_bbox_refinement"] = {
                    "status": "skipped",
                    "reasons": ["missing_geometry_bbox"],
                }
                continue

            projection["bbox_xyxy_geometry_px"] = list(geometry_bbox)

            debug_context = None
            class_name = obj.get("class", "unknown")
            if class_name == "ships":
                debug_context = {
                    "debug_dir": debug_ship_dir,
                    "frame_id": frame_id,
                    "object_id": str(obj.get("id", "unknown")),
                    "type_name": str(obj.get("type_name", "unknown")),
                }
            elif class_name in {"airplanes", "helicopters"}:
                debug_context = {
                    "debug_dir": debug_airborne_dir,
                    "frame_id": frame_id,
                    "object_id": str(obj.get("id", "unknown")),
                    "type_name": str(obj.get("type_name", "unknown")),
                    "class_name": str(class_name),
                    "projected_points_px": projection.get("projected_points_px") or [],
                }

            refined = self.refine_object(image, geometry_bbox, class_name, debug_context)
            if refined is None:
                if self.fallback_to_geometry_on_failure and obj.get("validation", {}).get("valid", False):
                    clipped_geometry_bbox = _clip_bbox(
                        geometry_bbox,
                        image.shape[1],
                        image.shape[0],
                        self.ignore_bottom_px,
                    )
                    if clipped_geometry_bbox is not None:
                        self._apply_geometry_fallback_bbox(obj, clipped_geometry_bbox, debug_context)
                        fallback_count += 1
                        continue

                failed_count += 1
                projection["bbox_xyxy_px"] = None
                projection["bbox_xyxy_visible_px"] = None
                projection["bbox_xywh_px"] = None
                projection["bbox_area_px2"] = 0.0
                projection["center_px"] = None
                failure_refinement = {
                    "status": "failed",
                    "reasons": ["visible_pixels_not_found"],
                }
                if debug_context and debug_context.get("class_name") in {"airplanes", "helicopters"}:
                    failure_refinement.update(debug_context.get("airborne_failure_diagnostics", {}))
                obj["visible_bbox_refinement"] = failure_refinement
                obj["quality_tier"] = "reject"
                obj["usable"] = False
                validation = obj.setdefault("validation", {})
                validation["valid"] = False
                validation["quality_tier"] = "reject"
                validation["checks"] = {
                    "bbox_exists": False,
                    "visible_refinement_pass": False,
                }
                validation["reasons"] = ["visible_bbox_refinement_failed"]
                continue

            self._apply_refined_bbox(obj, refined)
            refined_count += 1

        frame_metadata["bbox_refinement"] = {
            "status": "applied",
            "method": "visible_pixel_segmentation_from_rendered_image",
            "refined_objects": refined_count,
            "fallback_objects": fallback_count,
            "failed_objects": failed_count,
        }
        self._refresh_frame_stats(frame_metadata)
        return frame_metadata

    def refine_object(
        self,
        image: np.ndarray,
        geometry_bbox_xyxy_px: Sequence[float],
        class_name: str,
        debug_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[RefinedBBox]:
        height_px, width_px = image.shape[:2]
        clipped = _clip_bbox(geometry_bbox_xyxy_px, width_px, height_px, self.ignore_bottom_px)
        if clipped is None:
            return None

        if class_name in AIRBORNE_CLASSES:
            return self._refine_airborne(image, clipped, debug_context)
        if class_name == "ships":
            return self._refine_ship(image, clipped, debug_context)
        return self._refine_airborne(image, clipped, None)

    def _roi_for_bbox(
        self,
        image: np.ndarray,
        bbox: Sequence[float],
        margin_px: float,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        height_px, width_px = image.shape[:2]
        x_min, y_min, x_max, y_max = bbox
        x1 = max(0, int(math.floor(x_min - margin_px)))
        y1 = max(0, int(math.floor(y_min - margin_px)))
        x2 = min(width_px, int(math.ceil(x_max + margin_px)))
        y2 = min(max(1, height_px - self.ignore_bottom_px), int(math.ceil(y_max + margin_px)))
        return image[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _components_debug_union_bbox(self, components: Sequence[Dict[str, Any]]) -> Optional[List[float]]:
        if not components:
            return None
        return [
            min(float(component["bbox_xyxy_px"][0]) for component in components),
            min(float(component["bbox_xyxy_px"][1]) for component in components),
            max(float(component["bbox_xyxy_px"][2]) for component in components),
            max(float(component["bbox_xyxy_px"][3]) for component in components),
        ]

    def _limit_component_debug(self, components: Sequence[Dict[str, Any]], limit: int = 80) -> List[Dict[str, Any]]:
        ordered = sorted(components, key=lambda component: int(component.get("area_px", 0)), reverse=True)
        return [dict(component) for component in ordered[:limit]]

    def _safe_debug_token(self, value: Any) -> str:
        token = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))
        return (token[:96] or "unknown").strip("._-") or "unknown"

    def _draw_debug_bbox(self, image: np.ndarray, bbox: Sequence[float], color: Tuple[int, int, int], label: str) -> None:
        x1 = int(round(float(bbox[0])))
        y1 = int(round(float(bbox[1])))
        x2 = int(round(float(bbox[2])))
        y2 = int(round(float(bbox[3])))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # type: ignore[union-attr]
        cv2.putText(  # type: ignore[union-attr]
            image,
            label,
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    def _components_intersecting_bbox(
        self,
        components: Sequence[Tuple[int, int, int, int, int, np.ndarray]],
        roi_box: Tuple[int, int, int, int],
        bbox: Sequence[float],
    ) -> List[Tuple[int, int, int, int, int, np.ndarray]]:
        roi_x1, roi_y1, _roi_x2, _roi_y2 = roi_box
        x_min, y_min, x_max, y_max = bbox
        selected = []
        for component in components:
            x, y, w, h, area, _centroid = component
            abs_x1 = roi_x1 + x
            abs_y1 = roi_y1 + y
            abs_x2 = abs_x1 + w
            abs_y2 = abs_y1 + h
            overlap_x = max(0.0, min(abs_x2, x_max) - max(abs_x1, x_min))
            overlap_y = max(0.0, min(abs_y2, y_max) - max(abs_y1, y_min))
            if overlap_x * overlap_y > 0.0 or area > 500:
                selected.append(component)
        return selected

    def _bbox_from_components(
        self,
        components: Sequence[Tuple[int, int, int, int, int, np.ndarray]],
        roi_box: Tuple[int, int, int, int],
        width_px: int,
        height_px: int,
        padding_px: int,
        method: str,
    ) -> Optional[RefinedBBox]:
        if not components:
            return None
        roi_x1, roi_y1, _roi_x2, _roi_y2 = roi_box
        x1 = min(component[0] for component in components) + roi_x1
        y1 = min(component[1] for component in components) + roi_y1
        x2 = max(component[0] + component[2] for component in components) + roi_x1
        y2 = max(component[1] + component[3] for component in components) + roi_y1
        bbox = self._pad_bbox([x1, y1, x2, y2], width_px, height_px, padding_px)
        if _bbox_area(bbox) < self.min_refined_area_px:
            return None
        return RefinedBBox(
            bbox_xyxy_px=bbox,
            mask_area_px=sum(component[4] for component in components),
            method=method,
            confidence="visible_pixels",
            reasons=[],
        )

    def _pad_bbox(self, bbox: Sequence[float], width_px: int, height_px: int, padding_px: int) -> List[float]:
        padded = [
            float(bbox[0]) - float(padding_px),
            float(bbox[1]) - float(padding_px),
            float(bbox[2]) + float(padding_px),
            float(bbox[3]) + float(padding_px),
        ]
        clipped = _clip_bbox(padded, width_px, height_px, self.ignore_bottom_px)
        if clipped is None:
            return [float(value) for value in bbox]
        return clipped

    def _apply_refined_bbox(self, obj: Dict[str, Any], refined: RefinedBBox) -> None:
        projection = obj["projection"]
        projection["bbox_xyxy_visible_px"] = refined.bbox_xyxy_px
        projection["bbox_xyxy_px"] = refined.bbox_xyxy_px
        x_min, y_min, x_max, y_max = refined.bbox_xyxy_px
        width = x_max - x_min
        height = y_max - y_min
        projection["bbox_xywh_px"] = [x_min, y_min, width, height]
        projection["bbox_area_px2"] = width * height
        projection["center_px"] = [x_min + width * 0.5, y_min + height * 0.5]

        validation = obj.setdefault("validation", {})
        validation["valid"] = width >= 1.0 and height >= 1.0 and width * height >= self.min_refined_area_px
        validation["checks"] = {
            "bbox_exists": True,
            "visible_refinement_pass": validation["valid"],
        }
        validation["width_px"] = width
        validation["height_px"] = height
        validation["bbox_area_px2"] = width * height
        validation["visible_pixel_mask_area_px"] = refined.mask_area_px
        validation["quality_tier"] = "exact_visible" if validation["valid"] else "reject"
        if validation["valid"]:
            validation["reasons"] = []
            obj["quality_tier"] = "exact_visible"
            obj["confidence_source"] = "visible_pixels"
            obj["usable"] = True
        else:
            validation.setdefault("reasons", []).append("visible_bbox_refinement_failed")
            obj["quality_tier"] = "reject"
            obj["usable"] = False

        obj["visible_bbox_refinement"] = {
            "status": "applied",
            "method": refined.method,
            "confidence": refined.confidence,
            "mask_area_px": refined.mask_area_px,
            "reasons": refined.reasons,
        }
        if refined.diagnostics:
            obj["visible_bbox_refinement"].update(refined.diagnostics)

    def _apply_geometry_fallback_bbox(
        self,
        obj: Dict[str, Any],
        bbox_xyxy_px: Sequence[float],
        debug_context: Optional[Dict[str, Any]],
    ) -> None:
        projection = obj["projection"]
        projection["bbox_xyxy_px"] = [float(value) for value in bbox_xyxy_px]
        projection["bbox_xyxy_visible_px"] = None
        x_min, y_min, x_max, y_max = projection["bbox_xyxy_px"]
        width = x_max - x_min
        height = y_max - y_min
        projection["bbox_xywh_px"] = [x_min, y_min, width, height]
        projection["bbox_area_px2"] = width * height
        projection["center_px"] = [x_min + width * 0.5, y_min + height * 0.5]

        validation = obj.setdefault("validation", {})
        validation["valid"] = True
        validation.setdefault("checks", {})["bbox_exists"] = True
        validation["checks"]["visible_refinement_pass"] = False
        validation["width_px"] = width
        validation["height_px"] = height
        validation["bbox_area_px2"] = width * height
        validation["reasons"] = []
        obj["usable"] = True

        fallback_refinement = {
            "status": "fallback_geometry",
            "reasons": ["visible_pixels_not_found"],
            "fallback_bbox_source": "geometry_bbox",
            "confidence": obj.get("confidence_source", "geometry"),
        }
        if debug_context and debug_context.get("class_name") in {"airplanes", "helicopters"}:
            fallback_refinement.update(debug_context.get("airborne_failure_diagnostics", {}))
        obj["visible_bbox_refinement"] = fallback_refinement

    def _refresh_frame_stats(self, frame_metadata: Dict[str, Any]) -> None:
        objects = frame_metadata.get("objects", [])
        object_stats = frame_metadata.setdefault("object_stats", {})
        object_stats["total_objects"] = len(objects)
        object_stats["valid_objects"] = sum(1 for obj in objects if obj.get("validation", {}).get("valid"))
        object_stats["by_class"] = {}
        object_stats["valid_by_class"] = {}
        object_stats["by_dimensions_source"] = {}
        object_stats["by_quality_tier"] = {}
        for obj in objects:
            class_name = obj.get("class", "unknown")
            dimensions_source = obj.get("dimensions_source", "unknown")
            quality_tier = obj.get("quality_tier", "unknown")
            object_stats["by_class"][class_name] = object_stats["by_class"].get(class_name, 0) + 1
            object_stats["by_dimensions_source"][dimensions_source] = object_stats["by_dimensions_source"].get(dimensions_source, 0) + 1
            object_stats["by_quality_tier"][quality_tier] = object_stats["by_quality_tier"].get(quality_tier, 0) + 1
            if obj.get("validation", {}).get("valid"):
                object_stats["valid_by_class"][class_name] = object_stats["valid_by_class"].get(class_name, 0) + 1

        sync_ok = frame_metadata.get("sync", {}).get("strict_sync_valid", False)
        height_ok = "camera_height_not_15m" not in frame_metadata.get("invalid_reasons", [])
        has_valid_object = object_stats["valid_objects"] > 0
        invalid_reasons = [
            reason
            for reason in frame_metadata.get("invalid_reasons", [])
            if reason not in {"no_valid_objects"}
        ]
        if not has_valid_object and "no_valid_objects" not in invalid_reasons:
            invalid_reasons.append("no_valid_objects")
        frame_metadata["invalid_reasons"] = invalid_reasons
        frame_metadata["usable"] = bool(sync_ok and height_ok and has_valid_object)
