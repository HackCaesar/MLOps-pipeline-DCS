"""Ship bbox refinement mixin (edge projection + grabcut fallback).

Part of the visible-bbox refinement decomposition (Step 4). This mixin
carries only methods for the ship branch and MUST be combined with the
``VisibleBBoxRefiner`` facade via multiple inheritance — it relies on the
facade for shared helpers and configuration attributes
(``self._roi_for_bbox``, ``self._pad_bbox``, ``self._bbox_from_components``,
``self._components_intersecting_bbox``, ``self._components_debug_union_bbox``,
``self._limit_component_debug``, ``self._safe_debug_token``,
``self._draw_debug_bbox``, ``self.ship_padding_px``,
``self.airborne_padding_px``, ``self.ignore_bottom_px``,
``self.min_refined_area_px``, ``self.config``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None

from validators.bbox_geometry import (
    bbox_area as _bbox_area,
)
from validators.bbox_geometry import (
    bbox_center as _bbox_center,
)
from validators.bbox_geometry import (
    bbox_overlap_area as _bbox_overlap_area,
)
from validators.bbox_geometry import (
    clip_bbox as _clip_bbox,
)
from validators.components import (
    connected_components as _connected_components,
)
from validators.refined_bbox import RefinedBBox


class ShipRefinerMixin:
    def _refine_ship(
        self,
        image: np.ndarray,
        bbox: Sequence[float],
        debug_context: Optional[Dict[str, Any]],
    ) -> Optional[RefinedBBox]:
        return self._refine_ship_from_edges(image, bbox, debug_context)

    def _refine_ship_from_edges(
        self,
        image: np.ndarray,
        bbox: Sequence[float],
        debug_context: Optional[Dict[str, Any]],
    ) -> Optional[RefinedBBox]:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        small_or_front_view = self._ship_is_small_or_front_view(bbox, debug_context)
        geometry_primary = self._ship_geometry_primary_bbox(image, bbox, small_or_front_view)

        x_expand = max(20.0, 0.08 * width)
        y_expand_top = max(10.0, 0.05 * height)
        y_expand_bottom = self._ship_bottom_expand_limit(bbox, small_or_front_view)
        margin = max(x_expand, y_expand_top, y_expand_bottom, float(self.ship_padding_px) + 1.0)
        roi, roi_box = self._roi_for_bbox(image, bbox, margin)
        if roi.size == 0:
            return self._ship_geometry_primary_result(
                image=image,
                bbox=bbox,
                geometry_primary=geometry_primary,
                before_clamp_bbox=geometry_primary,
                debug_context=debug_context,
                source="ship_geometry_primary_small_or_front_view" if small_or_front_view else "ship_geometry_primary",
                reasons=["ship_geometry_primary_roi_empty"],
                small_or_front_view=small_or_front_view,
            )

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # type: ignore[union-attr]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # type: ignore[union-attr]
        _hue, saturation, _value = cv2.split(hsv)  # type: ignore[union-attr]
        edge_mask = cv2.Canny(gray, 50, 150) > 0  # type: ignore[union-attr]
        edge_mask = edge_mask & (saturation < 120)

        roi_x1, roi_y1, _roi_x2, _roi_y2 = roi_box
        allowed_x1 = max(0, int(math.floor(x_min - x_expand - roi_x1)))
        allowed_y1 = max(0, int(math.floor(y_min - y_expand_top - roi_y1)))
        allowed_x2 = min(edge_mask.shape[1], int(math.ceil(x_max + x_expand - roi_x1)))
        allowed_y2 = min(edge_mask.shape[0], int(math.ceil(y_max + y_expand_bottom - roi_y1)))
        allowed_region = np.zeros_like(edge_mask, dtype=bool)
        allowed_region[allowed_y1:allowed_y2, allowed_x1:allowed_x2] = True
        edge_mask = edge_mask & allowed_region

        components_all = self._diagnose_components(edge_mask, roi_box, bbox)

        if small_or_front_view:
            rejected_components = self._reject_ship_components(
                components_all,
                "ship_geometry_primary_small_or_front_view",
            )
            return self._ship_geometry_primary_result(
                image=image,
                bbox=bbox,
                geometry_primary=geometry_primary,
                before_clamp_bbox=geometry_primary,
                debug_context=debug_context,
                source="ship_geometry_primary_small_or_front_view",
                reasons=["ship_geometry_primary_small_or_front_view"],
                small_or_front_view=small_or_front_view,
                roi=roi,
                roi_box=roi_box,
                edge_mask=edge_mask,
                selected_components=[],
                rejected_components=rejected_components,
                bottom_limit_y=y_max + y_expand_bottom,
            )

        if int(edge_mask.sum()) < 20:
            return self._ship_geometry_primary_result(
                image=image,
                bbox=bbox,
                geometry_primary=geometry_primary,
                before_clamp_bbox=geometry_primary,
                debug_context=debug_context,
                source="ship_geometry_primary",
                reasons=["ship_edge_insufficient_pixels"],
                small_or_front_view=small_or_front_view,
                roi=roi,
                roi_box=roi_box,
                edge_mask=edge_mask,
                selected_components=[],
                rejected_components=components_all,
                bottom_limit_y=y_max + y_expand_bottom,
            )

        selected_components, rejected_components = self._split_ship_components_conservative(components_all, bbox, roi_box)
        selected_bbox = self._components_debug_union_bbox(selected_components)
        if selected_bbox is None:
            rejected_components = self._reject_ship_components(
                rejected_components,
                "ship_edge_rejected_geometry_fallback",
            )
            return self._ship_geometry_primary_result(
                image=image,
                bbox=bbox,
                geometry_primary=geometry_primary,
                before_clamp_bbox=geometry_primary,
                debug_context=debug_context,
                source="ship_edge_rejected_geometry_fallback",
                reasons=["ship_edge_rejected_geometry_fallback"],
                small_or_front_view=small_or_front_view,
                roi=roi,
                roi_box=roi_box,
                edge_mask=edge_mask,
                selected_components=[],
                rejected_components=rejected_components,
                bottom_limit_y=y_max + y_expand_bottom,
            )

        candidate = [
            min(float(x_min), selected_bbox[0]),
            min(float(y_min), selected_bbox[1]),
            max(float(x_max), selected_bbox[2]),
            max(float(y_max), selected_bbox[3]),
        ]
        candidate = _clip_bbox(candidate, image.shape[1], image.shape[0], self.ignore_bottom_px)
        if candidate is None or _bbox_area(candidate) < self.min_refined_area_px:
            rejected_components = self._reject_ship_components(
                selected_components + rejected_components,
                "ship_edge_rejected_geometry_fallback",
            )
            return self._ship_geometry_primary_result(
                image=image,
                bbox=bbox,
                geometry_primary=geometry_primary,
                before_clamp_bbox=geometry_primary,
                debug_context=debug_context,
                source="ship_edge_rejected_geometry_fallback",
                reasons=["ship_edge_candidate_invalid"],
                small_or_front_view=small_or_front_view,
                roi=roi,
                roi_box=roi_box,
                edge_mask=edge_mask,
                selected_components=[],
                rejected_components=rejected_components,
                bottom_limit_y=y_max + y_expand_bottom,
            )

        padded_candidate = self._pad_bbox(candidate, image.shape[1], image.shape[0], self.ship_padding_px)
        candidate_reject_reasons = self._ship_edge_candidate_reject_reasons(selected_bbox, bbox, small_or_front_view)
        if candidate_reject_reasons:
            rejected_components = self._reject_ship_components(
                selected_components + rejected_components,
                "ship_edge_rejected_geometry_fallback",
            )
            return self._ship_geometry_primary_result(
                image=image,
                bbox=bbox,
                geometry_primary=geometry_primary,
                before_clamp_bbox=padded_candidate,
                debug_context=debug_context,
                source="ship_edge_rejected_geometry_fallback",
                reasons=["ship_edge_rejected_geometry_fallback"] + candidate_reject_reasons,
                small_or_front_view=small_or_front_view,
                roi=roi,
                roi_box=roi_box,
                edge_mask=edge_mask,
                selected_components=[],
                rejected_components=rejected_components,
                bottom_limit_y=y_max + y_expand_bottom,
            )

        final_bbox, clamp_reasons, diagnostics = self._clamp_ship_bbox_to_geometry_conservative(
            padded_candidate,
            bbox,
            image.shape[1],
            image.shape[0],
            small_or_front_view,
        )
        if final_bbox == geometry_primary:
            final_source = "ship_geometry_primary"
            reasons = ["ship_geometry_primary"] + clamp_reasons
        else:
            final_source = "ship_edge_conservative_refinement"
            reasons = ["ship_edge_conservative_refinement"] + clamp_reasons
        diagnostics.update(
            self._ship_component_diagnostics(
                roi_box=roi_box,
                geometry_bbox=bbox,
                search_roi_bbox=[roi_box[0], roi_box[1], roi_box[2], roi_box[3]],
                selected_components=selected_components,
                rejected_components=rejected_components,
                final_bbox_source=final_source,
                bottom_limit_y=y_max + y_expand_bottom,
            )
        )
        diagnostics["ship_debug_artifacts"] = self._write_ship_debug_artifacts(
            image=image,
            roi=roi,
            roi_box=roi_box,
            edge_mask=edge_mask,
            foreground_mask=None,
            geometry_bbox=bbox,
            search_roi_bbox=[roi_box[0], roi_box[1], roi_box[2], roi_box[3]],
            selected_components=selected_components,
            rejected_components=rejected_components,
            before_clamp_bbox=padded_candidate,
            final_bbox=final_bbox,
            debug_context=debug_context,
            method="ship_edge_conservative_refinement",
        )

        return RefinedBBox(
            bbox_xyxy_px=final_bbox,
            mask_area_px=sum(int(component.get("area_px", 0)) for component in selected_components),
            method=final_source,
            confidence="visible_pixels",
            reasons=reasons,
            diagnostics=diagnostics,
        )

    def _ship_is_small_or_front_view(
        self,
        bbox: Sequence[float],
        debug_context: Optional[Dict[str, Any]],
    ) -> bool:
        width = max(1.0, float(bbox[2]) - float(bbox[0]))
        height = max(1.0, float(bbox[3]) - float(bbox[1]))
        type_name = str((debug_context or {}).get("type_name", "")).lower()
        return width / height < 0.70 or width < 80.0 or "kilo" in type_name

    def _ship_geometry_primary_bbox(
        self,
        image: np.ndarray,
        bbox: Sequence[float],
        small_or_front_view: bool,
    ) -> List[float]:
        padding_px = min(self.ship_padding_px, 2) if small_or_front_view else self.ship_padding_px
        return self._pad_bbox(bbox, image.shape[1], image.shape[0], padding_px)

    def _ship_bottom_expand_limit(self, bbox: Sequence[float], small_or_front_view: bool) -> float:
        geometry_height = max(1.0, float(bbox[3]) - float(bbox[1]))
        if small_or_front_view:
            return max(5.0, 0.05 * geometry_height)
        return max(8.0, 0.08 * geometry_height)

    def _ship_geometry_primary_result(
        self,
        image: np.ndarray,
        bbox: Sequence[float],
        geometry_primary: Sequence[float],
        before_clamp_bbox: Sequence[float],
        debug_context: Optional[Dict[str, Any]],
        source: str,
        reasons: Sequence[str],
        small_or_front_view: bool,
        roi: Optional[np.ndarray] = None,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        edge_mask: Optional[np.ndarray] = None,
        selected_components: Optional[Sequence[Dict[str, Any]]] = None,
        rejected_components: Optional[Sequence[Dict[str, Any]]] = None,
        bottom_limit_y: Optional[float] = None,
    ) -> RefinedBBox:
        final_bbox, clamp_reasons, diagnostics = self._clamp_ship_bbox_to_geometry_conservative(
            geometry_primary,
            bbox,
            image.shape[1],
            image.shape[0],
            small_or_front_view,
        )
        if list(before_clamp_bbox) != list(geometry_primary):
            diagnostics = self._ship_bbox_diagnostics(before_clamp_bbox, final_bbox, bbox)

        selected_components = list(selected_components or [])
        rejected_components = list(rejected_components or [])
        if roi_box is None:
            roi_box = (
                int(max(0, math.floor(float(bbox[0])))),
                int(max(0, math.floor(float(bbox[1])))),
                int(min(image.shape[1], math.ceil(float(bbox[2])))),
                int(min(image.shape[0], math.ceil(float(bbox[3])))),
            )
        if bottom_limit_y is None:
            bottom_limit_y = float(bbox[3]) + self._ship_bottom_expand_limit(bbox, small_or_front_view)

        diagnostics.update(
            self._ship_component_diagnostics(
                roi_box=roi_box,
                geometry_bbox=bbox,
                search_roi_bbox=[roi_box[0], roi_box[1], roi_box[2], roi_box[3]],
                selected_components=selected_components,
                rejected_components=rejected_components,
                final_bbox_source=source,
                bottom_limit_y=bottom_limit_y,
            )
        )
        if roi is not None:
            diagnostics["ship_debug_artifacts"] = self._write_ship_debug_artifacts(
                image=image,
                roi=roi,
                roi_box=roi_box,
                edge_mask=edge_mask,
                foreground_mask=None,
                geometry_bbox=bbox,
                search_roi_bbox=[roi_box[0], roi_box[1], roi_box[2], roi_box[3]],
                selected_components=selected_components,
                rejected_components=rejected_components,
                before_clamp_bbox=before_clamp_bbox,
                final_bbox=final_bbox,
                debug_context=debug_context,
                method=source,
            )

        return RefinedBBox(
            bbox_xyxy_px=final_bbox,
            mask_area_px=sum(int(component.get("area_px", 0)) for component in selected_components),
            method=source,
            confidence="geometry_primary" if not selected_components else "visible_pixels",
            reasons=list(reasons) + clamp_reasons,
            diagnostics=diagnostics,
        )

    def _reject_ship_components(
        self,
        components: Sequence[Dict[str, Any]],
        reason: str,
    ) -> List[Dict[str, Any]]:
        rejected = []
        for component in components:
            diagnostic = dict(component)
            reasons = set(str(item) for item in diagnostic.get("reject_reasons", []))
            reasons.add(reason)
            diagnostic["reject_reasons"] = sorted(reasons)
            diagnostic["selected"] = False
            rejected.append(diagnostic)
        return rejected

    def _split_ship_components_conservative(
        self,
        components: Sequence[Dict[str, Any]],
        geometry_bbox: Sequence[float],
        roi_box: Tuple[int, int, int, int],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        geometry_width = max(1.0, float(geometry_bbox[2]) - float(geometry_bbox[0]))
        geometry_height = max(1.0, float(geometry_bbox[3]) - float(geometry_bbox[1]))
        geometry_area = max(1.0, _bbox_area(geometry_bbox))
        geometry_center_x, geometry_center_y = _bbox_center(geometry_bbox)
        roi_height = max(1.0, float(roi_box[3] - roi_box[1]))
        selected: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        for component in components:
            diagnostic = dict(component)
            bbox = diagnostic["bbox_xyxy_px"]
            component_width = max(1.0, float(bbox[2]) - float(bbox[0]))
            component_height = max(1.0, float(bbox[3]) - float(bbox[1]))
            component_area = max(1.0, _bbox_area(bbox))
            component_center_x, component_center_y = diagnostic.get("centroid_px") or _bbox_center(bbox)
            aspect_ratio = component_width / component_height
            vertical_extent_ratio = component_height / geometry_height
            overlap_area = _bbox_overlap_area(bbox, geometry_bbox)
            overlap_ratio_component = overlap_area / component_area
            overlap_ratio_geometry = overlap_area / geometry_area
            reasons = set(str(item) for item in diagnostic.get("reject_reasons", []))

            diagnostic["ship_component_aspect_ratio"] = float(aspect_ratio)
            diagnostic["ship_component_vertical_extent_ratio"] = float(vertical_extent_ratio)
            diagnostic["overlap_area_with_geometry_px"] = float(overlap_area)
            diagnostic["overlap_ratio_with_component_bbox"] = float(overlap_ratio_component)
            diagnostic["overlap_ratio_with_geometry_bbox"] = float(overlap_ratio_geometry)

            if component_center_y > float(geometry_bbox[3]):
                reasons.add("rejected_below_ship_bottom")
            if aspect_ratio > 6.0 and component_height < 0.20 * geometry_height:
                reasons.add("rejected_horizontal_waterline")
            if overlap_area <= 0.0 or (overlap_ratio_component < 0.25 and overlap_ratio_geometry < 0.01):
                reasons.add("rejected_low_geometry_overlap")
            if (
                component_center_y > geometry_center_y
                and aspect_ratio > 3.0
                and component_height < 0.35 * geometry_height
                and overlap_ratio_component < 0.70
            ):
                reasons.add("rejected_water_reflection")
            if bbox[3] > float(roi_box[1]) + 0.80 * roi_height and component_height < max(3.0, 0.08 * geometry_height):
                reasons.add("rejected_low_vertical_continuity")

            center_x_margin = max(8.0, 0.10 * geometry_width)
            center_y_top_margin = max(6.0, 0.08 * geometry_height)
            center_y_bottom_margin = self._ship_bottom_expand_limit(geometry_bbox, False)
            center_near_geometry = (
                float(geometry_bbox[0]) - center_x_margin <= component_center_x <= float(geometry_bbox[2]) + center_x_margin
                and float(geometry_bbox[1]) - center_y_top_margin <= component_center_y <= float(geometry_bbox[3]) + center_y_bottom_margin
            )
            if not center_near_geometry:
                reasons.add("rejected_low_geometry_overlap")

            diagnostic["reject_reasons"] = sorted(reasons)
            if not reasons and overlap_area > 0.0:
                diagnostic["selected"] = True
                selected.append(diagnostic)
            else:
                diagnostic["selected"] = False
                rejected.append(diagnostic)

        return selected, rejected

    def _ship_edge_candidate_reject_reasons(
        self,
        selected_bbox: Sequence[float],
        geometry_bbox: Sequence[float],
        small_or_front_view: bool,
    ) -> List[str]:
        geometry_height = max(1.0, float(geometry_bbox[3]) - float(geometry_bbox[1]))
        selected_width = max(1.0, float(selected_bbox[2]) - float(selected_bbox[0]))
        selected_height = max(1.0, float(selected_bbox[3]) - float(selected_bbox[1]))
        selected_area = max(1.0, _bbox_area(selected_bbox))
        overlap_area = _bbox_overlap_area(selected_bbox, geometry_bbox)
        overlap_ratio = overlap_area / selected_area
        aspect_ratio = selected_width / selected_height
        reasons: List[str] = []

        if overlap_ratio < 0.55:
            reasons.append("rejected_low_geometry_overlap")
        if selected_height < max(6.0, 0.12 * geometry_height):
            reasons.append("rejected_low_vertical_continuity")
        if aspect_ratio > 6.0 and selected_height < 0.20 * geometry_height:
            reasons.append("rejected_horizontal_waterline")
        if float(selected_bbox[3]) > float(geometry_bbox[3]) + self._ship_bottom_expand_limit(geometry_bbox, small_or_front_view):
            reasons.append("rejected_below_ship_bottom")
        return reasons

    def _clamp_ship_bbox_to_geometry_conservative(
        self,
        refined_bbox: Sequence[float],
        geometry_bbox: Sequence[float],
        width_px: int,
        height_px: int,
        small_or_front_view: bool,
    ) -> Tuple[List[float], List[str], Dict[str, Any]]:
        geometry_width = max(1.0, float(geometry_bbox[2]) - float(geometry_bbox[0]))
        geometry_height = max(1.0, float(geometry_bbox[3]) - float(geometry_bbox[1]))
        before = [float(value) for value in refined_bbox]
        if small_or_front_view:
            max_left_right_expand = max(5.0, 0.04 * geometry_width)
            max_top_expand = max(5.0, 0.03 * geometry_height)
        else:
            max_left_right_expand = max(20.0, 0.08 * geometry_width)
            max_top_expand = max(10.0, 0.05 * geometry_height)
        max_bottom_expand = self._ship_bottom_expand_limit(geometry_bbox, small_or_front_view)
        clamped = [
            max(before[0], float(geometry_bbox[0]) - max_left_right_expand),
            max(before[1], float(geometry_bbox[1]) - max_top_expand),
            min(before[2], float(geometry_bbox[2]) + max_left_right_expand),
            min(before[3], float(geometry_bbox[3]) + max_bottom_expand),
        ]

        clipped = _clip_bbox(clamped, width_px, height_px, self.ignore_bottom_px)
        fallback_bbox = self._ship_geometry_primary_bbox_for_size(geometry_bbox, width_px, height_px, small_or_front_view)
        reasons: List[str] = []
        if clipped is None or _bbox_area(clipped) < self.min_refined_area_px:
            final_bbox = fallback_bbox
            reasons.append("ship_edge_rejected_geometry_fallback")
        else:
            final_bbox = clipped
            if before != clipped:
                reasons.append("ship_refinement_clamped_to_geometry")

        return final_bbox, reasons, self._ship_bbox_diagnostics(before, final_bbox, geometry_bbox)

    def _ship_geometry_primary_bbox_for_size(
        self,
        geometry_bbox: Sequence[float],
        width_px: int,
        height_px: int,
        small_or_front_view: bool,
    ) -> List[float]:
        padding_px = min(self.ship_padding_px, 2) if small_or_front_view else self.ship_padding_px
        return self._pad_bbox(geometry_bbox, width_px, height_px, padding_px)

    def _ship_bbox_diagnostics(
        self,
        before_bbox: Sequence[float],
        final_bbox: Sequence[float],
        geometry_bbox: Sequence[float],
    ) -> Dict[str, Any]:
        geometry_width = max(1.0, float(geometry_bbox[2]) - float(geometry_bbox[0]))
        geometry_height = max(1.0, float(geometry_bbox[3]) - float(geometry_bbox[1]))
        before = [float(value) for value in before_bbox]
        final = [float(value) for value in final_bbox]
        before_width = max(1.0, before[2] - before[0])
        before_height = max(1.0, before[3] - before[1])
        final_width = max(1.0, final[2] - final[0])
        final_height = max(1.0, final[3] - final[1])
        return {
            "ship_geometry_bbox_xyxy_px": [float(value) for value in geometry_bbox],
            "ship_refined_bbox_before_clamp_xyxy_px": before,
            "ship_refined_bbox_after_clamp_xyxy_px": final,
            "ship_expansion_ratio_w": before_width / geometry_width,
            "ship_expansion_ratio_h": before_height / geometry_height,
            "ship_expansion_ratio_w_after_clamp": final_width / geometry_width,
            "ship_expansion_ratio_h_after_clamp": final_height / geometry_height,
        }

    def _refine_ship_from_grabcut(
        self,
        image: np.ndarray,
        bbox: Sequence[float],
        debug_context: Optional[Dict[str, Any]],
    ) -> Optional[RefinedBBox]:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        margin = max(80.0, 0.40 * max(width, height))
        roi, roi_box = self._roi_for_bbox(image, bbox, margin)
        if roi.size == 0:
            return None

        roi_h, roi_w = roi.shape[:2]
        scale = min(1.0, 650.0 / float(max(roi_h, roi_w)))
        small = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)  # type: ignore[union-attr]
        roi_x1, roi_y1, _roi_x2, _roi_y2 = roi_box
        rect_margin = 0.15 * max(width, height)
        rect_x1 = max(1, int((x_min - rect_margin - roi_x1) * scale))
        rect_y1 = max(1, int((y_min - rect_margin - roi_y1) * scale))
        rect_x2 = min(small.shape[1] - 2, int((x_max + rect_margin - roi_x1) * scale))
        rect_y2 = min(small.shape[0] - 2, int((y_max + rect_margin - roi_y1) * scale))
        if rect_x2 <= rect_x1 + 2 or rect_y2 <= rect_y1 + 2:
            return None

        mask = np.zeros(small.shape[:2], dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(  # type: ignore[union-attr]
                small,
                mask,
                (rect_x1, rect_y1, rect_x2 - rect_x1, rect_y2 - rect_y1),
                bgd_model,
                fgd_model,
                2,
                cv2.GC_INIT_WITH_RECT,
            )
        except Exception:
            return None

        foreground = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype("uint8")  # type: ignore[union-attr]
        foreground = cv2.resize(foreground, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)  # type: ignore[union-attr]
        foreground = cv2.morphologyEx(  # type: ignore[union-attr]
            foreground * 255,
            cv2.MORPH_CLOSE,
            np.ones((7, 5), dtype=np.uint8),
            iterations=1,
        ) > 0

        components = _connected_components(foreground, min_area_px=50)
        if not components:
            return None
        selected = self._components_intersecting_bbox(components, roi_box, bbox)
        if not selected:
            selected = [max(components, key=lambda component: component[4])]
        refined = self._bbox_from_components(
            selected,
            roi_box,
            image.shape[1],
            image.shape[0],
            self.ship_padding_px,
            "ship_grabcut_visible_pixels",
        )
        if refined is None:
            return None

        final_bbox, reasons, diagnostics = self._clamp_ship_bbox_to_geometry(
            refined.bbox_xyxy_px,
            bbox,
            image.shape[1],
            image.shape[0],
        )
        selected_ids = {id(component) for component in selected}
        selected_diagnostics = [self._component_diagnostic(component, roi_box, bbox) for component in selected]
        rejected_diagnostics = [
            self._component_diagnostic(component, roi_box, bbox)
            for component in components
            if id(component) not in selected_ids
        ]
        final_source = self._ship_final_bbox_source("grabcut", reasons)
        diagnostics.update(
            self._ship_component_diagnostics(
                roi_box=roi_box,
                geometry_bbox=bbox,
                search_roi_bbox=[roi_box[0], roi_box[1], roi_box[2], roi_box[3]],
                selected_components=selected_diagnostics,
                rejected_components=rejected_diagnostics,
                final_bbox_source=final_source,
                bottom_limit_y=float(bbox[3]) + max(45.0, 0.20 * (float(bbox[3]) - float(bbox[1]))),
            )
        )
        diagnostics["ship_debug_artifacts"] = self._write_ship_debug_artifacts(
            image=image,
            roi=roi,
            roi_box=roi_box,
            edge_mask=None,
            foreground_mask=foreground,
            geometry_bbox=bbox,
            search_roi_bbox=[roi_box[0], roi_box[1], roi_box[2], roi_box[3]],
            selected_components=selected_diagnostics,
            rejected_components=rejected_diagnostics,
            before_clamp_bbox=refined.bbox_xyxy_px,
            final_bbox=final_bbox,
            debug_context=debug_context,
            method="ship_grabcut_visible_pixels",
        )
        return RefinedBBox(
            bbox_xyxy_px=final_bbox,
            mask_area_px=refined.mask_area_px,
            method=refined.method,
            confidence=refined.confidence,
            reasons=refined.reasons + reasons,
            diagnostics=diagnostics,
        )

    def _diagnose_components(
        self,
        mask: np.ndarray,
        roi_box: Tuple[int, int, int, int],
        geometry_bbox: Sequence[float],
    ) -> List[Dict[str, Any]]:
        if mask.size == 0:
            return []
        components = _connected_components(mask, min_area_px=1)
        return [self._component_diagnostic(component, roi_box, geometry_bbox) for component in components]

    def _split_components_by_bbox(
        self,
        components: Sequence[Dict[str, Any]],
        bbox: Sequence[float],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        selected: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for component in components:
            diagnostic = dict(component)
            overlap_area = _bbox_overlap_area(diagnostic["bbox_xyxy_px"], bbox)
            diagnostic["overlap_area_with_selected_bbox_px"] = float(overlap_area)
            if overlap_area > 0.0:
                diagnostic["selected"] = True
                selected.append(diagnostic)
                continue

            reject_reasons = list(diagnostic.get("reject_reasons", []))
            reject_reasons.append("outside_selected_bbox")
            diagnostic["reject_reasons"] = reject_reasons
            diagnostic["selected"] = False
            rejected.append(diagnostic)
        return selected, rejected

    def _component_diagnostic(
        self,
        component: Tuple[int, int, int, int, int, np.ndarray],
        roi_box: Tuple[int, int, int, int],
        geometry_bbox: Sequence[float],
    ) -> Dict[str, Any]:
        roi_x1, roi_y1, _roi_x2, _roi_y2 = roi_box
        x, y, width, height, area, centroid = component
        bbox = [float(roi_x1 + x), float(roi_y1 + y), float(roi_x1 + x + width), float(roi_y1 + y + height)]
        centroid_px = [float(roi_x1 + centroid[0]), float(roi_y1 + centroid[1])]
        component_bbox_area = max(1.0, _bbox_area(bbox))
        geometry_area = max(1.0, _bbox_area(geometry_bbox))
        overlap_area = _bbox_overlap_area(bbox, geometry_bbox)
        geometry_center = _bbox_center(geometry_bbox)
        center_distance = math.hypot(centroid_px[0] - geometry_center[0], centroid_px[1] - geometry_center[1])
        reject_reasons: List[str] = []
        if overlap_area <= 0.0:
            reject_reasons.append("outside_geometry_bbox")
        return {
            "bbox_xyxy_px": bbox,
            "bbox_xyxy_roi_px": [int(x), int(y), int(x + width), int(y + height)],
            "centroid_px": centroid_px,
            "area_px": int(area),
            "width_px": int(width),
            "height_px": int(height),
            "overlap_area_with_geometry_px": float(overlap_area),
            "overlap_ratio_with_component_bbox": float(overlap_area / component_bbox_area),
            "overlap_ratio_with_geometry_bbox": float(overlap_area / geometry_area),
            "center_distance_to_geometry_px": float(center_distance),
            "reject_reasons": reject_reasons,
        }

    def _ship_final_bbox_source(self, proposed_source: str, reasons: Sequence[str]) -> str:
        if "ship_refinement_fallback_geometry" in reasons:
            return "geometry_fallback"
        if "ship_refinement_clamped_to_geometry" in reasons:
            return f"{proposed_source}_clamped_to_geometry"
        return proposed_source

    def _ship_component_diagnostics(
        self,
        roi_box: Tuple[int, int, int, int],
        geometry_bbox: Sequence[float],
        search_roi_bbox: Sequence[float],
        selected_components: Sequence[Dict[str, Any]],
        rejected_components: Sequence[Dict[str, Any]],
        final_bbox_source: str,
        bottom_limit_y: float,
    ) -> Dict[str, Any]:
        selected_bbox = self._components_debug_union_bbox(selected_components)
        selected_area = sum(int(component.get("area_px", 0)) for component in selected_components)
        reject_reasons = sorted(
            {
                str(reason)
                for component in rejected_components
                for reason in component.get("reject_reasons", [])
            }
        )
        return {
            "ship_roi_xyxy_px": [float(value) for value in roi_box],
            "ship_search_roi_xyxy_px": [float(value) for value in search_roi_bbox],
            "ship_component_count": int(len(selected_components) + len(rejected_components)),
            "ship_selected_component_count": int(len(selected_components)),
            "ship_rejected_component_count": int(len(rejected_components)),
            "ship_selected_component_area_px": int(selected_area),
            "ship_selected_component_bbox_xyxy_px": selected_bbox,
            "ship_final_bbox_source": final_bbox_source,
            "ship_reject_reasons": reject_reasons,
            "ship_waterline_limit_y": float(geometry_bbox[3]),
            "ship_bottom_limit_y": float(bottom_limit_y),
            "ship_selected_components": self._limit_component_debug(selected_components),
            "ship_rejected_components": self._limit_component_debug(rejected_components),
        }

    def _write_ship_debug_artifacts(
        self,
        image: np.ndarray,
        roi: np.ndarray,
        roi_box: Tuple[int, int, int, int],
        edge_mask: Optional[np.ndarray],
        foreground_mask: Optional[np.ndarray],
        geometry_bbox: Sequence[float],
        search_roi_bbox: Sequence[float],
        selected_components: Sequence[Dict[str, Any]],
        rejected_components: Sequence[Dict[str, Any]],
        before_clamp_bbox: Sequence[float],
        final_bbox: Sequence[float],
        debug_context: Optional[Dict[str, Any]],
        method: str,
    ) -> Dict[str, str]:
        if not debug_context:
            return {}
        try:
            debug_dir = Path(debug_context["debug_dir"])
            debug_dir.mkdir(parents=True, exist_ok=True)
            prefix = self._ship_debug_prefix(debug_context, method)
            artifacts: Dict[str, str] = {}

            full_overlay = image.copy()
            self._draw_debug_bbox(full_overlay, search_roi_bbox, (255, 0, 255), "search")
            self._draw_debug_bbox(full_overlay, geometry_bbox, (0, 255, 255), "geom")
            self._draw_debug_bbox(full_overlay, before_clamp_bbox, (255, 255, 0), "pre")
            self._draw_debug_bbox(full_overlay, final_bbox, (0, 255, 0), "final")
            for component in self._limit_component_debug(rejected_components, limit=60):
                self._draw_debug_bbox(full_overlay, component["bbox_xyxy_px"], (0, 0, 255), "rej")
            for component in self._limit_component_debug(selected_components, limit=60):
                self._draw_debug_bbox(full_overlay, component["bbox_xyxy_px"], (255, 0, 0), "sel")
            full_overlay_path = debug_dir / f"{prefix}__overlay.png"
            cv2.imwrite(str(full_overlay_path), full_overlay)  # type: ignore[union-attr]
            artifacts["overlay_png"] = str(full_overlay_path)

            roi_overlay = roi.copy()
            roi_x1, roi_y1, _roi_x2, _roi_y2 = roi_box
            for label, bbox, color in (
                ("geom", geometry_bbox, (0, 255, 255)),
                ("pre", before_clamp_bbox, (255, 255, 0)),
                ("final", final_bbox, (0, 255, 0)),
            ):
                local_bbox = [bbox[0] - roi_x1, bbox[1] - roi_y1, bbox[2] - roi_x1, bbox[3] - roi_y1]
                self._draw_debug_bbox(roi_overlay, local_bbox, color, label)
            roi_overlay_path = debug_dir / f"{prefix}__roi_overlay.png"
            cv2.imwrite(str(roi_overlay_path), roi_overlay)  # type: ignore[union-attr]
            artifacts["roi_overlay_png"] = str(roi_overlay_path)

            if edge_mask is not None:
                edge_mask_path = debug_dir / f"{prefix}__edge_mask.png"
                cv2.imwrite(str(edge_mask_path), edge_mask.astype("uint8") * 255)  # type: ignore[union-attr]
                artifacts["edge_mask_png"] = str(edge_mask_path)
            if foreground_mask is not None:
                foreground_mask_path = debug_dir / f"{prefix}__foreground_mask.png"
                cv2.imwrite(str(foreground_mask_path), foreground_mask.astype("uint8") * 255)  # type: ignore[union-attr]
                artifacts["foreground_mask_png"] = str(foreground_mask_path)
            return artifacts
        except Exception as exc:  # pragma: no cover - best-effort diagnostics
            return {"error": str(exc)}

    def _write_ship_failure_debug(
        self,
        image: np.ndarray,
        bbox: Sequence[float],
        debug_context: Optional[Dict[str, Any]],
        reason: str,
    ) -> None:
        if not debug_context:
            return
        try:
            debug_dir = Path(debug_context["debug_dir"])
            debug_dir.mkdir(parents=True, exist_ok=True)
            prefix = self._ship_debug_prefix(debug_context, reason)
            overlay = image.copy()
            self._draw_debug_bbox(overlay, bbox, (0, 0, 255), reason)
            cv2.imwrite(str(debug_dir / f"{prefix}__failure.png"), overlay)  # type: ignore[union-attr]
        except Exception:
            return

    def _ship_debug_prefix(self, debug_context: Dict[str, Any], method: str) -> str:
        frame_id = self._safe_debug_token(debug_context.get("frame_id", "frame"))
        object_id = self._safe_debug_token(debug_context.get("object_id", "object"))
        type_name = self._safe_debug_token(debug_context.get("type_name", "unknown"))
        method_name = self._safe_debug_token(method)
        return f"{frame_id}__{object_id}__{type_name}__{method_name}"

    def _clamp_ship_bbox_to_geometry(
        self,
        refined_bbox: Sequence[float],
        geometry_bbox: Sequence[float],
        width_px: int,
        height_px: int,
    ) -> Tuple[List[float], List[str], Dict[str, Any]]:
        geometry_width = max(1.0, float(geometry_bbox[2]) - float(geometry_bbox[0]))
        geometry_height = max(1.0, float(geometry_bbox[3]) - float(geometry_bbox[1]))
        before = [float(value) for value in refined_bbox]

        max_left_right_expand = max(40.0, 0.15 * geometry_width)
        max_top_expand = max(25.0, 0.10 * geometry_height)
        max_bottom_expand = max(45.0, 0.20 * geometry_height)
        clamped = [
            max(before[0], float(geometry_bbox[0]) - max_left_right_expand),
            max(before[1], float(geometry_bbox[1]) - max_top_expand),
            min(before[2], float(geometry_bbox[2]) + max_left_right_expand),
            min(before[3], float(geometry_bbox[3]) + max_bottom_expand),
        ]

        clipped = _clip_bbox(clamped, width_px, height_px, self.ignore_bottom_px)
        fallback_bbox = self._pad_bbox(geometry_bbox, width_px, height_px, self.ship_padding_px)
        reasons: List[str] = []

        before_width = max(1.0, before[2] - before[0])
        before_height = max(1.0, before[3] - before[1])
        before_ratio_w = before_width / geometry_width
        before_ratio_h = before_height / geometry_height

        if clipped is None or _bbox_area(clipped) < self.min_refined_area_px:
            final_bbox = fallback_bbox
            reasons.append("ship_refinement_fallback_geometry")
        else:
            after_width = max(1.0, clipped[2] - clipped[0])
            after_height = max(1.0, clipped[3] - clipped[1])
            after_ratio_w = after_width / geometry_width
            after_ratio_h = after_height / geometry_height
            if before != clipped:
                reasons.append("ship_refinement_clamped_to_geometry")
            if after_ratio_w > 1.45 or after_ratio_h > 1.60:
                final_bbox = fallback_bbox
                reasons.append("ship_refinement_fallback_geometry")
            else:
                final_bbox = clipped

        final_width = max(1.0, final_bbox[2] - final_bbox[0])
        final_height = max(1.0, final_bbox[3] - final_bbox[1])
        diagnostics = {
            "ship_geometry_bbox_xyxy_px": [float(value) for value in geometry_bbox],
            "ship_refined_bbox_before_clamp_xyxy_px": before,
            "ship_refined_bbox_after_clamp_xyxy_px": [float(value) for value in final_bbox],
            "ship_expansion_ratio_w": before_ratio_w,
            "ship_expansion_ratio_h": before_ratio_h,
            "ship_expansion_ratio_w_after_clamp": final_width / geometry_width,
            "ship_expansion_ratio_h_after_clamp": final_height / geometry_height,
        }

        return final_bbox, reasons, diagnostics

    def _ship_bbox_plausible(self, candidate: Sequence[float], geometry_bbox: Sequence[float]) -> bool:
        geometry_width = float(geometry_bbox[2]) - float(geometry_bbox[0])
        geometry_height = float(geometry_bbox[3]) - float(geometry_bbox[1])
        candidate_width = float(candidate[2]) - float(candidate[0])
        candidate_height = float(candidate[3]) - float(candidate[1])
        if geometry_width >= 100.0 and candidate_width < max(20.0, 0.10 * geometry_width):
            return False
        if geometry_height >= 80.0 and candidate_height < max(20.0, 0.15 * geometry_height):
            return False
        top_expand = max(25.0, 0.10 * geometry_height)
        bottom_expand = max(45.0, 0.20 * geometry_height)
        x_expand = max(40.0, 0.15 * geometry_width)
        if float(candidate[1]) < float(geometry_bbox[1]) - top_expand:
            return False
        if float(candidate[3]) > float(geometry_bbox[3]) + bottom_expand:
            return False
        if float(candidate[0]) < float(geometry_bbox[0]) - x_expand:
            return False
        if float(candidate[2]) > float(geometry_bbox[2]) + x_expand:
            return False
        return True

