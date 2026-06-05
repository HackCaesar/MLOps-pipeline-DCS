"""Airborne (planes / helicopters) bbox refinement mixin.

Part of the visible-bbox refinement decomposition (Step 4). This mixin
carries only methods for the airborne branch and MUST be combined with the
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
from validators.components import (
    connected_components as _connected_components,
)
from validators.refined_bbox import RefinedBBox


class AirborneRefinerMixin:
    def _refine_airborne(
        self,
        image: np.ndarray,
        bbox: Sequence[float],
        debug_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[RefinedBBox]:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        margin = max(100.0, 2.5 * max(width, height))
        roi, roi_box = self._roi_for_bbox(image, bbox, margin)
        if roi.size == 0:
            self._record_airborne_failure_debug(
                image=image,
                bbox=bbox,
                debug_context=debug_context,
                reason="airborne_roi_empty",
            )
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # type: ignore[union-attr]
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)  # type: ignore[union-attr]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # type: ignore[union-attr]
        _hue, saturation, value = cv2.split(hsv)  # type: ignore[union-attr]
        roi_h, roi_w = gray.shape[:2]

        border_width = max(4, int(min(roi_h, roi_w) * 0.08))
        border_mask = np.zeros((roi_h, roi_w), dtype=bool)
        border_mask[:border_width, :] = True
        border_mask[-border_width:, :] = True
        border_mask[:, :border_width] = True
        border_mask[:, -border_width:] = True
        background_lab = lab[border_mask].reshape(-1, 3).astype(np.float32)
        background_median = np.median(background_lab, axis=0)
        color_distance = np.linalg.norm(lab.astype(np.float32) - background_median, axis=2)

        edges = cv2.Canny(gray, 40, 110) > 0  # type: ignore[union-attr]
        edge_neighborhood = cv2.dilate(  # type: ignore[union-attr]
            edges.astype("uint8"), np.ones((3, 3), dtype=np.uint8), iterations=1
        ) > 0
        mask = (
            ((color_distance > 18.0) & (value < 235))
            | (value < 115)
            | ((saturation < 55) & (value < 190))
        )
        mask = mask & (edge_neighborhood | (value < 120) | (color_distance > 35.0))
        mask = cv2.morphologyEx(  # type: ignore[union-attr]
            mask.astype("uint8") * 255,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )
        mask = cv2.morphologyEx(  # type: ignore[union-attr]
            mask,
            cv2.MORPH_OPEN,
            np.ones((2, 2), dtype=np.uint8),
            iterations=1,
        ) > 0

        min_area = max(8, int(0.00002 * roi_h * roi_w))
        components = _connected_components(mask, min_area)
        if not components:
            self._record_airborne_failure_debug(
                image=image,
                bbox=bbox,
                debug_context=debug_context,
                reason="airborne_no_connected_components",
                roi=roi,
                roi_box=roi_box,
                mask=mask,
                selected_components=[],
                rejected_components=[],
            )
            return None

        roi_x1, roi_y1, _roi_x2, _roi_y2 = roi_box
        expected_center = np.array([(x_min + x_max) * 0.5 - roi_x1, (y_min + y_max) * 0.5 - roi_y1])
        scored: List[Tuple[float, Tuple[int, int, int, int, int, np.ndarray]]] = []
        legacy_scored: List[Tuple[float, Tuple[int, int, int, int, int, np.ndarray]]] = []
        filter_reject_reasons: Dict[int, List[str]] = {}
        max_component_width = max(120.0, 4.0 * width)
        max_component_height = max(120.0, 4.0 * height)
        for component in components:
            x, y, w, h, area, centroid = component
            if w > roi_w * 0.80 or h > roi_h * 0.80:
                filter_reject_reasons[id(component)] = ["airborne_rejected_large_roi_fraction"]
                continue
            if w > max_component_width or h > max_component_height:
                reasons = []
                if w > max_component_width:
                    reasons.append("airborne_rejected_component_too_wide")
                if h > max_component_height:
                    reasons.append("airborne_rejected_component_too_tall")
                filter_reject_reasons[id(component)] = reasons
                continue
            distance = float(np.linalg.norm(centroid - expected_center))
            density = float(area) / max(1.0, float(w * h))
            legacy_score = float(area) * (0.4 + density) / (1.0 + distance / 180.0)
            legacy_scored.append((legacy_score, component))
            relation = self._airborne_component_relation(component, roi_box, bbox)
            geometry_reject_reasons = self._airborne_geometry_reject_reasons(relation, bbox)
            if geometry_reject_reasons:
                filter_reject_reasons[id(component)] = geometry_reject_reasons
                continue
            scored.append((self._airborne_component_geometry_score(component, relation), component))

        if not scored:
            rejected_components = self._airborne_component_diagnostics(
                components,
                roi_box,
                bbox,
                score_by_id={},
                selected_ids=set(),
                filter_reject_reasons=filter_reject_reasons,
            )
            self._record_airborne_failure_debug(
                image=image,
                bbox=bbox,
                debug_context=debug_context,
                reason="airborne_no_scored_components",
                roi=roi,
                roi_box=roi_box,
                mask=mask,
                selected_components=[],
                rejected_components=rejected_components,
            )
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        legacy_scored.sort(key=lambda item: item[0], reverse=True)
        for _best_score, best in scored[:8]:
            legacy_selected = self._select_airborne_components_legacy(best, legacy_scored[:8])
            legacy_bbox = self._bbox_from_components(
                legacy_selected,
                roi_box,
                image.shape[1],
                image.shape[0],
                self.airborne_padding_px,
                "airborne_visible_pixels",
            )
            selected, geometry_filter_rejects = self._select_airborne_components_geometry_filtered(best, scored[:8], bbox, roi_box)
            for component, reasons in geometry_filter_rejects:
                current_reasons = set(filter_reject_reasons.get(id(component), []))
                current_reasons.update(reasons)
                filter_reject_reasons[id(component)] = sorted(current_reasons)

            selected_bbox_result = self._bbox_from_components(
                selected,
                roi_box,
                image.shape[1],
                image.shape[0],
                self.airborne_padding_px,
                "airborne_visible_pixels",
            )
            if selected_bbox_result is None:
                continue
            bbox_before_filtering = legacy_bbox.bbox_xyxy_px if legacy_bbox is not None else selected_bbox_result.bbox_xyxy_px
            bbox_after_filtering = selected_bbox_result.bbox_xyxy_px
            final_bbox = selected_bbox_result.bbox_xyxy_px
            final_selected = list(selected)
            final_source = "airborne_visible_pixels_geometry_filtered" if bbox_before_filtering != bbox_after_filtering else "airborne_visible_pixels"
            reasons: List[str] = []
            if self._airborne_bbox_excessive(final_bbox, bbox):
                overlap_selected = [
                    component
                    for component in selected
                    if self._airborne_component_relation(component, roi_box, bbox)["overlap_area"] > 0.0
                ]
                overlap_bbox_result = self._bbox_from_components(
                    overlap_selected,
                    roi_box,
                    image.shape[1],
                    image.shape[0],
                    self.airborne_padding_px,
                    "airborne_visible_pixels_rebuilt_overlap_only",
                )
                if overlap_bbox_result is not None and not self._airborne_bbox_excessive(overlap_bbox_result.bbox_xyxy_px, bbox):
                    final_bbox = overlap_bbox_result.bbox_xyxy_px
                    final_selected = overlap_selected
                    final_source = "airborne_visible_pixels_rebuilt_overlap_only"
                    reasons.append("airborne_rebuilt_from_geometry_overlap_components")
                    for component in selected:
                        if id(component) not in {id(item) for item in overlap_selected}:
                            current_reasons = set(filter_reject_reasons.get(id(component), []))
                            current_reasons.update(
                                [
                                    "rejected_zero_geometry_overlap",
                                    "rejected_would_merge_neighbor_airborne_object",
                                    "rejected_excessive_expansion",
                                ]
                            )
                            filter_reject_reasons[id(component)] = sorted(current_reasons)
                else:
                    final_bbox = self._pad_bbox(bbox, image.shape[1], image.shape[0], self.airborne_padding_px)
                    final_selected = []
                    final_source = "airborne_geometry_fallback_neighbor_merge_guard"
                    reasons.append("airborne_fallback_geometry_due_to_excessive_expansion")
                    for component in selected:
                        current_reasons = set(filter_reject_reasons.get(id(component), []))
                        current_reasons.update(["rejected_excessive_expansion", "rejected_would_merge_neighbor_airborne_object"])
                        filter_reject_reasons[id(component)] = sorted(current_reasons)

            refined_width = final_bbox[2] - final_bbox[0]
            refined_height = final_bbox[3] - final_bbox[1]
            if refined_width > max(140.0, 4.5 * width) or refined_height > max(140.0, 4.5 * height):
                continue
            if refined_width < max(5.0, 0.25 * width) or refined_height < max(5.0, 0.25 * height):
                continue
            diagnostics = self._airborne_refinement_diagnostics(
                image=image,
                roi=roi,
                roi_box=roi_box,
                mask=mask,
                geometry_bbox=bbox,
                final_bbox=final_bbox,
                components=components,
                scored=scored,
                selected=final_selected,
                filter_reject_reasons=filter_reject_reasons,
                debug_context=debug_context,
                final_bbox_source=final_source,
                bbox_before_filtering=bbox_before_filtering,
                bbox_after_filtering=bbox_after_filtering,
            )
            return RefinedBBox(
                bbox_xyxy_px=final_bbox,
                mask_area_px=sum(component[4] for component in final_selected),
                method=final_source,
                confidence="visible_pixels" if final_selected else "geometry_fallback",
                reasons=reasons,
                diagnostics=diagnostics,
            )
        rejected_components = self._airborne_component_diagnostics(
            components,
            roi_box,
            bbox,
            score_by_id={id(component): score for score, component in scored},
            selected_ids=set(),
            filter_reject_reasons=filter_reject_reasons,
            default_reject_reason="airborne_rejected_candidate_size_gate",
        )
        self._record_airborne_failure_debug(
            image=image,
            bbox=bbox,
            debug_context=debug_context,
            reason="airborne_candidate_size_gate_failed",
            roi=roi,
            roi_box=roi_box,
            mask=mask,
            selected_components=[],
            rejected_components=rejected_components,
        )
        return None

    def _select_airborne_components_legacy(
        self,
        seed: Tuple[int, int, int, int, int, np.ndarray],
        scored: Sequence[Tuple[float, Tuple[int, int, int, int, int, np.ndarray]]],
    ) -> List[Tuple[int, int, int, int, int, np.ndarray]]:
        seed_center = seed[5]
        return [
            component
            for _score, component in scored
            if np.linalg.norm(component[5] - seed_center) < max(80.0, 3.0 * max(seed[2], seed[3]))
        ]

    def _select_airborne_components_geometry_filtered(
        self,
        seed: Tuple[int, int, int, int, int, np.ndarray],
        scored: Sequence[Tuple[float, Tuple[int, int, int, int, int, np.ndarray]]],
        geometry_bbox: Sequence[float],
        roi_box: Tuple[int, int, int, int],
    ) -> Tuple[List[Tuple[int, int, int, int, int, np.ndarray]], List[Tuple[Tuple[int, int, int, int, int, np.ndarray], List[str]]]]:
        selected = [seed]
        rejected: List[Tuple[Tuple[int, int, int, int, int, np.ndarray], List[str]]] = []
        seed_relation = self._airborne_component_relation(seed, roi_box, geometry_bbox)
        seed_has_overlap = seed_relation["overlap_area"] > 0.0 or seed_relation["overlap_ratio_component"] >= 0.10
        geometry_limit = self._airborne_geometry_distance_limit(geometry_bbox)
        seed_center = seed[5]
        close_to_seed_limit = max(12.0, 0.60 * max(float(geometry_bbox[2]) - float(geometry_bbox[0]), float(geometry_bbox[3]) - float(geometry_bbox[1])))

        for _score, component in scored:
            if component is seed:
                continue
            relation = self._airborne_component_relation(component, roi_box, geometry_bbox)
            reasons = self._airborne_geometry_reject_reasons(relation, geometry_bbox)
            has_overlap = relation["overlap_area"] > 0.0 or relation["overlap_ratio_component"] >= 0.10
            if seed_has_overlap and not has_overlap:
                reasons.extend(
                    [
                        "rejected_zero_geometry_overlap",
                        "rejected_would_merge_neighbor_airborne_object",
                    ]
                )
            seed_distance = float(np.linalg.norm(component[5] - seed_center))
            close_to_seed = seed_distance <= close_to_seed_limit
            close_to_geometry = relation["center_distance"] <= geometry_limit
            if reasons:
                rejected.append((component, sorted(set(reasons))))
                continue
            if has_overlap or (close_to_seed and close_to_geometry):
                selected.append(component)
            else:
                rejected.append(
                    (
                        component,
                        [
                            "rejected_far_from_geometry_center",
                            "rejected_would_merge_neighbor_airborne_object",
                        ],
                    )
                )
        return selected, rejected

    def _airborne_component_relation(
        self,
        component: Tuple[int, int, int, int, int, np.ndarray],
        roi_box: Tuple[int, int, int, int],
        geometry_bbox: Sequence[float],
    ) -> Dict[str, Any]:
        roi_x1, roi_y1, _roi_x2, _roi_y2 = roi_box
        x, y, width, height, _area, centroid = component
        bbox = [float(roi_x1 + x), float(roi_y1 + y), float(roi_x1 + x + width), float(roi_y1 + y + height)]
        component_center = _bbox_center(bbox)
        centroid_px = [float(roi_x1 + centroid[0]), float(roi_y1 + centroid[1])]
        geometry_center = _bbox_center(geometry_bbox)
        component_area = max(1.0, _bbox_area(bbox))
        geometry_area = max(1.0, _bbox_area(geometry_bbox))
        overlap_area = _bbox_overlap_area(bbox, geometry_bbox)
        return {
            "bbox": bbox,
            "component_center": component_center,
            "centroid_px": centroid_px,
            "geometry_center": geometry_center,
            "component_area": component_area,
            "geometry_area": geometry_area,
            "overlap_area": float(overlap_area),
            "overlap_ratio_component": float(overlap_area / component_area),
            "overlap_ratio_geometry": float(overlap_area / geometry_area),
            "center_distance": float(math.hypot(component_center[0] - geometry_center[0], component_center[1] - geometry_center[1])),
            "centroid_distance": float(math.hypot(centroid_px[0] - geometry_center[0], centroid_px[1] - geometry_center[1])),
        }

    def _airborne_geometry_distance_limit(self, geometry_bbox: Sequence[float]) -> float:
        geometry_width = max(1.0, float(geometry_bbox[2]) - float(geometry_bbox[0]))
        geometry_height = max(1.0, float(geometry_bbox[3]) - float(geometry_bbox[1]))
        return max(8.0, 0.45 * max(geometry_width, geometry_height))

    def _airborne_geometry_reject_reasons(self, relation: Dict[str, Any], geometry_bbox: Sequence[float]) -> List[str]:
        reasons: List[str] = []
        geometry_limit = self._airborne_geometry_distance_limit(geometry_bbox)
        if relation["overlap_area"] <= 0.0 and relation["overlap_ratio_component"] < 0.10:
            if relation["center_distance"] > geometry_limit:
                reasons.extend(
                    [
                        "rejected_zero_geometry_overlap",
                        "rejected_zero_geometry_overlap_far_component",
                        "rejected_far_from_geometry_center",
                        "rejected_would_merge_neighbor_airborne_object",
                    ]
                )
        return sorted(set(reasons))

    def _airborne_component_geometry_score(
        self,
        component: Tuple[int, int, int, int, int, np.ndarray],
        relation: Dict[str, Any],
    ) -> float:
        _x, _y, width, height, area, _centroid = component
        density = float(area) / max(1.0, float(width * height))
        distance_penalty = float(area) * (relation["center_distance"] / max(1.0, 0.5 * math.sqrt(relation["geometry_area"])))
        zero_overlap_penalty = float(area) * 2.0 if relation["overlap_area"] <= 0.0 else 0.0
        overlap_bonus = relation["overlap_area"] * 4.0 + relation["overlap_ratio_component"] * float(area) * 2.0
        return float(area) * (0.4 + density) + overlap_bonus - distance_penalty - zero_overlap_penalty

    def _airborne_bbox_excessive(self, candidate_bbox: Sequence[float], geometry_bbox: Sequence[float]) -> bool:
        geometry_width = max(1.0, float(geometry_bbox[2]) - float(geometry_bbox[0]))
        geometry_height = max(1.0, float(geometry_bbox[3]) - float(geometry_bbox[1]))
        candidate_width = max(1.0, float(candidate_bbox[2]) - float(candidate_bbox[0]))
        candidate_height = max(1.0, float(candidate_bbox[3]) - float(candidate_bbox[1]))
        geometry_center = _bbox_center(geometry_bbox)
        candidate_center = _bbox_center(candidate_bbox)
        center_shift = math.hypot(candidate_center[0] - geometry_center[0], candidate_center[1] - geometry_center[1])
        return (
            candidate_width / geometry_width > 1.8
            or candidate_height / geometry_height > 1.8
            or center_shift > max(20.0, 0.75 * max(geometry_width, geometry_height))
        )

    def _airborne_refinement_diagnostics(
        self,
        image: np.ndarray,
        roi: np.ndarray,
        roi_box: Tuple[int, int, int, int],
        mask: np.ndarray,
        geometry_bbox: Sequence[float],
        final_bbox: Sequence[float],
        components: Sequence[Tuple[int, int, int, int, int, np.ndarray]],
        scored: Sequence[Tuple[float, Tuple[int, int, int, int, int, np.ndarray]]],
        selected: Sequence[Tuple[int, int, int, int, int, np.ndarray]],
        filter_reject_reasons: Dict[int, List[str]],
        debug_context: Optional[Dict[str, Any]],
        final_bbox_source: str,
        bbox_before_filtering: Optional[Sequence[float]] = None,
        bbox_after_filtering: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        selected_ids = {id(component) for component in selected}
        score_by_id = {id(component): float(score) for score, component in scored}
        selected_components = self._airborne_component_diagnostics(
            selected,
            roi_box,
            geometry_bbox,
            score_by_id=score_by_id,
            selected_ids=selected_ids,
            filter_reject_reasons={},
        )
        rejected_components = self._airborne_component_diagnostics(
            components,
            roi_box,
            geometry_bbox,
            score_by_id=score_by_id,
            selected_ids=selected_ids,
            filter_reject_reasons=filter_reject_reasons,
            default_reject_reason="airborne_rejected_not_selected",
        )
        rejected_components = [component for component in rejected_components if not component.get("selected")]
        diagnostics = self._airborne_diagnostics_payload(
            roi_box=roi_box,
            geometry_bbox=geometry_bbox,
            final_bbox=final_bbox,
            components=components,
            selected_components=selected_components,
            rejected_components=rejected_components,
            final_bbox_source=final_bbox_source,
            bbox_before_filtering=bbox_before_filtering,
            bbox_after_filtering=bbox_after_filtering,
        )
        diagnostics["airborne_debug_artifacts"] = self._write_airborne_debug_artifacts(
            image=image,
            roi=roi,
            roi_box=roi_box,
            mask=mask,
            geometry_bbox=geometry_bbox,
            final_bbox=final_bbox,
            selected_components=selected_components,
            rejected_components=rejected_components,
            debug_context=debug_context,
            method=final_bbox_source,
        )
        return diagnostics

    def _record_airborne_failure_debug(
        self,
        image: np.ndarray,
        bbox: Sequence[float],
        debug_context: Optional[Dict[str, Any]],
        reason: str,
        roi: Optional[np.ndarray] = None,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        mask: Optional[np.ndarray] = None,
        selected_components: Optional[Sequence[Dict[str, Any]]] = None,
        rejected_components: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        if not debug_context:
            return
        if roi_box is None:
            roi_box = (
                int(max(0, math.floor(float(bbox[0])))),
                int(max(0, math.floor(float(bbox[1])))),
                int(min(image.shape[1], math.ceil(float(bbox[2])))),
                int(min(image.shape[0], math.ceil(float(bbox[3])))),
            )
        selected_components = list(selected_components or [])
        rejected_components = list(rejected_components or [])
        diagnostics = self._airborne_diagnostics_payload(
            roi_box=roi_box,
            geometry_bbox=bbox,
            final_bbox=None,
            components=[],
            selected_components=selected_components,
            rejected_components=rejected_components,
            final_bbox_source=reason,
        )
        diagnostics["airborne_debug_artifacts"] = self._write_airborne_debug_artifacts(
            image=image,
            roi=roi,
            roi_box=roi_box,
            mask=mask,
            geometry_bbox=bbox,
            final_bbox=None,
            selected_components=selected_components,
            rejected_components=rejected_components,
            debug_context=debug_context,
            method=reason,
        )
        debug_context["airborne_failure_diagnostics"] = diagnostics

    def _airborne_component_diagnostics(
        self,
        components: Sequence[Tuple[int, int, int, int, int, np.ndarray]],
        roi_box: Tuple[int, int, int, int],
        geometry_bbox: Sequence[float],
        score_by_id: Dict[int, float],
        selected_ids: set[int],
        filter_reject_reasons: Dict[int, List[str]],
        default_reject_reason: str = "airborne_rejected_not_selected",
    ) -> List[Dict[str, Any]]:
        diagnostics: List[Dict[str, Any]] = []
        for component in components:
            component_id = id(component)
            selected = component_id in selected_ids
            reject_reasons = [] if selected else list(filter_reject_reasons.get(component_id, [default_reject_reason]))
            diagnostics.append(
                self._airborne_component_diagnostic(
                    component,
                    roi_box,
                    geometry_bbox,
                    selected=selected,
                    score=score_by_id.get(component_id),
                    reject_reasons=reject_reasons,
                )
            )
        return diagnostics

    def _airborne_component_diagnostic(
        self,
        component: Tuple[int, int, int, int, int, np.ndarray],
        roi_box: Tuple[int, int, int, int],
        geometry_bbox: Sequence[float],
        selected: bool,
        score: Optional[float],
        reject_reasons: Sequence[str],
    ) -> Dict[str, Any]:
        roi_x1, roi_y1, _roi_x2, _roi_y2 = roi_box
        x, y, width, height, area, centroid = component
        relation = self._airborne_component_relation(component, roi_box, geometry_bbox)
        bbox = relation["bbox"]
        centroid_px = [float(roi_x1 + centroid[0]), float(roi_y1 + centroid[1])]
        return {
            "bbox_xyxy_px": bbox,
            "bbox_xyxy_roi_px": [int(x), int(y), int(x + width), int(y + height)],
            "component_bbox_center_px": [float(relation["component_center"][0]), float(relation["component_center"][1])],
            "centroid_px": centroid_px,
            "area_px": int(area),
            "width_px": int(width),
            "height_px": int(height),
            "score": score,
            "selected": selected,
            "overlap_area_with_geometry_px": float(relation["overlap_area"]),
            "overlap_ratio_with_component_bbox": float(relation["overlap_ratio_component"]),
            "overlap_ratio_with_geometry_bbox": float(relation["overlap_ratio_geometry"]),
            "center_distance_to_geometry_px": float(relation["center_distance"]),
            "centroid_distance_to_geometry_px": float(relation["centroid_distance"]),
            "reject_reasons": sorted(set(str(reason) for reason in reject_reasons)),
        }

    def _airborne_diagnostics_payload(
        self,
        roi_box: Tuple[int, int, int, int],
        geometry_bbox: Sequence[float],
        final_bbox: Optional[Sequence[float]],
        components: Sequence[Any],
        selected_components: Sequence[Dict[str, Any]],
        rejected_components: Sequence[Dict[str, Any]],
        final_bbox_source: str,
        bbox_before_filtering: Optional[Sequence[float]] = None,
        bbox_after_filtering: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        geometry_width = max(1.0, float(geometry_bbox[2]) - float(geometry_bbox[0]))
        geometry_height = max(1.0, float(geometry_bbox[3]) - float(geometry_bbox[1]))
        geometry_center = _bbox_center(geometry_bbox)
        before_bbox_list = [float(value) for value in bbox_before_filtering] if bbox_before_filtering is not None else None
        after_bbox_list = [float(value) for value in bbox_after_filtering] if bbox_after_filtering is not None else None

        def bbox_ratios(raw_bbox: Optional[Sequence[float]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
            if raw_bbox is None:
                return None, None, None
            bbox_width = max(1.0, float(raw_bbox[2]) - float(raw_bbox[0]))
            bbox_height = max(1.0, float(raw_bbox[3]) - float(raw_bbox[1]))
            bbox_center = _bbox_center(raw_bbox)
            center_shift = math.hypot(bbox_center[0] - geometry_center[0], bbox_center[1] - geometry_center[1])
            return float(bbox_width / geometry_width), float(bbox_height / geometry_height), float(center_shift)

        before_ratio_w, before_ratio_h, before_center_shift = bbox_ratios(bbox_before_filtering)
        after_ratio_w, after_ratio_h, after_center_shift = bbox_ratios(bbox_after_filtering)
        if final_bbox is None:
            final_bbox_list = None
            expansion_w = None
            expansion_h = None
            center_shift = None
            center_shift_magnitude = None
        else:
            final_bbox_list = [float(value) for value in final_bbox]
            final_width = max(1.0, float(final_bbox[2]) - float(final_bbox[0]))
            final_height = max(1.0, float(final_bbox[3]) - float(final_bbox[1]))
            final_center = _bbox_center(final_bbox)
            center_shift = [float(final_center[0] - geometry_center[0]), float(final_center[1] - geometry_center[1])]
            center_shift_magnitude = float(math.hypot(center_shift[0], center_shift[1]))
            expansion_w = float(final_width / geometry_width)
            expansion_h = float(final_height / geometry_height)
        reject_reasons = sorted(
            {
                str(reason)
                for component in rejected_components
                for reason in component.get("reject_reasons", [])
            }
        )
        return {
            "airborne_roi_xyxy_px": [float(value) for value in roi_box],
            "airborne_geometry_bbox_xyxy_px": [float(value) for value in geometry_bbox],
            "airborne_final_bbox_xyxy_px": final_bbox_list,
            "airborne_component_count": int(len(components) or len(selected_components) + len(rejected_components)),
            "airborne_selected_component_bbox_xyxy_px": self._components_debug_union_bbox(selected_components),
            "airborne_selected_component_area_px": int(sum(int(component.get("area_px", 0)) for component in selected_components)),
            "airborne_rejected_component_count": int(len(rejected_components)),
            "airborne_reject_reasons": reject_reasons,
            "airborne_expansion_ratio_w": expansion_w,
            "airborne_expansion_ratio_h": expansion_h,
            "airborne_center_shift_px": center_shift,
            "airborne_center_shift_magnitude_px": center_shift_magnitude,
            "airborne_final_bbox_source": final_bbox_source,
            "airborne_bbox_before_filtering_xyxy_px": before_bbox_list,
            "airborne_bbox_after_filtering_xyxy_px": after_bbox_list,
            "airborne_expansion_ratio_w_before_filtering": before_ratio_w,
            "airborne_expansion_ratio_h_before_filtering": before_ratio_h,
            "airborne_center_shift_magnitude_px_before_filtering": before_center_shift,
            "airborne_expansion_ratio_w_after_filtering": after_ratio_w,
            "airborne_expansion_ratio_h_after_filtering": after_ratio_h,
            "airborne_center_shift_magnitude_px_after_filtering": after_center_shift,
            "airborne_selected_components": self._limit_component_debug(selected_components),
            "airborne_rejected_components": self._limit_component_debug(rejected_components),
        }

    def _write_airborne_debug_artifacts(
        self,
        image: np.ndarray,
        roi: Optional[np.ndarray],
        roi_box: Tuple[int, int, int, int],
        mask: Optional[np.ndarray],
        geometry_bbox: Sequence[float],
        final_bbox: Optional[Sequence[float]],
        selected_components: Sequence[Dict[str, Any]],
        rejected_components: Sequence[Dict[str, Any]],
        debug_context: Optional[Dict[str, Any]],
        method: str,
    ) -> Dict[str, str]:
        if not debug_context:
            return {}
        try:
            debug_dir = Path(debug_context["debug_dir"])
            debug_dir.mkdir(parents=True, exist_ok=True)
            prefix = self._airborne_debug_prefix(debug_context, method)
            artifacts: Dict[str, str] = {}
            roi_bbox = [roi_box[0], roi_box[1], roi_box[2], roi_box[3]]

            overlay = image.copy()
            self._draw_debug_bbox(overlay, roi_bbox, (255, 0, 255), "search")
            self._draw_debug_bbox(overlay, geometry_bbox, (0, 255, 255), "geom")
            if final_bbox is not None:
                self._draw_debug_bbox(overlay, final_bbox, (0, 255, 0), "final")
            for component in self._limit_component_debug(rejected_components, limit=80):
                self._draw_debug_bbox(overlay, component["bbox_xyxy_px"], (0, 128, 255), "rej")
            for component in self._limit_component_debug(selected_components, limit=80):
                self._draw_debug_bbox(overlay, component["bbox_xyxy_px"], (255, 255, 0), "sel")
            self._draw_airborne_projected_points(overlay, debug_context, offset_xy=(0, 0))
            overlay_path = debug_dir / f"{prefix}__overlay.png"
            cv2.imwrite(str(overlay_path), overlay)  # type: ignore[union-attr]
            artifacts["overlay_png"] = str(overlay_path)

            if roi is not None:
                roi_overlay = roi.copy()
                roi_x1, roi_y1, _roi_x2, _roi_y2 = roi_box
                self._draw_debug_bbox(
                    roi_overlay,
                    [geometry_bbox[0] - roi_x1, geometry_bbox[1] - roi_y1, geometry_bbox[2] - roi_x1, geometry_bbox[3] - roi_y1],
                    (0, 255, 255),
                    "geom",
                )
                if final_bbox is not None:
                    self._draw_debug_bbox(
                        roi_overlay,
                        [final_bbox[0] - roi_x1, final_bbox[1] - roi_y1, final_bbox[2] - roi_x1, final_bbox[3] - roi_y1],
                        (0, 255, 0),
                        "final",
                    )
                for component in self._limit_component_debug(rejected_components, limit=80):
                    bbox = component["bbox_xyxy_px"]
                    self._draw_debug_bbox(
                        roi_overlay,
                        [bbox[0] - roi_x1, bbox[1] - roi_y1, bbox[2] - roi_x1, bbox[3] - roi_y1],
                        (0, 128, 255),
                        "rej",
                    )
                for component in self._limit_component_debug(selected_components, limit=80):
                    bbox = component["bbox_xyxy_px"]
                    self._draw_debug_bbox(
                        roi_overlay,
                        [bbox[0] - roi_x1, bbox[1] - roi_y1, bbox[2] - roi_x1, bbox[3] - roi_y1],
                        (255, 255, 0),
                        "sel",
                    )
                self._draw_airborne_projected_points(roi_overlay, debug_context, offset_xy=(roi_x1, roi_y1))
                roi_overlay_path = debug_dir / f"{prefix}__roi_overlay.png"
                cv2.imwrite(str(roi_overlay_path), roi_overlay)  # type: ignore[union-attr]
                artifacts["roi_overlay_png"] = str(roi_overlay_path)

                if mask is not None:
                    mask_path = debug_dir / f"{prefix}__mask.png"
                    cv2.imwrite(str(mask_path), mask.astype("uint8") * 255)  # type: ignore[union-attr]
                    artifacts["mask_png"] = str(mask_path)

                if selected_components or rejected_components:
                    component_overlay = roi.copy()
                    for component in self._limit_component_debug(rejected_components, limit=120):
                        bbox = component["bbox_xyxy_px"]
                        self._draw_debug_bbox(
                            component_overlay,
                            [bbox[0] - roi_x1, bbox[1] - roi_y1, bbox[2] - roi_x1, bbox[3] - roi_y1],
                            (0, 0, 255),
                            "rej",
                        )
                    for component in self._limit_component_debug(selected_components, limit=120):
                        bbox = component["bbox_xyxy_px"]
                        self._draw_debug_bbox(
                            component_overlay,
                            [bbox[0] - roi_x1, bbox[1] - roi_y1, bbox[2] - roi_x1, bbox[3] - roi_y1],
                            (255, 255, 0),
                            "sel",
                        )
                    component_overlay_path = debug_dir / f"{prefix}__component_overlay.png"
                    cv2.imwrite(str(component_overlay_path), component_overlay)  # type: ignore[union-attr]
                    artifacts["component_overlay_png"] = str(component_overlay_path)
            return artifacts
        except Exception as exc:  # pragma: no cover - best-effort diagnostics
            return {"error": str(exc)}

    def _airborne_debug_prefix(self, debug_context: Dict[str, Any], method: str) -> str:
        frame_id = self._safe_debug_token(debug_context.get("frame_id", "frame"))
        object_id = self._safe_debug_token(debug_context.get("object_id", "object"))
        class_name = self._safe_debug_token(debug_context.get("class_name", "airborne"))
        type_name = self._safe_debug_token(debug_context.get("type_name", "unknown"))
        method_name = self._safe_debug_token(method)
        return f"{frame_id}__{object_id}__{class_name}__{type_name}__{method_name}"

    def _draw_airborne_projected_points(
        self,
        image: np.ndarray,
        debug_context: Dict[str, Any],
        offset_xy: Tuple[int, int],
    ) -> None:
        offset_x, offset_y = offset_xy
        for point in debug_context.get("projected_points_px", []):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            px = int(round(float(point[0]) - float(offset_x)))
            py = int(round(float(point[1]) - float(offset_y)))
            if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                cv2.circle(image, (px, py), 2, (0, 0, 255), -1)  # type: ignore[union-attr]

