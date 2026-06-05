"""Frame and object metadata construction.

This module owns the *schema* of ``metadata/<frame_id>.json``:

- top-level frame fields (``schema_version``, ``frame_id``, ``timestamp_utc``,
  ``usable``, ``invalid_reasons``, ``scene``, ``sync``, ``image``, ``camera``,
  ``object_stats``, ``objects``);
- per-object fields (``id``, ``class``, ``type_name``, ``quality_tier``,
  ``confidence_source``, ``usable``, ``world_*``, ``basis_source``,
  ``dimensions_*``, ``distance_to_camera_m``, ``visibility``, ``projection``,
  ``validation``);
- the rules that decide ``validation.valid`` / ``quality_tier`` /
  ``confidence_source``.

These shapes are a contract: anything reading ``dataset/metadata/*.json``,
the YOLO/COCO exporter, the visible-bbox refiner and any downstream training
code depends on them. Any change here that alters output is an
**Algorithm Change**, not a refactor.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from projection import (
    CameraIntrinsics,
    CameraPose,
    ObjectCuboid,
    ObjectLocalBBox,
    project_cuboid,
    project_local_bbox,
)

from orchestrator.camera_context import (
    basis_from_hpb,
    basis_from_snapshot_axes,
    build_camera_context,
)
from orchestrator.json_io import utc_now_iso


class MetadataBuilder:
    """Stateful builder that owns the metadata schema and validation rules.

    State: ``config`` (full YAML), ``allowed_classes`` (ordered list),
    ``model_dims`` (lowercased name → catalog dimensions dict). All other
    parameters are passed per-call so the builder is reusable between frames.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        allowed_classes: Sequence[str],
        model_dims: Dict[str, Dict[str, Any]],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.allowed_classes = list(allowed_classes)
        self.model_dims = model_dims
        self.logger = logger or logging.getLogger(__name__)

    # ---- class & dimensions inference -----------------------------------

    def infer_class(self, type_name: str) -> Optional[str]:
        name = (type_name or "").lower()
        if any(token in name for token in ["burke", "ticonderoga", "cg-47", "kilo", "ship", "frigate", "destroyer", "carrier"]):
            return "ships"
        if any(token in name for token in ["uh-1", "mi-8", "ka-50", "mi-24", "ah-64", "helicopter", "helo", "uh-60"]):
            return "helicopters"
        if any(token in name for token in ["f-16", "a-10", "su-25", "f/a-18", "aircraft", "plane", "mig-"]):
            return "airplanes"
        return None

    def lookup_dimensions(self, type_name: str, class_name: str) -> Tuple[Dict[str, float], str]:
        name = (type_name or "").lower()
        normalized_name = re.sub(r"[_\-]+", " ", name)
        if name in self.model_dims:
            return self.model_dims[name], "model_catalog"
        if normalized_name in self.model_dims:
            return self.model_dims[normalized_name], "model_catalog_normalized"
        for key, dims in self.model_dims.items():
            if (key in name or key in normalized_name) and key not in self.allowed_classes:
                return dims, "model_catalog_partial_match"
        if class_name in self.model_dims:
            return self.model_dims[class_name], "class_catalog"
        raise KeyError(f"No dimensions for type={type_name}, class={class_name}")

    # ---- projection validation ------------------------------------------

    def validate_projection(self, class_name: str, projection_payload: Dict[str, Any]) -> Dict[str, Any]:
        bbox = projection_payload.get("bbox_xyxy_px")
        raw_bbox = projection_payload.get("bbox_xyxy_unclipped_px")
        if not bbox or not raw_bbox:
            return {
                "valid": False,
                "reasons": ["bbox_missing"],
                "checks": {"bbox_exists": False},
                "quality_tier": "reject",
            }

        x_min, y_min, x_max, y_max = bbox
        raw_x_min, raw_y_min, raw_x_max, raw_y_max = raw_bbox
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        raw_area = max(1.0, (raw_x_max - raw_x_min) * (raw_y_max - raw_y_min))
        truncation_pct = max(0.0, 1.0 - (area / raw_area))

        validation_cfg = self.config["validation"]
        reasons: List[str] = []
        if width < float(validation_cfg["min_bbox_width_px"]):
            reasons.append("bbox_too_narrow")
        if height < float(validation_cfg["min_bbox_height_px"]):
            reasons.append("bbox_too_short")
        if area < float(validation_cfg["min_bbox_area_px2"]):
            reasons.append("bbox_area_too_small")
        if truncation_pct > float(validation_cfg["reject_if_truncation_pct_over"]):
            reasons.append("bbox_too_truncated")

        quality_tier = "strong"
        if class_name in {"helicopters", "airplanes"}:
            quality_tier = "approximate"
        if reasons:
            quality_tier = "reject"

        return {
            "valid": len(reasons) == 0,
            "reasons": reasons,
            "checks": {
                "bbox_exists": True,
                "min_size_pass": width >= float(validation_cfg["min_bbox_width_px"])
                and height >= float(validation_cfg["min_bbox_height_px"]),
                "area_pass": area >= float(validation_cfg["min_bbox_area_px2"]),
                "truncation_pass": truncation_pct
                <= float(validation_cfg["reject_if_truncation_pct_over"]),
            },
            "width_px": width,
            "height_px": height,
            "bbox_area_px2": area,
            "truncation_pct": truncation_pct,
            "occlusion_pct_est": 0.0,
            "visibility_pct": max(0.0, 1.0 - truncation_pct),
            "quality_tier": quality_tier,
        }

    # ---- per-object metadata -------------------------------------------

    def build_object_metadata(
        self,
        raw_object: Dict[str, Any],
        camera_pose: CameraPose,
        intrinsics: CameraIntrinsics,
    ) -> Optional[Dict[str, Any]]:
        type_name = raw_object.get("type_name", "unknown")
        class_name = self.infer_class(type_name)
        if class_name not in self.allowed_classes:
            return None

        position_dict = raw_object.get("world_position_m")
        if position_dict is None:
            return None

        center = np.array(
            [position_dict["x"], position_dict["y"], position_dict["z"]],
            dtype=np.float64,
        )

        raw_basis = raw_object.get("world_basis_raw") or raw_object.get("world_basis")
        basis = basis_from_snapshot_axes(raw_basis)

        if basis is not None:
            basis_source = "snapshot_basis_raw" if raw_basis and "axis_x" in raw_basis else "snapshot_basis"
        else:
            basis = basis_from_hpb(
                float(raw_object.get("heading_rad", 0.0)),
                float(raw_object.get("pitch_rad", 0.0)),
                float(raw_object.get("bank_rad", 0.0)),
            )
            basis_source = "heading_pitch_bank"

        basis_forward = np.array(
            [basis["forward"]["x"], basis["forward"]["y"], basis["forward"]["z"]],
            dtype=np.float64,
        )
        basis_up = np.array(
            [basis["up"]["x"], basis["up"]["y"], basis["up"]["z"]],
            dtype=np.float64,
        )
        basis_right = np.array(
            [basis["right"]["x"], basis["right"]["y"], basis["right"]["z"]],
            dtype=np.float64,
        )

        bbox_min_local = raw_object.get("bbox_min_local")
        bbox_max_local = raw_object.get("bbox_max_local")

        if bbox_min_local and bbox_max_local:
            local_min = np.array(
                [bbox_min_local["x"], bbox_min_local["y"], bbox_min_local["z"]],
                dtype=np.float64,
            )
            local_max = np.array(
                [bbox_max_local["x"], bbox_max_local["y"], bbox_max_local["z"]],
                dtype=np.float64,
            )

            geometry_dimensions = {
                "length": float(abs(local_max[0] - local_min[0])),
                "width": float(abs(local_max[2] - local_min[2])),
                "height": float(abs(local_max[1] - local_min[1])),
            }

            geometry_source = raw_object.get("geometry_source") or "bbox_cache"

            local_bbox = ObjectLocalBBox.from_dcs_basis(
                object_id=str(raw_object.get("id", "unknown")),
                class_name=class_name,
                origin_w=center,
                axis_x_w=basis_forward,
                axis_y_w=basis_up,
                axis_z_w=basis_right,
                bbox_min_local=local_min,
                bbox_max_local=local_max,
                geometry_source=geometry_source,
            )

            projected = project_local_bbox(local_bbox, camera_pose, intrinsics)

            projection = {
                "visible": projected.visible,
                "projected_points_px": projected.projected_points_px,
                "bbox_xyxy_px": projected.bbox_xyxy_px,
                "bbox_xyxy_unclipped_px": projected.bbox_xyxy_unclipped_px,
                "reasons": projected.reasons,
            }

        else:
            dimensions, dimensions_source = self.lookup_dimensions(type_name, class_name)
            geometry_dimensions = dimensions
            geometry_source = dimensions_source

            cuboid = ObjectCuboid.from_basis(
                object_id=str(raw_object.get("id", "unknown")),
                class_name=class_name,
                center_w=center,
                forward_w=basis_forward,
                up_w=basis_up,
                right_w=basis_right,
                length_m=float(dimensions["length"]),
                width_m=float(dimensions["width"]),
                height_m=float(dimensions["height"]),
                geometry_source=dimensions_source,
            )

            projected = project_cuboid(cuboid, camera_pose, intrinsics)

            projection = {
                "visible": projected.visible,
                "projected_points_px": projected.projected_points_px,
                "bbox_xyxy_px": projected.bbox_xyxy_px,
                "bbox_xyxy_unclipped_px": projected.bbox_xyxy_unclipped_px,
                "reasons": projected.reasons,
            }

        bbox_xyxy = projection["bbox_xyxy_px"]
        bbox_xywh = None
        bbox_area = 0.0
        center_px = None

        if bbox_xyxy is not None:
            bbox_xywh = [
                bbox_xyxy[0],
                bbox_xyxy[1],
                bbox_xyxy[2] - bbox_xyxy[0],
                bbox_xyxy[3] - bbox_xyxy[1],
            ]
            bbox_area = bbox_xywh[2] * bbox_xywh[3]
            center_px = [
                bbox_xyxy[0] + bbox_xywh[2] / 2.0,
                bbox_xyxy[1] + bbox_xywh[3] / 2.0,
            ]

        projection_payload = {
            "projected_points_px": projection["projected_points_px"],
            "bbox_xyxy_px": bbox_xyxy,
            "bbox_xyxy_unclipped_px": projection["bbox_xyxy_unclipped_px"],
            "bbox_xywh_px": bbox_xywh,
            "bbox_area_px2": bbox_area,
            "center_px": center_px,
            "projection_reasons": projection["reasons"],
        }

        validation = self.validate_projection(class_name, projection_payload)

        confidence_source = (
            geometry_source
            if basis_source in {"snapshot_basis_raw", "snapshot_basis", "heading_pitch_bank"}
            else "estimated"
        )

        quality_tier = validation["quality_tier"]

        if validation["valid"] and geometry_source in {"bbox_cache", "desc_box"}:
            quality_tier = "strong"
            confidence_source = geometry_source
        elif (
            validation["valid"]
            and geometry_source.startswith("model_catalog")
            and basis_source in {"snapshot_basis_raw", "snapshot_basis"}
        ):
            quality_tier = "strong"
        elif validation["valid"]:
            quality_tier = "approximate"

        validation["quality_tier"] = quality_tier

        return {
            "id": str(raw_object.get("id", "unknown")),
            "class": class_name,
            "type_name": type_name,
            "quality_tier": quality_tier,
            "confidence_source": confidence_source,
            "usable": validation["valid"],

            "world_position_m": position_dict,
            "world_basis": basis,
            "world_basis_raw": raw_basis,
            "basis_source": basis_source,
            "position_source": raw_object.get("position_source"),
            "object_pose_cache_match": raw_object.get("object_pose_cache_match"),

            "heading_pitch_bank_rad": {
                "heading": float(raw_object.get("heading_rad", 0.0)),
                "pitch": float(raw_object.get("pitch_rad", 0.0)),
                "bank": float(raw_object.get("bank_rad", 0.0)),
            },

            "dimensions_m": {
                "length": float(geometry_dimensions["length"]),
                "width": float(geometry_dimensions["width"]),
                "height": float(geometry_dimensions["height"]),
            },
            "dimensions_source": geometry_source,

            "distance_to_camera_m": float(np.linalg.norm(center - camera_pose.position_w)),

            "visibility": {
                "in_frustum": projection["visible"],
                "truncation_pct": validation.get("truncation_pct", 1.0),
                "occlusion_pct_est": validation.get("occlusion_pct_est", 0.0),
                "visible_fraction_est": validation.get("visibility_pct", 0.0),
            },

            "projection": projection_payload,
            "validation": validation,
        }

    # ---- frame-level helpers -------------------------------------------

    def camera_height_ok(self, camera_state: Dict[str, Any]) -> bool:
        return abs(float(camera_state["height_asl_m"]) - float(self.config["camera"]["fixed_height_m"])) <= 0.05

    def build_scene_context(
        self,
        frame_id: str,
        mission_path: Optional[Path],
        primary_target: Optional[Dict[str, Any]],
        camera_ack: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "scene_id": f"scene_{frame_id}",
            "mission_id": str(mission_path) if mission_path else "attach_mode",
            "seed": self.config["pipeline"]["seed"],
            "map_id": self.config["generation"]["map_id"],
            "sea_only": self.config["generation"]["sea_only"],
            "weather": "unknown",
            "time_of_day": "unknown",
            "sea_state": None,
            "scenario_mode": "mvp_single_frame",
            "primary_target": primary_target,
            "camera_request_status": camera_ack,
        }

    def build_sync_context(
        self,
        frame_token: str,
        snapshot: Dict[str, Any],
        screenshot_ack: Dict[str, Any],
        pause_ack: Optional[Dict[str, Any]],
        screenshot_diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        telemetry_token = snapshot.get("frame_token")
        screenshot_token = screenshot_ack.get("frame_token")
        pause_token = pause_ack.get("frame_token") if pause_ack else None
        pause_ack_valid = (
            pause_ack is None
            or (pause_token == frame_token and pause_ack.get("status") == "paused" and bool(pause_ack.get("paused")))
        )
        stale_screenshot_check_passed = bool(screenshot_diagnostics.get("stale_screenshot_check_passed"))
        strict_sync_valid = frame_token == telemetry_token == screenshot_token and pause_ack_valid and stale_screenshot_check_passed
        return {
            "frame_token": frame_token,
            "telemetry_token": telemetry_token,
            "screenshot_token": screenshot_token,
            "pause_token": pause_token,
            "screenshot_name": screenshot_ack.get("screenshot_name"),
            "strict_sync_valid": strict_sync_valid,
            "pause_first_enabled": pause_ack is not None,
            "pause_ack_status": pause_ack.get("status") if pause_ack else None,
            "pause_ack_stage": pause_ack.get("stage") if pause_ack else None,
            **screenshot_diagnostics,
            "sync_method": "pause_ack_then_token_match_snapshot_then_hook_screenshot"
            if pause_ack is not None
            else "token_match_with_ordered_snapshot_then_hook_screenshot",
            "delta_frames": 0,
            "delta_sim_time_s": 0.0,
        }

    # ---- top-level entry point -----------------------------------------

    def build_frame_metadata(
        self,
        frame_id: str,
        snapshot: Dict[str, Any],
        screenshot_ack: Dict[str, Any],
        mission_path: Optional[Path],
        primary_target: Optional[Dict[str, Any]],
        camera_ack: Optional[Dict[str, Any]],
        pause_ack: Optional[Dict[str, Any]],
        screenshot_diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        camera_pose, intrinsics, camera_state = build_camera_context(
            snapshot["camera"], self.config["camera"]
        )
        objects: List[Dict[str, Any]] = []
        for raw_object in snapshot.get("objects", []):
            try:
                metadata = self.build_object_metadata(raw_object, camera_pose, intrinsics)
            except Exception as exc:
                self.logger.warning("Object projection failed for %s: %s", raw_object.get("type_name"), exc)
                metadata = None
            if metadata is not None:
                objects.append(metadata)

        sync_context = self.build_sync_context(
            snapshot["frame_token"], snapshot, screenshot_ack, pause_ack, screenshot_diagnostics
        )
        scene_context = self.build_scene_context(frame_id, mission_path, primary_target, camera_ack)
        object_stats: Dict[str, Any] = {
            "total_objects": len(objects),
            "valid_objects": sum(1 for obj in objects if obj["validation"]["valid"]),
            "by_class": {},
            "valid_by_class": {},
            "by_dimensions_source": {},
            "by_quality_tier": {},
        }
        for obj in objects:
            class_name = obj["class"]
            dimensions_source = obj["dimensions_source"]
            quality_tier = obj["quality_tier"]
            object_stats["by_class"][class_name] = object_stats["by_class"].get(class_name, 0) + 1
            object_stats["by_dimensions_source"][dimensions_source] = object_stats["by_dimensions_source"].get(dimensions_source, 0) + 1
            object_stats["by_quality_tier"][quality_tier] = object_stats["by_quality_tier"].get(quality_tier, 0) + 1
            if obj["validation"]["valid"]:
                object_stats["valid_by_class"][class_name] = object_stats["valid_by_class"].get(class_name, 0) + 1

        image_cfg = self.config["camera"]["image"]
        usable = (
            sync_context["strict_sync_valid"]
            and self.camera_height_ok(camera_state)
            and any(obj["validation"]["valid"] for obj in objects)
        )
        invalid_reasons: List[str] = []
        if not sync_context["strict_sync_valid"]:
            invalid_reasons.append("sync_invalid")
        if not sync_context.get("stale_screenshot_check_passed", False):
            invalid_reasons.append("stale_or_missing_screenshot")
        if not self.camera_height_ok(camera_state):
            invalid_reasons.append("camera_height_not_15m")
        if not any(obj["validation"]["valid"] for obj in objects):
            invalid_reasons.append("no_valid_objects")

        return {
            "schema_version": "1.0",
            "frame_id": frame_id,
            "timestamp_utc": utc_now_iso(),
            "simulation_time_s": snapshot.get("simulation_time_s"),
            "usable": usable,
            "invalid_reasons": invalid_reasons,
            "scene": scene_context,
            "sync": sync_context,
            "image": {
                "file_name": f"{frame_id}.{self.config['export']['image_format']}",
                "width": int(image_cfg["width"]),
                "height": int(image_cfg["height"]),
                "format": self.config["export"]["image_format"],
            },
            "camera": camera_state,
            "object_stats": object_stats,
            "objects": objects,
        }
