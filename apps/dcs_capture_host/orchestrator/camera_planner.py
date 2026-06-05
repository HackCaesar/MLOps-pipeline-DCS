"""Camera-pose planning for one capture frame.

``CameraPlanner`` chooses a primary target from the snapshot and produces a
camera pose request that keeps the camera at exactly the configured ASL
height (15 m for the ship-camera task) with ``pitch ≈ 0``.

The planner intentionally evaluates several candidate distances (``150 m`` up
to ``1200 m``, capped by ``camera_to_target_range_m.max``) and several
candidate look-targets, then scores each by:

1. number of valid objects;
2. sum of estimated visible fraction;
3. sum of valid bbox areas;
4. proximity to the configured ``mvp_distance_to_primary_target_m``.

This module does **not** start DCS or talk to it. ``probe_scene_and_move_camera``
in ``orchestrator.main`` owns the full request/ack handshake.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from projection import CameraIntrinsics, CameraPose

from orchestrator.camera_context import basis_from_snapshot_axes
from orchestrator.metadata_builder import MetadataBuilder


def _normalize_vector(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(array)
    if norm == 0.0:
        raise ValueError("Zero-length vector")
    return array / norm


class CameraPlanner:
    """Plans a horizon-locked camera pose around the highest-priority target."""

    def __init__(
        self,
        config: Dict[str, Any],
        allowed_classes: Sequence[str],
        metadata_builder: MetadataBuilder,
    ) -> None:
        self.config = config
        self.allowed_classes = list(allowed_classes)
        self.metadata_builder = metadata_builder
        self._orbit_frame_index = 0

    def select_primary_target(self, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """First in-snapshot object that is in ``allowed_classes`` and has a
        world position. Used by the probe-only flow."""
        for raw_object in snapshot.get("objects", []):
            class_name = self.metadata_builder.infer_class(raw_object.get("type_name", ""))
            if class_name in self.allowed_classes and raw_object.get("world_position_m") is not None:
                return {
                    "id": raw_object.get("id"),
                    "type_name": raw_object.get("type_name"),
                    "class": class_name,
                    "world_position_m": raw_object.get("world_position_m"),
                }
        return None

    def plan_camera_pose(
        self,
        snapshot: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Return ``(pose_request, primary_target_summary)`` or ``(None, None)``.

        The pose request follows the DCS Lua camera-request schema:
        ``position`` + ``axis_x/y/z`` where ``axis_x`` is forward, ``axis_y``
        is up, ``axis_z`` is right.
        """
        primary_target = None
        class_priority = {"ships": 0, "helicopters": 1, "airplanes": 2}
        candidates: List[Tuple[int, Dict[str, Any]]] = []
        for raw_object in snapshot.get("objects", []):
            class_name = self.metadata_builder.infer_class(raw_object.get("type_name", ""))
            if class_name in self.allowed_classes and raw_object.get("world_position_m") is not None:
                candidates.append((class_priority.get(class_name, 99), raw_object))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            primary_target = candidates[0][1]

        if primary_target is None:
            return None, None

        target_position = primary_target["world_position_m"]

        raw_target_basis = primary_target.get("world_basis_raw") or primary_target.get("world_basis")
        target_basis = basis_from_snapshot_axes(raw_target_basis) or {
            "forward": {"x": 1.0, "y": 0.0, "z": 0.0}
        }
        forward = np.array(
            [
                target_basis["forward"]["x"],
                0.0,
                target_basis["forward"]["z"],
            ],
            dtype=np.float64,
        )
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        forward = _normalize_vector(forward)

        target_center = np.array(
            [target_position["x"], target_position["y"], target_position["z"]],
            dtype=np.float64,
        )
        candidate_centers = []
        for _, candidate in candidates:
            position = candidate.get("world_position_m")
            if position is not None:
                candidate_centers.append([position["x"], position["y"], position["z"]])
        if len(candidate_centers) > 1:
            default_look_target = np.mean(np.asarray(candidate_centers, dtype=np.float64), axis=0)
        else:
            default_look_target = target_center.copy()

        image_cfg = self.config["camera"]["image"]
        intrinsics = CameraIntrinsics.from_vertical_fov(
            width_px=int(image_cfg["width"]),
            height_px=int(image_cfg["height"]),
            vertical_fov_deg=float(self.config["camera"]["fov"]["vertical_deg"]),
            near_m=self.config["camera"]["clip_planes_m"]["near"],
            far_m=self.config["camera"]["clip_planes_m"]["far"],
            source=self.config["camera"]["fov"]["source"],
        )
        configured_distance = float(self.config["camera"]["mvp_distance_to_primary_target_m"])
        range_cfg = self.config.get("generation", {}).get("camera_to_target_range_m", {})
        min_configured_distance = float(range_cfg.get("min", 0.0))
        max_configured_distance = float(range_cfg.get("max", 3500.0))
        candidate_distances = {150.0, 170.0, 200.0, 240.0, 300.0, 500.0, 800.0, 1200.0, configured_distance}
        distance_options = sorted(
            distance for distance in candidate_distances
            if min_configured_distance <= distance <= max_configured_distance
        )
        if not distance_options:
            distance_options = [min(max(configured_distance, min_configured_distance), max_configured_distance)]
        look_target_options = [default_look_target]
        if len(candidate_centers) > 1:
            lower_target = default_look_target.copy()
            lower_target[1] = (default_look_target[1] + target_center[1]) / 2.0
            look_target_options.append(lower_target)
            ship_target = default_look_target.copy()
            ship_target[1] = target_center[1]
            look_target_options.append(ship_target)

        primary_target_summary = {
            "id": primary_target.get("id"),
            "type_name": primary_target.get("type_name"),
            "class": self.metadata_builder.infer_class(primary_target.get("type_name", "")),
            "world_position_m": target_position,
        }

        best_pose_request = None
        best_score: Tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        fixed_height_m = float(self.config["camera"]["fixed_height_m"])

        def evaluate_pose(
            camera_position: np.ndarray,
            look_target: np.ndarray,
        ) -> Tuple[Tuple[float, float, float, float], Dict[str, Any]]:
            view_direction = look_target - camera_position
            view_direction[1] = 0.0
            if np.linalg.norm(view_direction) < 1e-6:
                view_direction = forward.copy()
            view_forward = _normalize_vector(view_direction)
            right = _normalize_vector(np.cross(view_forward, world_up))
            up = world_up.copy()
            camera_pose = CameraPose.from_basis(
                position_w=camera_position,
                forward_w=view_forward,
                up_w=up,
                right_w=right,
            )
            object_metadata = []
            for _, candidate in candidates:
                try:
                    metadata = self.metadata_builder.build_object_metadata(
                        candidate, camera_pose, intrinsics
                    )
                except Exception:
                    metadata = None
                if metadata is not None:
                    object_metadata.append(metadata)
            valid_count = sum(1 for metadata in object_metadata if metadata["validation"]["valid"])
            visible_fraction = sum(
                metadata["visibility"].get("visible_fraction_est", 0.0) for metadata in object_metadata
            )
            valid_area = sum(
                metadata["projection"].get("bbox_area_px2", 0.0)
                for metadata in object_metadata
                if metadata["validation"]["valid"]
            )
            horizontal_delta = camera_position[[0, 2]] - target_center[[0, 2]]
            actual_distance = float(np.linalg.norm(horizontal_delta))
            distance_penalty = -abs(actual_distance - configured_distance)
            score = (float(valid_count), float(visible_fraction), float(valid_area), float(distance_penalty))
            pose_request = {
                "position": {
                    "x": float(camera_position[0]),
                    "y": float(camera_position[1]),
                    "z": float(camera_position[2]),
                },
                "axis_x": {
                    "x": float(view_forward[0]),
                    "y": float(view_forward[1]),
                    "z": float(view_forward[2]),
                },
                "axis_y": {
                    "x": float(up[0]),
                    "y": float(up[1]),
                    "z": float(up[2]),
                },
                "axis_z": {
                    "x": float(right[0]),
                    "y": float(right[1]),
                    "z": float(right[2]),
                },
            }
            return score, pose_request

        orbit_cfg = self.config.get("camera", {}).get("orbit_sequence", {})
        if orbit_cfg.get("enabled", False):
            distances = [float(value) for value in orbit_cfg.get("distances_m", [])]
            azimuths = [float(value) for value in orbit_cfg.get("azimuths_deg", [])]
            if distances and azimuths:
                # Anchor the orbit at the PRIMARY TARGET ONLY (not the average of
                # all candidate centres). Averaging dragged the camera hundreds of
                # km away whenever ships and aircraft lived in unrelated regions.
                sequence_index = self._orbit_frame_index
                self._orbit_frame_index += 1
                distance = max(
                    min_configured_distance,
                    min(max_configured_distance, distances[sequence_index % len(distances)]),
                )
                azimuth_rad = math.radians(azimuths[sequence_index % len(azimuths)])
                orbit_direction = _normalize_vector(
                    [math.cos(azimuth_rad), 0.0, math.sin(azimuth_rad)]
                )
                camera_position = target_center + orbit_direction * distance
                camera_position[1] = fixed_height_m
                # Look AT the primary target itself. Looking at the average of
                # remote clusters used to leave the primary off-screen.
                score, orbit_pose_request = evaluate_pose(camera_position, target_center)
                # ALWAYS honour the planned orbit pose. The old fallback path
                # below produced identical frames whenever the orbit candidate
                # found zero valid bboxes, which is exactly the bug we are fixing.
                return orbit_pose_request, primary_target_summary

        for distance in distance_options:
            camera_position = target_center - forward * distance
            camera_position[1] = fixed_height_m
            for look_target in look_target_options:
                score, pose_request = evaluate_pose(camera_position, look_target)
                if score > best_score:
                    best_score = score
                    best_pose_request = pose_request

        pose_request = best_pose_request
        return pose_request, primary_target_summary
