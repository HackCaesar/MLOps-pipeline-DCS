"""Camera basis math and snapshot → ``CameraPose``/``CameraIntrinsics`` builder.

This module owns the small pieces of math that translate between DCS-native
basis formats (``world_basis_raw.axis_x/y/z``, HPB euler angles, classic
``forward/up/right``) and the projection module's ``CameraPose``.

All helpers are pure and side-effect free. ``build_camera_context`` is the
single non-pure-ish entry point that the orchestrator uses to build a full
``camera`` metadata block out of a raw DCS snapshot.

Axis conventions match those documented in ``projection/projector.py``:

- World ``X = north``, ``Y = up``, ``Z = east``.
- DCS raw basis: ``axis_x → forward``, ``axis_y → up``, ``axis_z → right``.
- HPB: heading is clockwise from north around world-up. Pitch and bank are
  applied in the original order (pitch around the local right axis, bank
  around the local forward axis).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from projection import CameraIntrinsics, CameraPose

# ---- vector primitives -----------------------------------------------------


def _normalize_vector(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(array)
    if norm == 0.0:
        raise ValueError("Zero-length vector")
    return array / norm


def rotate_vector_around_axis(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation of ``vector`` around the unit ``axis`` by ``angle_rad``."""
    axis = axis / np.linalg.norm(axis)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return (
        vector * cos_a
        + np.cross(axis, vector) * sin_a
        + axis * np.dot(axis, vector) * (1.0 - cos_a)
    )


# ---- basis conversions -----------------------------------------------------


def basis_to_euler_deg(forward: np.ndarray, right: np.ndarray) -> Dict[str, float]:
    """Approximate yaw/pitch/roll in degrees from a ``forward/right`` pair.

    Yaw is taken from forward's horizontal projection, pitch from its
    vertical component, roll from the right vector's vertical component.
    Used only for human-readable ``camera.euler_deg`` in the metadata.
    """
    yaw = math.degrees(math.atan2(float(forward[2]), float(forward[0])))
    pitch = math.degrees(math.atan2(float(forward[1]), float(np.linalg.norm(forward[[0, 2]]))))
    roll = math.degrees(math.atan2(float(right[1]), 1.0))
    return {"yaw": yaw, "pitch": pitch, "roll": roll}


def basis_from_hpb(heading_rad: float, pitch_rad: float, bank_rad: float) -> Dict[str, Dict[str, float]]:
    """Construct a ``forward/up/right`` basis dict from DCS HPB angles.

    Pitch is applied with a negated angle because DCS positive pitch points
    the nose **down** in this convention. Bank is applied around the forward
    axis. Only applied when the absolute value exceeds ``1e-9`` to keep the
    no-rotation case bit-stable.
    """
    forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Heading in DCS is clockwise from north around world-up.
    forward = _normalize_vector(np.array([math.cos(heading_rad), 0.0, math.sin(heading_rad)], dtype=np.float64))
    right = _normalize_vector(np.cross(forward, up))
    up = _normalize_vector(np.cross(right, forward))

    if abs(pitch_rad) > 1e-9:
        forward = _normalize_vector(rotate_vector_around_axis(forward, right, -pitch_rad))
        up = _normalize_vector(rotate_vector_around_axis(up, right, -pitch_rad))

    if abs(bank_rad) > 1e-9:
        right = _normalize_vector(rotate_vector_around_axis(right, forward, bank_rad))
        up = _normalize_vector(rotate_vector_around_axis(up, forward, bank_rad))

    return {
        "forward": {"x": float(forward[0]), "y": float(forward[1]), "z": float(forward[2])},
        "up": {"x": float(up[0]), "y": float(up[1]), "z": float(up[2])},
        "right": {"x": float(right[0]), "y": float(right[1]), "z": float(right[2])},
    }


def basis_from_snapshot_axes(raw_basis: Optional[Dict[str, Any]]) -> Optional[Dict[str, Dict[str, float]]]:
    """Translate a raw snapshot basis dict into a canonical ``forward/up/right``
    dict.

    Accepts two formats:

    - DCS-native ``{"axis_x": {x,y,z}, "axis_y": {x,y,z}, "axis_z": {x,y,z}}``
      where ``axis_x → forward``, ``axis_y → up``, ``axis_z → right``.
    - Pre-normalized ``{"forward": {x,y,z}, "up": ..., "right": ...}``.

    Returns ``None`` if ``raw_basis`` is ``None`` or in neither format.
    """
    if raw_basis is None:
        return None

    if all(key in raw_basis for key in ("axis_x", "axis_y", "axis_z")):
        return {
            "forward": {
                "x": float(raw_basis["axis_x"]["x"]),
                "y": float(raw_basis["axis_x"]["y"]),
                "z": float(raw_basis["axis_x"]["z"]),
            },
            "up": {
                "x": float(raw_basis["axis_y"]["x"]),
                "y": float(raw_basis["axis_y"]["y"]),
                "z": float(raw_basis["axis_y"]["z"]),
            },
            "right": {
                "x": float(raw_basis["axis_z"]["x"]),
                "y": float(raw_basis["axis_z"]["y"]),
                "z": float(raw_basis["axis_z"]["z"]),
            },
        }

    if all(key in raw_basis for key in ("forward", "up", "right")):
        return {
            "forward": {
                "x": float(raw_basis["forward"]["x"]),
                "y": float(raw_basis["forward"]["y"]),
                "z": float(raw_basis["forward"]["z"]),
            },
            "up": {
                "x": float(raw_basis["up"]["x"]),
                "y": float(raw_basis["up"]["y"]),
                "z": float(raw_basis["up"]["z"]),
            },
            "right": {
                "x": float(raw_basis["right"]["x"]),
                "y": float(raw_basis["right"]["y"]),
                "z": float(raw_basis["right"]["z"]),
            },
        }

    return None


# ---- snapshot → CameraPose + camera_state metadata block -------------------


def build_camera_context(
    snapshot_camera: Dict[str, Any],
    camera_config: Dict[str, Any],
) -> Tuple[CameraPose, CameraIntrinsics, Dict[str, Any]]:
    """Build ``(CameraPose, CameraIntrinsics, camera_state_dict)`` from a snapshot.

    ``camera_config`` is the ``camera:`` section of the YAML config; the only
    keys consulted are ``image.width/height``, ``fov.vertical_deg``,
    ``fov.source``, ``clip_planes_m.near/far``, ``height_reference``. No
    other side effects.

    Raises ``KeyError`` if the snapshot has no usable basis.
    """
    position = snapshot_camera["world_position_m"]

    raw_basis = snapshot_camera.get("world_basis_raw") or snapshot_camera.get("world_basis")
    basis = basis_from_snapshot_axes(raw_basis)
    if basis is None:
        raise KeyError("Camera basis missing or has unsupported format: expected world_basis_raw.axis_x/y/z")

    pose = CameraPose.from_basis(
        position_w=[position["x"], position["y"], position["z"]],
        forward_w=[basis["forward"]["x"], basis["forward"]["y"], basis["forward"]["z"]],
        up_w=[basis["up"]["x"], basis["up"]["y"], basis["up"]["z"]],
        right_w=[basis["right"]["x"], basis["right"]["y"], basis["right"]["z"]],
    )

    image_cfg = camera_config["image"]
    intrinsics = CameraIntrinsics.from_vertical_fov(
        width_px=int(image_cfg["width"]),
        height_px=int(image_cfg["height"]),
        vertical_fov_deg=float(camera_config["fov"]["vertical_deg"]),
        near_m=camera_config["clip_planes_m"]["near"],
        far_m=camera_config["clip_planes_m"]["far"],
        source=camera_config["fov"]["source"],
    )

    camera_state = {
        "height_reference": camera_config["height_reference"],
        "height_asl_m": float(position["y"]),
        "world_position_m": position,
        "world_basis": basis,
        "world_basis_raw": raw_basis,
        "euler_deg": basis_to_euler_deg(pose.forward_w, pose.right_w),
        "rotation_source": "derived_from_basis_raw" if raw_basis and "axis_x" in raw_basis else "derived_from_basis",
        "intrinsics": {
            "fx_px": intrinsics.fx_px,
            "fy_px": intrinsics.fy_px,
            "cx_px": intrinsics.cx_px,
            "cy_px": intrinsics.cy_px,
            "fov_vertical_deg": float(camera_config["fov"]["vertical_deg"]),
            "fov_horizontal_deg": math.degrees(
                2.0 * math.atan(image_cfg["width"] / (2.0 * intrinsics.fx_px))
            ),
            "source": camera_config["fov"]["source"],
        },
        "clip_planes_m": {
            "near": camera_config["clip_planes_m"]["near"],
            "far": camera_config["clip_planes_m"]["far"],
            "source": "configured_or_unknown",
        },
        "extrinsics_convention": "x_c=dot(right,p-cam), y_c=dot(up,p-cam), z_c=dot(forward,p-cam)",
    }
    return pose, intrinsics, camera_state
