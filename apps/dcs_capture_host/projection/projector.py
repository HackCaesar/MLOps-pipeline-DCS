"""Projection utilities for DCS synthetic dataset generation.

Coordinate conventions used by this module:

- World frame ``W``:
  - ``X_w``: north
  - ``Y_w``: up
  - ``Z_w``: east
  - units: meters
- Camera frame ``C`` for CV projection:
  - ``x_c``: right
  - ``y_c``: up
  - ``z_c``: forward
  - units: meters

If DCS camera basis is provided as:

- ``forward_w = cam.x``
- ``up_w = cam.y``
- ``right_w = cam.z``

then the transform from world to camera is:

- ``p_rel = p_world - cam_position``
- ``x_c = dot(right_w, p_rel)``
- ``y_c = dot(up_w, p_rel)``
- ``z_c = dot(forward_w, p_rel)``

Pixel projection uses a pinhole camera:

- ``u = fx * (x_c / z_c) + cx``
- ``v = cy - fy * (y_c / z_c)``

The minus sign in ``v`` is required because image coordinates grow downward.

Module layout:

- Vector helpers: ``_as_np3``, ``_normalize``.
- Geometry helpers (pure, no IO): ``_rotation_from_axes``, ``_local_to_world``,
  ``_centered_box_corners``, ``_local_bbox_corners``.
- Pinhole helpers (pure): ``_depth_mask``, ``_apply_pinhole``,
  ``_bbox_from_pixels``, ``_make_invisible_result``.
- Dataclasses: ``CameraIntrinsics``, ``CameraPose``, ``ObjectCuboid``,
  ``ObjectLocalBBox``, ``ProjectionResult``.
- Public projection API: ``project_points_world_to_image``,
  ``clip_bbox_to_image``, ``project_object_corners``, ``project_cuboid``,
  ``project_local_bbox``.

Corner orders (do not reorder without an Algorithm Change Proposal: they end
up in ``ProjectionResult.projected_points_px`` and propagate to metadata):

- ``ObjectCuboid``: two CCW face loops at ``z = -L/2`` and ``z = +L/2`` in the
  order ``(-w,-h), (+w,-h), (+w,+h), (-w,+h)``.
- ``ObjectLocalBBox``: lexicographic over ``(x, y, z)`` from min to max.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

ArrayLike = Sequence[float]


def _as_np3(vector: ArrayLike) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"Expected shape (3,), got {array.shape}")
    return array


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("Zero-length vector cannot be normalized")
    return vector / norm


def _rotation_from_axes(
    axis_local_x_w: np.ndarray,
    axis_local_y_w: np.ndarray,
    axis_local_z_w: np.ndarray,
) -> np.ndarray:
    """Build a 3x3 rotation matrix whose columns are the world-space directions
    of the local X, Y, Z axes. Multiplying ``rotation @ p_local`` maps a point
    expressed in the object's local frame into world coordinates."""
    return np.column_stack([axis_local_x_w, axis_local_y_w, axis_local_z_w])


def _local_to_world(
    local_corners: np.ndarray,
    rotation: np.ndarray,
    origin_w: np.ndarray,
) -> np.ndarray:
    """Apply rotation + translation to ``(N, 3)`` local corners."""
    return (rotation @ local_corners.T).T + origin_w


def _centered_box_corners(length_m: float, width_m: float, height_m: float) -> np.ndarray:
    """Return the 8 corners of an axis-aligned box centered at the origin.

    The local frame is ``(x=width, y=height, z=length)``. The order is two CCW
    loops viewed from ``+z``: first at ``z = -L/2``, then at ``z = +L/2``.
    """
    half_length = length_m / 2.0
    half_width = width_m / 2.0
    half_height = height_m / 2.0
    return np.array(
        [
            [-half_width, -half_height, -half_length],
            [half_width, -half_height, -half_length],
            [half_width, half_height, -half_length],
            [-half_width, half_height, -half_length],
            [-half_width, -half_height, half_length],
            [half_width, -half_height, half_length],
            [half_width, half_height, half_length],
            [-half_width, half_height, half_length],
        ],
        dtype=np.float64,
    )


def _local_bbox_corners(bbox_min_local: np.ndarray, bbox_max_local: np.ndarray) -> np.ndarray:
    """Return the 8 corners of an axis-aligned bbox in lexicographic order
    over ``(x, y, z)`` from min to max."""
    min_x, min_y, min_z = bbox_min_local
    max_x, max_y, max_z = bbox_max_local
    return np.array(
        [
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ],
        dtype=np.float64,
    )


def _depth_mask(
    z_camera: np.ndarray,
    near_m: Optional[float],
    far_m: Optional[float],
) -> np.ndarray:
    """Boolean mask of points with usable forward depth ``z_c``.

    ``near_m`` defaults to ``1e-6`` (never accept zero or negative depth).
    ``far_m`` is enforced only when configured.
    """
    near = max(1e-6, near_m or 1e-6)
    mask = z_camera > near
    if far_m is not None:
        mask = mask & (z_camera < far_m)
    return mask


def _apply_pinhole(
    camera_points: np.ndarray,
    intrinsics: "CameraIntrinsics",
    valid_depth: np.ndarray,
) -> np.ndarray:
    """Project ``(N, 3)`` camera-space points to ``(N, 2)`` pixel coordinates.

    Entries where ``valid_depth`` is ``False`` remain ``NaN``. The pinhole
    formulas are exactly ``u = fx * x_c/z_c + cx`` and
    ``v = cy - fy * y_c/z_c``; the minus sign in ``v`` accounts for image
    pixels growing downward.
    """
    pixels = np.full((camera_points.shape[0], 2), np.nan, dtype=np.float64)
    x_c = camera_points[:, 0]
    y_c = camera_points[:, 1]
    z_c = camera_points[:, 2]
    pixels[valid_depth, 0] = (
        intrinsics.fx_px * (x_c[valid_depth] / z_c[valid_depth]) + intrinsics.cx_px
    )
    pixels[valid_depth, 1] = (
        intrinsics.cy_px - intrinsics.fy_px * (y_c[valid_depth] / z_c[valid_depth])
    )
    return pixels


def _bbox_from_pixels(valid_pixels: np.ndarray) -> List[float]:
    """``(M, 2)`` pixel array → ``[x_min, y_min, x_max, y_max]`` using
    ``nanmin``/``nanmax`` (so any residual NaN entries are ignored)."""
    return [
        float(np.nanmin(valid_pixels[:, 0])),
        float(np.nanmin(valid_pixels[:, 1])),
        float(np.nanmax(valid_pixels[:, 0])),
        float(np.nanmax(valid_pixels[:, 1])),
    ]


def _make_invisible_result(object_id: str, reasons: List[str]) -> "ProjectionResult":
    """Canonical 'object is not visible' result. Used when every projected
    point fell behind the near plane."""
    return ProjectionResult(
        object_id=object_id,
        visible=False,
        projected_points_px=[],
        bbox_xyxy_px=None,
        bbox_xyxy_unclipped_px=None,
        depth_range_m=None,
        truncated=False,
        reasons=reasons,
    )


@dataclass(frozen=True)
class CameraIntrinsics:
    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    near_m: Optional[float] = None
    far_m: Optional[float] = None
    source: str = "configured"

    @classmethod
    def from_vertical_fov(
        cls,
        width_px: int,
        height_px: int,
        vertical_fov_deg: float,
        near_m: Optional[float] = None,
        far_m: Optional[float] = None,
        source: str = "configured_fov",
    ) -> "CameraIntrinsics":
        vertical_fov_rad = math.radians(vertical_fov_deg)
        fy_px = height_px / (2.0 * math.tan(vertical_fov_rad / 2.0))
        # Square pixels imply the same focal length in pixel units for X and Y.
        # The horizontal FOV then changes with the image width, not with a scaled fx.
        fx_px = fy_px
        return cls(
            width_px=width_px,
            height_px=height_px,
            fx_px=fx_px,
            fy_px=fy_px,
            cx_px=width_px / 2.0,
            cy_px=height_px / 2.0,
            near_m=near_m,
            far_m=far_m,
            source=source,
        )


@dataclass(frozen=True)
class CameraPose:
    position_w: np.ndarray
    forward_w: np.ndarray
    up_w: np.ndarray
    right_w: np.ndarray

    @classmethod
    def from_basis(
        cls,
        position_w: ArrayLike,
        forward_w: ArrayLike,
        up_w: ArrayLike,
        right_w: ArrayLike,
    ) -> "CameraPose":
        return cls(
            position_w=_as_np3(position_w),
            forward_w=_normalize(_as_np3(forward_w)),
            up_w=_normalize(_as_np3(up_w)),
            right_w=_normalize(_as_np3(right_w)),
        )

    def world_to_camera(self, points_world: np.ndarray) -> np.ndarray:
        rel = points_world - self.position_w[None, :]
        x_c = rel @ self.right_w
        y_c = rel @ self.up_w
        z_c = rel @ self.forward_w
        return np.stack([x_c, y_c, z_c], axis=1)


@dataclass(frozen=True)
class ObjectCuboid:
    object_id: str
    class_name: str
    center_w: np.ndarray
    forward_w: np.ndarray
    up_w: np.ndarray
    right_w: np.ndarray
    length_m: float
    width_m: float
    height_m: float
    geometry_source: str = "approximate_dims"

    @classmethod
    def from_basis(
        cls,
        object_id: str,
        class_name: str,
        center_w: ArrayLike,
        forward_w: ArrayLike,
        up_w: ArrayLike,
        right_w: ArrayLike,
        length_m: float,
        width_m: float,
        height_m: float,
        geometry_source: str = "approximate_dims",
    ) -> "ObjectCuboid":
        return cls(
            object_id=object_id,
            class_name=class_name,
            center_w=_as_np3(center_w),
            forward_w=_normalize(_as_np3(forward_w)),
            up_w=_normalize(_as_np3(up_w)),
            right_w=_normalize(_as_np3(right_w)),
            length_m=float(length_m),
            width_m=float(width_m),
            height_m=float(height_m),
            geometry_source=geometry_source,
        )

    def corners_world(self) -> np.ndarray:
        local_corners = _centered_box_corners(self.length_m, self.width_m, self.height_m)
        rotation = _rotation_from_axes(self.right_w, self.up_w, self.forward_w)
        return _local_to_world(local_corners, rotation, self.center_w)


@dataclass(frozen=True)
class ObjectLocalBBox:
    object_id: str
    class_name: str
    origin_w: np.ndarray
    axis_x_w: np.ndarray
    axis_y_w: np.ndarray
    axis_z_w: np.ndarray
    bbox_min_local: np.ndarray
    bbox_max_local: np.ndarray
    geometry_source: str = "bbox_cache"

    @classmethod
    def from_dcs_basis(
        cls,
        object_id: str,
        class_name: str,
        origin_w: ArrayLike,
        axis_x_w: ArrayLike,
        axis_y_w: ArrayLike,
        axis_z_w: ArrayLike,
        bbox_min_local: ArrayLike,
        bbox_max_local: ArrayLike,
        geometry_source: str = "bbox_cache",
    ) -> "ObjectLocalBBox":
        return cls(
            object_id=object_id,
            class_name=class_name,
            origin_w=_as_np3(origin_w),
            axis_x_w=_normalize(_as_np3(axis_x_w)),
            axis_y_w=_normalize(_as_np3(axis_y_w)),
            axis_z_w=_normalize(_as_np3(axis_z_w)),
            bbox_min_local=_as_np3(bbox_min_local),
            bbox_max_local=_as_np3(bbox_max_local),
            geometry_source=geometry_source,
        )

    def corners_world(self) -> np.ndarray:
        local_corners = _local_bbox_corners(self.bbox_min_local, self.bbox_max_local)
        rotation = _rotation_from_axes(self.axis_x_w, self.axis_y_w, self.axis_z_w)
        return _local_to_world(local_corners, rotation, self.origin_w)


@dataclass
class ProjectionResult:
    object_id: str
    visible: bool
    projected_points_px: List[List[float]]
    bbox_xyxy_px: Optional[List[float]]
    bbox_xyxy_unclipped_px: Optional[List[float]]
    depth_range_m: Optional[List[float]]
    truncated: bool
    reasons: List[str]


def project_points_world_to_image(
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
    points_world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    points_world = np.asarray(points_world, dtype=np.float64)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(f"Expected points_world shape (N, 3), got {points_world.shape}")

    camera_points = pose.world_to_camera(points_world)
    valid_depth = _depth_mask(camera_points[:, 2], intrinsics.near_m, intrinsics.far_m)
    pixels = _apply_pinhole(camera_points, intrinsics, valid_depth)
    return pixels, valid_depth


def clip_bbox_to_image(
    bbox_xyxy_px: Sequence[float],
    width_px: int,
    height_px: int,
) -> Tuple[Optional[List[float]], bool]:
    x_min, y_min, x_max, y_max = bbox_xyxy_px
    clipped = [
        max(0.0, min(float(width_px), float(x_min))),
        max(0.0, min(float(height_px), float(y_min))),
        max(0.0, min(float(width_px), float(x_max))),
        max(0.0, min(float(height_px), float(y_max))),
    ]
    truncated = clipped != [float(x_min), float(y_min), float(x_max), float(y_max)]
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        return None, True
    return clipped, truncated


def project_object_corners(
    object_id: str,
    corners_world: np.ndarray,
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> ProjectionResult:
    pixels, valid_depth = project_points_world_to_image(pose, intrinsics, corners_world)

    reasons: List[str] = []
    if not np.any(valid_depth):
        reasons.append("all_points_behind_camera")
        return _make_invisible_result(object_id, reasons)

    valid_pixels = pixels[valid_depth]
    bbox_raw = _bbox_from_pixels(valid_pixels)

    bbox_clipped, truncated = clip_bbox_to_image(
        bbox_raw,
        width_px=intrinsics.width_px,
        height_px=intrinsics.height_px,
    )

    if truncated:
        reasons.append("bbox_clipped_to_image")
    if np.any(~valid_depth):
        reasons.append("partial_geometry_behind_camera")

    camera_points = pose.world_to_camera(corners_world)
    valid_depth_values = camera_points[valid_depth, 2]

    return ProjectionResult(
        object_id=object_id,
        visible=bbox_clipped is not None,
        projected_points_px=valid_pixels.tolist(),
        bbox_xyxy_px=bbox_clipped,
        bbox_xyxy_unclipped_px=bbox_raw,
        depth_range_m=[
            float(np.min(valid_depth_values)),
            float(np.max(valid_depth_values)),
        ],
        truncated=truncated,
        reasons=reasons,
    )


def project_cuboid(
    cuboid: ObjectCuboid,
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> ProjectionResult:
    return project_object_corners(
        object_id=cuboid.object_id,
        corners_world=cuboid.corners_world(),
        pose=pose,
        intrinsics=intrinsics,
    )


def project_local_bbox(
    bbox: ObjectLocalBBox,
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> ProjectionResult:
    return project_object_corners(
        object_id=bbox.object_id,
        corners_world=bbox.corners_world(),
        pose=pose,
        intrinsics=intrinsics,
    )
