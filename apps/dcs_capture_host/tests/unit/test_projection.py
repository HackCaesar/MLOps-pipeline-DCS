"""Unit tests for projection helpers and end-to-end projection.

These cover the pure pieces extracted in Step 2 of the refactor:
corner orders, rotation matrix, depth mask, pinhole formula and bbox
extraction. They also pin down the end-to-end behavior of
``project_local_bbox`` / ``project_cuboid`` on simple synthetic inputs,
so any silent change in axis convention or projection sign is caught.

Runs with pytest (``pytest tests/unit/test_projection.py``) or directly
(``python tests/unit/test_projection.py``).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projection import (
    CameraIntrinsics,
    CameraPose,
    ObjectCuboid,
    ObjectLocalBBox,
    clip_bbox_to_image,
    project_cuboid,
    project_local_bbox,
    project_object_corners,
    project_points_world_to_image,
)
from projection.projector import (
    _apply_pinhole,
    _bbox_from_pixels,
    _centered_box_corners,
    _depth_mask,
    _local_bbox_corners,
    _local_to_world,
    _make_invisible_result,
    _rotation_from_axes,
)

# ---------- helper tests ----------------------------------------------------


def test_centered_box_corners_order_and_count() -> None:
    corners = _centered_box_corners(length_m=4.0, width_m=2.0, height_m=6.0)
    assert corners.shape == (8, 3)
    # Two CCW loops over z = -L/2 then z = +L/2.
    expected = np.array(
        [
            [-1.0, -3.0, -2.0],
            [1.0, -3.0, -2.0],
            [1.0, 3.0, -2.0],
            [-1.0, 3.0, -2.0],
            [-1.0, -3.0, 2.0],
            [1.0, -3.0, 2.0],
            [1.0, 3.0, 2.0],
            [-1.0, 3.0, 2.0],
        ]
    )
    assert np.allclose(corners, expected)


def test_local_bbox_corners_lexicographic_order() -> None:
    corners = _local_bbox_corners(np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0]))
    assert corners.shape == (8, 3)
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 3.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 3.0],
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 3.0],
        ]
    )
    assert np.allclose(corners, expected)


def test_rotation_from_identity_axes_is_identity() -> None:
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    rotation = _rotation_from_axes(x, y, z)
    assert np.allclose(rotation, np.eye(3))


def test_rotation_columns_map_local_axes_to_world() -> None:
    # 90-degree rotation around world Y axis: local +x (1,0,0) -> world +z (0,0,1).
    x_w = np.array([0.0, 0.0, 1.0])
    y_w = np.array([0.0, 1.0, 0.0])
    z_w = np.array([-1.0, 0.0, 0.0])
    rotation = _rotation_from_axes(x_w, y_w, z_w)
    assert np.allclose(rotation @ np.array([1.0, 0.0, 0.0]), x_w)
    assert np.allclose(rotation @ np.array([0.0, 1.0, 0.0]), y_w)
    assert np.allclose(rotation @ np.array([0.0, 0.0, 1.0]), z_w)


def test_local_to_world_translates_after_rotation() -> None:
    rotation = np.eye(3)
    origin = np.array([10.0, 20.0, 30.0])
    local = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 0.0]])
    world = _local_to_world(local, rotation, origin)
    assert np.allclose(world, np.array([[11.0, 22.0, 33.0], [9.0, 20.0, 30.0]]))


def test_depth_mask_zero_and_negative_filtered() -> None:
    z = np.array([-1.0, 0.0, 1e-8, 1.0, 100.0])
    mask = _depth_mask(z, near_m=None, far_m=None)
    assert mask.tolist() == [False, False, False, True, True]


def test_depth_mask_respects_far() -> None:
    z = np.array([0.5, 5.0, 50.0])
    mask = _depth_mask(z, near_m=1.0, far_m=10.0)
    assert mask.tolist() == [False, True, False]


def test_apply_pinhole_basic_principal_ray() -> None:
    intrinsics = CameraIntrinsics(
        width_px=200, height_px=100, fx_px=100.0, fy_px=100.0, cx_px=100.0, cy_px=50.0
    )
    cam_pts = np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [0.0, 1.0, 10.0]])
    valid = np.ones(3, dtype=bool)
    pixels = _apply_pinhole(cam_pts, intrinsics, valid)
    # Principal ray hits (cx, cy).
    assert np.allclose(pixels[0], [100.0, 50.0])
    # +x_c moves u to the right.
    assert np.allclose(pixels[1], [110.0, 50.0])
    # +y_c moves v UP (smaller v); minus sign is the key invariant.
    assert np.allclose(pixels[2], [100.0, 40.0])


def test_apply_pinhole_keeps_nan_where_invalid() -> None:
    intrinsics = CameraIntrinsics(
        width_px=200, height_px=100, fx_px=100.0, fy_px=100.0, cx_px=100.0, cy_px=50.0
    )
    cam_pts = np.array([[1.0, 1.0, -5.0], [1.0, 1.0, 10.0]])
    valid = np.array([False, True])
    pixels = _apply_pinhole(cam_pts, intrinsics, valid)
    assert math.isnan(pixels[0, 0]) and math.isnan(pixels[0, 1])
    assert not math.isnan(pixels[1, 0]) and not math.isnan(pixels[1, 1])


def test_bbox_from_pixels_min_max() -> None:
    pixels = np.array([[10.0, 5.0], [20.0, 8.0], [15.0, 7.0]])
    assert _bbox_from_pixels(pixels) == [10.0, 5.0, 20.0, 8.0]


def test_bbox_from_pixels_ignores_nan() -> None:
    pixels = np.array([[10.0, 5.0], [np.nan, np.nan], [15.0, 7.0]])
    assert _bbox_from_pixels(pixels) == [10.0, 5.0, 15.0, 7.0]


def test_make_invisible_result_shape() -> None:
    res = _make_invisible_result("obj-x", reasons=["all_points_behind_camera"])
    assert res.object_id == "obj-x"
    assert res.visible is False
    assert res.bbox_xyxy_px is None and res.bbox_xyxy_unclipped_px is None
    assert res.depth_range_m is None
    assert res.truncated is False
    assert res.reasons == ["all_points_behind_camera"]
    assert res.projected_points_px == []


def test_clip_bbox_inside_returns_unchanged() -> None:
    clipped, truncated = clip_bbox_to_image([10.0, 10.0, 50.0, 60.0], 200, 100)
    assert clipped == [10.0, 10.0, 50.0, 60.0]
    assert truncated is False


def test_clip_bbox_partially_out_marks_truncated() -> None:
    clipped, truncated = clip_bbox_to_image([-5.0, 50.0, 250.0, 80.0], 200, 100)
    assert clipped == [0.0, 50.0, 200.0, 80.0]
    assert truncated is True


def test_clip_bbox_fully_out_returns_none() -> None:
    clipped, truncated = clip_bbox_to_image([-50.0, -50.0, -10.0, -10.0], 200, 100)
    assert clipped is None and truncated is True


# ---------- end-to-end projection -------------------------------------------


def _identity_camera() -> CameraPose:
    return CameraPose.from_basis(
        position_w=[0.0, 0.0, 0.0],
        forward_w=[0.0, 0.0, 1.0],
        up_w=[0.0, 1.0, 0.0],
        right_w=[1.0, 0.0, 0.0],
    )


def _hd_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width_px=2560,
        height_px=1440,
        fx_px=1500.0,
        fy_px=1500.0,
        cx_px=1280.0,
        cy_px=720.0,
        near_m=None,
        far_m=None,
    )


def test_project_points_behind_camera_are_nan_and_masked() -> None:
    pose = _identity_camera()
    intr = _hd_intrinsics()
    pts = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 10.0]])
    pixels, valid = project_points_world_to_image(pose, intr, pts)
    assert valid.tolist() == [False, True]
    assert math.isnan(pixels[0, 0])
    assert np.allclose(pixels[1], [1280.0, 720.0])


def test_project_points_rejects_bad_shape() -> None:
    pose = _identity_camera()
    intr = _hd_intrinsics()
    try:
        project_points_world_to_image(pose, intr, np.array([1.0, 2.0, 3.0]))
    except ValueError as exc:
        assert "(N, 3)" in str(exc)
        return
    raise AssertionError("expected ValueError for non-(N,3) shape")


def test_project_local_bbox_center_in_front_of_camera() -> None:
    """A unit cube centered 10 m ahead, aligned with world axes, projects to a
    centered bbox roughly 150 px wide (fx/fy=1500 over depth ~10)."""
    pose = _identity_camera()
    intr = _hd_intrinsics()
    bbox = ObjectLocalBBox.from_dcs_basis(
        object_id="cube",
        class_name="ships",
        origin_w=[0.0, 0.0, 10.0],
        axis_x_w=[1.0, 0.0, 0.0],
        axis_y_w=[0.0, 1.0, 0.0],
        axis_z_w=[0.0, 0.0, 1.0],
        bbox_min_local=[-0.5, -0.5, -0.5],
        bbox_max_local=[0.5, 0.5, 0.5],
    )
    res = project_local_bbox(bbox, pose, intr)
    assert res.visible is True
    assert res.bbox_xyxy_px is not None
    # Nearest face at z=9.5, farthest at z=10.5.
    near_half = 1500.0 * 0.5 / 9.5
    assert math.isclose(res.bbox_xyxy_px[0], 1280.0 - near_half, abs_tol=1e-6)
    assert math.isclose(res.bbox_xyxy_px[2], 1280.0 + near_half, abs_tol=1e-6)
    # 8 corners, 8 projected entries.
    assert len(res.projected_points_px) == 8
    assert res.depth_range_m == [9.5, 10.5]
    assert res.truncated is False


def test_project_cuboid_matches_equivalent_local_bbox() -> None:
    """``ObjectCuboid`` (length=Z, width=X, height=Y) of size (2,2,2) centered at
    (0,0,10) must produce the same bbox extremes as a 2x2x2 ``ObjectLocalBBox``
    aligned with world axes — corner *order* differs between them, but bbox
    extremes are order-invariant."""
    pose = _identity_camera()
    intr = _hd_intrinsics()
    cuboid = ObjectCuboid.from_basis(
        object_id="c1",
        class_name="ships",
        center_w=[0.0, 0.0, 10.0],
        forward_w=[0.0, 0.0, 1.0],
        up_w=[0.0, 1.0, 0.0],
        right_w=[1.0, 0.0, 0.0],
        length_m=2.0,
        width_m=2.0,
        height_m=2.0,
    )
    local = ObjectLocalBBox.from_dcs_basis(
        object_id="b1",
        class_name="ships",
        origin_w=[0.0, 0.0, 10.0],
        axis_x_w=[1.0, 0.0, 0.0],
        axis_y_w=[0.0, 1.0, 0.0],
        axis_z_w=[0.0, 0.0, 1.0],
        bbox_min_local=[-1.0, -1.0, -1.0],
        bbox_max_local=[1.0, 1.0, 1.0],
    )
    r_cuboid = project_cuboid(cuboid, pose, intr)
    r_local = project_local_bbox(local, pose, intr)
    assert r_cuboid.bbox_xyxy_px is not None and r_local.bbox_xyxy_px is not None
    for a, b in zip(r_cuboid.bbox_xyxy_px, r_local.bbox_xyxy_px):
        assert math.isclose(a, b, abs_tol=1e-9)


def test_project_object_all_behind_returns_invisible() -> None:
    pose = _identity_camera()
    intr = _hd_intrinsics()
    behind = np.array(
        [[i % 2 - 0.5, i // 2 % 2 - 0.5, -1.0 - i] for i in range(8)],
        dtype=np.float64,
    )
    res = project_object_corners("o", behind, pose, intr)
    assert res.visible is False
    assert res.bbox_xyxy_px is None
    assert res.bbox_xyxy_unclipped_px is None
    assert res.depth_range_m is None
    assert res.projected_points_px == []
    assert "all_points_behind_camera" in res.reasons


def test_project_object_partial_behind_records_reason() -> None:
    pose = _identity_camera()
    intr = _hd_intrinsics()
    # 4 corners in front, 4 behind.
    corners = np.array(
        [
            [-0.5, -0.5, 10.0],
            [0.5, -0.5, 10.0],
            [0.5, 0.5, 10.0],
            [-0.5, 0.5, 10.0],
            [-0.5, -0.5, -1.0],
            [0.5, -0.5, -1.0],
            [0.5, 0.5, -1.0],
            [-0.5, 0.5, -1.0],
        ],
        dtype=np.float64,
    )
    res = project_object_corners("o", corners, pose, intr)
    assert res.visible is True
    assert "partial_geometry_behind_camera" in res.reasons


# ---------- runner ----------------------------------------------------------


if __name__ == "__main__":
    failures = 0
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL {name}: {e!r}")
        except Exception as e:
            failures += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)
