"""Unit tests for packages.common.bbox_ops — boundary cases."""
from __future__ import annotations

import pytest

from packages.common.bbox_ops import (
    area_xyxy,
    clip_xyxy,
    coco_xywh_to_xyxy,
    from_yolo_normalized,
    intersect_xyxy,
    iou_xyxy,
    is_valid_box,
    project_xywh_to_tile,
    resize_xyxy,
    to_yolo_normalized,
    translate_xyxy,
    visible_ratio,
    xyxy_to_coco_xywh,
)

# ---- format conversions ------------------------------------------------

def test_coco_xywh_to_xyxy() -> None:
    assert coco_xywh_to_xyxy((10, 20, 30, 40)) == (10.0, 20.0, 40.0, 60.0)


def test_xyxy_to_coco_xywh() -> None:
    assert xyxy_to_coco_xywh((10, 20, 40, 60)) == (10.0, 20.0, 30.0, 40.0)


def test_roundtrip_coco_xyxy_coco() -> None:
    src = (1.5, 2.5, 3.25, 4.75)
    assert xyxy_to_coco_xywh(coco_xywh_to_xyxy(src)) == src


# ---- clip --------------------------------------------------------------

def test_clip_inside_unchanged() -> None:
    assert clip_xyxy((1, 2, 8, 9), 10, 10) == (1.0, 2.0, 8.0, 9.0)


def test_clip_clamps_negative_and_overflow() -> None:
    assert clip_xyxy((-5, -3, 12, 20), 10, 10) == (0.0, 0.0, 10.0, 10.0)


def test_clip_fully_outside_collapses_to_corner() -> None:
    out = clip_xyxy((20, 20, 30, 30), 10, 10)
    assert out == (10.0, 10.0, 10.0, 10.0)
    assert area_xyxy(out) == 0.0


# ---- area --------------------------------------------------------------

def test_area_positive() -> None:
    assert area_xyxy((0, 0, 10, 5)) == 50.0


def test_area_zero_on_collapsed() -> None:
    assert area_xyxy((5, 5, 5, 5)) == 0.0


def test_area_negative_clamped_to_zero() -> None:
    assert area_xyxy((10, 10, 5, 5)) == 0.0


# ---- resize / translate ------------------------------------------------

def test_resize_uniform() -> None:
    assert resize_xyxy((10, 20, 30, 40), 0.5, 0.5) == (5.0, 10.0, 15.0, 20.0)


def test_resize_non_uniform() -> None:
    assert resize_xyxy((10, 20, 30, 40), 0.5, 2.0) == (5.0, 40.0, 15.0, 80.0)


def test_translate() -> None:
    assert translate_xyxy((10, 20, 30, 40), -5, 5) == (5.0, 25.0, 25.0, 45.0)


# ---- intersect / iou ---------------------------------------------------

def test_intersect_overlap() -> None:
    assert intersect_xyxy((0, 0, 10, 10), (5, 5, 15, 15)) == (5.0, 5.0, 10.0, 10.0)


def test_intersect_disjoint_returns_none() -> None:
    assert intersect_xyxy((0, 0, 5, 5), (10, 10, 15, 15)) is None


def test_intersect_touch_only_returns_none() -> None:
    # Sharing an edge but no area → not an intersection.
    assert intersect_xyxy((0, 0, 5, 5), (5, 0, 10, 5)) is None


def test_iou_perfect() -> None:
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint() -> None:
    assert iou_xyxy((0, 0, 5, 5), (10, 10, 15, 15)) == 0.0


def test_iou_half_overlap() -> None:
    val = iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10))
    assert val == pytest.approx(50 / 150)


# ---- visible_ratio + validation ----------------------------------------

def test_visible_ratio_full_inside() -> None:
    assert visible_ratio((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_visible_ratio_partial() -> None:
    # 50% visible
    assert visible_ratio((0, 0, 10, 10), (0, 0, 5, 10)) == 0.5


def test_visible_ratio_zero_ref_area() -> None:
    assert visible_ratio((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


def test_visible_ratio_above_one_clamped() -> None:
    # If the "visible" box is somehow bigger than reference, clamp to 1.0.
    assert visible_ratio((0, 0, 5, 5), (0, 0, 10, 10)) == 1.0


def test_is_valid_box() -> None:
    assert is_valid_box((0, 0, 5, 5))
    assert not is_valid_box((0, 0, 5, 5), min_area=26.0)
    assert not is_valid_box((5, 5, 5, 5))


# ---- project_xywh_to_tile ----------------------------------------------

def test_project_keeps_bbox_fully_inside_tile() -> None:
    out = project_xywh_to_tile(
        (100.0, 100.0, 50.0, 50.0),
        resize_scale=(1.0, 1.0),
        tile_offset=(50, 50),
        tile_size=640,
        visible_threshold=0.80,
    )
    assert out is not None
    new_box, ratio = out
    assert ratio == 1.0
    # After translate by -50, the bbox starts at (50, 50) in tile coords.
    assert new_box == (50.0, 50.0, 50.0, 50.0)


def test_project_drops_when_visible_ratio_below_threshold() -> None:
    # bbox of width 100 starting at x=600, tile at x=[0..640] => visible width=40 → 40%
    out = project_xywh_to_tile(
        (600.0, 100.0, 100.0, 50.0),
        resize_scale=(1.0, 1.0),
        tile_offset=(0, 0),
        tile_size=640,
        visible_threshold=0.80,
    )
    assert out is None


def test_project_just_above_threshold_kept() -> None:
    # bbox visible portion is 81% of original
    out = project_xywh_to_tile(
        (560.0, 100.0, 100.0, 50.0),  # x=560..660; visible in tile [0..640] is 560..640 = 80px / 100 = 0.8
        resize_scale=(1.0, 1.0),
        tile_offset=(0, 0),
        tile_size=640,
        visible_threshold=0.80,
    )
    assert out is not None
    new_box, ratio = out
    assert ratio == pytest.approx(0.80)
    assert new_box == (560.0, 100.0, 80.0, 50.0)


def test_project_with_resize_scale() -> None:
    # Source at (200, 200, 100, 100); 0.5 resize → (100, 100, 50, 50) in scaled space.
    out = project_xywh_to_tile(
        (200.0, 200.0, 100.0, 100.0),
        resize_scale=(0.5, 0.5),
        tile_offset=(0, 0),
        tile_size=640,
        visible_threshold=0.80,
    )
    assert out is not None
    new_box, ratio = out
    assert ratio == 1.0
    assert new_box == (100.0, 100.0, 50.0, 50.0)


def test_project_disjoint_returns_none() -> None:
    out = project_xywh_to_tile(
        (1000.0, 1000.0, 10.0, 10.0),
        resize_scale=(1.0, 1.0),
        tile_offset=(0, 0),
        tile_size=640,
    )
    assert out is None


def test_project_invalid_tile_size_raises() -> None:
    with pytest.raises(ValueError):
        project_xywh_to_tile(
            (0, 0, 1, 1),
            resize_scale=(1, 1),
            tile_offset=(0, 0),
            tile_size=0,
        )


# ---- YOLO normalized round-trip ----------------------------------------

def test_yolo_norm_roundtrip() -> None:
    box = (100.0, 200.0, 300.0, 400.0)
    norm = to_yolo_normalized(box, 640, 480)
    back = from_yolo_normalized(norm, 640, 480)
    assert back == pytest.approx(box, abs=1e-9)


def test_yolo_centerpoint() -> None:
    norm = to_yolo_normalized((100, 100, 300, 300), 400, 400)
    cxn, cyn, wn, hn = norm
    assert cxn == 0.5 and cyn == 0.5
    assert wn == 0.5 and hn == 0.5
