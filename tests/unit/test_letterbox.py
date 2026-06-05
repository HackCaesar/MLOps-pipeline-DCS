"""Unit tests for packages.common.letterbox — geometry + bbox round-trip."""
from __future__ import annotations

import pytest

from packages.common.letterbox import (
    LetterboxTransform,
    apply_letterbox,
    compute_letterbox,
    letterbox_box_to_source,
    source_box_to_letterbox,
)


def test_landscape_letterbox_geometry() -> None:
    t = compute_letterbox(2560, 1440, 640)
    assert t.scale == pytest.approx(640 / 2560)
    assert t.new_width == 640
    assert t.new_height == round(1440 * (640 / 2560))  # 360
    # vertical centering
    assert t.pad_y == (640 - t.new_height) // 2
    assert t.pad_x == 0


def test_portrait_letterbox_geometry() -> None:
    t = compute_letterbox(720, 1280, 640)
    assert t.scale == pytest.approx(640 / 1280)
    assert t.new_height == 640
    assert t.new_width == round(720 * (640 / 1280))  # 360
    assert t.pad_x == (640 - t.new_width) // 2
    assert t.pad_y == 0


def test_square_letterbox_identity() -> None:
    t = compute_letterbox(640, 640, 640)
    assert t.scale == 1.0
    assert t.pad_x == 0 and t.pad_y == 0
    assert t.new_width == 640 and t.new_height == 640


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        compute_letterbox(0, 100, 640)
    with pytest.raises(ValueError):
        compute_letterbox(100, 100, 0)


def test_box_roundtrip_source_letterbox_source() -> None:
    t = compute_letterbox(2560, 1440, 640)
    box = (1000.0, 500.0, 1500.0, 900.0)
    lb = source_box_to_letterbox(box, t)
    back = letterbox_box_to_source(lb, t)
    assert back == pytest.approx(box, abs=1e-9)


def test_box_projection_within_bounds() -> None:
    t = compute_letterbox(2560, 1440, 640)
    lb = source_box_to_letterbox((0, 0, 2560, 1440), t)
    assert lb[0] == pytest.approx(t.pad_x, abs=1e-9)
    assert lb[1] == pytest.approx(t.pad_y, abs=1e-9)
    assert lb[2] == pytest.approx(t.pad_x + t.new_width, abs=1e-9)
    assert lb[3] == pytest.approx(t.pad_y + t.new_height, abs=1e-9)


def test_apply_letterbox_produces_correct_canvas() -> None:
    import numpy as np
    img = np.full((1440, 2560, 3), fill_value=200, dtype=np.uint8)
    canvas, t = apply_letterbox(img, target_size=640, pad_value=114)
    assert canvas.shape == (640, 640, 3)
    # Inside the resized rectangle: original color 200.
    inside_region = canvas[t.pad_y + 1:t.pad_y + t.new_height - 1,
                           t.pad_x + 1:t.pad_x + t.new_width - 1]
    assert (inside_region == 200).all()
    # Pad regions: 114.
    if t.pad_y > 0:
        assert (canvas[0:t.pad_y, :] == 114).all()
    if t.pad_x > 0:
        assert (canvas[:, 0:t.pad_x] == 114).all()


def test_transform_dataclass_fields() -> None:
    t = LetterboxTransform(2560, 1440, 640, 0.25, 640, 360, 0, 140)
    assert t.src_width == 2560 and t.target_size == 640
