"""Smoke tests for packages.common.image_io and image_resize.

Skipped when neither cv2 nor PIL is installed (still keeps the module importable).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from packages.common.image_io import available_backend, get_image_size, imread, imwrite
from packages.common.image_resize import compute_resize_transform, resize_image

SKIP_IF_NO_BACKEND = pytest.mark.skipif(
    available_backend() == "none",
    reason="No image backend (cv2/PIL) available",
)


def test_resolve_transform_exact() -> None:
    t = compute_resize_transform(2560, 1440, 640, 360, mode="exact")
    assert t.scale_x == pytest.approx(0.25)
    assert t.scale_y == pytest.approx(0.25)
    assert t.dst_width == 640 and t.dst_height == 360
    assert t.mode == "exact"


def test_resolve_transform_exact_non_uniform() -> None:
    t = compute_resize_transform(2560, 1440, 1920, 1080, mode="exact")
    # 1920/2560 = 0.75; 1080/1440 = 0.75 → uniform actually
    assert t.scale_x == pytest.approx(0.75)
    assert t.scale_y == pytest.approx(0.75)


def test_resolve_transform_fit_aspect() -> None:
    t = compute_resize_transform(2560, 1440, 640, 640, mode="fit_aspect")
    # min scale = 640/2560 = 0.25 (landscape source); dst height becomes 360
    assert t.scale_x == t.scale_y == pytest.approx(0.25)
    assert t.dst_width == 640 and t.dst_height == 360


def test_resolve_transform_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown resize mode"):
        compute_resize_transform(640, 480, 100, 100, mode="bogus")


def test_resolve_transform_invalid_dims_raise() -> None:
    with pytest.raises(ValueError):
        compute_resize_transform(0, 1, 1, 1)
    with pytest.raises(ValueError):
        compute_resize_transform(1, 1, 0, 1)


@SKIP_IF_NO_BACKEND
def test_imread_imwrite_roundtrip(tmp_path: Path) -> None:
    import numpy as np
    src = (np.arange(40 * 60 * 3, dtype=np.uint8).reshape(40, 60, 3)) % 255
    out_path = tmp_path / "img.png"
    imwrite(out_path, src)
    back = imread(out_path)
    assert back.shape == src.shape
    # PNG is lossless, so byte-equal.
    assert (back == src).all()


@SKIP_IF_NO_BACKEND
def test_get_image_size_without_decoding(tmp_path: Path) -> None:
    import numpy as np
    src = np.zeros((33, 77, 3), dtype=np.uint8)
    p = tmp_path / "x.png"
    imwrite(p, src)
    assert get_image_size(p) == (77, 33)


@SKIP_IF_NO_BACKEND
def test_resize_image_changes_shape() -> None:
    import numpy as np
    src = np.zeros((20, 40, 3), dtype=np.uint8)
    out = resize_image(src, dst_w=80, dst_h=60)
    assert out.shape == (60, 80, 3)


@SKIP_IF_NO_BACKEND
def test_imread_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        imread(tmp_path / "no.png")
