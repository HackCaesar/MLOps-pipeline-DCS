"""Unit tests for the pixel-materialization path in apps.evaluation_export.evaluate."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pytest

from apps.evaluation_export.evaluate import (
    EvalScale,
    _build_crop_inputs,
    _extract_tile_pixels,
    _get_or_resize,
)
from packages.common.tiling import Tile


def _solid(w: int, h: int, value: int = 128) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


# ---- _extract_tile_pixels ---------------------------------------------

def test_extract_tile_pixels_no_pad() -> None:
    arr = _solid(640, 640, 50)
    t = Tile(crop_offset=(0, 0), crop_size=(640, 640), tile_size=640, pad=(0, 0))
    out = _extract_tile_pixels(arr, t, tile_size=640)
    assert out.shape == (640, 640, 3)
    assert (out == 50).all()


def test_extract_tile_pixels_with_pad() -> None:
    arr = _solid(500, 500, 50)
    t = Tile(crop_offset=(0, 0), crop_size=(500, 500), tile_size=640, pad=(140, 140))
    out = _extract_tile_pixels(arr, t, tile_size=640, pad_value=114)
    assert out.shape == (640, 640, 3)
    assert (out[:500, :500] == 50).all()
    assert (out[500:, :] == 114).all()
    assert (out[:, 500:] == 114).all()


def test_extract_tile_pixels_offset() -> None:
    """Tile at (300, 200) of size 640×640 in a 2560×1440 image."""
    arr = np.zeros((1440, 2560, 3), dtype=np.uint8)
    arr[200:840, 300:940] = 99  # mark expected slice
    t = Tile(crop_offset=(300, 200), crop_size=(640, 640), tile_size=640, pad=(0, 0))
    out = _extract_tile_pixels(arr, t, tile_size=640)
    assert (out == 99).all()


# ---- _get_or_resize cache ---------------------------------------------

def test_get_or_resize_returns_none_when_source_is_none() -> None:
    cache: Dict[Tuple[int, int], np.ndarray] = {}
    assert _get_or_resize(None, cache, 100, 100) is None
    assert cache == {}


def test_get_or_resize_caches_resized() -> None:
    src = _solid(2560, 1440)
    cache: Dict[Tuple[int, int], np.ndarray] = {}
    first = _get_or_resize(src, cache, 1920, 1080)
    assert first is not None and first.shape == (1080, 1920, 3)
    second = _get_or_resize(src, cache, 1920, 1080)
    assert second is first  # cached reference, not a re-resize


def test_get_or_resize_without_cache_still_works() -> None:
    src = _solid(2560, 1440)
    out = _get_or_resize(src, None, 640, 360)
    assert out is not None and out.shape == (360, 640, 3)


# ---- _build_crop_inputs without source_array (mock path) --------------

def test_build_crop_inputs_no_pixels_for_source_full_image() -> None:
    scale = EvalScale(name="full_source", mode="source_full_image")
    crops, _ = _build_crop_inputs(scale, image_id=1, src_w=2560, src_h=1440,
                                   crop_size=640, stride=320, include_edge_tiles=True)
    assert len(crops) == 1
    assert crops[0].image is None
    assert crops[0].extra.get("letterbox") is True


def test_build_crop_inputs_no_pixels_for_crops() -> None:
    scale = EvalScale(name="original", mode="keep_original", crop=True)
    crops, _ = _build_crop_inputs(scale, image_id=1, src_w=2560, src_h=1440,
                                   crop_size=640, stride=320, include_edge_tiles=True)
    assert len(crops) > 1
    for c in crops:
        assert c.image is None


# ---- _build_crop_inputs WITH source_array (real-predictor path) -------

def test_build_crop_inputs_populates_pixels_for_keep_original() -> None:
    src = _solid(2560, 1440, 77)
    scale = EvalScale(name="original", mode="keep_original", crop=True)
    crops, _ = _build_crop_inputs(scale, image_id=1, src_w=2560, src_h=1440,
                                   crop_size=640, stride=320, include_edge_tiles=True,
                                   source_array=src)
    assert len(crops) > 1
    for c in crops:
        assert c.image is not None
        assert c.image.shape == (640, 640, 3)


def test_build_crop_inputs_resize_reuses_scale_cache() -> None:
    src = _solid(2560, 1440, 33)
    scale_a = EvalScale(name="full_hd", mode="resize", size=(1920, 1080), crop=True)
    cache: Dict[Tuple[int, int], np.ndarray] = {}
    crops_a, _ = _build_crop_inputs(scale_a, image_id=1, src_w=2560, src_h=1440,
                                      crop_size=640, stride=320, include_edge_tiles=True,
                                      source_array=src, scale_cache=cache)
    cached = cache.get((1920, 1080))
    assert cached is not None
    crops_b, _ = _build_crop_inputs(scale_a, image_id=1, src_w=2560, src_h=1440,
                                      crop_size=640, stride=320, include_edge_tiles=True,
                                      source_array=src, scale_cache=cache)
    # Second call should reuse cached resized array (we don't have direct hook
    # but cache size shouldn't grow on a second identical call).
    assert len(cache) == 1


def test_build_crop_inputs_source_full_image_passes_full_array() -> None:
    src = _solid(2560, 1440, 100)
    scale = EvalScale(name="full_source", mode="source_full_image")
    crops, _ = _build_crop_inputs(scale, image_id=1, src_w=2560, src_h=1440,
                                   crop_size=640, stride=320, include_edge_tiles=True,
                                   source_array=src)
    assert len(crops) == 1
    assert crops[0].image is src
    assert crops[0].extra["letterbox"] is True


def test_build_crop_inputs_crop_false_with_mismatched_scale_raises() -> None:
    src = _solid(2560, 1440, 100)
    scale = EvalScale(name="bad", mode="resize", size=(1280, 720), crop=False)
    with pytest.raises(ValueError, match="tile_size"):
        _build_crop_inputs(scale, image_id=1, src_w=2560, src_h=1440,
                            crop_size=640, stride=320, include_edge_tiles=True,
                            source_array=src)


def test_build_crop_inputs_crop_false_final_640_ok() -> None:
    src = _solid(2560, 1440, 100)
    scale = EvalScale(name="final_640", mode="resize", size=(640, 640), crop=False)
    crops, _ = _build_crop_inputs(scale, image_id=1, src_w=2560, src_h=1440,
                                   crop_size=640, stride=320, include_edge_tiles=True,
                                   source_array=src)
    assert len(crops) == 1
    assert crops[0].image is not None
    assert crops[0].image.shape == (640, 640, 3)


def test_build_crop_inputs_crop_false_no_source_ok() -> None:
    """Without source_array, crop:false with mismatched scale is still allowed
    (mock path — no pixel validation needed)."""
    scale = EvalScale(name="weird", mode="resize", size=(800, 600), crop=False)
    crops, _ = _build_crop_inputs(scale, image_id=1, src_w=2560, src_h=1440,
                                   crop_size=640, stride=320, include_edge_tiles=True,
                                   source_array=None)
    assert len(crops) == 1
    assert crops[0].image is None
