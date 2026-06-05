"""Unit tests for packages.common.tiling.generate_tiles."""
from __future__ import annotations

import pytest

from packages.common.tiling import Tile, generate_tiles


def test_image_smaller_than_tile_yields_single_tile() -> None:
    tiles = generate_tiles(width=500, height=300, crop_size=640, stride=320)
    assert len(tiles) == 1
    t = tiles[0]
    assert t.crop_offset == (0, 0)
    assert t.crop_size == (500, 300)
    assert t.pad == (140, 340)


def test_exact_tile_size_yields_single_tile() -> None:
    tiles = generate_tiles(640, 640, 640, 320)
    assert len(tiles) == 1
    assert tiles[0].pad == (0, 0)
    assert tiles[0].crop_offset == (0, 0)


def test_sliding_window_no_edge_tiles_no_dup() -> None:
    # 1280 wide, crop=640, stride=320 → positions 0, 320, 640. Adding 640 again
    # (edge tile == 1280-640=640) is a dup, so it must not appear twice.
    tiles = generate_tiles(1280, 640, 640, 320, include_edge_tiles=True)
    xs = sorted({t.crop_offset[0] for t in tiles})
    assert xs == [0, 320, 640]


def test_edge_tile_pinned_to_right() -> None:
    # 1500 wide, crop=640, stride=320 → 0, 320, 640, 860 (=1500-640)
    tiles = generate_tiles(1500, 640, 640, 320, include_edge_tiles=True)
    xs = sorted({t.crop_offset[0] for t in tiles})
    assert xs == [0, 320, 640, 860]
    # last tile fully covers right edge
    assert max(t.crop_offset[0] + t.crop_size[0] for t in tiles) == 1500


def test_no_edge_tile_when_disabled() -> None:
    tiles = generate_tiles(1500, 640, 640, 320, include_edge_tiles=False)
    xs = sorted({t.crop_offset[0] for t in tiles})
    assert xs == [0, 320, 640]
    # right edge NOT fully covered
    assert max(t.crop_offset[0] + t.crop_size[0] for t in tiles) == 1280


def test_unique_crop_ids() -> None:
    tiles = generate_tiles(1920, 1080, 640, 320)
    ids = [t.crop_id for t in tiles]
    assert ids == list(range(len(tiles))) == sorted(set(ids))


def test_tiles_cover_full_image_with_edge_tiles() -> None:
    tiles = generate_tiles(2560, 1440, 640, 320, include_edge_tiles=True)
    max_x = max(t.crop_offset[0] + t.crop_size[0] for t in tiles)
    max_y = max(t.crop_offset[1] + t.crop_size[1] for t in tiles)
    assert max_x == 2560
    assert max_y == 1440


def test_tile_size_is_constant() -> None:
    tiles = generate_tiles(2560, 1440, 640, 320, include_edge_tiles=True)
    assert all(t.tile_size == 640 for t in tiles)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        generate_tiles(0, 100, 640, 320)
    with pytest.raises(ValueError):
        generate_tiles(100, 100, 0, 320)
    with pytest.raises(ValueError):
        generate_tiles(100, 100, 640, 0)


def test_tile_to_dict_serializable() -> None:
    t = Tile(crop_offset=(0, 0), crop_size=(640, 640), tile_size=640)
    d = t.to_dict()
    assert d["crop_offset"] == [0, 0]
    assert d["crop_size"] == [640, 640]
    assert d["pad"] == [0, 0]
