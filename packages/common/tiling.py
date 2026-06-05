"""Sliding-window tile generation for multi-scale enrichment + evaluation.

Migrated and simplified from ``yolox_eval/src/tiler.py``. We only need:

- ``Tile`` dataclass with ``crop_offset``, ``crop_size``, ``pad``;
- ``generate_tiles(width, height, crop_size, stride, include_edge_tiles)``.

Multi-scale planning lives in the calling app (``apps/dataset_enrichment``).
That keeps this module pure-data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Tile:
    """One tile in an image.

    Coordinates are in the *containing image* coords (the image at the scale we
    are tiling). ``pad`` records right/bottom padding when an edge tile starts
    before image edge but extends past it — the actual pixel array provided to
    a model still has ``tile_size × tile_size`` size with the padded region
    filled with a configured pad value (typically 114 for YOLOX).
    """

    crop_offset: tuple[int, int]   # (x, y) top-left of the tile in image coords
    crop_size: tuple[int, int]     # actual image area covered (before pad)
    tile_size: int                 # nominal tile size (e.g. 640)
    pad: tuple[int, int] = (0, 0)  # (pad_right, pad_bottom)
    crop_id: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["crop_offset"] = list(self.crop_offset)
        d["crop_size"] = list(self.crop_size)
        d["pad"] = list(self.pad)
        return d


def generate_tiles(
    width: int,
    height: int,
    crop_size: int,
    stride: int,
    include_edge_tiles: bool = True,
) -> list[Tile]:
    """Generate tiles covering an image of ``width × height``.

    Sliding window with ``stride`` along each axis. When ``include_edge_tiles``
    is true, the last tile on each axis is pinned to ``image_size - crop_size``
    so that the bottom/right edges are always covered (no half-tile drops).

    If ``width <= crop_size`` or ``height <= crop_size`` along an axis, exactly
    one tile is emitted on that axis (with ``pad`` recorded for the missing
    pixels).
    """
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if width <= 0 or height <= 0:
        raise ValueError(f"image must have positive dimensions, got {width}×{height}")

    xs = _axis_positions(width, crop_size, stride, include_edge_tiles)
    ys = _axis_positions(height, crop_size, stride, include_edge_tiles)

    tiles: list[Tile] = []
    crop_id = 0
    for y in ys:
        for x in xs:
            crop_w = min(crop_size, width - x)
            crop_h = min(crop_size, height - y)
            pad_r = max(0, crop_size - crop_w)
            pad_b = max(0, crop_size - crop_h)
            tiles.append(Tile(
                crop_offset=(x, y),
                crop_size=(crop_w, crop_h),
                tile_size=crop_size,
                pad=(pad_r, pad_b),
                crop_id=crop_id,
            ))
            crop_id += 1
    return tiles


def _axis_positions(
    axis_length: int, crop_size: int, stride: int, include_edge_tiles: bool,
) -> list[int]:
    if axis_length <= crop_size:
        return [0]

    positions: list[int] = []
    x = 0
    while x + crop_size <= axis_length:
        positions.append(x)
        x += stride

    if include_edge_tiles:
        last_start = axis_length - crop_size
        if not positions or positions[-1] != last_start:
            positions.append(last_start)
    return positions
