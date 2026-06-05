"""Integration smoke: build_or_reuse_cache with the REAL tiler — created → reused.

Needs numpy/opencv (the tiler), so it self-skips where they're absent; it runs in
CI (which installs the `dataset` extra) and in the dataset-processor image.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")

from apps.dataset_enrichment.tile_cache import build_or_reuse_cache  # noqa: E402
from tests.fixtures.mini_raw_dataset import write_mini_raw_dataset  # noqa: E402

_CFG = {
    "crop_size": 640,
    "stride": 320,
    "visible_area_threshold": 0.80,
    "include_edge_tiles": True,
    "keep_background_images": True,
    "background_policy": {"resize_to": [640, 640], "method": "letterbox", "pad_value": 114},
    "scales": [
        {"name": "original",  "mode": "keep_original", "crop": True},
        {"name": "final_640", "size": [640, 640],      "crop": False},
    ],
}


def test_build_cache_created_then_reused(tmp_path: Path) -> None:
    raw = write_mini_raw_dataset(
        tmp_path / "raw",
        images_per_split={"train": 2, "val": 1, "test": 1},
        annotations_per_image=2,
    )
    kw = dict(raw_dataset_dir=raw, dataset_id="ds_smoke", enrichment_cfg=_CFG,
              cache_root=tmp_path / "cache" / "tiles",
              runs_dir=tmp_path / "runs", run_id="smoke")

    first = build_or_reuse_cache(**kw)
    assert first["status"] == "created"
    cache_dir = Path(first["cache_dir"])
    assert (cache_dir / "cache_meta.json").is_file()
    assert (cache_dir / "tile_manifest.json").is_file()
    for s in ("train", "val", "test"):
        assert (cache_dir / "annotations" / f"instances_{s}.json").is_file()
        assert (cache_dir / "images" / s).is_dir()

    second = build_or_reuse_cache(**kw)
    assert second["status"] == "reused"
    assert second["tile_cache_id"] == first["tile_cache_id"]
    assert (tmp_path / "runs" / "smoke" / "tile_cache.json").is_file()
