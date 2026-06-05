"""Tile cache build/reuse, atomicity, validity, breadcrumb (apps.dataset_enrichment.tile_cache).

Uses a stub ``enrich_fn`` so it runs without numpy/cv2.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.dataset_enrichment import tile_cache
from packages.common import hashing

_CFG = {
    "crop_size": 640, "stride": 320, "visible_area_threshold": 0.8,
    "include_edge_tiles": True, "keep_background_images": True,
    "background_policy": {"resize_to": [640, 640], "method": "letterbox", "pad_value": 114},
    "scales": [{"name": "original", "mode": "keep_original", "crop": True}],
}


class _Result:
    num_dropped_tiles = 2
    num_background_images = 1


def _stub_enrich(raw_dataset_dir: Path, out_dir: Path, dataset_id: str, cfg) -> _Result:
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_dir / "images" / s / f"{s}_tile.png").write_bytes(b"tile")
        (out_dir / "annotations" / f"instances_{s}.json").write_text(json.dumps(
            {"images": [{"id": 1, "file_name": f"{s}_tile.png", "width": 640, "height": 640}],
             "annotations": [], "categories": [{"id": 0, "name": "ships"}]}))
    (out_dir / "dropped_tiles_manifest.jsonl").write_text("")
    return _Result()


def _write_raw(root: Path) -> Path:
    """A minimal NON-empty raw dataset so dataset_hash != EMPTY_SHA256."""
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        (root / "images" / s).mkdir(parents=True, exist_ok=True)
        (root / "images" / s / f"{s}.png").write_bytes(b"src-" + s.encode())
        (root / "annotations" / f"instances_{s}.json").write_text(json.dumps(
            {"images": [{"id": 1, "file_name": f"{s}.png", "width": 8, "height": 8}],
             "annotations": [], "categories": [{"id": 0, "name": "ships"}]}))
    return root


def _build(tmp_path: Path, *, cfg=_CFG, run_id="r1"):
    raw = _write_raw(tmp_path / "raw" / "ds")
    return tile_cache.build_or_reuse_cache(
        raw_dataset_dir=raw, dataset_id="ds", enrichment_cfg=cfg,
        cache_root=tmp_path / "cache" / "tiles", runs_dir=tmp_path / "runs",
        run_id=run_id, enrich_fn=_stub_enrich)


def test_created_then_reused(tmp_path: Path) -> None:
    info1 = _build(tmp_path)
    assert info1["status"] == "created"
    cache_dir = Path(info1["cache_dir"])

    meta_p = cache_dir / "cache_meta.json"
    meta = json.loads(meta_p.read_text())
    assert meta["status"] == "complete"
    assert meta["schema_version"] == 1 and meta["layout_version"] == 1
    assert meta["dataset_hash"] and meta["split_hash"] and meta["tile_config_hash"]
    assert (cache_dir / "tile_manifest.json").is_file()
    assert info1["num_train_tiles"] == 1 and info1["num_val_tiles"] == 1

    # layout
    for s in ("train", "val", "test"):
        assert (cache_dir / "annotations" / f"instances_{s}.json").is_file()
        assert (cache_dir / "images" / s).is_dir()

    mtime = meta_p.stat().st_mtime_ns
    info2 = _build(tmp_path)
    assert info2["status"] == "reused"
    assert info2["cache_dir"] == info1["cache_dir"]
    assert meta_p.stat().st_mtime_ns == mtime, "reuse must not rewrite the cache"

    # no leftover .building, breadcrumb present
    assert not list((tmp_path / "cache" / "tiles").glob("*.building"))
    bc = tmp_path / "runs" / "r1" / "tile_cache.json"
    assert json.loads(bc.read_text())["status"] == "reused"


def test_config_change_produces_new_cache(tmp_path: Path) -> None:
    info1 = _build(tmp_path)
    info2 = _build(tmp_path, cfg=dict(_CFG, crop_size=512))
    assert info2["tile_cache_id"] != info1["tile_cache_id"]
    assert info2["status"] == "created"


def test_incomplete_cache_is_not_reused(tmp_path: Path) -> None:
    info1 = _build(tmp_path)
    meta_p = Path(info1["cache_dir"]) / "cache_meta.json"
    meta = json.loads(meta_p.read_text())
    meta["status"] = "building"          # simulate a crashed/partial build
    meta_p.write_text(json.dumps(meta))
    assert _build(tmp_path)["status"] == "created"   # rebuilt, not reused


def test_build_refuses_empty_dataset(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "empty"
    raw.mkdir(parents=True)   # no annotations, no images -> empty dataset_hash
    with pytest.raises(ValueError, match="appears empty"):
        tile_cache.build_or_reuse_cache(
            raw_dataset_dir=raw, dataset_id="ds", enrichment_cfg=_CFG,
            cache_root=tmp_path / "cache" / "tiles", runs_dir=tmp_path / "runs",
            run_id="r", enrich_fn=_stub_enrich)


def test_stale_building_dir_removed(tmp_path: Path) -> None:
    raw = _write_raw(tmp_path / "raw" / "ds")
    cache_root = tmp_path / "cache" / "tiles"
    cache_root.mkdir(parents=True)
    tcid = hashing.compute_hashes(raw, _CFG)["tile_cache_id"]
    stale = cache_root / f"{tcid}.building"
    stale.mkdir()
    (stale / "junk.txt").write_text("leftover")

    info = tile_cache.build_or_reuse_cache(
        raw_dataset_dir=raw, dataset_id="ds", enrichment_cfg=_CFG,
        cache_root=cache_root, runs_dir=tmp_path / "runs", run_id="r",
        enrich_fn=_stub_enrich)
    assert info["status"] == "created"
    assert not (cache_root / f"{tcid}.building").exists()
    assert (cache_root / tcid / "cache_meta.json").is_file()
