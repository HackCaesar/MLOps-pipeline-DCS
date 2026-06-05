"""Determinism + sensitivity of the tile-cache hashes (packages.common.hashing)."""
from __future__ import annotations

import json
from pathlib import Path

from packages.common import hashing

_CFG = {
    "enabled": True, "mode": "temporary_materialization",
    "crop_size": 640, "stride": 320, "visible_area_threshold": 0.8,
    "include_edge_tiles": True, "keep_background_images": True,
    "background_policy": {"resize_to": [640, 640], "method": "letterbox", "pad_value": 114},
    "scales": [{"name": "original", "mode": "keep_original", "crop": True}],
}


def _write_raw(root: Path, *, splits: dict[str, list[str]]) -> Path:
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        (root / "images" / s).mkdir(parents=True, exist_ok=True)
        imgs = []
        for i, fn in enumerate(splits.get(s, [])):
            (root / "images" / s / fn).write_bytes(b"img-" + fn.encode())
            imgs.append({"id": i + 1, "file_name": fn, "width": 8, "height": 8})
        (root / "annotations" / f"instances_{s}.json").write_text(
            json.dumps({"images": imgs, "annotations": [], "categories": [{"id": 0, "name": "ships"}]}))
    return root


def test_hashes_deterministic(tmp_path: Path) -> None:
    r = _write_raw(tmp_path / "ds", splits={"train": ["a.png", "b.png"], "val": ["c.png"], "test": ["d.png"]})
    assert hashing.compute_hashes(r, _CFG) == hashing.compute_hashes(r, _CFG)
    assert hashing.compute_hashes(r, _CFG)["tile_cache_id"].count("__") == 2


def test_tile_config_hash_invariant_to_key_order_and_float_format() -> None:
    a = hashing.tile_config_hash({"crop_size": 640, "stride": 320, "visible_area_threshold": 0.8})
    b = hashing.tile_config_hash({"visible_area_threshold": 0.80, "stride": 320, "crop_size": 640})
    assert a == b


def test_tile_config_hash_ignores_non_geometry_keys() -> None:
    other = dict(_CFG, enabled=False, mode="different")
    assert hashing.tile_config_hash(_CFG) == hashing.tile_config_hash(other)


def test_tile_config_hash_changes_on_geometry() -> None:
    assert hashing.tile_config_hash(dict(_CFG, crop_size=512)) != hashing.tile_config_hash(_CFG)
    assert hashing.tile_config_hash(dict(_CFG, stride=160)) != hashing.tile_config_hash(_CFG)


def test_split_hash_changes_when_frame_moves_split(tmp_path: Path) -> None:
    r1 = _write_raw(tmp_path / "a", splits={"train": ["x.png"], "val": ["y.png"], "test": []})
    r2 = _write_raw(tmp_path / "b", splits={"train": ["y.png"], "val": ["x.png"], "test": []})
    assert hashing.split_hash(r1) != hashing.split_hash(r2)


def test_dataset_hash_changes_on_annotation_edit(tmp_path: Path) -> None:
    r = _write_raw(tmp_path / "ds", splits={"train": ["a.png"], "val": [], "test": []})
    h1 = hashing.dataset_hash(r)
    p = r / "annotations" / "instances_train.json"
    d = json.loads(p.read_text())
    d["annotations"].append({"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0})
    p.write_text(json.dumps(d))
    assert hashing.dataset_hash(r) != h1


def test_empty_dataset_hash_is_the_empty_sha256(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert hashing.dataset_hash(empty) == hashing.EMPTY_SHA256


def test_dataset_hash_nonempty_for_real_dataset(tmp_path: Path) -> None:
    r = _write_raw(tmp_path / "ds", splits={"train": ["a.png"], "val": ["b.png"], "test": ["c.png"]})
    assert hashing.dataset_hash(r) != hashing.EMPTY_SHA256


def test_tile_config_hash_includes_algorithm_version(monkeypatch) -> None:
    # A tiling-CODE change (bumped TILING_ALGORITHM_VERSION) must invalidate the
    # cache key even if the geometry config is byte-for-byte identical.
    before = hashing.tile_config_hash(_CFG)
    monkeypatch.setattr(hashing, "TILING_ALGORITHM_VERSION",
                        hashing.TILING_ALGORITHM_VERSION + 1)
    after = hashing.tile_config_hash(_CFG)
    assert before != after
