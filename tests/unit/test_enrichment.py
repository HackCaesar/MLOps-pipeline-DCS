"""Unit tests for apps.dataset_enrichment.enrich."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apps.dataset_enrichment.enrich import (
    EnrichmentResult,
    _apply_background_policy,
    _extract_tile,
    enrich_dataset,
    parse_scales,
)
from packages.common.coco_io import load_coco
from packages.common.manifest import count_records, read_manifest
from packages.common.tiling import Tile
from tests.fixtures.mini_raw_dataset import write_mini_raw_dataset

_ENRICHMENT_CFG = {
    "crop_size": 640,
    "stride": 320,
    "visible_area_threshold": 0.80,
    "include_edge_tiles": True,
    "keep_background_images": True,
    "background_policy": {"resize_to": [640, 640], "method": "letterbox", "pad_value": 114},
    "scales": [
        {"name": "original",     "mode": "keep_original", "crop": True},
        {"name": "full_hd",      "size": [1920, 1080],    "crop": True},
        {"name": "intermediate", "size": [1536,  864],    "crop": True},
        {"name": "final_640",    "size": [ 640,  640],    "crop": False},
    ],
}


# ---- parse_scales ------------------------------------------------------

def test_parse_scales_basic() -> None:
    scales = parse_scales(_ENRICHMENT_CFG["scales"])
    assert len(scales) == 4
    assert [s.name for s in scales] == ["original", "full_hd", "intermediate", "final_640"]
    assert scales[0].mode == "keep_original" and scales[0].size is None
    assert scales[1].size == (1920, 1080) and scales[1].crop is True
    assert scales[3].size == (640, 640) and scales[3].crop is False


def test_parse_scales_missing_size_raises() -> None:
    with pytest.raises(ValueError, match="size"):
        parse_scales([{"name": "x", "crop": True}])


# ---- _extract_tile -----------------------------------------------------

def test_extract_tile_no_pad() -> None:
    img = np.arange(640 * 640 * 3, dtype=np.uint8).reshape(640, 640, 3) % 255
    tile = Tile(crop_offset=(0, 0), crop_size=(640, 640), tile_size=640, pad=(0, 0))
    out = _extract_tile(img, tile)
    assert out.shape == (640, 640, 3)
    assert (out == img).all()


def test_extract_tile_with_pad() -> None:
    img = np.full((500, 500, 3), 50, dtype=np.uint8)
    tile = Tile(crop_offset=(0, 0), crop_size=(500, 500), tile_size=640, pad=(140, 140))
    out = _extract_tile(img, tile, pad_value=114)
    assert out.shape == (640, 640, 3)
    assert (out[:500, :500] == 50).all()
    # padded region filled with pad_value
    assert (out[500:, :] == 114).all()
    assert (out[:, 500:] == 114).all()


def test_extract_tile_returns_copy(tmp_path: Path) -> None:
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    tile = Tile(crop_offset=(0, 0), crop_size=(640, 640), tile_size=640)
    out = _extract_tile(img, tile)
    out[0, 0, 0] = 99
    assert img[0, 0, 0] == 0  # source unchanged


# ---- _apply_background_policy ------------------------------------------

def test_background_policy_letterboxes_to_square() -> None:
    src = np.full((1440, 2560, 3), 200, dtype=np.uint8)
    out = _apply_background_policy(src, target_size=[640, 640], pad_value=114)
    assert out.shape == (640, 640, 3)
    # Top/bottom bands should be the pad value; the middle band contains image pixels.
    top_band = out[:10, :]
    assert (top_band == 114).all()
    middle_band = out[300:340, 200:400]
    assert (middle_band == 200).all()


def test_background_policy_rejects_nonsquare_target() -> None:
    src = np.zeros((100, 200, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="square"):
        _apply_background_policy(src, target_size=[640, 480], pad_value=114)


# ---- enrich_dataset: dry-run -------------------------------------------

def test_dry_run_no_writes(tmp_path: Path) -> None:
    raw = write_mini_raw_dataset(
        tmp_path / "raw",
        images_per_split={"train": 2, "val": 1, "test": 1},
        annotations_per_image=2,
        include_one_background_in_train=True,
    )
    tmp_root = tmp_path / "tmp" / "run_xxx" / "enriched_dataset"
    result = enrich_dataset(
        raw_dataset_dir=raw, tmp_root=tmp_root,
        dataset_id="ds_test", run_id="run_xxx",
        enrichment_cfg=_ENRICHMENT_CFG,
        write_images=False, write_coco=False,
    )
    # No images written, no COCO written.
    assert not any((tmp_root / "images").rglob("*.png"))
    assert not (tmp_root / "annotations" / "instances_train.json").exists()
    assert not (tmp_root / "enrichment_summary.json").exists()
    # But manifest exists (with records).
    assert (tmp_root / "manifest.jsonl").exists()
    assert isinstance(result, EnrichmentResult)


# ---- enrich_dataset: full build ----------------------------------------

def test_full_build_produces_full_layout(tmp_path: Path) -> None:
    raw = write_mini_raw_dataset(
        tmp_path / "raw",
        images_per_split={"train": 2, "val": 1, "test": 1},
    )
    tmp_root = tmp_path / "tmp" / "run_001" / "enriched_dataset"
    enrich_dataset(
        raw_dataset_dir=raw, tmp_root=tmp_root,
        dataset_id="ds_test", run_id="run_001",
        enrichment_cfg=_ENRICHMENT_CFG,
    )

    # Output files
    for split in ("train", "val", "test"):
        assert (tmp_root / "annotations" / f"instances_{split}.json").is_file()
    assert (tmp_root / "manifest.jsonl").is_file()
    assert (tmp_root / "dropped_tiles_manifest.jsonl").is_file()
    assert (tmp_root / "enrichment_summary.json").is_file()

    # Images written for each enriched record
    for split in ("train", "val", "test"):
        coco = load_coco(tmp_root / "annotations" / f"instances_{split}.json")
        for img in coco["images"]:
            assert (tmp_root / "images" / split / img["file_name"]).is_file()

    # Counts in summary match counts in COCO
    summary = json.loads((tmp_root / "enrichment_summary.json").read_text())
    for split in ("train", "val", "test"):
        coco = load_coco(tmp_root / "annotations" / f"instances_{split}.json")
        # background images count towards num_enriched_images via background channel
        assert (summary["result"]["num_enriched_images"][split]
                + summary["result"]["per_scale_counts"]["background"][split]
                ) == len(coco["images"])


def test_background_image_produces_record_and_no_annotations(tmp_path: Path) -> None:
    raw = write_mini_raw_dataset(
        tmp_path / "raw",
        images_per_split={"train": 1, "val": 0, "test": 0},
        include_one_background_in_train=True,
        annotations_per_image=0,  # everything is background
    )
    tmp_root = tmp_path / "tmp" / "bg_run" / "enriched_dataset"
    result = enrich_dataset(
        raw_dataset_dir=raw, tmp_root=tmp_root,
        dataset_id="ds_test", run_id="bg_run",
        enrichment_cfg=_ENRICHMENT_CFG,
    )
    # In val/test we expected zero source images, so they don't contribute.
    assert result.num_background_images >= 1
    train_coco = load_coco(tmp_root / "annotations" / "instances_train.json")
    # Exactly the 1 background frame → 1 enriched image; 0 annotations.
    assert len(train_coco["images"]) == 1
    assert train_coco["annotations"] == []
    # Manifest record marks it.
    records = [r for r in read_manifest(tmp_root / "manifest.jsonl") if r["split"] == "train"]
    assert any(r["is_background"] for r in records)


def test_image_with_objects_emits_records_per_scale(tmp_path: Path) -> None:
    raw = write_mini_raw_dataset(
        tmp_path / "raw",
        images_per_split={"train": 1, "val": 0, "test": 0},
        annotations_per_image=2,
        include_one_background_in_train=False,
    )
    tmp_root = tmp_path / "tmp" / "scales_run" / "enriched_dataset"
    enrich_dataset(
        raw_dataset_dir=raw, tmp_root=tmp_root,
        dataset_id="ds_test", run_id="scales_run",
        enrichment_cfg=_ENRICHMENT_CFG,
    )
    records = list(read_manifest(tmp_root / "manifest.jsonl"))
    scale_names = {r["scale_name"] for r in records if r["split"] == "train"}
    # Both crop scales and final_640 should appear; background should NOT for object-images
    assert "final_640" in scale_names
    # At least one of the crop scales should have produced a record (the 2K image with bboxes near top-left
    # will at least produce an "original" tile_0 record).
    assert scale_names & {"original", "full_hd", "intermediate"}


def test_no_split_leakage_in_enriched_manifest(tmp_path: Path) -> None:
    raw = write_mini_raw_dataset(
        tmp_path / "raw",
        images_per_split={"train": 2, "val": 1, "test": 1},
    )
    tmp_root = tmp_path / "tmp" / "leak_run" / "enriched_dataset"
    enrich_dataset(
        raw_dataset_dir=raw, tmp_root=tmp_root,
        dataset_id="ds_test", run_id="leak_run",
        enrichment_cfg=_ENRICHMENT_CFG,
    )
    # Each source_image_id appears in exactly one split throughout the manifest.
    seen: dict[int, set[str]] = {}
    for r in read_manifest(tmp_root / "manifest.jsonl"):
        seen.setdefault(r["source_image_id"], set()).add(r["split"])
    for sid, splits in seen.items():
        assert len(splits) == 1, f"source_image_id {sid} appeared in multiple splits: {splits}"


def test_visible_threshold_drops_tile_with_partial_bbox(tmp_path: Path) -> None:
    # Annotations placed at right edge: only partially visible in original tiles,
    # high visible_threshold should drop those tiles → entry in dropped manifest.
    raw = write_mini_raw_dataset(
        tmp_path / "raw",
        images_per_split={"train": 1, "val": 0, "test": 0},
        image_size=(2560, 1440),
        annotations_per_image=0,
        include_one_background_in_train=False,
    )
    # Hand-craft COCO so single bbox sits across a tile boundary.
    coco_path = raw / "annotations" / "instances_train.json"
    coco = load_coco(coco_path)
    coco["annotations"].append({
        "id": 1, "image_id": coco["images"][0]["id"], "category_id": 0,
        # x=600, w=200 → spans [600..800] in source; with stride 320,
        # tile [320..960] contains all → visible=1.0 → kept;
        # tile [960..1600] cuts the right edge by zero → no overlap → drop.
        "bbox": [600.0, 200.0, 200.0, 50.0],
        "area": 10000.0, "iscrowd": 0,
    })
    coco_path.write_text(json.dumps(coco))

    tmp_root = tmp_path / "tmp" / "vt_run" / "enriched_dataset"
    enrich_dataset(
        raw_dataset_dir=raw, tmp_root=tmp_root,
        dataset_id="ds_test", run_id="vt_run",
        enrichment_cfg={**_ENRICHMENT_CFG, "visible_area_threshold": 0.80},
    )
    dropped = count_records(tmp_root / "dropped_tiles_manifest.jsonl")
    assert dropped > 0


def test_summary_records_per_scale_counts(tmp_path: Path) -> None:
    raw = write_mini_raw_dataset(
        tmp_path / "raw",
        images_per_split={"train": 1, "val": 0, "test": 0},
        include_one_background_in_train=False,
    )
    tmp_root = tmp_path / "tmp" / "psc_run" / "enriched_dataset"
    result = enrich_dataset(
        raw_dataset_dir=raw, tmp_root=tmp_root,
        dataset_id="ds_test", run_id="psc_run",
        enrichment_cfg=_ENRICHMENT_CFG,
    )
    assert "final_640" in result.per_scale_counts
    assert result.per_scale_counts["final_640"]["train"] == 1


def test_num_images_per_split_limits_processing(tmp_path: Path) -> None:
    raw = write_mini_raw_dataset(
        tmp_path / "raw",
        images_per_split={"train": 5, "val": 5, "test": 5},
    )
    tmp_root = tmp_path / "tmp" / "limit_run" / "enriched_dataset"
    result = enrich_dataset(
        raw_dataset_dir=raw, tmp_root=tmp_root,
        dataset_id="ds_test", run_id="limit_run",
        enrichment_cfg=_ENRICHMENT_CFG,
        num_images_per_split=2,
        write_images=False, write_coco=False,
    )
    for split in ("train", "val", "test"):
        assert result.num_source_images[split] == 2


def test_missing_split_coco_is_warning_not_crash(tmp_path: Path) -> None:
    raw = write_mini_raw_dataset(tmp_path / "raw")
    (raw / "annotations" / "instances_val.json").unlink()
    tmp_root = tmp_path / "tmp" / "missing_run" / "enriched_dataset"
    result = enrich_dataset(
        raw_dataset_dir=raw, tmp_root=tmp_root,
        dataset_id="ds_test", run_id="missing_run",
        enrichment_cfg=_ENRICHMENT_CFG,
    )
    assert any("Missing split COCO" in w for w in result.warnings)
