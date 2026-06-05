"""Integration smoke test: full enrichment cycle from mini raw dataset → tmp enriched."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from apps.dataset_enrichment.enrich import enrich_dataset
from packages.common.coco_io import load_coco
from packages.common.manifest import count_records
from tests.fixtures.mini_raw_dataset import write_mini_raw_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def test_end_to_end_enrichment_layout(tmp_path: Path) -> None:
    raw = write_mini_raw_dataset(
        tmp_path / "raw",
        images_per_split={"train": 3, "val": 1, "test": 1},
        annotations_per_image=2,
        include_one_background_in_train=True,
    )
    tmp_root = tmp_path / "tmp" / "smoke_run" / "enriched_dataset"

    result = enrich_dataset(
        raw_dataset_dir=raw, tmp_root=tmp_root,
        dataset_id="ds_smoke", run_id="smoke_run",
        enrichment_cfg=_ENRICHMENT_CFG,
    )

    # 1) Required output files
    for required in [
        "manifest.jsonl",
        "dropped_tiles_manifest.jsonl",
        "enrichment_summary.json",
        "annotations/instances_train.json",
        "annotations/instances_val.json",
        "annotations/instances_test.json",
    ]:
        assert (tmp_root / required).is_file(), f"missing {required}"

    # 2) Every enriched image referenced in COCO actually exists on disk
    for split in ("train", "val", "test"):
        coco = load_coco(tmp_root / "annotations" / f"instances_{split}.json")
        for img in coco["images"]:
            f = tmp_root / "images" / split / img["file_name"]
            assert f.is_file(), f"image not written: {f}"
            # All enriched images are crop_size×crop_size or scale.size
            assert img["width"] > 0 and img["height"] > 0

    # 3) Manifest count == sum of (per-scale enriched + backgrounds)
    manifest_records = count_records(tmp_root / "manifest.jsonl")
    total_enriched = sum(result.num_enriched_images.values()) + result.num_background_images
    assert manifest_records == total_enriched

    # 4) Every dropped tile has a row in dropped_tiles_manifest
    dropped_records = count_records(tmp_root / "dropped_tiles_manifest.jsonl")
    assert dropped_records == result.num_dropped_tiles

    # 5) Categories preserved in every enriched COCO
    raw_categories = load_coco(raw / "annotations" / "instances_train.json")["categories"]
    for split in ("train", "val", "test"):
        coco = load_coco(tmp_root / "annotations" / f"instances_{split}.json")
        assert coco["categories"] == raw_categories


def test_dry_run_cli(tmp_path: Path) -> None:
    """End-to-end CLI: build a temporary YAML, run dry-run, expect non-empty summary."""
    raw = write_mini_raw_dataset(
        tmp_path / "raw",
        images_per_split={"train": 2, "val": 1, "test": 1},
    )
    cfg = {
        "pipeline": {"run_id": "cli_dry", "seed": 42},
        "storage":  {"root_dir": str(tmp_path), "cache_dir": str(tmp_path / "cache")},
        "data":     {"raw_dataset_dir": str(raw)},
        "enrichment": _ENRICHMENT_CFG,
    }
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    proc = subprocess.run(
        [sys.executable, "-m", "apps.dataset_enrichment.cli", "dry-run",
         "--config", str(cfg_path), "--num-images", "1"],
        cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    summary = json.loads(proc.stdout)
    assert summary["target_dir"]
    assert summary["num_source_images"]["train"] == 1
    # Dry-run must not materialize a tile cache.
    assert not (tmp_path / "cache").exists()
