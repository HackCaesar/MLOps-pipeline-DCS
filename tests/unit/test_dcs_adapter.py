"""Unit tests for apps.dcs_capture.adapter.normalize_dataset."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pytest

from apps.dcs_capture.adapter import assign_split, normalize_dataset
from packages.common.coco_io import load_coco
from packages.common.manifest import count_records, read_manifest
from tests.fixtures.mini_dcs_dataset import write_mini_dcs_dataset

# ---- assign_split ------------------------------------------------------

def test_assign_split_deterministic() -> None:
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    a = [assign_split(f"frame_{i:03d}", ratios) for i in range(200)]
    b = [assign_split(f"frame_{i:03d}", ratios) for i in range(200)]
    assert a == b


def test_assign_split_distribution_within_5pct() -> None:
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    counts = Counter(assign_split(f"f_{i}", ratios) for i in range(5000))
    total = sum(counts.values())
    train_frac = counts["train"] / total
    val_frac = counts["val"] / total
    test_frac = counts["test"] / total
    assert abs(train_frac - 0.70) < 0.05
    assert abs(val_frac - 0.15) < 0.05
    assert abs(test_frac - 0.15) < 0.05


def test_assign_split_raises_when_ratios_dont_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        assign_split("x", {"train": 0.5, "val": 0.1, "test": 0.1})


def test_assign_split_single_bucket() -> None:
    # 100% train should send everything to train.
    for i in range(50):
        assert assign_split(f"x_{i}", {"train": 1.0, "val": 0.0, "test": 0.0}) == "train"


# ---- normalize_dataset --------------------------------------------------

def _run_adapter(tmp_path: Path, num_frames: int = 12) -> tuple[Path, Path]:
    source = write_mini_dcs_dataset(tmp_path / "src", num_frames=num_frames)
    target = tmp_path / "out" / "ds_001"
    result = normalize_dataset(
        source_dir=source, target_dir=target,
        dataset_id="ds_001", name="test dataset",
    )
    assert result.target_dir == target
    assert result.num_images == num_frames
    return source, target


def test_creates_full_layout(tmp_path: Path) -> None:
    _, target = _run_adapter(tmp_path, num_frames=10)
    # required dirs
    for d in ("images/train", "images/val", "images/test",
              "annotations", "metadata"):
        assert (target / d).is_dir(), f"missing dir: {d}"
    # required files
    for f in ("metadata/dataset_info.json", "metadata/classes.json",
              "metadata/capture_manifest.jsonl",
              "annotations/instances_train.json",
              "annotations/instances_val.json",
              "annotations/instances_test.json"):
        assert (target / f).is_file(), f"missing file: {f}"


def test_no_split_leakage(tmp_path: Path) -> None:
    _, target = _run_adapter(tmp_path, num_frames=20)
    frames_per_split: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        coco = load_coco(target / "annotations" / f"instances_{split}.json")
        frames_per_split[split] = {Path(img["file_name"]).stem for img in coco["images"]}
    intersect_tv = frames_per_split["train"] & frames_per_split["val"]
    intersect_tt = frames_per_split["train"] & frames_per_split["test"]
    intersect_vt = frames_per_split["val"] & frames_per_split["test"]
    assert not intersect_tv and not intersect_tt and not intersect_vt


def test_image_copied_to_correct_split_folder(tmp_path: Path) -> None:
    _, target = _run_adapter(tmp_path, num_frames=8)
    for split in ("train", "val", "test"):
        coco = load_coco(target / "annotations" / f"instances_{split}.json")
        for img in coco["images"]:
            f = target / "images" / split / img["file_name"]
            assert f.is_file(), f"image not copied to {split}: {img['file_name']}"


def test_hardlink_placement_links_image(tmp_path: Path) -> None:
    source = write_mini_dcs_dataset(tmp_path / "src", num_frames=8)
    target = tmp_path / "out" / "ds_001"
    normalize_dataset(
        source_dir=source, target_dir=target,
        dataset_id="ds_001", image_placement="hardlink",
    )
    if not hasattr(os, "link"):
        pytest.skip("os.link not supported on this platform")
    found_hardlink = False
    for split in ("train", "val", "test"):
        coco = load_coco(target / "annotations" / f"instances_{split}.json")
        for img in coco["images"]:
            src_img = source / "images" / img["file_name"]
            dst_img = target / "images" / split / img["file_name"]
            assert dst_img.is_file()
            try:
                same_inode = os.stat(src_img).st_ino == os.stat(dst_img).st_ino
            except OSError:
                continue
            if same_inode:
                found_hardlink = True
    if not found_hardlink:
        pytest.skip("hardlinks unsupported (copy fallback used) on this filesystem")
    assert found_hardlink


def test_copy_placement_produces_real_copy(tmp_path: Path) -> None:
    source = write_mini_dcs_dataset(tmp_path / "src", num_frames=8)
    target = tmp_path / "out" / "ds_001"
    normalize_dataset(
        source_dir=source, target_dir=target,
        dataset_id="ds_001", image_placement="copy",
    )
    for split in ("train", "val", "test"):
        coco = load_coco(target / "annotations" / f"instances_{split}.json")
        for img in coco["images"]:
            src_img = source / "images" / img["file_name"]
            dst_img = target / "images" / split / img["file_name"]
            assert dst_img.is_file()
            assert os.stat(src_img).st_ino != os.stat(dst_img).st_ino


def test_categories_preserved_in_every_split(tmp_path: Path) -> None:
    source, target = _run_adapter(tmp_path, num_frames=8)
    src_coco = load_coco(source / "annotations" / "_annotations.coco.json")
    for split in ("train", "val", "test"):
        coco = load_coco(target / "annotations" / f"instances_{split}.json")
        assert coco["categories"] == src_coco["categories"]


def test_dataset_info_counts_match_splits(tmp_path: Path) -> None:
    _, target = _run_adapter(tmp_path, num_frames=20)
    info = json.loads((target / "metadata" / "dataset_info.json").read_text())
    total_imgs = sum(c["num_images"] for c in info["splits"].values())
    total_anns = sum(c["num_annotations"] for c in info["splits"].values())
    assert info["num_images"] == total_imgs
    assert info["num_annotations"] == total_anns
    assert info["dataset_id"] == "ds_001"
    assert info["split_strategy"] == "by_source_frame_hash"
    assert info["annotation_format"] == "coco"


def test_capture_manifest_one_record_per_image(tmp_path: Path) -> None:
    _, target = _run_adapter(tmp_path, num_frames=15)
    manifest_path = target / "metadata" / "capture_manifest.jsonl"
    assert count_records(manifest_path) == 15
    records = list(read_manifest(manifest_path))
    assert all("frame_id" in r and "split" in r and "source_image" in r for r in records)
    # Rich-metadata fields populated from per-frame metadata
    assert all(r.get("has_rich_metadata") for r in records)
    assert all(r.get("scene_id", "").startswith("scene_") for r in records)


def test_classes_json_matches_categories(tmp_path: Path) -> None:
    _, target = _run_adapter(tmp_path, num_frames=4)
    classes = json.loads((target / "metadata" / "classes.json").read_text())
    assert [c["name"] for c in classes["categories"]] == [
        "ships", "helicopters", "airplanes",
    ]


def test_overwrite_required_when_target_exists(tmp_path: Path) -> None:
    source = write_mini_dcs_dataset(tmp_path / "src", num_frames=3)
    target = tmp_path / "out"
    normalize_dataset(source_dir=source, target_dir=target, dataset_id="d")
    with pytest.raises(FileExistsError):
        normalize_dataset(source_dir=source, target_dir=target, dataset_id="d")


def test_overwrite_allows_redo(tmp_path: Path) -> None:
    source = write_mini_dcs_dataset(tmp_path / "src", num_frames=3)
    target = tmp_path / "out"
    normalize_dataset(source_dir=source, target_dir=target, dataset_id="d")
    # Append a stray file inside the old target — it must be gone after overwrite.
    (target / "stray.txt").write_text("garbage")
    normalize_dataset(source_dir=source, target_dir=target,
                      dataset_id="d", overwrite=True)
    assert not (target / "stray.txt").exists()


def test_warns_on_missing_image_referenced_in_coco(tmp_path: Path) -> None:
    source = write_mini_dcs_dataset(tmp_path / "src", num_frames=3)
    # Remove one image file but keep the COCO reference.
    (source / "images" / "frame_001.png").unlink()
    target = tmp_path / "out"
    result = normalize_dataset(
        source_dir=source, target_dir=target, dataset_id="d",
    )
    assert any("Source image missing" in w for w in result.warnings)


def test_missing_source_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        normalize_dataset(
            source_dir=tmp_path / "nope", target_dir=tmp_path / "out",
            dataset_id="d",
        )


def test_dcs_config_snapshot_copied(tmp_path: Path) -> None:
    source = write_mini_dcs_dataset(tmp_path / "src", num_frames=3)
    dcs_cfg = tmp_path / "dcs_config.yaml"
    dcs_cfg.write_text("paths:\n  output_root: ./dataset\n")
    target = tmp_path / "out"
    result = normalize_dataset(
        source_dir=source, target_dir=target, dataset_id="d",
        dcs_config_path=dcs_cfg,
    )
    assert (target / "metadata" / "dcs_run_config.yaml").is_file()
    assert not result.warnings


def test_no_images_mode_skips_copy(tmp_path: Path) -> None:
    source = write_mini_dcs_dataset(tmp_path / "src", num_frames=4)
    target = tmp_path / "out"
    normalize_dataset(
        source_dir=source, target_dir=target, dataset_id="d",
        copy_images=False,
    )
    for split in ("train", "val", "test"):
        # split folders exist (created upfront) but images are not copied
        split_dir = target / "images" / split
        assert split_dir.is_dir()
        assert list(split_dir.iterdir()) == []
