"""Tests for scripts.migrate_legacy_dataset (legacy 4-class -> canonical 3-class)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.common.classes import ClassContractError
from scripts.migrate_legacy_dataset import migrate_legacy_dataset

SPLITS = ("train", "val", "test")

_LEGACY4 = [
    {"id": 0, "name": "ships"},
    {"id": 1, "name": "buoys"},
    {"id": 2, "name": "helicopters"},
    {"id": 3, "name": "airplanes"},
]
_CANON = [
    {"id": 0, "name": "ships"},
    {"id": 1, "name": "helicopters"},
    {"id": 2, "name": "airplanes"},
]


def _write_raw(root: Path, dataset_id: str, categories: list[dict],
              anns_by_split: dict[str, list[int]]) -> Path:
    """Build a minimal raw-contract dataset; one image per annotation, dummy bytes."""
    d = root / dataset_id
    (d / "metadata").mkdir(parents=True, exist_ok=True)
    img_id = 1
    ann_id = 1
    for split in SPLITS:
        (d / "images" / split).mkdir(parents=True, exist_ok=True)
        (d / "annotations").mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        for cat in anns_by_split.get(split, []):
            fn = f"{split}_{img_id:03d}.png"
            (d / "images" / split / fn).write_bytes(b"\x89PNG\r\n\x1a\n_dummy_")
            images.append({"id": img_id, "file_name": fn, "width": 64, "height": 64})
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": cat,
                                 "bbox": [1.0, 2.0, 3.0, 4.0], "area": 12.0, "iscrowd": 0})
            img_id += 1
            ann_id += 1
        (d / "annotations" / f"instances_{split}.json").write_text(
            json.dumps({"images": images, "annotations": annotations,
                        "categories": categories}), encoding="utf-8")
    (d / "metadata" / "classes.json").write_text(
        json.dumps({"categories": categories}), encoding="utf-8")
    (d / "metadata" / "dataset_info.json").write_text(
        json.dumps({"dataset_id": dataset_id, "num_classes": len(categories)}),
        encoding="utf-8")
    return d


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def test_legacy4_remaps_drops_buoys_and_relocates(tmp_path: Path) -> None:
    src = _write_raw(tmp_path / "legacy", "ds", _LEGACY4,
                     {"train": [0, 1, 2, 3], "val": [0, 2], "test": [3]})
    target_root = tmp_path / "raw"

    report = migrate_legacy_dataset(src, target_root, dataset_id="ds",
                                    link_mode="copy")
    assert report["source_kind"] == "legacy4"
    assert report["status"] == "done"
    assert report["dropped_annotations_total"] == 1   # the single buoys annotation

    tgt = target_root / "ds"
    # categories canonical everywhere
    assert _load(tgt / "metadata" / "classes.json")["categories"] == _CANON
    for split in SPLITS:
        coco = _load(tgt / "annotations" / f"instances_{split}.json")
        assert coco["categories"] == _CANON
        for a in coco["annotations"]:
            assert a["category_id"] in (0, 1, 2)
    # train: 0->0, 2->1, 3->2 ; buoys(1) dropped -> 3 anns remain
    train = _load(tgt / "annotations" / "instances_train.json")
    assert sorted(a["category_id"] for a in train["annotations"]) == [0, 1, 2]
    # images relocated
    assert any((tgt / "images" / "train").glob("*.png"))
    # dataset_info recounted to canonical
    info = _load(tgt / "metadata" / "dataset_info.json")
    assert info["num_classes"] == 3
    assert info["migrated_from"] == str(src.resolve())
    # report + source preserved (no --delete-source)
    assert (tgt / "migration_report.json").is_file()
    assert src.is_dir()
    assert report["source_deleted"] is False


def test_canonical_source_is_identity(tmp_path: Path) -> None:
    src = _write_raw(tmp_path / "canon", "ds", _CANON,
                     {"train": [0, 1, 2], "val": [1], "test": [2]})
    report = migrate_legacy_dataset(src, tmp_path / "raw", dataset_id="ds",
                                    link_mode="copy")
    assert report["source_kind"] == "canonical"
    assert report["dropped_annotations_total"] == 0
    train = _load(tmp_path / "raw" / "ds" / "annotations" / "instances_train.json")
    assert sorted(a["category_id"] for a in train["annotations"]) == [0, 1, 2]


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    src = _write_raw(tmp_path / "legacy", "ds", _LEGACY4, {"train": [0, 2, 3]})
    report = migrate_legacy_dataset(src, tmp_path / "raw", dataset_id="ds",
                                    link_mode="copy", dry_run=True)
    assert report["status"] == "planned"
    assert not (tmp_path / "raw" / "ds").exists()


def test_unknown_categories_raise(tmp_path: Path) -> None:
    weird = [{"id": 0, "name": "ships"}, {"id": 1, "name": "submarines"}]
    src = _write_raw(tmp_path / "weird", "ds", weird, {"train": [0, 1]})
    with pytest.raises(ClassContractError):
        migrate_legacy_dataset(src, tmp_path / "raw", dataset_id="ds", link_mode="copy")
