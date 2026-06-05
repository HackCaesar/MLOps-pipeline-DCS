"""Unit tests for scripts.validate_raw_dataset.validate_raw_dataset."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.dcs_capture.adapter import normalize_dataset
from packages.common.coco_io import save_coco
from scripts.validate_raw_dataset import validate_raw_dataset
from tests.fixtures.mini_dcs_dataset import write_mini_dcs_dataset


@pytest.fixture
def good_dataset(tmp_path: Path) -> Path:
    source = write_mini_dcs_dataset(tmp_path / "src", num_frames=12)
    target = tmp_path / "out"
    normalize_dataset(source_dir=source, target_dir=target, dataset_id="d_ok")
    return target


def test_clean_dataset_passes(good_dataset: Path) -> None:
    n_err, n_warn, lines = validate_raw_dataset(good_dataset)
    assert n_err == 0, "Expected no errors, got:\n" + "\n".join(lines)


def test_missing_dataset_info_is_error(good_dataset: Path) -> None:
    (good_dataset / "metadata" / "dataset_info.json").unlink()
    n_err, _, lines = validate_raw_dataset(good_dataset)
    assert n_err >= 1
    assert any("dataset_info.json" in ln for ln in lines)


def test_missing_split_coco_is_error(good_dataset: Path) -> None:
    (good_dataset / "annotations" / "instances_val.json").unlink()
    n_err, _, lines = validate_raw_dataset(good_dataset)
    assert n_err >= 1
    assert any("instances_val.json" in ln for ln in lines)


def test_split_leakage_detected(good_dataset: Path) -> None:
    # Inject a frame from train into val COCO + folder.
    train_coco_path = good_dataset / "annotations" / "instances_train.json"
    val_coco_path = good_dataset / "annotations" / "instances_val.json"
    train_coco = json.loads(train_coco_path.read_text())
    val_coco = json.loads(val_coco_path.read_text())
    # Take first train image and copy into val.
    leaked = train_coco["images"][0]
    val_coco["images"].append(leaked)
    save_coco(val_coco, val_coco_path)
    # Also create the image file in val/ to avoid file-existence error.
    src_img = good_dataset / "images" / "train" / leaked["file_name"]
    (good_dataset / "images" / "val" / leaked["file_name"]).write_bytes(src_img.read_bytes())

    n_err, _, lines = validate_raw_dataset(good_dataset)
    assert n_err >= 1
    assert any("multiple splits" in ln for ln in lines)


def test_image_file_missing_is_error(good_dataset: Path) -> None:
    train_coco = json.loads((good_dataset / "annotations" / "instances_train.json").read_text())
    if not train_coco["images"]:
        pytest.skip("Fixture has no train images this run")
    target = good_dataset / "images" / "train" / train_coco["images"][0]["file_name"]
    target.unlink()
    n_err, _, lines = validate_raw_dataset(good_dataset)
    assert n_err >= 1
    assert any("file does not exist" in ln for ln in lines)


def test_classes_id_mismatch_is_error(good_dataset: Path) -> None:
    val_coco_path = good_dataset / "annotations" / "instances_val.json"
    val_coco = json.loads(val_coco_path.read_text())
    val_coco["categories"] = [{"id": 999, "name": "alien"}]
    save_coco(val_coco, val_coco_path)
    n_err, _, lines = validate_raw_dataset(good_dataset)
    assert n_err >= 1
    assert any("category ids" in ln for ln in lines)


def test_count_mismatch_in_dataset_info(good_dataset: Path) -> None:
    info_path = good_dataset / "metadata" / "dataset_info.json"
    info = json.loads(info_path.read_text())
    info["num_images"] = info["num_images"] + 1000  # bogus
    info_path.write_text(json.dumps(info))
    n_err, _, lines = validate_raw_dataset(good_dataset)
    assert n_err >= 1
    assert any("num_images" in ln for ln in lines)


def test_validate_returns_zero_for_nonexistent_dataset(tmp_path: Path) -> None:
    n_err, _, lines = validate_raw_dataset(tmp_path / "does_not_exist")
    assert n_err >= 1
