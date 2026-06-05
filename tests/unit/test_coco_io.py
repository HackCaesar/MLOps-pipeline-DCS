"""Unit tests for packages.common.coco_io."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.common.coco_io import (
    CocoValidationError,
    empty_coco,
    index_by_image,
    load_coco,
    save_coco,
    split_by_image_ids,
    validate_coco,
)


def _minimal_coco() -> dict:
    return {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 2560, "height": 1440},
            {"id": 2, "file_name": "b.png", "width": 2560, "height": 1440},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 0,
             "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0},
            {"id": 11, "image_id": 2, "category_id": 1,
             "bbox": [5, 5, 10, 10], "area": 100, "iscrowd": 0},
        ],
        "categories": [
            {"id": 0, "name": "ships"},
            {"id": 1, "name": "buoys"},
        ],
    }


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    coco = _minimal_coco()
    p = save_coco(coco, tmp_path / "subdir" / "out.json")
    assert p.exists()
    loaded = load_coco(p)
    assert loaded == coco


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_coco(tmp_path / "no.json")


def test_load_missing_required_keys(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"images": []}))
    with pytest.raises(ValueError, match="missing required key"):
        load_coco(p)


def test_validate_minimal_ok() -> None:
    issues = validate_coco(_minimal_coco(), strict=True)
    # No errors. There may be no warnings since iscrowd is set.
    assert all(i.severity != "error" for i in issues)


def test_validate_duplicate_image_id() -> None:
    coco = _minimal_coco()
    coco["images"].append({"id": 1, "file_name": "dup.png", "width": 10, "height": 10})
    with pytest.raises(CocoValidationError, match="duplicate image"):
        validate_coco(coco, strict=True)


def test_validate_orphan_annotation() -> None:
    coco = _minimal_coco()
    coco["annotations"][0]["image_id"] = 999
    with pytest.raises(CocoValidationError, match="image_id"):
        validate_coco(coco, strict=True)


def test_validate_bad_bbox_shape() -> None:
    coco = _minimal_coco()
    coco["annotations"][0]["bbox"] = [1, 2, 3]
    with pytest.raises(CocoValidationError, match="bbox"):
        validate_coco(coco, strict=True)


def test_validate_zero_dim_bbox() -> None:
    coco = _minimal_coco()
    coco["annotations"][0]["bbox"] = [10, 10, 0, 5]
    with pytest.raises(CocoValidationError, match="non-positive bbox"):
        validate_coco(coco, strict=True)


def test_validate_missing_iscrowd_is_warning_not_error() -> None:
    coco = _minimal_coco()
    del coco["annotations"][0]["iscrowd"]
    issues = validate_coco(coco, strict=False)
    severities = {i.severity for i in issues}
    assert "error" not in severities
    assert any("iscrowd" in i.message for i in issues)


def test_validate_image_file_existence(tmp_path: Path) -> None:
    coco = _minimal_coco()
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    # Only one of two files exists.
    (images_dir / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(CocoValidationError, match="file does not exist"):
        validate_coco(coco, images_dir=images_dir, strict=True)


def test_empty_coco_with_categories() -> None:
    cats = [{"id": 0, "name": "ships"}]
    coco = empty_coco(cats)
    assert coco["categories"] == cats
    assert coco["images"] == [] and coco["annotations"] == []


def test_index_by_image() -> None:
    idx = index_by_image(_minimal_coco())
    assert sorted(idx.keys()) == [1, 2]
    assert len(idx[1]) == 1 and idx[1][0]["id"] == 10
    assert len(idx[2]) == 1 and idx[2][0]["id"] == 11


def test_split_by_image_ids() -> None:
    coco = _minimal_coco()
    splits = split_by_image_ids(coco, {"train": [1], "val": [2], "test": []})

    assert [img["id"] for img in splits["train"]["images"]] == [1]
    assert [ann["id"] for ann in splits["train"]["annotations"]] == [10]
    assert [img["id"] for img in splits["val"]["images"]] == [2]
    assert [ann["id"] for ann in splits["val"]["annotations"]] == [11]
    assert splits["test"]["images"] == []
    # categories preserved in every split
    for name in ("train", "val", "test"):
        assert splits[name]["categories"] == coco["categories"]


def test_validate_extra_fields_preserved(tmp_path: Path) -> None:
    coco = _minimal_coco()
    coco["annotations"][0]["quality_tier"] = "exact_visible"
    coco["annotations"][0]["confidence_source"] = "visible_pixels"
    # Should not flag extras as errors.
    issues = validate_coco(coco, strict=True)
    assert all(i.severity != "error" for i in issues)
    # Round-trip preserves extras.
    p = save_coco(coco, tmp_path / "out.json")
    reloaded = load_coco(p)
    assert reloaded["annotations"][0]["quality_tier"] == "exact_visible"
