"""COCO JSON I/O + structural validation.

This module owns the **schema** of the raw/enriched dataset COCO files:

- ``images[]``: ``id``, ``file_name``, ``width``, ``height``;
- ``annotations[]``: ``id``, ``image_id``, ``category_id``, ``bbox`` (xywh),
  ``area``, ``iscrowd``;
- ``categories[]``: ``id``, ``name``.

Extra fields (``quality_tier``, ``confidence_source``) from the DCS exporter
are preserved on read/write but not required by the validator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REQUIRED_IMAGE_FIELDS = ("id", "file_name", "width", "height")
REQUIRED_ANNOTATION_FIELDS = ("id", "image_id", "category_id", "bbox")
REQUIRED_CATEGORY_FIELDS = ("id", "name")


@dataclass(frozen=True)
class CocoValidationIssue:
    severity: str  # "error" | "warning"
    location: str
    message: str


class CocoValidationError(ValueError):
    def __init__(self, issues: Sequence[CocoValidationIssue]) -> None:
        self.issues = list(issues)
        msg = "\n".join(f"  [{i.severity}] {i.location}: {i.message}" for i in self.issues)
        super().__init__(f"COCO validation failed:\n{msg}")


def load_coco(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"COCO file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"COCO root must be an object, got {type(data).__name__}")
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(f"COCO root missing required key {key!r}: {path}")
    return data


def save_coco(coco: dict[str, Any], path: str | Path, indent: int | None = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=indent)
    return path


def validate_coco(
    coco: dict[str, Any],
    *,
    images_dir: Path | None = None,
    strict: bool = True,
) -> list[CocoValidationIssue]:
    """Validate structural invariants. Returns list of issues; raises in ``strict`` mode.

    Checks:
    - required top-level keys exist;
    - per-image: required fields present, positive dimensions, file_name non-empty,
      file exists in ``images_dir`` if provided;
    - per-annotation: required fields present, bbox is length-4 with positive w/h,
      ``image_id`` references an existing image, ``category_id`` references an existing
      category, ``iscrowd`` defaults to 0 if missing (warning), area positive;
    - image_id uniqueness;
    - annotation_id uniqueness;
    - category_id uniqueness.
    """
    issues: list[CocoValidationIssue] = []

    for key in ("images", "annotations", "categories"):
        if key not in coco:
            issues.append(CocoValidationIssue("error", key, f"missing required key {key!r}"))
    if issues:
        if strict:
            raise CocoValidationError(issues)
        return issues

    images_by_id: dict[int, dict] = {}
    for idx, img in enumerate(coco.get("images", [])):
        loc = f"images[{idx}]"
        for field in REQUIRED_IMAGE_FIELDS:
            if field not in img:
                issues.append(CocoValidationIssue("error", loc, f"missing {field!r}"))
        if not all(field in img for field in REQUIRED_IMAGE_FIELDS):
            continue
        if int(img["width"]) <= 0 or int(img["height"]) <= 0:
            issues.append(CocoValidationIssue("error", loc,
                                              f"non-positive dimensions {img['width']}×{img['height']}"))
        if not str(img.get("file_name", "")).strip():
            issues.append(CocoValidationIssue("error", loc, "empty file_name"))
        if img["id"] in images_by_id:
            issues.append(CocoValidationIssue("error", loc, f"duplicate image id {img['id']!r}"))
        images_by_id[img["id"]] = img
        if images_dir is not None:
            f_path = Path(images_dir) / str(img["file_name"])
            if not f_path.exists():
                issues.append(CocoValidationIssue("error", loc, f"file does not exist: {f_path}"))

    categories_by_id: dict[int, dict] = {}
    for idx, cat in enumerate(coco.get("categories", [])):
        loc = f"categories[{idx}]"
        for field in REQUIRED_CATEGORY_FIELDS:
            if field not in cat:
                issues.append(CocoValidationIssue("error", loc, f"missing {field!r}"))
                continue
        if cat["id"] in categories_by_id:
            issues.append(CocoValidationIssue("error", loc, f"duplicate category id {cat['id']!r}"))
        categories_by_id[cat["id"]] = cat

    annotation_ids: set = set()
    for idx, ann in enumerate(coco.get("annotations", [])):
        loc = f"annotations[{idx}]"
        for field in REQUIRED_ANNOTATION_FIELDS:
            if field not in ann:
                issues.append(CocoValidationIssue("error", loc, f"missing {field!r}"))
        if not all(field in ann for field in REQUIRED_ANNOTATION_FIELDS):
            continue
        if ann["id"] in annotation_ids:
            issues.append(CocoValidationIssue("error", loc, f"duplicate annotation id {ann['id']!r}"))
        annotation_ids.add(ann["id"])
        if ann["image_id"] not in images_by_id:
            issues.append(CocoValidationIssue("error", loc, f"image_id {ann['image_id']!r} not in images"))
        if ann["category_id"] not in categories_by_id:
            issues.append(CocoValidationIssue("error", loc, f"category_id {ann['category_id']!r} not in categories"))
        bbox = ann["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            issues.append(CocoValidationIssue("error", loc, f"bbox must be [x,y,w,h], got {bbox!r}"))
            continue
        x, y, w, h = (float(v) for v in bbox)
        if w <= 0 or h <= 0:
            issues.append(CocoValidationIssue("error", loc,
                                              f"non-positive bbox size {w}×{h}"))
        area = ann.get("area")
        if area is not None and float(area) <= 0:
            issues.append(CocoValidationIssue("warning", loc, f"area is non-positive: {area!r}"))
        if "iscrowd" not in ann:
            issues.append(CocoValidationIssue("warning", loc, "iscrowd missing, defaulting to 0"))

    errors = [i for i in issues if i.severity == "error"]
    if strict and errors:
        raise CocoValidationError(issues)
    return issues


def empty_coco(categories: Iterable[dict] | None = None) -> dict[str, Any]:
    """Return an empty COCO dict, optionally pre-populated with categories."""
    return {
        "images": [],
        "annotations": [],
        "categories": list(categories) if categories is not None else [],
    }


def index_by_image(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Return ``image_id → [annotations]`` dict."""
    out: dict[int, list[dict[str, Any]]] = {img["id"]: [] for img in coco.get("images", [])}
    for ann in coco.get("annotations", []):
        out.setdefault(ann["image_id"], []).append(ann)
    return out


def split_by_image_ids(
    coco: dict[str, Any], image_ids_by_split: dict[str, Iterable[int]],
) -> dict[str, dict[str, Any]]:
    """Split a single COCO file into multiple per-split COCO files.

    Annotations follow images. Categories are copied as-is to every split.
    """
    categories = list(coco.get("categories", []))
    splits: dict[str, dict[str, Any]] = {
        name: empty_coco(categories) for name in image_ids_by_split
    }
    by_image: dict[int, list[dict]] = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    for name, ids in image_ids_by_split.items():
        wanted = set(ids)
        for img in coco["images"]:
            if img["id"] in wanted:
                splits[name]["images"].append(img)
                splits[name]["annotations"].extend(by_image.get(img["id"], []))
    return splits
