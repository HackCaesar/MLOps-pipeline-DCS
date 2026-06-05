"""Validate a raw dataset directory against the raw-dataset contract.

Checks:

- ``metadata/dataset_info.json`` and ``metadata/classes.json`` exist and parse;
- ``annotations/instances_{train,val,test}.json`` exist;
- each per-split COCO file passes structural validation (``coco_io.validate_coco``);
- referenced image files exist under ``images/{split}/``;
- categories in every split match ``classes.json``;
- the canonical 3-class contract holds (no ``buoys`` and no category outside
  ships/helicopters/airplanes — see ``packages/common/classes.py``);
- counts in ``dataset_info.json`` agree with the COCO files;
- ``source_frame_id`` (from filename stem) is **not shared** between splits — i.e.
  no data leakage from one source frame into multiple splits.

Exit codes:
  0 — clean (no errors; warnings printed);
  1 — structural errors found;
  2 — config or arguments problem.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from packages.common.classes import (
    ClassContractError,
    assert_categories_allowed,
    assert_category_ids_allowed,
)
from packages.common.coco_io import (
    CocoValidationIssue,
    load_coco,
    validate_coco,
)
from packages.common.config import ConfigError, load_config
from packages.common.logging_utils import get_logger, setup_logging
from packages.common.manifest import read_manifest
from packages.common.paths import resolve_path

LOG = get_logger(__name__)
SPLITS = ("train", "val", "test")


def validate_raw_dataset(dataset_dir: Path) -> tuple[int, int, list[str]]:
    """Run all validation checks. Returns (n_errors, n_warnings, lines)."""
    issues: list[CocoValidationIssue] = []
    lines: list[str] = []

    def add(severity: str, location: str, message: str) -> None:
        issues.append(CocoValidationIssue(severity, location, message))
        lines.append(f"[{severity}] {location}: {message}")

    if not dataset_dir.is_dir():
        add("error", str(dataset_dir), "dataset directory does not exist")
        return _summarize(issues, lines)

    # ---- required metadata files ----
    info_path = dataset_dir / "metadata" / "dataset_info.json"
    classes_path = dataset_dir / "metadata" / "classes.json"
    if not info_path.is_file():
        add("error", "metadata/dataset_info.json", "missing")
    if not classes_path.is_file():
        add("error", "metadata/classes.json", "missing")

    info: dict[str, Any] = {}
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            add("error", "metadata/dataset_info.json", f"malformed JSON: {exc}")

    classes_categories: list[dict] = []
    if classes_path.is_file():
        try:
            classes_doc = json.loads(classes_path.read_text(encoding="utf-8"))
            classes_categories = list(classes_doc.get("categories", []))
        except json.JSONDecodeError as exc:
            add("error", "metadata/classes.json", f"malformed JSON: {exc}")

    # ---- canonical 3-class contract (classes.json) ----
    if classes_categories:
        try:
            assert_categories_allowed(classes_categories)
        except ClassContractError as exc:
            add("error", "metadata/classes.json", str(exc))

    # ---- per-split COCO ----
    split_counts: dict[str, dict[str, int]] = {}
    image_ids_seen_global: set[Any] = set()
    annotation_ids_seen_global: set[Any] = set()
    frame_ids_by_split: dict[str, set[str]] = defaultdict(set)

    for split in SPLITS:
        coco_path = dataset_dir / "annotations" / f"instances_{split}.json"
        if not coco_path.is_file():
            add("error", f"annotations/instances_{split}.json", "missing")
            continue
        try:
            coco = load_coco(coco_path)
        except (ValueError, FileNotFoundError) as exc:
            add("error", f"annotations/instances_{split}.json", str(exc))
            continue

        split_images_dir = dataset_dir / "images" / split
        coco_issues = validate_coco(coco, images_dir=split_images_dir, strict=False)
        for ci in coco_issues:
            add(ci.severity, f"annotations/instances_{split}.json::{ci.location}", ci.message)

        split_counts[split] = {
            "num_images": len(coco["images"]),
            "num_annotations": len(coco["annotations"]),
        }

        # cross-split uniqueness
        for img in coco["images"]:
            key = (img["id"],)
            if img["id"] in image_ids_seen_global:
                add("warning", f"annotations/instances_{split}.json::images",
                    f"image id {img['id']} also seen in another split (acceptable but unusual)")
            image_ids_seen_global.add(img["id"])
            frame_ids_by_split[split].add(Path(img["file_name"]).stem)

        for ann in coco["annotations"]:
            if ann["id"] in annotation_ids_seen_global:
                add("error", f"annotations/instances_{split}.json::annotations",
                    f"annotation id {ann['id']} duplicated across splits")
            annotation_ids_seen_global.add(ann["id"])

        # categories vs classes.json
        if classes_categories:
            split_cat_ids = sorted(c["id"] for c in coco["categories"])
            expected_cat_ids = sorted(c["id"] for c in classes_categories)
            if split_cat_ids != expected_cat_ids:
                add("error", f"annotations/instances_{split}.json::categories",
                    f"category ids {split_cat_ids} ≠ classes.json {expected_cat_ids}")

        # canonical contract: no annotation may use a category outside the 3
        try:
            assert_category_ids_allowed(a["category_id"] for a in coco["annotations"])
        except ClassContractError as exc:
            add("error", f"annotations/instances_{split}.json::categories", str(exc))

    # ---- leakage: same frame_id in multiple splits? ----
    seen_frames: dict[str, list[str]] = defaultdict(list)
    for split, frames in frame_ids_by_split.items():
        for f in frames:
            seen_frames[f].append(split)
    for frame_id, splits in seen_frames.items():
        if len(splits) > 1:
            add("error", "split_leakage",
                f"source_frame_id={frame_id!r} appears in multiple splits: {splits}")

    # ---- counts vs dataset_info.json ----
    if info:
        for key in ("num_images", "num_annotations", "num_classes"):
            if key not in info:
                add("warning", "metadata/dataset_info.json", f"missing key {key!r}")
        info_splits = info.get("splits", {})
        for split, counts in split_counts.items():
            info_counts = info_splits.get(split, {})
            for k in ("num_images", "num_annotations"):
                if k in info_counts and info_counts[k] != counts[k]:
                    add("error", "metadata/dataset_info.json::splits",
                        f"{split}.{k} mismatch: dataset_info={info_counts[k]} vs COCO={counts[k]}")
        total_images = sum(c["num_images"] for c in split_counts.values())
        if "num_images" in info and info["num_images"] != total_images:
            add("error", "metadata/dataset_info.json",
                f"num_images={info['num_images']} ≠ sum across splits {total_images}")

    # ---- optional capture_manifest cross-check ----
    manifest_path = dataset_dir / "metadata" / "capture_manifest.jsonl"
    if manifest_path.is_file():
        try:
            manifest_splits: dict[str, set[str]] = defaultdict(set)
            for rec in read_manifest(manifest_path):
                manifest_splits[rec.get("split", "?")].add(rec.get("frame_id", ""))
            for split in SPLITS:
                coco_frames = frame_ids_by_split.get(split, set())
                man_frames = manifest_splits.get(split, set())
                missing_in_manifest = coco_frames - man_frames
                if missing_in_manifest:
                    add("warning", "metadata/capture_manifest.jsonl",
                        f"{split}: {len(missing_in_manifest)} frames missing from manifest")
        except (ValueError, FileNotFoundError) as exc:
            add("warning", "metadata/capture_manifest.jsonl", str(exc))

    return _summarize(issues, lines)


def _summarize(issues: list[CocoValidationIssue], lines: list[str]) -> tuple[int, int, list[str]]:
    n_err = sum(1 for i in issues if i.severity == "error")
    n_warn = sum(1 for i in issues if i.severity == "warning")
    return n_err, n_warn, lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=False, help="pipeline.yaml (used to default --dataset-dir)")
    p.add_argument("--dataset-dir", required=False,
                   help="explicit path to storage/datasets/raw/{dataset_id}")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-warn", action="store_true",
                   help="exit 0 even if warnings present (errors still fail)")
    args = p.parse_args(argv)

    setup_logging(level="WARNING" if args.quiet else "INFO")

    dataset_dir_raw = args.dataset_dir
    if not dataset_dir_raw and args.config:
        try:
            cfg = load_config(args.config)
        except ConfigError as exc:
            LOG.error("CONFIG ERROR: %s", exc)
            return 2
        dataset_dir_raw = (cfg.get("data") or {}).get("raw_dataset_dir")
    if not dataset_dir_raw:
        LOG.error("--dataset-dir is required (no value in config either)")
        return 2

    dataset_dir = resolve_path(dataset_dir_raw)
    assert dataset_dir is not None

    LOG.info("Validating raw dataset: %s", dataset_dir)
    n_err, n_warn, lines = validate_raw_dataset(dataset_dir)

    for ln in lines:
        # warnings to stderr only when not --quiet; errors always to stderr.
        if ln.startswith("[error]"):
            print(ln, file=sys.stderr)
        elif not args.quiet:
            print(ln)

    LOG.info("Summary: %d errors, %d warnings", n_err, n_warn)
    if n_err:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
