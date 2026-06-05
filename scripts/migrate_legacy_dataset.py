"""Migrate a legacy raw dataset into the canonical storage, applying the 3-class
contract — WITHOUT copying images in full and WITHOUT destroying the source.

What it does
------------
Given a legacy raw-contract dataset (``images/{train,val,test}/``,
``annotations/instances_{split}.json``, ``metadata/{classes,dataset_info}.json``)
it produces the same dataset under ``{target_root}/{dataset_id}/`` with:

* images placed via **hardlink** (fallback to copy across filesystems), or
  ``--link-mode move`` / ``copy`` — never a blind full ``cp -r``;
* COCO categories + annotations remapped to the canonical 3 classes when the
  source is the legacy 4-class layout (``buoys`` dropped; helicopters/airplanes
  renumbered 2,3 -> 1,2 — see ``packages/common/classes.py``). An already-canonical
  source is relocated unchanged.

Safety
------
* never deletes the source unless ``--delete-source`` is passed explicitly;
* refuses to overwrite a *different* existing target unless ``--overwrite``
  (an identical target — same content hash — is treated as a no-op);
* ``--dry-run`` prints the plan and writes nothing;
* writes ``migration_report.json`` into the target.

This script is **not** wired into any pipeline and never runs automatically.

Usage
-----
    python -m scripts.migrate_legacy_dataset \
        --source-dir D:/yolo/storage/datasets/raw/dcs_caucasus_100 \
        --dataset-id dcs_caucasus_100 \
        --target-root D:/MLOps_storage/datasets/raw \
        --dry-run
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

# Make `packages`/`apps` importable when run as a plain script too.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apps.dcs_capture.adapter import _place_image  # reuse hardlink/copy/move helper
from packages.common.classes import (
    CANONICAL_CATEGORIES,
    LEGACY_CATEGORY_ID_REMAP,
    ClassContractError,
)

SPLITS = ("train", "val", "test")

_CANON_BY_ID = {int(c["id"]): str(c["name"]) for c in CANONICAL_CATEGORIES}
_LEGACY4_BY_ID = {0: "ships", 1: "buoys", 2: "helicopters", 3: "airplanes"}


# --------------------------------------------------------------------------- #
# classification + remap
# --------------------------------------------------------------------------- #

def classify_source(categories: list[dict]) -> str:
    """Return 'canonical' (3 classes), 'legacy4' (4 classes incl buoys), or 'unknown'."""
    by_id = {int(c["id"]): str(c["name"]) for c in categories}
    if by_id == _CANON_BY_ID:
        return "canonical"
    if by_id == _LEGACY4_BY_ID:
        return "legacy4"
    return "unknown"


def remap_split_coco(coco: dict, *, source_kind: str) -> tuple[dict, int]:
    """Return (new_coco, dropped_annotations). Identity for a canonical source."""
    out = dict(coco)
    if source_kind == "canonical":
        out["categories"] = [dict(c) for c in CANONICAL_CATEGORIES]
        return out, 0

    # legacy4 -> canonical
    new_anns: list[dict] = []
    dropped = 0
    for ann in coco.get("annotations", []):
        old = int(ann["category_id"])
        if old not in LEGACY_CATEGORY_ID_REMAP:
            raise ClassContractError(
                f"annotation category_id {old} has no legacy→canonical mapping"
            )
        new_id = LEGACY_CATEGORY_ID_REMAP[old]
        if new_id is None:  # buoys -> dropped
            dropped += 1
            continue
        a = dict(ann)
        a["category_id"] = new_id
        new_anns.append(a)
    out["annotations"] = new_anns
    out["categories"] = [dict(c) for c in CANONICAL_CATEGORIES]
    return out, dropped


# --------------------------------------------------------------------------- #
# hashing
# --------------------------------------------------------------------------- #

def dataset_content_hash(dataset_dir: Path) -> str:
    """Stable hash over annotation JSONs + (image relpath, size) pairs."""
    h = hashlib.sha256()
    ann_dir = dataset_dir / "annotations"
    for split in SPLITS:
        p = ann_dir / f"instances_{split}.json"
        if p.is_file():
            h.update(p.read_bytes())
    images_dir = dataset_dir / "images"
    if images_dir.is_dir():
        for img in sorted(images_dir.rglob("*")):
            if img.is_file():
                rel = img.relative_to(images_dir).as_posix()
                h.update(rel.encode("utf-8"))
                h.update(str(img.stat().st_size).encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# migration
# --------------------------------------------------------------------------- #

def migrate_legacy_dataset(
    source_dir: str | Path,
    target_root: str | Path,
    *,
    dataset_id: str,
    link_mode: str = "hardlink",
    allow_copy_fallback: bool = False,
    delete_source: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    source = Path(source_dir).resolve()
    target = (Path(target_root) / dataset_id).resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"source dataset not found: {source}")
    classes_path = source / "metadata" / "classes.json"
    if not classes_path.is_file():
        raise FileNotFoundError(f"missing {classes_path} (not a raw-contract dataset?)")

    src_categories = json.loads(classes_path.read_text(encoding="utf-8")).get("categories", [])
    source_kind = classify_source(src_categories)
    if source_kind == "unknown":
        raise ClassContractError(
            "source categories are neither the canonical 3 nor the legacy 4-class "
            f"layout: {src_categories}"
        )

    src_hash = dataset_content_hash(source)

    report: dict[str, Any] = {
        "source_dir": str(source),
        "target_dir": str(target),
        "dataset_id": dataset_id,
        "source_kind": source_kind,
        "link_mode": link_mode,
        "allow_copy_fallback": allow_copy_fallback,
        "source_content_hash": src_hash,
        "dry_run": dry_run,
        "delete_source": delete_source,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "splits": {},
        "dropped_annotations_total": 0,
        "status": "planned" if dry_run else "running",
    }

    # target already present?
    if target.exists():
        if target.is_dir() and (target / "metadata" / "classes.json").is_file():
            if dataset_content_hash(target) == src_hash and source_kind == "canonical":
                report["status"] = "noop_identical_target"
                return report
        if not overwrite:
            raise FileExistsError(
                f"target {target} already exists and differs; pass --overwrite to replace"
            )
        if not dry_run:
            shutil.rmtree(target)

    # plan / execute per split
    total_dropped = 0
    for split in SPLITS:
        coco_path = source / "annotations" / f"instances_{split}.json"
        split_info: dict[str, Any] = {"present": coco_path.is_file()}
        if coco_path.is_file():
            coco = json.loads(coco_path.read_text(encoding="utf-8"))
            new_coco, dropped = remap_split_coco(coco, source_kind=source_kind)
            total_dropped += dropped
            split_info.update({
                "num_images": len(new_coco.get("images", [])),
                "num_annotations": len(new_coco.get("annotations", [])),
                "dropped_annotations": dropped,
            })
            if not dry_run:
                (target / "annotations").mkdir(parents=True, exist_ok=True)
                (target / "images" / split).mkdir(parents=True, exist_ok=True)
                (target / "annotations" / f"instances_{split}.json").write_text(
                    json.dumps(new_coco, indent=2, ensure_ascii=False), encoding="utf-8",
                )
                for img in new_coco.get("images", []):
                    src_img = source / "images" / split / img["file_name"]
                    dst_img = target / "images" / split / img["file_name"]
                    if src_img.is_file():
                        _place_image(src_img, dst_img, link_mode,
                                     allow_copy_fallback=allow_copy_fallback)
        report["splits"][split] = split_info

    report["dropped_annotations_total"] = total_dropped

    # metadata: classes.json (canonical) + dataset_info.json (num_classes=3, recount)
    if not dry_run:
        meta_dir = target / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "classes.json").write_text(
            json.dumps({"categories": [dict(c) for c in CANONICAL_CATEGORIES]},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        info_path = source / "metadata" / "dataset_info.json"
        info: dict[str, Any] = {}
        if info_path.is_file():
            info = json.loads(info_path.read_text(encoding="utf-8"))
        info["dataset_id"] = dataset_id
        info["num_classes"] = len(CANONICAL_CATEGORIES)
        total_imgs = sum(s.get("num_images", 0) for s in report["splits"].values())
        total_anns = sum(s.get("num_annotations", 0) for s in report["splits"].values())
        info["num_images"] = total_imgs
        info["num_annotations"] = total_anns
        info["splits"] = {
            sp: {"num_images": report["splits"][sp].get("num_images", 0),
                 "num_annotations": report["splits"][sp].get("num_annotations", 0)}
            for sp in SPLITS if report["splits"][sp].get("present")
        }
        info["migrated_from"] = str(source)
        (meta_dir / "dataset_info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        # carry over capture_manifest if present (small, just copy)
        man = source / "metadata" / "capture_manifest.jsonl"
        if man.is_file():
            shutil.copy2(man, meta_dir / "capture_manifest.jsonl")

        report["target_content_hash"] = dataset_content_hash(target)
        (target / "migration_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
        )

    # source deletion only on explicit request
    if delete_source and not dry_run:
        shutil.rmtree(source)
        report["source_deleted"] = True
    else:
        report["source_deleted"] = False

    report["status"] = "planned" if dry_run else "done"
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", required=True, help="legacy raw-contract dataset dir")
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--target-root", required=True,
                   help="canonical raw root, e.g. D:/MLOps_storage/datasets/raw")
    p.add_argument("--link-mode", choices=["hardlink", "copy", "move"], default="hardlink",
                   help="how to place images (default: hardlink; fails across filesystems unless --allow-copy-fallback)")
    p.add_argument("--allow-copy-fallback", action="store_true",
                   help="permit copy when hardlink fails (cross-filesystem); off by default")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing, differing target")
    p.add_argument("--delete-source", action="store_true",
                   help="DANGEROUS: remove the source after a successful migration")
    p.add_argument("--dry-run", action="store_true", help="print plan, write nothing")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = migrate_legacy_dataset(
            source_dir=args.source_dir,
            target_root=args.target_root,
            dataset_id=args.dataset_id,
            link_mode=args.link_mode,
            allow_copy_fallback=args.allow_copy_fallback,
            delete_source=args.delete_source,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, FileExistsError, ClassContractError) as exc:
        print(f"[migrate] ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
