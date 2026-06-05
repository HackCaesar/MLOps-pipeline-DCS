"""Register a validated raw dataset into the SQLite metadata store.

Reads ``metadata/dataset_info.json`` for top-level fields, scans each
per-split COCO file, and writes:

- one row in ``datasets``;
- one row per image in ``images``;
- one row per annotation in ``annotations``;
- ``datasets.content_hash`` is the sha256 of the dataset directory (via
  ``LocalStorageBackend.compute_directory_hash``).

Idempotent re-runs: by default fails with a clear error if the dataset_id is
already registered. Pass ``--replace`` to drop and re-insert.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from packages.common.coco_io import index_by_image, load_coco
from packages.common.config import ConfigError, load_config
from packages.common.logging_utils import get_logger, setup_logging
from packages.common.metadata_store import SQLiteMetadataStore
from packages.common.paths import resolve_path
from packages.common.storage import LocalStorageBackend

LOG = get_logger(__name__)
SPLITS = ("train", "val", "test")


def register_dataset(
    dataset_dir: Path,
    sqlite_path: Path,
    *,
    replace: bool = False,
    compute_hash: bool = True,
) -> dict[str, Any]:
    """Insert dataset rows into SQLite. Returns a summary dict for the CLI."""
    info_path = dataset_dir / "metadata" / "dataset_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"dataset_info.json not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    dataset_id = info["dataset_id"]

    content_hash: str | None = None
    if compute_hash:
        LOG.info("Computing directory content hash (this may take a moment)…")
        backend = LocalStorageBackend(dataset_dir.parent)
        content_hash = backend.compute_directory_hash(dataset_dir.name)

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "dataset_id": dataset_id,
        "images_registered":      0,
        "annotations_registered": 0,
        "splits": {},
        "content_hash":           content_hash,
        "replaced":               False,
    }

    with SQLiteMetadataStore(sqlite_path) as store:
        store.init_db()

        if store.get_dataset(dataset_id) is not None:
            if not replace:
                raise RuntimeError(
                    f"Dataset {dataset_id!r} is already registered. Pass --replace to overwrite."
                )
            LOG.warning("Dataset %s already exists — replacing.", dataset_id)
            _drop_dataset_rows(store, dataset_id)
            summary["replaced"] = True

        store.register_dataset(
            dataset_id=dataset_id,
            name=info.get("name", dataset_id),
            created_at=info["created_at"],
            source=info.get("source", "unknown"),
            storage_path=str(dataset_dir),
            annotation_format=info.get("annotation_format", "coco"),
            num_images=int(info["num_images"]),
            num_annotations=int(info["num_annotations"]),
            num_classes=int(info["num_classes"]),
            status="registered",
            content_hash=content_hash,
            notes=info.get("notes"),
        )

        for split in SPLITS:
            coco_path = dataset_dir / "annotations" / f"instances_{split}.json"
            if not coco_path.is_file():
                continue
            coco = load_coco(coco_path)
            anns_by_image = index_by_image(coco)
            for img in coco["images"]:
                image_id = _image_uid(dataset_id, split, img["id"])
                store.register_image(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    split=split,
                    file_name=img["file_name"],
                    width=int(img["width"]),
                    height=int(img["height"]),
                    storage_path=f"images/{split}/{img['file_name']}",
                    has_objects=bool(anns_by_image.get(img["id"], [])),
                    source_frame_id=Path(img["file_name"]).stem,
                )
                summary["images_registered"] += 1

            for ann in coco["annotations"]:
                ann_uid = _annotation_uid(dataset_id, split, ann["id"])
                store.register_annotation(
                    annotation_id=ann_uid,
                    dataset_id=dataset_id,
                    image_id=_image_uid(dataset_id, split, ann["image_id"]),
                    category_id=int(ann["category_id"]),
                    bbox_xywh=tuple(ann["bbox"]),
                    area=float(ann.get("area", 0.0)),
                    iscrowd=bool(ann.get("iscrowd", 0)),
                    source_annotation_id=str(ann["id"]),
                )
                summary["annotations_registered"] += 1

            summary["splits"][split] = {
                "num_images":      len(coco["images"]),
                "num_annotations": len(coco["annotations"]),
            }

    return summary


def _image_uid(dataset_id: str, split: str, raw_id: Any) -> str:
    return f"{dataset_id}:{split}:{raw_id}"


def _annotation_uid(dataset_id: str, split: str, raw_id: Any) -> str:
    return f"{dataset_id}:{split}:ann:{raw_id}"


def _drop_dataset_rows(store: SQLiteMetadataStore, dataset_id: str) -> None:
    # Replace the dataset's CHILD rows only (annotations → images, FK-safe order).
    # The datasets row itself is intentionally kept: tile_caches / pipeline_runs
    # reference it (FK, no ON DELETE CASCADE), so deleting it raises "FOREIGN KEY
    # constraint failed" on any re-register. register_dataset() upserts the row
    # in place (ON CONFLICT DO UPDATE), which preserves those references.
    conn = store._conn  # noqa: SLF001 — intentional reach-through for cleanup
    with conn:
        conn.execute("DELETE FROM annotations WHERE dataset_id = ?", (dataset_id,))
        conn.execute("DELETE FROM images WHERE dataset_id = ?", (dataset_id,))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="pipeline.yaml")
    p.add_argument("--dataset-dir", required=True,
                   help="path to the validated raw dataset directory")
    p.add_argument("--replace", action="store_true",
                   help="drop and re-insert if dataset_id already registered")
    p.add_argument("--skip-hash", action="store_true",
                   help="skip content_hash computation (faster for huge datasets in dev)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    setup_logging(level="WARNING" if args.quiet else "INFO")

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        LOG.error("CONFIG ERROR: %s", exc)
        return 2

    sqlite_path = resolve_path(cfg["metadata_store"]["sqlite_path"])
    assert sqlite_path is not None
    dataset_dir = resolve_path(args.dataset_dir)
    assert dataset_dir is not None

    try:
        summary = register_dataset(
            dataset_dir, sqlite_path,
            replace=args.replace, compute_hash=not args.skip_hash,
        )
    except FileNotFoundError as exc:
        LOG.error("%s", exc)
        return 1
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 1
    except sqlite3.IntegrityError as exc:
        LOG.error("SQLite integrity error: %s", exc)
        return 1

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    LOG.info("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
