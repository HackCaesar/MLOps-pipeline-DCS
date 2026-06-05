"""Reassemble ``dataset/annotations/_annotations.coco.json`` from per-frame
metadata produced by the capture pipeline.

Why this exists
---------------
``orchestrator/main.py`` creates a fresh ``DatasetExporter`` on every process
start and ``write_coco`` truncates ``_annotations.coco.json`` with whatever
that single process accumulated. When the batch runner calls the orchestrator
once per mission, the final COCO file ends up containing only the **last**
mission's frames — every previous batch is overwritten.

This script avoids re-running DCS. It scans ``dataset/metadata/*.json`` (the
authoritative per-frame metadata that the exporter itself reads) and rebuilds
``_annotations.coco.json`` using the same rules as ``DatasetExporter``:

* a frame is exported only when ``usable=True``;
* an object is exported only when ``validation.valid=True``,
  ``projection.bbox_xyxy_px`` is non-null, and ``quality_tier`` is allowed.

CLI:

    python scripts/rebuild_coco_from_metadata.py \\
        --metadata-dir dataset/metadata \\
        --output dataset/annotations/_annotations.coco.json \\
        --classes ships helicopters airplanes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Use the existing exporter's filter rules instead of duplicating them —
# avoids the previous AI's "I'll just rewrite the filter inline" mistake.
from exporters.coco_yolo import (  # noqa: E402  (sys.path tweak above)
    _is_usable_frame,
    _is_valid_object,
    _make_coco_annotation,
    _make_coco_categories,
    _make_coco_image,
    _object_bbox_xyxy,
)

DEFAULT_CLASSES = ("ships", "helicopters", "airplanes")
DEFAULT_QUALITY_TIERS = ("exact_visible", "exact", "strong", "approximate")


def load_metadata(metadata_dir: Path) -> List[Mapping[str, object]]:
    files = sorted(metadata_dir.glob("*.json"))
    frames: List[Mapping[str, object]] = []
    for path in files:
        try:
            frames.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] cannot read {path}: {exc}", file=sys.stderr)
    return frames


def _object_quality_allowed(obj: Mapping[str, object], allowed: Sequence[str]) -> bool:
    if not allowed:
        return True
    return obj.get("quality_tier", "approximate") in allowed


def rebuild(
    frames: Iterable[Mapping[str, object]],
    classes: Sequence[str],
    quality_tiers: Sequence[str],
) -> dict:
    category_map = {name: index for index, name in enumerate(classes)}
    coco_images: List[dict] = []
    coco_annotations: List[dict] = []
    next_annotation_id = 1
    skipped_frames = 0
    skipped_objects = 0

    # Stable ordering: by frame_id when available, otherwise by image file_name.
    def _frame_sort_key(frame: Mapping[str, object]) -> str:
        return str(frame.get("frame_id") or frame.get("image", {}).get("file_name", ""))

    for frame in sorted(frames, key=_frame_sort_key):
        if not _is_usable_frame(frame):
            skipped_frames += 1
            continue
        image_info = frame.get("image") or {}
        if not image_info.get("file_name"):
            skipped_frames += 1
            continue
        image_id = len(coco_images) + 1
        coco_images.append(_make_coco_image(image_id, image_info))

        for obj in frame.get("objects", []):
            if not _is_valid_object(obj):
                skipped_objects += 1
                continue
            if not _object_quality_allowed(obj, quality_tiers):
                skipped_objects += 1
                continue
            bbox = _object_bbox_xyxy(obj)
            if not bbox:
                skipped_objects += 1
                continue
            class_name = obj.get("class")
            if class_name not in category_map:
                skipped_objects += 1
                continue
            coco_annotations.append(
                _make_coco_annotation(
                    annotation_id=next_annotation_id,
                    image_id=image_id,
                    category_id=category_map[class_name],
                    bbox_xyxy=bbox,
                    obj=obj,
                )
            )
            next_annotation_id += 1

    return {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": _make_coco_categories(category_map),
        "_stats": {
            "frames_seen": len(coco_images) + skipped_frames,
            "frames_exported": len(coco_images),
            "frames_skipped": skipped_frames,
            "annotations_exported": len(coco_annotations),
            "objects_skipped": skipped_objects,
        },
    }


def write_coco(coco: dict, output_path: Path, drop_stats: bool = True) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(coco)
    stats = payload.pop("_stats", None) if drop_stats else None
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if stats is not None:
        sidecar = output_path.with_suffix(output_path.suffix + ".rebuild_stats.json")
        sidecar.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-dir",
        default=str(PROJECT_ROOT / "dataset" / "metadata"),
        help="Directory containing one *.json file per captured frame.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "dataset" / "annotations" / "_annotations.coco.json"),
        help="Where to write the merged COCO file.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="Ordered class names; the index becomes category_id.",
    )
    parser.add_argument(
        "--quality-tiers",
        nargs="+",
        default=list(DEFAULT_QUALITY_TIERS),
        help="Allowed quality_tier values. Pass --quality-tiers '' to keep all.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    metadata_dir = Path(args.metadata_dir)
    if not metadata_dir.is_dir():
        print(f"[error] metadata dir not found: {metadata_dir}", file=sys.stderr)
        return 2

    quality_tiers = [tier for tier in args.quality_tiers if tier]
    frames = load_metadata(metadata_dir)
    if not frames:
        print(f"[warn] no metadata JSONs in {metadata_dir}", file=sys.stderr)

    coco = rebuild(frames, args.classes, quality_tiers)
    output_path = Path(args.output)
    write_coco(coco, output_path, drop_stats=True)

    stats = coco["_stats"]
    print(json.dumps({
        "output": str(output_path),
        "frames_seen": stats["frames_seen"],
        "frames_exported": stats["frames_exported"],
        "frames_skipped": stats["frames_skipped"],
        "annotations": stats["annotations_exported"],
        "objects_skipped": stats["objects_skipped"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
