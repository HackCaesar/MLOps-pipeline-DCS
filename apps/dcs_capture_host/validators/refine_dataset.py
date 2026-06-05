"""Recompute existing dataset labels with visible-pixel bboxes.

Batch CLI that re-runs ``VisibleBBoxRefiner`` + ``DatasetExporter`` on an
already-captured ``dataset/`` directory. Useful when refinement logic
changes and the screenshots are still good.

Implementation note: this script reuses shared helpers from the orchestrator
package (``json_io.load_yaml_config``, ``runtime_paths.resolve_config_path``,
``overlay.render_overlay``) instead of carrying its own copies, so behavior
stays in lock-step with a freshly captured frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exporters.coco_yolo import DatasetExporter
from orchestrator.json_io import load_yaml_config
from orchestrator.overlay import render_overlay
from orchestrator.runtime_paths import resolve_config_path

from validators import VisibleBBoxRefiner


def iter_metadata(metadata_dir: Path, frame_id: Optional[str]) -> Iterable[Path]:
    if frame_id:
        yield metadata_dir / f"{frame_id}.json"
        return
    yield from sorted(metadata_dir.glob("*.json"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine existing DCS dataset bboxes from visible pixels")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument("--frame-id", help="Optional single frame id to refine")
    parser.add_argument("--no-overlays", action="store_true", help="Do not rewrite debug overlays")
    args = parser.parse_args()

    config = load_yaml_config(Path(args.config))
    output_root = resolve_config_path(config["paths"]["output_root"])
    if output_root is None:
        raise SystemExit("paths.output_root must be configured")
    image_format = config["export"]["image_format"]
    images_dir = output_root / "images"
    metadata_dir = output_root / "metadata"
    yolo_dir = output_root / "yolo_labels"
    annotations_dir = output_root / "annotations"
    overlays_dir = output_root / "debug_overlays"

    refiner = VisibleBBoxRefiner(config)
    yolo_exporter = DatasetExporter(
        config["generation"]["allowed_classes"],
        allowed_quality_tiers=config["pipeline"].get("allowed_quality_tiers_for_training"),
    )
    processed = 0
    refined = 0
    failed = 0
    usable = 0

    for metadata_path in iter_metadata(metadata_dir, args.frame_id):
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame_id = metadata["frame_id"]
        image_path = images_dir / f"{frame_id}.{image_format}"
        metadata = refiner.refine_frame(image_path, metadata)
        write_json(metadata_path, metadata)
        yolo_exporter.write_yolo(metadata, yolo_dir / f"{frame_id}.txt")
        if not args.no_overlays:
            render_overlay(
                image_path=image_path,
                frame_metadata=metadata,
                overlay_path=overlays_dir / f"{frame_id}_overlay.png",
            )
        processed += 1
        refined += metadata.get("bbox_refinement", {}).get("refined_objects", 0)
        failed += metadata.get("bbox_refinement", {}).get("failed_objects", 0)
        usable += int(bool(metadata.get("usable", False)))

    coco_exporter = DatasetExporter(
        config["generation"]["allowed_classes"],
        allowed_quality_tiers=config["pipeline"].get("allowed_quality_tiers_for_training"),
    )
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        coco_exporter.add_frame(metadata)
    coco_exporter.write_coco(annotations_dir / "_annotations.coco.json")
    print(
        json.dumps(
            {
                "processed_frames": processed,
                "usable_frames": usable,
                "refined_objects": refined,
                "failed_objects": failed,
                "coco": str(annotations_dir / "_annotations.coco.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
