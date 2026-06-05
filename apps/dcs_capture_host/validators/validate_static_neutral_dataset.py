"""Validate the 10-mission static neutral dataset output."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

ALLOWED_CLASSES = {"ships", "helicopters", "airplanes"}
IMAGE_WIDTH = 2560.0
IMAGE_HEIGHT = 1440.0


def bbox_area(bbox: Sequence[float]) -> float:
    return (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1]))


def yolo_line_valid(line: str) -> bool:
    parts = line.split()
    if len(parts) != 5:
        return False
    try:
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
    except ValueError:
        return False
    if class_id not in {0, 1, 2}:
        return False
    x_center, y_center, width, height = values
    return 0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < width < 0.5 and 0.0 < height < 0.5


def validate(dataset_root: Path, mission_prefix: str) -> Dict[str, Any]:
    images = sorted((dataset_root / "images").glob(f"{mission_prefix}*.png"))
    metadata_paths = sorted((dataset_root / "metadata").glob(f"{mission_prefix}*.json"))
    yolo_paths = sorted((dataset_root / "yolo_labels").glob(f"{mission_prefix}*.txt"))
    coco_path = dataset_root / "annotations" / "_annotations.coco.json"

    errors: List[str] = []
    warnings: List[str] = []
    frame_counts: Counter[str] = Counter()
    valid_objects = 0
    class_counts: Counter[str] = Counter()
    bbox_area_ratios: List[float] = []
    camera_distances: Dict[str, List[float]] = defaultdict(list)
    camera_positions: Dict[str, set[tuple[float, float, float]]] = defaultdict(set)

    if len(images) != 100:
        errors.append(f"expected 100 images, found {len(images)}")
    if len(metadata_paths) != 100:
        errors.append(f"expected 100 metadata files, found {len(metadata_paths)}")
    if len(yolo_paths) != 100:
        errors.append(f"expected 100 YOLO label files, found {len(yolo_paths)}")
    if not coco_path.exists():
        errors.append(f"missing COCO annotations: {coco_path}")

    image_names = {path.stem for path in images}
    metadata_names = {path.stem for path in metadata_paths}
    yolo_names = {path.stem for path in yolo_paths}
    for missing in sorted(metadata_names - image_names):
        errors.append(f"metadata without image: {missing}")
    for missing in sorted(image_names - metadata_names):
        errors.append(f"image without metadata: {missing}")
    for missing in sorted(image_names - yolo_names):
        errors.append(f"image without YOLO labels: {missing}")

    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame_id = metadata.get("frame_id", metadata_path.stem)
        mission_id = Path(metadata.get("scene", {}).get("mission_id", "unknown")).stem
        frame_counts[mission_id] += 1

        if not metadata.get("usable"):
            errors.append(f"unusable frame: {frame_id} reasons={metadata.get('invalid_reasons')}")
        sync = metadata.get("sync", {})
        if not sync.get("strict_sync_valid"):
            errors.append(f"sync invalid: {frame_id}")
        if not sync.get("stale_screenshot_check_passed"):
            errors.append(f"stale screenshot check failed: {frame_id}")

        camera = metadata.get("camera", {})
        height = float(camera.get("height_asl_m", math.nan))
        if not math.isfinite(height) or abs(height - 15.0) > 0.10:
            errors.append(f"camera height not 15m: {frame_id} height={height}")
        pitch = float(camera.get("euler_deg", {}).get("pitch", math.nan))
        if not math.isfinite(pitch) or abs(pitch) > 0.25:
            errors.append(f"camera pitch not horizontal: {frame_id} pitch={pitch}")
        position = camera.get("world_position_m", {})
        if position:
            camera_positions[mission_id].add(
                (round(float(position.get("x", 0.0)), 1), round(float(position.get("y", 0.0)), 1), round(float(position.get("z", 0.0)), 1))
            )

        objects = metadata.get("objects", [])
        valid_in_frame = 0
        for obj in objects:
            cls = obj.get("class")
            if cls not in ALLOWED_CLASSES:
                errors.append(f"unexpected class in metadata: {frame_id} class={cls}")
                continue
            class_counts[cls] += 1
            if not obj.get("validation", {}).get("valid"):
                continue
            bbox = obj.get("projection", {}).get("bbox_xyxy_px")
            if not bbox:
                errors.append(f"valid object without bbox: {frame_id} object={obj.get('id')}")
                continue
            area = bbox_area(bbox)
            if area <= 0.0:
                errors.append(f"zero/negative bbox area: {frame_id} object={obj.get('id')} bbox={bbox}")
                continue
            ratio = area / (IMAGE_WIDTH * IMAGE_HEIGHT)
            bbox_area_ratios.append(ratio)
            if ratio >= 0.50:
                errors.append(f"bbox too huge: {frame_id} object={obj.get('id')} ratio={ratio:.3f}")
            if ratio <= 0.00005:
                warnings.append(f"bbox very small: {frame_id} object={obj.get('id')} ratio={ratio:.6f}")
            valid_in_frame += 1
            valid_objects += 1
            distance = obj.get("distance_to_camera_m")
            if distance is not None:
                camera_distances[mission_id].append(float(distance))
        if valid_in_frame == 0:
            errors.append(f"frame has no valid bbox: {frame_id}")

        yolo_path = dataset_root / "yolo_labels" / f"{frame_id}.txt"
        if yolo_path.exists():
            lines = [line.strip() for line in yolo_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                errors.append(f"empty YOLO labels: {frame_id}")
            for line in lines:
                if not yolo_line_valid(line):
                    errors.append(f"invalid YOLO line: {frame_id}: {line}")

    if len(frame_counts) != 10:
        errors.append(f"expected 10 missions in metadata, found {len(frame_counts)}")
    for mission_id, count in sorted(frame_counts.items()):
        if count != 10:
            errors.append(f"mission {mission_id} produced {count} frames, expected 10")
        if len(camera_positions[mission_id]) != 10:
            errors.append(f"mission {mission_id} has {len(camera_positions[mission_id])} unique camera positions, expected 10")
        if camera_distances[mission_id]:
            min_distance = min(camera_distances[mission_id])
            max_distance = max(camera_distances[mission_id])
            if min_distance < 250.0 or max_distance > 900.0:
                warnings.append(f"mission {mission_id} object distance range is broad: {min_distance:.1f}-{max_distance:.1f} m")

    coco_images = 0
    coco_annotations = 0
    if coco_path.exists():
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        coco_images = len(coco.get("images", []))
        coco_annotations = len(coco.get("annotations", []))
        if coco_images != 100:
            errors.append(f"COCO contains {coco_images} images, expected 100")
        if coco_annotations <= 0:
            errors.append("COCO has no annotations")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "total_images": len(images),
        "total_metadata": len(metadata_paths),
        "total_yolo_files": len(yolo_paths),
        "total_valid_objects": valid_objects,
        "class_counts": dict(sorted(class_counts.items())),
        "frames_per_mission": dict(sorted(frame_counts.items())),
        "coco_images": coco_images,
        "coco_annotations": coco_annotations,
        "bbox_area_ratio_min": min(bbox_area_ratios) if bbox_area_ratios else None,
        "bbox_area_ratio_max": max(bbox_area_ratios) if bbox_area_ratios else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate static neutral dataset output")
    parser.add_argument("--dataset-root", default=str(Path(__file__).resolve().parents[1] / "dataset"))
    parser.add_argument("--mission-prefix", default="caucasus_static_neutral_")
    parser.add_argument("--report", default=str(Path(__file__).resolve().parents[1] / "diagnostics" / "static_neutral_dataset_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate(Path(args.dataset_root), args.mission_prefix)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
