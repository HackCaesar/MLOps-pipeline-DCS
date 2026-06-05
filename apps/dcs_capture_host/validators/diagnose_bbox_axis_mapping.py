"""Offline diagnostic for DCS local bbox axis/sign mapping.

This script does not change production projection. It reprojects snapshot local bbox
corners through several candidate mappings and compares each result to the
visible/final bbox already stored in frame metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projection import CameraIntrinsics, CameraPose, project_object_corners

DEFAULT_FRAMES = ["pose_sync_unique_01", "pose_sync_unique_03", "pose_sync_unique_05"]


@dataclass(frozen=True)
class OrderVariant:
    name: str
    indices: Tuple[int, int, int]
    signs: Tuple[float, float, float]


@dataclass(frozen=True)
class BasisVariant:
    name: str
    axes: Tuple[str, str, str]


ORDER_VARIANTS = [
    OrderVariant("x_y_z", (0, 1, 2), (1.0, 1.0, 1.0)),
    OrderVariant("negx_y_z", (0, 1, 2), (-1.0, 1.0, 1.0)),
    OrderVariant("x_y_negz", (0, 1, 2), (1.0, 1.0, -1.0)),
    OrderVariant("negx_y_negz", (0, 1, 2), (-1.0, 1.0, -1.0)),
    OrderVariant("z_y_x", (2, 1, 0), (1.0, 1.0, 1.0)),
    OrderVariant("negz_y_x", (2, 1, 0), (-1.0, 1.0, 1.0)),
    OrderVariant("z_y_negx", (2, 1, 0), (1.0, 1.0, -1.0)),
    OrderVariant("negz_y_negx", (2, 1, 0), (-1.0, 1.0, -1.0)),
]

BASIS_VARIANTS = [
    BasisVariant("axis_x_forward_axis_y_up_axis_z_right", ("axis_x", "axis_y", "axis_z")),
    BasisVariant("axis_x_right_axis_y_up_axis_z_forward", ("axis_z", "axis_y", "axis_x")),
    BasisVariant("axis_z_forward_axis_y_up_axis_x_right", ("axis_z", "axis_y", "axis_x")),
]

CURRENT_MAPPING_NAME = "axis_x_forward_axis_y_up_axis_z_right__x_y_z"


def is_windows_absolute(raw_path: str) -> bool:
    return len(raw_path) > 1 and raw_path[1] == ":"


def resolve_config_path(raw_path: Optional[str]) -> Optional[Path]:
    if raw_path in (None, "", "null"):
        return None
    if is_windows_absolute(raw_path):
        return Path(raw_path)
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyYAML is required to load config files") from exc
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def vec3(raw: Dict[str, Any]) -> np.ndarray:
    return np.asarray([raw["x"], raw["y"], raw["z"]], dtype=np.float64)


def bbox_area(bbox: Optional[Sequence[float]]) -> float:
    if not bbox:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5)


def bbox_iou(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    if not a or not b:
        return None
    inter = bbox_area(
        [
            max(float(a[0]), float(b[0])),
            max(float(a[1]), float(b[1])),
            min(float(a[2]), float(b[2])),
            min(float(a[3]), float(b[3])),
        ]
    )
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0.0:
        return None
    return inter / union


def center_error(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    if not a or not b:
        return None
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def width_height_ratios(candidate: Optional[Sequence[float]], target: Optional[Sequence[float]]) -> Tuple[Optional[float], Optional[float]]:
    if not candidate or not target:
        return None, None
    target_w = max(1e-6, float(target[2]) - float(target[0]))
    target_h = max(1e-6, float(target[3]) - float(target[1]))
    return (float(candidate[2]) - float(candidate[0])) / target_w, (float(candidate[3]) - float(candidate[1])) / target_h


def area_ratio(candidate: Optional[Sequence[float]], target: Optional[Sequence[float]]) -> Optional[float]:
    target_area = bbox_area(target)
    if not candidate or target_area <= 0.0:
        return None
    return bbox_area(candidate) / target_area


def make_local_corners(bbox_min: Dict[str, Any], bbox_max: Dict[str, Any]) -> np.ndarray:
    min_x, min_y, min_z = float(bbox_min["x"]), float(bbox_min["y"]), float(bbox_min["z"])
    max_x, max_y, max_z = float(bbox_max["x"]), float(bbox_max["y"]), float(bbox_max["z"])
    return np.asarray(
        [
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ],
        dtype=np.float64,
    )


def mapped_corners_world(
    local_corners: np.ndarray,
    origin_w: np.ndarray,
    raw_basis: Dict[str, Any],
    order: OrderVariant,
    basis: BasisVariant,
) -> np.ndarray:
    reordered = local_corners[:, list(order.indices)] * np.asarray(order.signs, dtype=np.float64)[None, :]
    axes = [vec3(raw_basis[axis_name]) for axis_name in basis.axes]
    rotation = np.column_stack(axes)
    return (rotation @ reordered.T).T + origin_w


def camera_context(metadata: Dict[str, Any]) -> Tuple[CameraPose, CameraIntrinsics]:
    camera = metadata["camera"]
    position = camera["world_position_m"]
    basis = camera["world_basis"]
    image = metadata["image"]
    intr = camera["intrinsics"]
    clip = camera.get("clip_planes_m", {})
    pose = CameraPose.from_basis(
        position_w=[position["x"], position["y"], position["z"]],
        forward_w=[basis["forward"]["x"], basis["forward"]["y"], basis["forward"]["z"]],
        up_w=[basis["up"]["x"], basis["up"]["y"], basis["up"]["z"]],
        right_w=[basis["right"]["x"], basis["right"]["y"], basis["right"]["z"]],
    )
    intrinsics = CameraIntrinsics(
        width_px=int(image["width"]),
        height_px=int(image["height"]),
        fx_px=float(intr["fx_px"]),
        fy_px=float(intr["fy_px"]),
        cx_px=float(intr["cx_px"]),
        cy_px=float(intr["cy_px"]),
        near_m=clip.get("near"),
        far_m=clip.get("far"),
        source=str(intr.get("source", "metadata")),
    )
    return pose, intrinsics


def mapping_name(basis: BasisVariant, order: OrderVariant) -> str:
    return f"{basis.name}__{order.name}"


def target_bbox(projection: Dict[str, Any]) -> Tuple[Optional[List[float]], str]:
    if projection.get("bbox_xyxy_visible_px"):
        return projection["bbox_xyxy_visible_px"], "visible"
    if projection.get("bbox_xyxy_px"):
        return projection["bbox_xyxy_px"], "final"
    return None, "missing"


def metric_row(
    mapping: str,
    candidate: Optional[Sequence[float]],
    target: Optional[Sequence[float]],
    visible: Optional[Sequence[float]],
    final: Optional[Sequence[float]],
) -> Dict[str, Any]:
    width_ratio, height_ratio = width_height_ratios(candidate, target)
    return {
        "mapping": mapping,
        "projected_bbox_xyxy_px": [float(value) for value in candidate] if candidate else None,
        "iou_target": bbox_iou(candidate, target),
        "iou_visible": bbox_iou(candidate, visible),
        "iou_final": bbox_iou(candidate, final),
        "center_error_px": center_error(candidate, target),
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "bbox_area_ratio": area_ratio(candidate, target),
    }


def object_mapping_metrics(
    frame_id: str,
    metadata_obj: Dict[str, Any],
    snapshot_obj: Dict[str, Any],
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> Optional[Dict[str, Any]]:
    raw_basis = metadata_obj.get("world_basis_raw")
    position = metadata_obj.get("world_position_m")
    bbox_min = snapshot_obj.get("bbox_min_local")
    bbox_max = snapshot_obj.get("bbox_max_local")
    if not raw_basis or not position or not bbox_min or not bbox_max:
        return None

    local_corners = make_local_corners(bbox_min, bbox_max)
    origin = vec3(position)
    projection = metadata_obj.get("projection", {})
    visible_bbox = projection.get("bbox_xyxy_visible_px")
    final_bbox = projection.get("bbox_xyxy_px")
    current_geometry = projection.get("bbox_xyxy_geometry_px") or projection.get("bbox_xyxy_unclipped_px")
    target, target_kind = target_bbox(projection)

    mapping_metrics: List[Dict[str, Any]] = []
    for basis in BASIS_VARIANTS:
        for order in ORDER_VARIANTS:
            name = mapping_name(basis, order)
            corners_world = mapped_corners_world(local_corners, origin, raw_basis, order, basis)
            projected = project_object_corners(
                object_id=str(metadata_obj.get("id", "unknown")),
                corners_world=corners_world,
                pose=pose,
                intrinsics=intrinsics,
            )
            mapping_metrics.append(metric_row(name, projected.bbox_xyxy_px, target, visible_bbox, final_bbox))

    current = metric_row("production_metadata_geometry", current_geometry, target, visible_bbox, final_bbox)
    current_mapping = next((row for row in mapping_metrics if row["mapping"] == CURRENT_MAPPING_NAME), None)

    valid_candidates = [row for row in mapping_metrics if row["iou_target"] is not None]
    best = max(valid_candidates, key=lambda row: row["iou_target"]) if valid_candidates else None

    return {
        "frame_id": frame_id,
        "object_id": str(metadata_obj.get("id", "unknown")),
        "class": metadata_obj.get("class"),
        "type_name": metadata_obj.get("type_name"),
        "target_kind": target_kind,
        "basis_source": metadata_obj.get("basis_source"),
        "position_source": metadata_obj.get("position_source"),
        "object_pose_cache_match": metadata_obj.get("object_pose_cache_match"),
        "target_bbox_xyxy_px": [float(value) for value in target] if target else None,
        "visible_bbox_xyxy_px": visible_bbox,
        "final_bbox_xyxy_px": final_bbox,
        "production_geometry_bbox_xyxy_px": current_geometry,
        "current_mapping_iou": current_mapping["iou_target"] if current_mapping else None,
        "production_geometry_iou": current["iou_target"],
        "best_mapping": best["mapping"] if best else None,
        "best_iou": best["iou_target"] if best else None,
        "center_error_current": current_mapping["center_error_px"] if current_mapping else None,
        "center_error_best": best["center_error_px"] if best else None,
        "width_ratio_best": best["width_ratio"] if best else None,
        "height_ratio_best": best["height_ratio"] if best else None,
        "bbox_area_ratio_best": best["bbox_area_ratio"] if best else None,
        "mapping_metrics": mapping_metrics,
    }


def aggregate(objects: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    objects = list(objects)
    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for obj in objects:
        by_class[str(obj.get("class", "unknown"))].append(obj)

    def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        current_values = [row["current_mapping_iou"] for row in rows if row.get("current_mapping_iou") is not None]
        best_values = [row["best_iou"] for row in rows if row.get("best_iou") is not None]
        return {
            "objects": len(rows),
            "best_mapping_counts": dict(Counter(row.get("best_mapping") for row in rows)),
            "avg_current_mapping_iou": sum(current_values) / len(current_values) if current_values else None,
            "avg_best_iou": sum(best_values) / len(best_values) if best_values else None,
            "avg_iou_improvement": (
                (sum(best_values) / len(best_values)) - (sum(current_values) / len(current_values))
                if current_values and best_values
                else None
            ),
        }

    return {
        "overall": summarize(objects),
        "by_class": {class_name: summarize(rows) for class_name, rows in sorted(by_class.items())},
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "frame_id",
        "object_id",
        "class",
        "type_name",
        "target_kind",
        "current_mapping_iou",
        "production_geometry_iou",
        "best_mapping",
        "best_iou",
        "center_error_current",
        "center_error_best",
        "width_ratio_best",
        "height_ratio_best",
        "bbox_area_ratio_best",
        "basis_source",
        "position_source",
        "object_pose_cache_match",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose DCS local bbox axis/sign mapping offline")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "bbox_test_capture.yaml"))
    parser.add_argument("--frames", nargs="+", default=DEFAULT_FRAMES)
    parser.add_argument("--output-json", default=str(PROJECT_ROOT / "diagnostics" / "axis_mapping_report.json"))
    parser.add_argument("--output-csv", default=str(PROJECT_ROOT / "diagnostics" / "axis_mapping_report.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(Path(args.config))
    output_root = resolve_config_path(config["paths"]["output_root"])
    snapshot_dir = resolve_config_path(config["integration"]["snapshot_dir"])
    if output_root is None or snapshot_dir is None:
        raise ValueError("output_root and snapshot_dir must be configured")

    object_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for frame_id in args.frames:
        metadata_path = output_root / "metadata" / f"{frame_id}.json"
        metadata = load_json(metadata_path)
        frame_token = metadata.get("sync", {}).get("frame_token")
        if not frame_token:
            raise ValueError(f"Missing sync.frame_token in {metadata_path}")
        snapshot_path = snapshot_dir / f"snapshot_{frame_token}.json"
        snapshot = load_json(snapshot_path)
        snapshot_by_id = {str(obj.get("id")): obj for obj in snapshot.get("objects", [])}
        pose, intrinsics = camera_context(metadata)

        for metadata_obj in metadata.get("objects", []):
            object_id = str(metadata_obj.get("id"))
            snapshot_obj = snapshot_by_id.get(object_id)
            if not snapshot_obj:
                skipped.append({"frame_id": frame_id, "object_id": object_id, "reason": "missing_snapshot_object"})
                continue
            result = object_mapping_metrics(frame_id, metadata_obj, snapshot_obj, pose, intrinsics)
            if result is None:
                skipped.append({"frame_id": frame_id, "object_id": object_id, "reason": "missing_basis_or_bbox"})
                continue
            object_rows.append(result)

    report = {
        "frames": args.frames,
        "current_mapping": CURRENT_MAPPING_NAME,
        "basis_variants": [basis.name for basis in BASIS_VARIANTS],
        "order_variants": [order.name for order in ORDER_VARIANTS],
        "aggregate": aggregate(object_rows),
        "objects": object_rows,
        "skipped": skipped,
        "notes": [
            "Production mapping is not changed by this diagnostic.",
            "Best mapping is selected against visible bbox when present, otherwise final bbox.",
            "Visible/final bboxes may include refiner errors, so mapping conclusions should be verified visually.",
        ],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(Path(args.output_csv), object_rows)

    print(json.dumps(report["aggregate"], indent=2))
    print(f"Wrote {output_json}")
    print(f"Wrote {Path(args.output_csv)}")


if __name__ == "__main__":
    main()
