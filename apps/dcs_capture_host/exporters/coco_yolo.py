"""COCO and YOLO exporters for rich frame metadata.

The public API is unchanged:

- ``DatasetExporter(classes, allowed_quality_tiers=None)``
- ``add_frame(frame_metadata)`` — accumulates COCO image + annotations
- ``write_yolo(frame_metadata, output_path)`` — writes a YOLO ``.txt`` per frame
- ``write_coco(output_path)`` — writes the aggregated COCO json

Object filtering rules are also unchanged: an object is exported only if the
frame is ``usable``, ``validation.valid`` is true, its ``quality_tier`` is in
``allowed_quality_tiers`` (when configured) and ``projection.bbox_xyxy_px`` is
present and truthy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _is_usable_frame(frame_metadata: Mapping[str, Any]) -> bool:
    return bool(frame_metadata.get("usable", False))


def _is_valid_object(obj: Mapping[str, Any]) -> bool:
    return bool(obj.get("validation", {}).get("valid", False))


def _object_bbox_xyxy(obj: Mapping[str, Any]) -> Optional[Sequence[float]]:
    return obj.get("projection", {}).get("bbox_xyxy_px")


def _xyxy_to_xywh(bbox_xyxy: Sequence[float]) -> Tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = bbox_xyxy
    return x_min, y_min, x_max - x_min, y_max - y_min


def _xyxy_to_yolo_norm(
    bbox_xyxy: Sequence[float],
    width_px: float,
    height_px: float,
) -> Tuple[float, float, float, float]:
    x_min, y_min, w, h = _xyxy_to_xywh(bbox_xyxy)
    x_center = x_min + w / 2.0
    y_center = y_min + h / 2.0
    return x_center / width_px, y_center / height_px, w / width_px, h / height_px


def _format_yolo_line(class_id: int, yolo_norm: Tuple[float, float, float, float]) -> str:
    xn, yn, wn, hn = yolo_norm
    return f"{class_id} {xn:.6f} {yn:.6f} {wn:.6f} {hn:.6f}"


def _make_coco_image(image_id: int, image_info: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": image_id,
        "file_name": image_info["file_name"],
        "width": image_info["width"],
        "height": image_info["height"],
    }


def _make_coco_annotation(
    annotation_id: int,
    image_id: int,
    category_id: int,
    bbox_xyxy: Sequence[float],
    obj: Mapping[str, Any],
) -> Dict[str, Any]:
    x_min, y_min, width, height = _xyxy_to_xywh(bbox_xyxy)
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x_min, y_min, width, height],
        "area": width * height,
        "iscrowd": 0,
        "quality_tier": obj.get("quality_tier", "approximate"),
        "confidence_source": obj.get("confidence_source", "unknown"),
    }


def _make_coco_categories(category_map: Mapping[str, int]) -> List[Dict[str, Any]]:
    return [{"id": category_id, "name": name} for name, category_id in category_map.items()]


@dataclass
class DatasetExporter:
    classes: Sequence[str]
    allowed_quality_tiers: Optional[Sequence[str]] = None
    coco_images: List[Dict[str, Any]] = field(default_factory=list)
    coco_annotations: List[Dict[str, Any]] = field(default_factory=list)
    next_annotation_id: int = 1

    def __post_init__(self) -> None:
        self.category_map: Dict[str, int] = {
            name: index for index, name in enumerate(self.classes)
        }

    def add_frame(self, frame_metadata: Dict[str, Any]) -> None:
        if not _is_usable_frame(frame_metadata):
            return

        image_info = frame_metadata["image"]
        image_id = len(self.coco_images) + 1
        self.coco_images.append(_make_coco_image(image_id, image_info))

        for obj in frame_metadata.get("objects", []):
            if not self._object_exportable(obj):
                continue
            bbox = _object_bbox_xyxy(obj)
            category_id = self.category_map[obj["class"]]
            self.coco_annotations.append(
                _make_coco_annotation(
                    annotation_id=self.next_annotation_id,
                    image_id=image_id,
                    category_id=category_id,
                    bbox_xyxy=bbox,
                    obj=obj,
                )
            )
            self.next_annotation_id += 1

    def write_yolo(self, frame_metadata: Dict[str, Any], output_path: Path) -> None:
        if not _is_usable_frame(frame_metadata):
            _ensure_parent(output_path)
            output_path.write_text("", encoding="utf-8")
            return

        image_info = frame_metadata["image"]
        width_px = float(image_info["width"])
        height_px = float(image_info["height"])

        lines: List[str] = []
        for obj in frame_metadata.get("objects", []):
            if not self._object_exportable(obj):
                continue
            bbox = _object_bbox_xyxy(obj)
            class_id = self.category_map[obj["class"]]
            yolo_norm = _xyxy_to_yolo_norm(bbox, width_px, height_px)
            lines.append(_format_yolo_line(class_id, yolo_norm))

        _ensure_parent(output_path)
        output_path.write_text("\n".join(lines), encoding="utf-8")

    def write_coco(self, output_path: Path) -> None:
        payload = {
            "images": self.coco_images,
            "annotations": self.coco_annotations,
            "categories": _make_coco_categories(self.category_map),
        }
        _ensure_parent(output_path)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _quality_tier_allowed(self, obj: Mapping[str, Any]) -> bool:
        if self.allowed_quality_tiers is None:
            return True
        return obj.get("quality_tier", "approximate") in self.allowed_quality_tiers

    def _object_exportable(self, obj: Mapping[str, Any]) -> bool:
        if not _is_valid_object(obj):
            return False
        if not self._quality_tier_allowed(obj):
            return False
        return bool(_object_bbox_xyxy(obj))
