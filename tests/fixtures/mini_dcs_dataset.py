"""Build a tiny DCS_AutoDataset-style fixture in a temp directory."""
from __future__ import annotations

import json
from pathlib import Path


def write_mini_dcs_dataset(root: Path, num_frames: int = 6) -> Path:
    """Create root/dataset/ matching the DCS_AutoDataset layout, with ``num_frames`` images.

    Categories are the canonical 3 (ships, helicopters, airplanes).
    Each frame gets 1-3 bbox annotations, deterministically. A 1-pixel PNG
    payload stands in for real screenshots — we never decode them in the tests
    that use this fixture.
    """
    root = Path(root)
    base = root / "dataset"
    images_dir = base / "images"
    metadata_dir = base / "metadata"
    annotations_dir = base / "annotations"
    yolo_dir = base / "yolo_labels"
    for d in (images_dir, metadata_dir, annotations_dir, yolo_dir):
        d.mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": 0, "name": "ships"},
        {"id": 1, "name": "helicopters"},
        {"id": 2, "name": "airplanes"},
    ]

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    next_ann_id = 1

    png_payload = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x5b\x0c\x02\xc4\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    for i in range(num_frames):
        frame_id = f"frame_{i:03d}"
        file_name = f"{frame_id}.png"
        (images_dir / file_name).write_bytes(png_payload)
        (yolo_dir / f"{frame_id}.txt").write_text("")

        coco_images.append({
            "id": i + 1,
            "file_name": file_name,
            "width": 2560,
            "height": 1440,
        })

        # 1-3 annotations per image; class cycles ships/helicopters/airplanes
        num_anns = (i % 3) + 1
        for k in range(num_anns):
            cat = (i + k) % 3
            bbox = [100.0 + 50 * k, 150.0 + 30 * k, 30.0 + 5 * k, 20.0 + 3 * k]
            coco_annotations.append({
                "id": next_ann_id,
                "image_id": i + 1,
                "category_id": cat,
                "bbox": bbox,
                "area": float(bbox[2] * bbox[3]),
                "iscrowd": 0,
                "quality_tier": "exact_visible",
                "confidence_source": "visible_pixels",
            })
            next_ann_id += 1

        # Per-frame DCS metadata (subset of the real schema)
        per_frame = {
            "schema_version": "1.0",
            "frame_id": frame_id,
            "timestamp_utc": "2026-04-22T10:15:30.125Z",
            "usable": True,
            "scene": {
                "scene_id":    f"scene_{i:04d}",
                "mission_id":  "test_mission_001",
                "weather":     ["clear", "cloudy", "rain"][i % 3],
                "time_of_day": "12:00",
                "sea_state":   i % 4,
                "scenario_mode": "multi_object",
            },
            "objects": [
                {"quality_tier": "exact_visible"} for _ in range(num_anns)
            ],
        }
        (metadata_dir / f"{frame_id}.json").write_text(
            json.dumps(per_frame, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    (annotations_dir / "_annotations.coco.json").write_text(
        json.dumps({
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories,
        }, indent=2),
        encoding="utf-8",
    )
    return base
