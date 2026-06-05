"""Build a tiny **raw dataset contract** layout (post-adapter) in a temp directory.

This is for enrichment tests — DCS adapter tests use `mini_dcs_dataset.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from packages.common.image_io import imwrite


def write_mini_raw_dataset(
    root: Path,
    *,
    dataset_id: str = "ds_test",
    image_size: tuple[int, int] = (2560, 1440),
    images_per_split: dict[str, int] | None = None,
    annotations_per_image: int = 2,
    include_one_background_in_train: bool = True,
) -> Path:
    """Create root/{dataset_id}/ matching the raw dataset contract.

    Returns path to the dataset directory.
    """
    if images_per_split is None:
        images_per_split = {"train": 3, "val": 1, "test": 1}

    dataset_dir = root / dataset_id
    for sub in ("images/train", "images/val", "images/test", "annotations", "metadata"):
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": 0, "name": "ships"},
        {"id": 1, "name": "helicopters"},
        {"id": 2, "name": "airplanes"},
    ]
    image_w, image_h = image_size

    splits_coco: dict[str, dict] = {
        s: {"images": [], "annotations": [], "categories": categories}
        for s in ("train", "val", "test")
    }

    next_image_id = 1
    next_ann_id = 1
    for split, n_images in images_per_split.items():
        for i in range(n_images):
            file_name = f"frame_{split}_{i:03d}.png"
            file_path = dataset_dir / "images" / split / file_name
            # Write a solid-color image (so resize/crop are predictable).
            arr = np.full((image_h, image_w, 3), fill_value=128, dtype=np.uint8)
            imwrite(file_path, arr)

            splits_coco[split]["images"].append({
                "id": next_image_id,
                "file_name": file_name,
                "width": image_w,
                "height": image_h,
            })

            is_background = (
                include_one_background_in_train and split == "train" and i == 0
            )
            if not is_background:
                # Lay annotations in a deterministic spot near the top-left of the image.
                for k in range(annotations_per_image):
                    bbox = [200.0 + 100 * k, 200.0 + 80 * k, 80.0, 60.0]
                    splits_coco[split]["annotations"].append({
                        "id": next_ann_id,
                        "image_id": next_image_id,
                        "category_id": k % 3,
                        "bbox": bbox,
                        "area": float(bbox[2] * bbox[3]),
                        "iscrowd": 0,
                    })
                    next_ann_id += 1
            next_image_id += 1

    for split in ("train", "val", "test"):
        (dataset_dir / "annotations" / f"instances_{split}.json").write_text(
            json.dumps(splits_coco[split], indent=2), encoding="utf-8",
        )

    info = {
        "dataset_id": dataset_id,
        "name": "mini raw dataset",
        "created_at": "2026-05-24T00:00:00Z",
        "source": "test_fixture",
        "num_images": sum(images_per_split.values()),
        "num_annotations": sum(len(c["annotations"]) for c in splits_coco.values()),
        "num_classes": len(categories),
        "split_strategy": "fixed_by_fixture",
        "split_ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
        "splits": {
            s: {
                "num_images": len(c["images"]),
                "num_annotations": len(c["annotations"]),
            }
            for s, c in splits_coco.items()
        },
        "annotation_format": "coco",
        "image_format": "png",
    }
    (dataset_dir / "metadata" / "dataset_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8",
    )
    (dataset_dir / "metadata" / "classes.json").write_text(
        json.dumps({"categories": categories}, indent=2), encoding="utf-8",
    )
    return dataset_dir
