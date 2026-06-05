"""Multi-scale tiling enrichment for the YOLOX 640×640 trainer.

Inputs: a raw dataset directory matching the raw-dataset contract.
Outputs: tiled images + per-split COCO written into the given output directory
(the tile-cache build dir; see ``apps.dataset_enrichment.tile_cache``).

Algorithm:

For each image WITH objects, for each scale (original, full_hd, intermediate, final_640):
    1. Resize source image accordingly. ``crop: true`` scales preserve aspect via
       per-axis scale_x/scale_y derived from ``target_size / source_size``.
       ``crop: false`` final_640 stretches into 640×640 (annotations scale to match).
    2. If ``crop: true``: generate tiles with ``include_edge_tiles``, project each
       bbox via ``bbox_ops.project_xywh_to_tile`` (visible_ratio ≥ threshold), drop
       tiles with no surviving bboxes.
    3. If ``crop: false``: emit the resized image as-is (one image, all scaled
       annotations).

For each image WITHOUT objects (background): emit a single 640×640 letterboxed
image with no annotations, recorded in manifest as background.

Splits are taken from the raw dataset already (set by the adapter) — enrichment
**does not redo splits** to preserve no-leakage guarantee.
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from packages.common.bbox_ops import (
    coco_xywh_to_xyxy,
    project_xywh_to_tile,
    resize_xyxy,
    xyxy_to_coco_xywh,
)
from packages.common.coco_io import (
    empty_coco,
    index_by_image,
    load_coco,
    save_coco,
)
from packages.common.image_io import imread, imwrite
from packages.common.image_resize import resize_image
from packages.common.letterbox import apply_letterbox
from packages.common.logging_utils import get_logger
from packages.common.manifest import ManifestWriter
from packages.common.tiling import Tile, generate_tiles

LOG = get_logger(__name__)
SPLIT_NAMES = ("train", "val", "test")


# ---------------------------------------------------------------------- #
# config + result types
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class ScaleSpec:
    """One scale entry from the enrichment.scales config list."""
    name: str
    mode: str   # "keep_original" | "resize"
    size: Optional[tuple[int, int]]   # (W, H) — None means keep_original
    crop: bool


def parse_scales(scales_cfg: Sequence[Mapping[str, Any]]) -> list[ScaleSpec]:
    out: list[ScaleSpec] = []
    for entry in scales_cfg:
        if entry.get("mode") == "keep_original":
            size = None
            mode = "keep_original"
        else:
            sz = entry.get("size")
            if not (isinstance(sz, (list, tuple)) and len(sz) == 2):
                raise ValueError(f"Scale {entry.get('name')!r} must have size: [W, H]")
            size = (int(sz[0]), int(sz[1]))
            mode = "resize"
        out.append(ScaleSpec(
            name=str(entry["name"]),
            mode=mode,
            size=size,
            crop=bool(entry.get("crop", False)),
        ))
    return out


@dataclass
class EnrichmentResult:
    target_dir: Path
    num_source_images: dict[str, int] = field(default_factory=lambda: {s: 0 for s in SPLIT_NAMES})
    num_enriched_images: dict[str, int] = field(default_factory=lambda: {s: 0 for s in SPLIT_NAMES})
    num_enriched_annotations: dict[str, int] = field(default_factory=lambda: {s: 0 for s in SPLIT_NAMES})
    num_dropped_tiles: int = 0
    num_background_images: int = 0
    per_scale_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        return {
            "target_dir": str(self.target_dir),
            "num_source_images":        self.num_source_images,
            "num_enriched_images":      self.num_enriched_images,
            "num_enriched_annotations": self.num_enriched_annotations,
            "num_dropped_tiles":        self.num_dropped_tiles,
            "num_background_images":    self.num_background_images,
            "per_scale_counts":         self.per_scale_counts,
            "warnings":                 self.warnings,
        }


# ---------------------------------------------------------------------- #
# core enrichment
# ---------------------------------------------------------------------- #

def enrich_dataset(
    raw_dataset_dir: str | Path,
    tmp_root: str | Path,
    *,
    dataset_id: str,
    run_id: str,
    enrichment_cfg: Mapping[str, Any],
    num_images_per_split: Optional[int] = None,
    write_images: bool = True,
    write_coco: bool = True,
) -> EnrichmentResult:
    """Run multi-scale enrichment over an adapted raw dataset.

    Parameters
    ----------
    raw_dataset_dir : path to ``storage/datasets/raw/{dataset_id}``
    tmp_root        : output directory to write tiles into (the tile-cache build dir)
    enrichment_cfg  : the ``enrichment`` section of pipeline.yaml
    num_images_per_split : if not None, only process the first N images of each
                           split (useful for dry-run).
    write_images    : when False, skips writing image files (dry-run).
    write_coco      : when False, skips writing instances_{split}.json.
    """
    raw_dataset_dir = Path(raw_dataset_dir)
    tmp_root = Path(tmp_root)

    scales = parse_scales(enrichment_cfg.get("scales") or [])
    if not scales:
        raise ValueError("enrichment.scales must be a non-empty list")

    crop_size = int(enrichment_cfg.get("crop_size", 640))
    stride    = int(enrichment_cfg.get("stride", 320))
    include_edge_tiles = bool(enrichment_cfg.get("include_edge_tiles", True))
    visible_threshold = float(enrichment_cfg.get("visible_area_threshold", 0.80))
    keep_background = bool(enrichment_cfg.get("keep_background_images", True))
    bg_policy = enrichment_cfg.get("background_policy") or {}
    bg_target = bg_policy.get("resize_to") or [crop_size, crop_size]
    bg_pad    = int(bg_policy.get("pad_value", 114))

    result = EnrichmentResult(target_dir=tmp_root)
    result.per_scale_counts = {s.name: {sp: 0 for sp in SPLIT_NAMES} for s in scales}
    result.per_scale_counts["background"] = {sp: 0 for sp in SPLIT_NAMES}

    # Layout
    for sp in SPLIT_NAMES:
        (tmp_root / "images" / sp).mkdir(parents=True, exist_ok=True)
    (tmp_root / "annotations").mkdir(parents=True, exist_ok=True)

    manifest_path = tmp_root / "manifest.jsonl"
    dropped_path = tmp_root / "dropped_tiles_manifest.jsonl"
    # Reset manifests at the start of a run.
    if manifest_path.exists():
        manifest_path.unlink()
    if dropped_path.exists():
        dropped_path.unlink()

    with ManifestWriter(manifest_path) as mw, ManifestWriter(dropped_path) as dw:
        for split in SPLIT_NAMES:
            coco_path = raw_dataset_dir / "annotations" / f"instances_{split}.json"
            if not coco_path.is_file():
                result.warnings.append(f"Missing split COCO: {coco_path}")
                continue
            src_coco = load_coco(coco_path)
            ann_by_image = index_by_image(src_coco)
            enriched_coco = empty_coco(src_coco.get("categories", []))

            images = src_coco["images"]
            if num_images_per_split is not None:
                images = images[:num_images_per_split]
            result.num_source_images[split] = len(images)

            new_image_id = 0
            new_ann_id = 0

            for src_img in images:
                anns = ann_by_image.get(src_img["id"], [])
                src_image_path = raw_dataset_dir / "images" / split / src_img["file_name"]
                if not src_image_path.is_file():
                    result.warnings.append(f"Missing source image: {src_image_path}")
                    continue

                if not anns:
                    if not keep_background:
                        continue
                    new_image_id += 1
                    file_name = _bg_file_name(new_image_id, src_img)
                    if write_images:
                        source = imread(src_image_path)
                        bg_image = _apply_background_policy(source, bg_target, bg_pad)
                        imwrite(tmp_root / "images" / split / file_name, bg_image)
                    enriched_coco["images"].append({
                        "id": new_image_id,
                        "file_name": file_name,
                        "width": int(bg_target[0]),
                        "height": int(bg_target[1]),
                    })
                    record = _make_manifest_record(
                        run_id=run_id, dataset_id=dataset_id, split=split,
                        src_img=src_img, scale_name="background",
                        scale_size=(int(bg_target[0]), int(bg_target[1])),
                        resize_transform={"scale_x": float(bg_target[0]) / src_img["width"],
                                          "scale_y": float(bg_target[1]) / src_img["height"]},
                        tile_xyxy=None,
                        new_image_id=new_image_id, new_file_name=file_name,
                        kept_anns=[], dropped=[], visible_ratios={},
                        drop_reasons={},
                        is_background=True,
                    )
                    mw.write(record)
                    result.num_background_images += 1
                    result.per_scale_counts["background"][split] += 1
                    continue

                # Image WITH objects — full multi-scale pipeline.
                source_array = imread(src_image_path) if write_images else None
                for scale in scales:
                    if scale.mode == "keep_original":
                        scaled_w, scaled_h = src_img["width"], src_img["height"]
                        sx, sy = 1.0, 1.0
                        scaled_array = source_array
                    else:
                        scaled_w, scaled_h = scale.size  # type: ignore[misc]
                        sx = scaled_w / src_img["width"]
                        sy = scaled_h / src_img["height"]
                        scaled_array = resize_image(source_array, scaled_w, scaled_h) if write_images else None

                    if scale.crop:
                        tiles = generate_tiles(
                            scaled_w, scaled_h, crop_size, stride, include_edge_tiles,
                        )
                        for tile in tiles:
                            tile_ann_specs: list[_KeptAnnotation] = []
                            drop_specs: list[_DroppedAnnotation] = []
                            for ann in anns:
                                out = project_xywh_to_tile(
                                    ann["bbox"],
                                    resize_scale=(sx, sy),
                                    tile_offset=tile.crop_offset,
                                    tile_size=crop_size,
                                    visible_threshold=visible_threshold,
                                )
                                if out is None:
                                    drop_specs.append(_DroppedAnnotation(
                                        ann_id=ann["id"], reason="visible_below_or_disjoint",
                                    ))
                                else:
                                    new_box, ratio = out
                                    tile_ann_specs.append(_KeptAnnotation(
                                        ann_id=ann["id"], new_bbox=new_box,
                                        visible_ratio=ratio,
                                        category_id=int(ann["category_id"]),
                                    ))
                            if not tile_ann_specs:
                                dw.write({
                                    "run_id": run_id, "dataset_id": dataset_id, "split": split,
                                    "source_image_id": src_img["id"],
                                    "source_file_name": src_img["file_name"],
                                    "scale_name": scale.name,
                                    "tile_xyxy": [tile.crop_offset[0], tile.crop_offset[1],
                                                  tile.crop_offset[0] + crop_size,
                                                  tile.crop_offset[1] + crop_size],
                                    "reason": "no_objects" if not drop_specs else "all_objects_below_visible_threshold",
                                    "num_objects_before_filter": len(anns),
                                    "num_objects_after_filter": 0,
                                })
                                result.num_dropped_tiles += 1
                                continue

                            new_image_id += 1
                            file_name = _tile_file_name(new_image_id, src_img, scale.name, tile.crop_id)
                            if write_images:
                                tile_image = _extract_tile(scaled_array, tile, pad_value=bg_pad)
                                imwrite(tmp_root / "images" / split / file_name, tile_image)
                            enriched_coco["images"].append({
                                "id": new_image_id,
                                "file_name": file_name,
                                "width": crop_size, "height": crop_size,
                            })

                            kept_ann_new_ids: list[int] = []
                            for spec in tile_ann_specs:
                                new_ann_id += 1
                                area = spec.new_bbox[2] * spec.new_bbox[3]
                                enriched_coco["annotations"].append({
                                    "id": new_ann_id, "image_id": new_image_id,
                                    "category_id": spec.category_id,
                                    "bbox": list(spec.new_bbox),
                                    "area": area,
                                    "iscrowd": 0,
                                    "source_annotation_id": int(spec.ann_id),
                                })
                                kept_ann_new_ids.append(new_ann_id)

                            mw.write(_make_manifest_record(
                                run_id=run_id, dataset_id=dataset_id, split=split,
                                src_img=src_img, scale_name=scale.name,
                                scale_size=(scaled_w, scaled_h),
                                resize_transform={"scale_x": sx, "scale_y": sy},
                                tile_xyxy=(tile.crop_offset[0], tile.crop_offset[1],
                                           tile.crop_offset[0] + crop_size,
                                           tile.crop_offset[1] + crop_size),
                                new_image_id=new_image_id, new_file_name=file_name,
                                kept_anns=[(s.ann_id, kept_ann_new_ids[i])
                                           for i, s in enumerate(tile_ann_specs)],
                                dropped=[d.ann_id for d in drop_specs],
                                visible_ratios={str(s.ann_id): s.visible_ratio for s in tile_ann_specs},
                                drop_reasons={str(d.ann_id): d.reason for d in drop_specs},
                                is_background=False,
                            ))
                            result.num_enriched_images[split] += 1
                            result.num_enriched_annotations[split] += len(tile_ann_specs)
                            result.per_scale_counts[scale.name][split] += 1

                    else:
                        # crop: false — one image at scale.size, all anns scaled.
                        target_w, target_h = scale.size  # type: ignore[misc]
                        new_image_id += 1
                        file_name = _resized_file_name(new_image_id, src_img, scale.name)
                        if write_images:
                            imwrite(tmp_root / "images" / split / file_name, scaled_array)
                        enriched_coco["images"].append({
                            "id": new_image_id, "file_name": file_name,
                            "width": target_w, "height": target_h,
                        })
                        kept_ann_new_ids: list[int] = []
                        kept_anns_for_manifest: list[tuple[int, int]] = []
                        visible_ratios: dict[str, float] = {}
                        for ann in anns:
                            # Whole-image resize: bbox scales by (sx, sy).
                            new_box_xyxy = resize_xyxy(coco_xywh_to_xyxy(ann["bbox"]), sx, sy)
                            new_box = xyxy_to_coco_xywh(new_box_xyxy)
                            new_ann_id += 1
                            area = new_box[2] * new_box[3]
                            enriched_coco["annotations"].append({
                                "id": new_ann_id, "image_id": new_image_id,
                                "category_id": int(ann["category_id"]),
                                "bbox": list(new_box),
                                "area": area,
                                "iscrowd": 0,
                                "source_annotation_id": int(ann["id"]),
                            })
                            kept_ann_new_ids.append(new_ann_id)
                            kept_anns_for_manifest.append((ann["id"], new_ann_id))
                            visible_ratios[str(ann["id"])] = 1.0

                        mw.write(_make_manifest_record(
                            run_id=run_id, dataset_id=dataset_id, split=split,
                            src_img=src_img, scale_name=scale.name,
                            scale_size=(target_w, target_h),
                            resize_transform={"scale_x": sx, "scale_y": sy},
                            tile_xyxy=None,
                            new_image_id=new_image_id, new_file_name=file_name,
                            kept_anns=kept_anns_for_manifest,
                            dropped=[], visible_ratios=visible_ratios,
                            drop_reasons={},
                            is_background=False,
                        ))
                        result.num_enriched_images[split] += 1
                        result.num_enriched_annotations[split] += len(anns)
                        result.per_scale_counts[scale.name][split] += 1

            if write_coco:
                save_coco(enriched_coco, tmp_root / "annotations" / f"instances_{split}.json")

    if write_coco:
        _write_summary_json(result, tmp_root, enrichment_cfg)
    return result


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class _KeptAnnotation:
    ann_id: int
    new_bbox: tuple[float, float, float, float]
    visible_ratio: float
    category_id: int


@dataclass(frozen=True)
class _DroppedAnnotation:
    ann_id: int
    reason: str


def _extract_tile(scaled_image: "np.ndarray", tile: Tile, pad_value: int = 114) -> "np.ndarray":
    """Crop a tile_size×tile_size patch from ``scaled_image``; pad missing region with pad_value."""
    ts = tile.tile_size
    ox, oy = tile.crop_offset
    cw, ch = tile.crop_size
    crop = scaled_image[oy:oy + ch, ox:ox + cw]
    if tile.pad == (0, 0):
        return crop.copy()
    if scaled_image.ndim == 3:
        canvas = np.full((ts, ts, scaled_image.shape[2]), pad_value, dtype=scaled_image.dtype)
    else:
        canvas = np.full((ts, ts), pad_value, dtype=scaled_image.dtype)
    canvas[:ch, :cw] = crop
    return canvas


def _apply_background_policy(source: "np.ndarray", target_size: Sequence[int], pad_value: int) -> "np.ndarray":
    """Letterbox the background image into a square ``target_size`` canvas."""
    target_w, target_h = int(target_size[0]), int(target_size[1])
    if target_w != target_h:
        raise ValueError(f"Background letterbox supports square targets only, got {target_w}×{target_h}")
    canvas, _ = apply_letterbox(source, target_w, pad_value=pad_value)
    return canvas


def _bg_file_name(new_image_id: int, src_img: Mapping[str, Any]) -> str:
    stem = Path(src_img["file_name"]).stem
    return f"{new_image_id:08d}_{stem}_background.png"


def _resized_file_name(new_image_id: int, src_img: Mapping[str, Any], scale_name: str) -> str:
    stem = Path(src_img["file_name"]).stem
    return f"{new_image_id:08d}_{stem}_{scale_name}.png"


def _tile_file_name(new_image_id: int, src_img: Mapping[str, Any], scale_name: str, crop_id: int) -> str:
    stem = Path(src_img["file_name"]).stem
    return f"{new_image_id:08d}_{stem}_{scale_name}_t{crop_id:03d}.png"


def _make_manifest_record(
    *,
    run_id: str, dataset_id: str, split: str,
    src_img: Mapping[str, Any], scale_name: str,
    scale_size: tuple[int, int],
    resize_transform: dict[str, float],
    tile_xyxy: Optional[tuple[int, int, int, int]],
    new_image_id: int, new_file_name: str,
    kept_anns: list[tuple[int, int]],
    dropped: list[int],
    visible_ratios: dict[str, float],
    drop_reasons: dict[str, str],
    is_background: bool,
) -> dict[str, Any]:
    return {
        "run_id":               run_id,
        "dataset_id":           dataset_id,
        "split":                split,
        "source_image_id":      src_img["id"],
        "source_file_name":     src_img["file_name"],
        "source_width":         src_img["width"],
        "source_height":        src_img["height"],
        "scale_name":           scale_name,
        "scale_width":          scale_size[0],
        "scale_height":         scale_size[1],
        "resize_transform":     resize_transform,
        "tile_xyxy_in_scaled_image": list(tile_xyxy) if tile_xyxy else None,
        "new_image_id":         new_image_id,
        "new_file_name":        new_file_name,
        "kept_source_annotation_ids":  [int(a[0]) for a in kept_anns],
        "kept_annotation_ids":         [int(a[1]) for a in kept_anns],
        "dropped_annotation_ids":      [int(d) for d in dropped],
        "visible_ratios":              visible_ratios,
        "drop_reasons":                drop_reasons,
        "is_background":               is_background,
    }


def _write_summary_json(result: EnrichmentResult, tmp_root: Path, enrichment_cfg: Mapping[str, Any]) -> None:
    summary = {
        "created_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "config_used": {
            "crop_size":              enrichment_cfg.get("crop_size"),
            "stride":                 enrichment_cfg.get("stride"),
            "visible_area_threshold": enrichment_cfg.get("visible_area_threshold"),
            "include_edge_tiles":     enrichment_cfg.get("include_edge_tiles"),
            "keep_background_images": enrichment_cfg.get("keep_background_images"),
            "scales":                 list(enrichment_cfg.get("scales") or []),
            "background_policy":      enrichment_cfg.get("background_policy"),
        },
        "result": result.to_summary(),
    }
    (tmp_root / "enrichment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def cleanup_tmp(tmp_root: str | Path) -> None:
    """Remove a temporary enriched dataset directory (idempotent)."""
    p = Path(tmp_root)
    if p.exists():
        shutil.rmtree(p)
