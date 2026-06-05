"""End-to-end evaluation orchestrator.

The evaluation procedure:

For each test image:
  1. **Full-source inference first** — letterbox source to model size, predict,
     unmap detections back to source coords.
  2. **Multi-scale crop inference** — for each configured scale:
     - resize source (preserving aspect for crop=true scales);
     - tile into ``crop_size``-sized inputs;
     - predict on each tile;
     - map detections back to source coords via crop→scale→source.
  3. **Merge** — collect all source-coord detections; apply NMS/Soft-NMS/WBF +
     fragment fusion (``apps.evaluation_export.merge``).
  4. **Metrics + reports** — compute against ground truth in source coords
     (``apps.evaluation_export.metrics``), write JSON/CSV/PNG/visualizations.

This module only orchestrates. Actual inference is delegated to a ``Predictor``
(see ``inference.py``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from apps.evaluation_export.inference import CropDetection, CropInput, Predictor
from apps.evaluation_export.merge import (
    Detection,
    FusionConfig,
    FusionStats,
    apply_postprocessing,
    merge_fragmented_boxes,
)
from apps.evaluation_export.metrics import (
    GTBox,
    build_confusion_matrix,
    compute_metrics,
)
from apps.evaluation_export.report import (
    RunSummary,
    TimingStats,
    print_summary,
    write_confusion_matrix_files,
    write_false_examples_jsonl,
    write_per_class_csv,
    write_predictions_files,
    write_summary_csv,
    write_summary_json,
)
from packages.common.bbox_ops import resize_xyxy, translate_xyxy
from packages.common.coco_io import load_coco
from packages.common.image_resize import resize_image
from packages.common.letterbox import compute_letterbox, letterbox_box_to_source
from packages.common.logging_utils import get_logger
from packages.common.mlflow_utils import MLflowRun
from packages.common.tiling import Tile, generate_tiles

LOG = get_logger(__name__)


# ---------------------------------------------------------------------- #
# config types
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class EvalScale:
    """One scale stage in the eval pipeline."""
    name: str
    mode: str        # "source_full_image" | "keep_original" | "resize"
    size: Optional[Tuple[int, int]] = None
    crop: bool = True


def parse_eval_scales(scales_cfg: Sequence[Mapping[str, Any]]) -> List[EvalScale]:
    out: List[EvalScale] = []
    for entry in scales_cfg:
        mode = entry.get("mode")
        if mode == "source_full_image":
            out.append(EvalScale(name=str(entry["name"]), mode=mode))
            continue
        if mode == "keep_original":
            out.append(EvalScale(name=str(entry["name"]), mode=mode,
                                 crop=bool(entry.get("crop", True))))
            continue
        sz = entry.get("size")
        if not (isinstance(sz, (list, tuple)) and len(sz) == 2):
            raise ValueError(f"Eval scale {entry.get('name')!r} requires size: [W, H]")
        out.append(EvalScale(
            name=str(entry["name"]), mode="resize",
            size=(int(sz[0]), int(sz[1])),
            crop=bool(entry.get("crop", True)),
        ))
    return out


@dataclass
class EvaluationResult:
    summary: RunSummary
    output_dir: Path
    artifacts: Dict[str, Path] = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# main entry
# ---------------------------------------------------------------------- #

def evaluate(
    *,
    predictor: Predictor,
    raw_dataset_dir: str | Path,
    split: str,
    classes: Sequence[str],
    eval_cfg: Mapping[str, Any],
    output_dir: str | Path,
    backend_name: str = "mock",
    model_path: Optional[str] = None,
    predictions_path: Optional[str] = None,
    images_for_visualizations: Optional[Sequence[int]] = None,
    mlflow_run: Optional[MLflowRun] = None,
) -> EvaluationResult:
    """Run the evaluator on a raw-dataset split and write artifacts."""
    raw_dataset_dir = Path(raw_dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coco = load_coco(raw_dataset_dir / "annotations" / f"instances_{split}.json")
    gts_by_image = _build_gts(coco, classes)

    scales = parse_eval_scales(eval_cfg.get("scales") or [])
    if not scales:
        raise ValueError("evaluation.scales is empty")

    crop_size  = int(eval_cfg.get("crop_size", 640))
    stride     = int(eval_cfg.get("stride", 320))
    include_edge_tiles = bool(eval_cfg.get("include_edge_tiles", True))
    full_image_first   = bool(eval_cfg.get("full_image_first", True))

    merge_cfg = eval_cfg.get("merge") or {}
    method    = merge_cfg.get("method", "class_aware_nms")
    iou_thr   = float(merge_cfg.get("iou_threshold", 0.5))
    conf_thr  = float(merge_cfg.get("confidence_threshold", 0.25))
    max_det   = int(merge_cfg.get("max_detections_per_image", 300))
    fusion_cfg = _parse_fusion_cfg(merge_cfg.get("fragment_fusion") or {})

    metrics_cfg = eval_cfg.get("metrics") or {}
    iou_thresholds = [float(t) for t in metrics_cfg.get("iou_thresholds")
                       or [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]]

    # ---- iterate images ----
    predictions_by_image: Dict[int, List[Detection]] = {}
    raw_tile_payload: dict = {"images": []}
    timing = TimingStats()
    timing.num_images = len(coco["images"])
    total_before_nms = 0
    total_after_nms  = 0
    total_after_merge = 0
    fusion_stats_aggregate = FusionStats()
    per_image_fusion: List[dict] = []

    needs_pixels = bool(getattr(predictor, "needs_pixels", True))
    if needs_pixels:
        from packages.common.image_io import imread
        LOG.info("Predictor reports needs_pixels=True; will load source images from %s/images/%s/",
                 raw_dataset_dir, split)
    images_dir = raw_dataset_dir / "images" / split

    e2e_start = time.perf_counter()
    for img_meta in coco["images"]:
        image_id = int(img_meta["id"])
        src_w = int(img_meta["width"])
        src_h = int(img_meta["height"])

        img_payload = {
            "image_id": image_id,
            "image_size": [src_w, src_h],
            "tiles": [],
        }

        # ---- 1) Optionally start with full-source-image inference ----
        ordered_scales = list(scales)
        if full_image_first:
            ordered_scales.sort(
                key=lambda s: 0 if s.mode == "source_full_image" else 1
            )

        all_dets_for_image: List[Detection] = []

        # Load source pixels once per image; cache scaled arrays per scale_size.
        source_array: Optional[np.ndarray] = None
        scale_cache: Dict[Tuple[int, int], np.ndarray] = {}
        if needs_pixels:
            src_img_path = images_dir / img_meta["file_name"]
            t_pre = time.perf_counter()
            source_array = imread(src_img_path)
            timing.preprocessing_total += time.perf_counter() - t_pre
            if source_array.shape[1] != src_w or source_array.shape[0] != src_h:
                LOG.warning("Image %s on disk is %dx%d but COCO says %dx%d — using disk dims",
                            src_img_path, source_array.shape[1], source_array.shape[0], src_w, src_h)
                src_w, src_h = int(source_array.shape[1]), int(source_array.shape[0])

        for sc in ordered_scales:
            t_pre = time.perf_counter()
            crops, tiles_for_payload = _build_crop_inputs(
                sc, image_id, src_w, src_h,
                crop_size=crop_size, stride=stride,
                include_edge_tiles=include_edge_tiles,
                source_array=source_array if needs_pixels else None,
                scale_cache=scale_cache if needs_pixels else None,
            )
            timing.preprocessing_total += time.perf_counter() - t_pre
            t0 = time.perf_counter()
            preds = predictor.predict_batch(crops)
            timing.inference_total += time.perf_counter() - t0

            for crop_inp, dets in zip(crops, preds):
                source_dets = _crop_dets_to_source(
                    dets, crop_inp,
                    classes=classes,
                    src_w=src_w, src_h=src_h,
                )
                all_dets_for_image.extend(source_dets)
                # add to raw_tile_payload for traceability
                img_payload["tiles"].append({
                    "scale_name": crop_inp.scale_name,
                    "scale_size": list(crop_inp.scale_size),
                    "crop_offset": list(crop_inp.crop_offset),
                    "crop_size":  list(crop_inp.crop_size),
                    "tile_size":  crop_inp.tile_size,
                    "detections": [
                        {"class_id": d.class_id, "confidence": d.confidence,
                         "bbox_crop": list(d.bbox_crop)} for d in dets
                    ],
                })

        raw_tile_payload["images"].append(img_payload)
        total_before_nms += len(all_dets_for_image)

        # ---- 2) postprocess (NMS / etc) ----
        t_pp = time.perf_counter()
        post = apply_postprocessing(
            all_dets_for_image,
            method=method,
            iou_threshold=iou_thr,
            score_threshold=conf_thr,
            max_detections=max_det,
        )
        timing.postprocessing_total += time.perf_counter() - t_pp
        total_after_nms += len(post)

        # ---- 3) fragment fusion ----
        t_m = time.perf_counter()
        fused, fstats = merge_fragmented_boxes(post, fusion_cfg)
        timing.stitch_merge_total += time.perf_counter() - t_m
        total_after_merge += len(fused)
        per_image_fusion.append({
            "image_id": image_id,
            "before_nms": fstats.after_nms if False else len(all_dets_for_image),
            "after_nms": fstats.after_nms,
            "after_merge": fstats.after_merge,
            "merged_groups": fstats.merged_groups,
        })
        fusion_stats_aggregate.before_nms += len(all_dets_for_image)
        fusion_stats_aggregate.after_nms  += fstats.after_nms
        fusion_stats_aggregate.after_merge += fstats.after_merge

        predictions_by_image[image_id] = fused

    timing.end_to_end_total = time.perf_counter() - e2e_start

    # ---- 4) metrics ----
    metrics_report = compute_metrics(
        predictions_by_image=predictions_by_image,
        gts_by_image=gts_by_image,
        classes=classes,
        iou_threshold_confusion=iou_thr,
        map_iou_thresholds=iou_thresholds,
    )
    cm = build_confusion_matrix(
        predictions_by_image=predictions_by_image,
        gts_by_image=gts_by_image,
        classes=classes,
        iou_threshold=iou_thr,
    )

    summary = RunSummary(
        backend=backend_name,
        model_path=model_path,
        predictions_path=predictions_path,
        num_images=timing.num_images,
        num_gt=sum(len(v) for v in gts_by_image.values()),
        num_predictions_before_nms=total_before_nms,
        num_predictions_after_nms=total_after_nms,
        num_predictions_after_merge=total_after_merge,
        metrics=metrics_report,
        timing=timing,
        fusion_stats=fusion_stats_aggregate,
        per_image_fusion=per_image_fusion,
    )

    # ---- 5) reports ----
    artifacts: Dict[str, Path] = {}
    artifacts["summary_json"] = write_summary_json(summary, output_dir)
    artifacts["summary_csv"]  = write_summary_csv(summary, output_dir)
    artifacts["per_class_csv"] = write_per_class_csv(summary, output_dir)
    if metrics_cfg.get("save_confusion_matrix", True):
        cm_paths = write_confusion_matrix_files(cm, output_dir)
        artifacts["confusion_csv"] = cm_paths["csv"]
        artifacts["confusion_png"] = cm_paths["png"]
    if metrics_cfg.get("save_fp_fn_examples", True):
        fp_fn = write_false_examples_jsonl(
            predictions_by_image, gts_by_image, output_dir, iou_threshold=iou_thr,
        )
        artifacts["false_positives"] = fp_fn["fp"]
        artifacts["false_negatives"] = fp_fn["fn"]
    pred_paths = write_predictions_files(
        raw_tile_payload, predictions_by_image, coco.get("categories", []),
        output_dir,
    )
    artifacts.update({f"predictions_{k}": v for k, v in pred_paths.items()})

    print_summary(summary)

    # ---- 6) optional MLflow logging ----
    if mlflow_run is not None and mlflow_run.enabled:
        _log_evaluation_to_mlflow(mlflow_run, summary, artifacts, classes)

    return EvaluationResult(summary=summary, output_dir=output_dir, artifacts=artifacts)


def _log_evaluation_to_mlflow(mlrun: "MLflowRun", summary: RunSummary,
                                artifacts: Dict[str, Path],
                                classes: Sequence[str]) -> None:
    m = summary.metrics
    metrics: Dict[str, float] = {
        "eval/overall_precision":          m.overall_precision,
        "eval/overall_recall":             m.overall_recall,
        "eval/overall_f1":                 m.overall_f1,
        "eval/overall_detection_accuracy": m.overall_detection_accuracy,
        "eval/mAP50":                      m.map50,
        "eval/mAP50_95":                   m.map5095,
        "eval/total_tp":                   float(m.total_tp),
        "eval/total_fp":                   float(m.total_fp),
        "eval/total_fn":                   float(m.total_fn),
        "eval/num_predictions_before_nms": float(summary.num_predictions_before_nms),
        "eval/num_predictions_after_nms":  float(summary.num_predictions_after_nms),
        "eval/num_predictions_after_merge": float(summary.num_predictions_after_merge),
        "eval/fps":                        summary.timing.fps,
    }
    for cid, pcm in m.per_class.items():
        name = pcm.class_name or str(cid)
        metrics[f"eval/per_class/{name}/precision"] = pcm.precision
        metrics[f"eval/per_class/{name}/recall"]    = pcm.recall
        metrics[f"eval/per_class/{name}/f1"]        = pcm.f1
        metrics[f"eval/per_class/{name}/ap50"]      = pcm.ap50
        metrics[f"eval/per_class/{name}/ap5095"]    = pcm.ap5095
    mlrun.log_metrics(metrics)

    # Log each artifact at top of "reports" namespace.
    for key, path in artifacts.items():
        mlrun.log_artifact(path, artifact_path="reports")


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #

def _build_gts(coco: dict, classes: Sequence[str]) -> Dict[int, List[GTBox]]:
    out: Dict[int, List[GTBox]] = {int(img["id"]): [] for img in coco["images"]}
    cat_name_by_id = {c["id"]: c["name"] for c in coco.get("categories", [])}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        cid = int(ann["category_id"])
        name = cat_name_by_id.get(cid)
        out.setdefault(int(ann["image_id"]), []).append(GTBox(
            image_id=int(ann["image_id"]),
            class_id=cid, class_name=name,
            bbox=(float(x), float(y), float(x + w), float(y + h)),
            difficult=bool(ann.get("iscrowd", 0)),
        ))
    return out


def _build_crop_inputs(
    scale: EvalScale, image_id: int, src_w: int, src_h: int,
    *, crop_size: int, stride: int, include_edge_tiles: bool,
    source_array: Optional["np.ndarray"] = None,
    scale_cache: Optional[Dict[Tuple[int, int], "np.ndarray"]] = None,
    pad_value: int = 114,
) -> Tuple[List[CropInput], List[dict]]:
    """Generate ``CropInput`` list for one image at one scale stage.

    If ``source_array`` is provided, each returned CropInput is populated with
    the actual pixel array required for inference (full-source for ``source_full_image``,
    tile_size×tile_size patch otherwise). ``scale_cache`` is a per-image dict
    that lets repeated calls at the same ``scale.size`` share the resized array.

    Returns ``(inputs, [])`` — the second list is a placeholder for future
    raw_tile_payload extension (now built in the main loop).
    """
    inputs: List[CropInput] = []
    serial: List[dict] = []

    # ---- source_full_image: pass the WHOLE source, predictor letterboxes ----
    if scale.mode == "source_full_image":
        inputs.append(CropInput(
            image_id=image_id, scale_name=scale.name,
            scale_size=(src_w, src_h),
            crop_offset=(0, 0), crop_size=(src_w, src_h),
            tile_size=crop_size,
            image=source_array,
            extra={"letterbox": True, "src_w": src_w, "src_h": src_h},
        ))
        return inputs, serial

    # ---- determine the working scale and the pixel array at that scale ----
    if scale.mode == "keep_original":
        scale_w, scale_h = src_w, src_h
        scaled_array = source_array
    else:  # "resize"
        assert scale.size is not None
        scale_w, scale_h = scale.size
        scaled_array = _get_or_resize(source_array, scale_cache, scale_w, scale_h)

    # ---- crop:true → sliding-window tiles, each tile_size×tile_size ----
    if scale.crop:
        tiles = generate_tiles(scale_w, scale_h, crop_size, stride, include_edge_tiles)
        for t in tiles:
            tile_image = None
            if scaled_array is not None:
                tile_image = _extract_tile_pixels(
                    scaled_array, t, crop_size, pad_value=pad_value,
                )
            inputs.append(CropInput(
                image_id=image_id, scale_name=scale.name,
                scale_size=(scale_w, scale_h),
                crop_offset=tuple(t.crop_offset),
                crop_size=tuple(t.crop_size),
                tile_size=crop_size,
                image=tile_image,
            ))
        return inputs, serial

    # ---- crop:false → single image at scale.size ----
    # Restriction: predictor input must be tile_size×tile_size. For ``crop:false``
    # we require ``scale_size == (tile_size, tile_size)`` so the resized image
    # IS the model input. The standard "final_640" scale satisfies this.
    if scaled_array is not None and (scale_w, scale_h) != (crop_size, crop_size):
        raise ValueError(
            f"Scale {scale.name!r} has crop=false but scale_size=({scale_w},{scale_h}) "
            f"≠ tile_size ({crop_size},{crop_size}). Pre-cropped inputs must already "
            f"match the model input size. Set scale_size to ({crop_size},{crop_size}) "
            f"or enable crop=true."
        )
    inputs.append(CropInput(
        image_id=image_id, scale_name=scale.name,
        scale_size=(scale_w, scale_h),
        crop_offset=(0, 0),
        crop_size=(scale_w, scale_h),
        tile_size=crop_size,
        image=scaled_array,
    ))
    return inputs, serial


def _get_or_resize(
    source_array: Optional["np.ndarray"],
    cache: Optional[Dict[Tuple[int, int], "np.ndarray"]],
    target_w: int, target_h: int,
) -> Optional["np.ndarray"]:
    if source_array is None:
        return None
    key = (target_w, target_h)
    if cache is not None and key in cache:
        return cache[key]
    arr = resize_image(source_array, target_w, target_h)
    if cache is not None:
        cache[key] = arr
    return arr


def _extract_tile_pixels(
    scaled_array: "np.ndarray", tile: Tile, tile_size: int, *, pad_value: int = 114,
) -> "np.ndarray":
    """Crop a ``tile_size × tile_size`` patch from ``scaled_array``; pad with ``pad_value``."""
    ox, oy = tile.crop_offset
    cw, ch = tile.crop_size
    crop = scaled_array[oy:oy + ch, ox:ox + cw]
    if tile.pad == (0, 0):
        return crop.copy()
    if scaled_array.ndim == 3:
        canvas = np.full((tile_size, tile_size, scaled_array.shape[2]),
                         pad_value, dtype=scaled_array.dtype)
    else:
        canvas = np.full((tile_size, tile_size), pad_value, dtype=scaled_array.dtype)
    canvas[:ch, :cw] = crop
    return canvas


def _crop_dets_to_source(
    crop_dets: Sequence[CropDetection],
    crop_inp: CropInput,
    *,
    classes: Sequence[str],
    src_w: int,
    src_h: int,
) -> List[Detection]:
    """Map predictions from crop coords back into source-image coords."""
    out: List[Detection] = []
    is_letterbox = crop_inp.extra.get("letterbox") is True
    if is_letterbox:
        t = compute_letterbox(src_w, src_h, crop_inp.tile_size)
    for d in crop_dets:
        bbox_xyxy = d.bbox_crop
        if is_letterbox:
            bbox_src = letterbox_box_to_source(bbox_xyxy, t)
        else:
            # 1) crop coords → scale coords (translate by crop_offset)
            cx, cy = crop_inp.crop_offset
            scale_box = translate_xyxy(bbox_xyxy, cx, cy)
            # 2) scale → source: per-axis ratio
            sx = src_w / crop_inp.scale_size[0]
            sy = src_h / crop_inp.scale_size[1]
            bbox_src = resize_xyxy(scale_box, sx, sy)

        # clip to source bounds
        x1 = max(0.0, min(bbox_src[0], src_w))
        y1 = max(0.0, min(bbox_src[1], src_h))
        x2 = max(0.0, min(bbox_src[2], src_w))
        y2 = max(0.0, min(bbox_src[3], src_h))
        if x2 <= x1 or y2 <= y1:
            continue

        out.append(Detection(
            class_id=d.class_id,
            confidence=d.confidence,
            bbox=(x1, y1, x2, y2),
            class_name=classes[d.class_id] if 0 <= d.class_id < len(classes) else None,
            scale_level=-1,
            crop_id=-1,
            source="tile",
            original_size=(src_w, src_h),
            scale_size=crop_inp.scale_size,
            crop_offset=crop_inp.crop_offset,
        ))
    return out


def _parse_fusion_cfg(cfg: Mapping[str, Any]) -> FusionConfig:
    return FusionConfig(
        enabled=bool(cfg.get("enabled", True)),
        max_distance_px=float(cfg.get("max_distance_px", 50.0)),
        min_iou=float(cfg.get("min_iou", 0.05)),
        min_ios=float(cfg.get("min_ios", 0.3)),
        fusion_strategy=str(cfg.get("strategy", "union")),
        max_aspect_ratio=float(cfg.get("max_aspect_ratio", 30.0)),
        max_area_ratio=float(cfg.get("max_area_ratio", 50.0)),
        disabled_classes=list(cfg.get("disabled_classes", [])),
    )
