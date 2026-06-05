"""Report writers + RunSummary aggregation.

Migrated from ``yolox_eval/src/report_writer.py``. ``loguru`` swapped for
stdlib logging via ``packages.common.logging_utils.get_logger``.

Outputs (under ``reports_dir/``):

- ``metrics/summary.json``
- ``metrics/summary.csv``
- ``metrics/per_class_metrics.csv``
- ``metrics/confusion_matrix.csv``
- ``metrics/confusion_matrix.png``
- ``metrics/pr_curve.png``
- ``predictions/predictions_raw_tiles.json``
- ``predictions/predictions_stitched.json``
- ``predictions/predictions_coco.json``
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from apps.evaluation_export.merge import Detection, FusionStats
from apps.evaluation_export.metrics import (
    ConfusionMatrix,
    MetricsReport,
    save_confusion_heatmap,
)
from packages.common.logging_utils import get_logger

LOG = get_logger(__name__)


@dataclass
class TimingStats:
    num_images: int = 0
    preprocessing_total:  float = 0.0
    inference_total:      float = 0.0
    postprocessing_total: float = 0.0
    stitch_merge_total:   float = 0.0
    end_to_end_total:     float = 0.0

    @property
    def fps(self) -> float:
        if self.end_to_end_total <= 0 or self.num_images <= 0:
            return 0.0
        return self.num_images / self.end_to_end_total

    def to_dict(self) -> dict:
        n = max(1, self.num_images)
        return {
            "num_images":            self.num_images,
            "avg_preprocessing_ms":  1000.0 * self.preprocessing_total  / n,
            "avg_inference_ms":      1000.0 * self.inference_total      / n,
            "avg_postprocessing_ms": 1000.0 * self.postprocessing_total / n,
            "avg_stitch_merge_ms":   1000.0 * self.stitch_merge_total   / n,
            "avg_end_to_end_ms":     1000.0 * self.end_to_end_total     / n,
            "fps": self.fps,
        }


@dataclass
class RunSummary:
    backend: str
    model_path:        Optional[str]
    predictions_path:  Optional[str]
    num_images: int
    num_gt:     int
    num_predictions_before_nms:  int
    num_predictions_after_nms:   int
    num_predictions_after_merge: int
    metrics: MetricsReport
    timing:  TimingStats
    fusion_stats: Optional[FusionStats] = None
    per_image_fusion: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "model_path": self.model_path,
            "predictions_path": self.predictions_path,
            "num_images": self.num_images,
            "num_gt": self.num_gt,
            "num_predictions_before_nms":  self.num_predictions_before_nms,
            "num_predictions_after_nms":   self.num_predictions_after_nms,
            "num_predictions_after_merge": self.num_predictions_after_merge,
            "metrics": self.metrics.to_dict(),
            "timing":  self.timing.to_dict(),
            "fusion_stats": self.fusion_stats.to_dict() if self.fusion_stats else None,
            "per_image_fusion": self.per_image_fusion,
        }


# ---------------------------------------------------------------------- #
# writers
# ---------------------------------------------------------------------- #

def write_summary_json(summary: RunSummary, out_dir: str | Path) -> Path:
    path = Path(out_dir) / "metrics" / "summary.json"
    _save_json(summary.to_dict(), path)
    return path


def write_summary_csv(summary: RunSummary, out_dir: str | Path) -> Path:
    path = Path(out_dir) / "metrics" / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    t = summary.timing.to_dict()
    rows = [
        ("backend", summary.backend),
        ("model_path", summary.model_path or ""),
        ("predictions_path", summary.predictions_path or ""),
        ("num_images", summary.num_images),
        ("num_gt", summary.num_gt),
        ("num_predictions_before_nms",  summary.num_predictions_before_nms),
        ("num_predictions_after_nms",   summary.num_predictions_after_nms),
        ("num_predictions_after_merge", summary.num_predictions_after_merge),
        ("overall_precision",          summary.metrics.overall_precision),
        ("overall_recall",             summary.metrics.overall_recall),
        ("overall_f1",                 summary.metrics.overall_f1),
        ("overall_detection_accuracy", summary.metrics.overall_detection_accuracy),
        ("mAP@50",                     summary.metrics.map50),
        ("mAP@50:95",                  summary.metrics.map5095),
        ("fps",                        summary.timing.fps),
        ("avg_inference_ms",      t["avg_inference_ms"]),
        ("avg_postprocessing_ms", t["avg_postprocessing_ms"]),
        ("avg_stitch_merge_ms",   t["avg_stitch_merge_ms"]),
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in rows:
            w.writerow([k, v])
    return path


def write_per_class_csv(summary: RunSummary, out_dir: str | Path) -> Path:
    path = Path(out_dir) / "metrics" / "per_class_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["class_id", "class_name", "num_gt", "num_pred", "tp", "fp", "fn",
              "precision", "recall", "f1", "ap50", "ap5095", "detection_accuracy"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cid in sorted(summary.metrics.per_class.keys()):
            pcm = summary.metrics.per_class[cid]
            w.writerow({k: getattr(pcm, k) for k in fields})
    return path


def write_confusion_matrix_files(cm: ConfusionMatrix, out_dir: str | Path,
                                  normalize: str = "row") -> Dict[str, Path]:
    csv_path = Path(out_dir) / "metrics" / "confusion_matrix.csv"
    png_path = Path(out_dir) / "metrics" / "confusion_matrix.png"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cm.to_csv(csv_path)
    save_confusion_heatmap(cm, png_path, normalize=normalize)
    return {"csv": csv_path, "png": png_path}


def write_predictions_files(
    raw_tiles_payload: dict,
    stitched_by_image: Dict[int, List[Detection]],
    coco_categories: List[dict],
    out_dir: str | Path,
) -> Dict[str, Path]:
    out = Path(out_dir) / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "predictions_raw_tiles.json"
    _save_json(raw_tiles_payload, raw_path)

    stitched_payload = {
        "images": [
            {"image_id": int(img_id),
             "detections": [d.to_dict() for d in dets]}
            for img_id, dets in sorted(stitched_by_image.items())
        ]
    }
    stitched_path = out / "predictions_stitched.json"
    _save_json(stitched_payload, stitched_path)

    coco_preds = []
    for img_id, dets in stitched_by_image.items():
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            coco_preds.append({
                "image_id": int(img_id),
                "category_id": int(d.class_id),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(d.confidence),
            })
    coco_path = out / "predictions_coco.json"
    _save_json(coco_preds, coco_path)

    return {"raw_tiles": raw_path, "stitched": stitched_path, "coco": coco_path}


def write_false_examples_jsonl(
    predictions_by_image: Dict[int, List[Detection]],
    gts_by_image: Dict[int, Any],   # GTBox per image
    out_dir: str | Path,
    iou_threshold: float = 0.5,
) -> Dict[str, Path]:
    """Emit FP / FN per-image rows for diagnostics. Pure JSONL, no images."""
    from apps.evaluation_export.merge import iou_matrix as _iou_mat

    out = Path(out_dir) / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    fp_path = out / "false_positives.jsonl"
    fn_path = out / "false_negatives.jsonl"

    with fp_path.open("w", encoding="utf-8") as fp_f, fn_path.open("w", encoding="utf-8") as fn_f:
        for image_id in sorted(set(predictions_by_image.keys()) | set(gts_by_image.keys())):
            preds = sorted(predictions_by_image.get(image_id, []),
                           key=lambda d: d.confidence, reverse=True)
            gts   = list(gts_by_image.get(image_id, []))
            if preds and gts:
                iou_mat = _iou_mat(
                    np.array([p.bbox for p in preds], dtype=np.float32),
                    np.array([g.bbox for g in gts], dtype=np.float32),
                )
            else:
                iou_mat = np.zeros((len(preds), len(gts)), dtype=np.float32)

            matched_gt: set = set()
            for pi, p in enumerate(preds):
                best_iou = 0.0; best_gi = -1
                for gi, g in enumerate(gts):
                    if gi in matched_gt or g.class_id != p.class_id:
                        continue
                    if iou_mat[pi, gi] >= iou_threshold and iou_mat[pi, gi] > best_iou:
                        best_iou = float(iou_mat[pi, gi]); best_gi = gi
                if best_gi >= 0:
                    matched_gt.add(best_gi)
                else:
                    fp_f.write(json.dumps({
                        "image_id": image_id, "class_id": p.class_id,
                        "confidence": p.confidence, "bbox": list(p.bbox),
                    }, ensure_ascii=False) + "\n")
            for gi, g in enumerate(gts):
                if gi in matched_gt:
                    continue
                fn_f.write(json.dumps({
                    "image_id": image_id, "class_id": g.class_id,
                    "bbox": list(g.bbox),
                }, ensure_ascii=False) + "\n")

    return {"fp": fp_path, "fn": fn_path}


def print_summary(summary: RunSummary) -> None:
    m = summary.metrics
    LOG.info("=" * 60)
    LOG.info("backend                 : %s", summary.backend)
    LOG.info("model_path              : %s", summary.model_path)
    LOG.info("predictions_path        : %s", summary.predictions_path)
    LOG.info("num_images              : %d", summary.num_images)
    LOG.info("num_gt                  : %d", summary.num_gt)
    LOG.info("num_pred before / after / after merge: %d / %d / %d",
             summary.num_predictions_before_nms,
             summary.num_predictions_after_nms,
             summary.num_predictions_after_merge)
    LOG.info("precision               : %.4f", m.overall_precision)
    LOG.info("recall                  : %.4f", m.overall_recall)
    LOG.info("F1                      : %.4f", m.overall_f1)
    LOG.info("detection accuracy*     : %.4f   (TP/(TP+FP+FN))",
             m.overall_detection_accuracy)
    LOG.info("mAP@50                  : %.4f", m.map50)
    LOG.info("mAP@50:95               : %.4f", m.map5095)
    LOG.info("FPS                     : %.2f", summary.timing.fps)
    LOG.info("per-class:")
    for cid in sorted(m.per_class.keys()):
        pcm = m.per_class[cid]
        LOG.info("  [%d] %-20s P=%.3f R=%.3f F1=%.3f AP50=%.3f AP50-95=%.3f TP/FP/FN=%d/%d/%d",
                 cid, pcm.class_name, pcm.precision, pcm.recall, pcm.f1,
                 pcm.ap50, pcm.ap5095, pcm.tp, pcm.fp, pcm.fn)
    LOG.info("=" * 60)


def _save_json(obj, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
