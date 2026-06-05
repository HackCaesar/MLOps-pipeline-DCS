"""Detection metrics + confusion matrix.

Migrated from ``yolox_eval/src/{metrics,confusion_matrix}.py``. Combined into one
module because both share the matching logic.

Metrics: precision/recall/F1 at one IoU threshold, COCO-style AP@50 and AP@50:95
via 101-point interpolation, per-class breakdown, PR curve.

Confusion matrix: GT × pred rows, with a ``background`` row/col for FN/FP.
PNG heatmap via matplotlib (Agg backend, headless).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from apps.evaluation_export.merge import Detection, iou_matrix
from packages.common.logging_utils import get_logger

LOG = get_logger(__name__)

BACKGROUND = "background"


# ---------------------------------------------------------------------- #
# data types
# ---------------------------------------------------------------------- #

@dataclass
class GTBox:
    image_id: int
    class_id: int
    bbox: Tuple[float, float, float, float]
    class_name: Optional[str] = None
    difficult: bool = False


@dataclass
class PerClassMetrics:
    class_id: int
    class_name: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    ap50: float = 0.0
    ap5095: float = 0.0
    detection_accuracy: float = 0.0
    num_gt: int = 0
    num_pred: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class MetricsReport:
    iou_threshold_confusion: float
    map_iou_thresholds: List[float]
    per_class: Dict[int, PerClassMetrics] = field(default_factory=dict)
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1: float = 0.0
    overall_detection_accuracy: float = 0.0
    map50: float = 0.0
    map5095: float = 0.0
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    num_gt: int = 0
    num_pred: int = 0
    pr_curve: Optional[Dict[str, list]] = None

    def to_dict(self) -> dict:
        return {
            "iou_threshold_confusion": self.iou_threshold_confusion,
            "map_iou_thresholds": self.map_iou_thresholds,
            "per_class": {int(k): v.to_dict() for k, v in self.per_class.items()},
            "overall_precision":          self.overall_precision,
            "overall_recall":             self.overall_recall,
            "overall_f1":                 self.overall_f1,
            "overall_detection_accuracy": self.overall_detection_accuracy,
            "map50":   self.map50,
            "map5095": self.map5095,
            "total_tp": self.total_tp, "total_fp": self.total_fp, "total_fn": self.total_fn,
            "num_gt":   self.num_gt,   "num_pred":  self.num_pred,
            "pr_curve": self.pr_curve,
        }


@dataclass
class ConfusionMatrix:
    classes: List[str]    # length N+1; last element is "background"
    iou_threshold: float
    matrix: np.ndarray    # shape (N+1, N+1), int64

    def to_csv(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gt\\pred"] + self.classes)
            for i, name in enumerate(self.classes):
                w.writerow([name] + self.matrix[i].tolist())

    def to_dict(self) -> dict:
        return {
            "classes": self.classes,
            "iou_threshold": self.iou_threshold,
            "matrix": self.matrix.tolist(),
        }


# ---------------------------------------------------------------------- #
# compute_metrics
# ---------------------------------------------------------------------- #

def compute_metrics(
    predictions_by_image: Dict[int, List[Detection]],
    gts_by_image: Dict[int, List[GTBox]],
    classes: Sequence[str],
    iou_threshold_confusion: float = 0.5,
    map_iou_thresholds: Sequence[float] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10)),
) -> MetricsReport:
    """Compute every metric the report cares about.

    See module docstring for the algorithmic details.
    """
    report = MetricsReport(
        iou_threshold_confusion=float(iou_threshold_confusion),
        map_iou_thresholds=[float(t) for t in map_iou_thresholds],
    )
    for cid, cname in enumerate(classes):
        report.per_class[cid] = PerClassMetrics(class_id=cid, class_name=cname)

    image_ids = sorted(set(predictions_by_image.keys()) | set(gts_by_image.keys()))

    # ---- 1) TP/FP/FN at iou_threshold_confusion ----
    pred_records_by_class: Dict[int, List[Tuple[float, int]]] = {cid: [] for cid in report.per_class}
    num_gt_by_class: Dict[int, int] = {cid: 0 for cid in report.per_class}
    overall_records: List[Tuple[float, int]] = []
    overall_gt_count = 0

    for image_id in image_ids:
        preds = sorted(predictions_by_image.get(image_id, []),
                       key=lambda d: d.confidence, reverse=True)
        gts = list(gts_by_image.get(image_id, []))

        for g in gts:
            if g.class_id in num_gt_by_class:
                num_gt_by_class[g.class_id] += 1
            overall_gt_count += 1

        if not preds and not gts:
            continue

        if preds and gts:
            pred_boxes = np.array([p.bbox for p in preds], dtype=np.float32)
            gt_boxes   = np.array([g.bbox for g in gts],   dtype=np.float32)
            iou_mat = iou_matrix(pred_boxes, gt_boxes)
        else:
            iou_mat = np.zeros((len(preds), len(gts)), dtype=np.float32)

        matched_gt: set = set()
        for pi, p in enumerate(preds):
            best_iou = 0.0
            best_gi = -1
            for gi, g in enumerate(gts):
                if gi in matched_gt or g.class_id != p.class_id:
                    continue
                if iou_mat[pi, gi] >= iou_threshold_confusion and iou_mat[pi, gi] > best_iou:
                    best_iou = float(iou_mat[pi, gi])
                    best_gi = gi
            is_tp = 1 if best_gi >= 0 else 0
            if is_tp:
                matched_gt.add(best_gi)
            if p.class_id in pred_records_by_class:
                pred_records_by_class[p.class_id].append((float(p.confidence), is_tp))
            overall_records.append((float(p.confidence), is_tp))

        for gi, g in enumerate(gts):
            if gi in matched_gt:
                continue
            if g.class_id in report.per_class:
                report.per_class[g.class_id].fn += 1

    # ---- per-class tallies ----
    for cid, pcm in report.per_class.items():
        recs = pred_records_by_class.get(cid, [])
        pcm.num_pred = len(recs)
        pcm.num_gt = num_gt_by_class[cid]
        pcm.tp = sum(r[1] for r in recs)
        pcm.fp = pcm.num_pred - pcm.tp
        pcm.precision = _safe_div(pcm.tp, pcm.tp + pcm.fp)
        pcm.recall    = _safe_div(pcm.tp, pcm.tp + pcm.fn)
        pcm.f1 = _f1(pcm.precision, pcm.recall)
        pcm.detection_accuracy = _safe_div(pcm.tp, pcm.tp + pcm.fp + pcm.fn)

    # ---- overall (micro) ----
    total_tp = sum(p.tp for p in report.per_class.values())
    total_fp = sum(p.fp for p in report.per_class.values())
    total_fn = sum(p.fn for p in report.per_class.values())
    report.total_tp, report.total_fp, report.total_fn = total_tp, total_fp, total_fn
    report.num_gt   = sum(p.num_gt   for p in report.per_class.values())
    report.num_pred = sum(p.num_pred for p in report.per_class.values())
    report.overall_precision = _safe_div(total_tp, total_tp + total_fp)
    report.overall_recall    = _safe_div(total_tp, total_tp + total_fn)
    report.overall_f1        = _f1(report.overall_precision, report.overall_recall)
    report.overall_detection_accuracy = _safe_div(total_tp, total_tp + total_fp + total_fn)

    # ---- AP@50 and mAP@50:95 ----
    ap50_per_class:   Dict[int, float] = {}
    ap5095_per_class: Dict[int, float] = {}
    for cid in report.per_class:
        aps_at = [
            _compute_ap_for_class(predictions_by_image, gts_by_image, cid, iou_t)
            for iou_t in report.map_iou_thresholds
        ]
        ap50_per_class[cid]   = float(aps_at[0]) if aps_at else 0.0
        ap5095_per_class[cid] = float(np.mean(aps_at)) if aps_at else 0.0
    for cid, pcm in report.per_class.items():
        pcm.ap50   = ap50_per_class.get(cid,   0.0)
        pcm.ap5095 = ap5095_per_class.get(cid, 0.0)

    valid_classes = [cid for cid, pcm in report.per_class.items() if pcm.num_gt > 0]
    if valid_classes:
        report.map50   = float(np.mean([ap50_per_class[c]   for c in valid_classes]))
        report.map5095 = float(np.mean([ap5095_per_class[c] for c in valid_classes]))

    # ---- overall PR curve ----
    report.pr_curve = _pr_curve_from_records(overall_records, overall_gt_count)
    return report


def _compute_ap_for_class(
    preds_by_image: Dict[int, List[Detection]],
    gts_by_image:    Dict[int, List[GTBox]],
    class_id: int,
    iou_threshold: float,
) -> float:
    """COCO-style 101-point interpolated AP for a single (class, IoU)."""
    records: List[Tuple[float, int]] = []
    n_gt = 0
    image_ids = sorted(set(preds_by_image.keys()) | set(gts_by_image.keys()))
    for image_id in image_ids:
        preds = [p for p in preds_by_image.get(image_id, []) if p.class_id == class_id]
        preds.sort(key=lambda d: d.confidence, reverse=True)
        gts = [g for g in gts_by_image.get(image_id, []) if g.class_id == class_id]
        n_gt += len(gts)
        if not preds:
            continue
        if not gts:
            for p in preds:
                records.append((float(p.confidence), 0))
            continue

        pred_boxes = np.array([p.bbox for p in preds], dtype=np.float32)
        gt_boxes   = np.array([g.bbox for g in gts],   dtype=np.float32)
        iou_mat = iou_matrix(pred_boxes, gt_boxes)
        matched: set = set()
        for pi, p in enumerate(preds):
            best_iou = 0.0
            best_gi = -1
            for gi in range(len(gts)):
                if gi in matched:
                    continue
                if iou_mat[pi, gi] >= iou_threshold and iou_mat[pi, gi] > best_iou:
                    best_iou = float(iou_mat[pi, gi])
                    best_gi = gi
            if best_gi >= 0:
                matched.add(best_gi)
                records.append((float(p.confidence), 1))
            else:
                records.append((float(p.confidence), 0))

    if n_gt == 0 or not records:
        return 0.0

    records.sort(key=lambda r: r[0], reverse=True)
    tp_arr = np.array([r[1] for r in records], dtype=np.float64)
    fp_arr = 1.0 - tp_arr
    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)
    recall    = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # Make precision monotonically non-increasing from right to left.
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    rec_thresholds = np.linspace(0.0, 1.0, 101)
    p_interp = np.zeros_like(rec_thresholds)
    for k, t in enumerate(rec_thresholds):
        idx = np.searchsorted(recall, t, side="left")
        p_interp[k] = precision[idx] if idx < len(precision) else 0.0
    return float(p_interp.mean())


def _pr_curve_from_records(records: List[Tuple[float, int]], n_gt: int) -> Dict[str, list]:
    if n_gt == 0 or not records:
        return {"recall": [], "precision": []}
    records.sort(key=lambda r: r[0], reverse=True)
    tp = np.array([r[1] for r in records], dtype=np.float64)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall    = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return {"recall": recall.tolist(), "precision": precision.tolist()}


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------- #
# Confusion matrix
# ---------------------------------------------------------------------- #

def build_confusion_matrix(
    predictions_by_image: Dict[int, List[Detection]],
    gts_by_image: Dict[int, List[GTBox]],
    classes: Sequence[str],
    iou_threshold: float = 0.5,
) -> ConfusionMatrix:
    """Greedy GT/pred matching by IoU; ``background`` row/col holds FN/FP."""
    names = list(classes) + [BACKGROUND]
    n = len(names)
    bg = n - 1
    M = np.zeros((n, n), dtype=np.int64)

    image_ids = sorted(set(predictions_by_image.keys()) | set(gts_by_image.keys()))
    for image_id in image_ids:
        preds = sorted(predictions_by_image.get(image_id, []),
                       key=lambda d: d.confidence, reverse=True)
        gts = list(gts_by_image.get(image_id, []))
        matched_gt: set = set()

        if preds and gts:
            pred_boxes = np.array([p.bbox for p in preds], dtype=np.float32)
            gt_boxes   = np.array([g.bbox for g in gts],   dtype=np.float32)
            iou_mat = iou_matrix(pred_boxes, gt_boxes)
        else:
            iou_mat = np.zeros((len(preds), len(gts)), dtype=np.float32)

        for pi, p in enumerate(preds):
            best_iou = 0.0
            best_gi = -1
            for gi in range(len(gts)):
                if gi in matched_gt:
                    continue
                if iou_mat[pi, gi] >= iou_threshold and iou_mat[pi, gi] > best_iou:
                    best_iou = float(iou_mat[pi, gi])
                    best_gi = gi
            if best_gi >= 0:
                gt_cls = gts[best_gi].class_id
                pred_cls = p.class_id
                M[gt_cls if 0 <= gt_cls < bg else bg,
                  pred_cls if 0 <= pred_cls < bg else bg] += 1
                matched_gt.add(best_gi)
            else:
                pred_cls = p.class_id
                M[bg, pred_cls if 0 <= pred_cls < bg else bg] += 1

        for gi, g in enumerate(gts):
            if gi in matched_gt:
                continue
            gt_cls = g.class_id
            M[gt_cls if 0 <= gt_cls < bg else bg, bg] += 1

    return ConfusionMatrix(classes=names, iou_threshold=float(iou_threshold), matrix=M)


def save_confusion_heatmap(cm: ConfusionMatrix, path: str | Path,
                            normalize: str = "row") -> None:
    """Save PNG heatmap. ``normalize`` ∈ {"row", "col", "none"}.

    Best-effort: if matplotlib is not installed, log a warning and skip the PNG
    (the confusion-matrix data itself is still produced) — never fail evaluation.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.warning("matplotlib not installed — skipping confusion heatmap %s", path)
        return

    M = cm.matrix.astype(np.float64).copy()
    if normalize == "row":
        row_sums = M.sum(axis=1, keepdims=True)
        M_norm = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)
        label = "row-normalized (recall view)"
    elif normalize == "col":
        col_sums = M.sum(axis=0, keepdims=True)
        M_norm = np.divide(M, col_sums, out=np.zeros_like(M), where=col_sums > 0)
        label = "column-normalized (precision view)"
    else:
        M_norm = M
        label = "raw counts"

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(cm.classes) + 4),
                                    max(5, 0.6 * len(cm.classes) + 3)))
    im = ax.imshow(M_norm, cmap="Blues", interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(len(cm.classes)))
    ax.set_yticks(np.arange(len(cm.classes)))
    ax.set_xticklabels(cm.classes, rotation=45, ha="right")
    ax.set_yticklabels(cm.classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix @IoU={cm.iou_threshold:.2f} — {label}")

    threshold = M_norm.max() / 2 if M_norm.max() > 0 else 0.5
    for i in range(M_norm.shape[0]):
        for j in range(M_norm.shape[1]):
            val = M_norm[i, j]
            raw = cm.matrix[i, j]
            color = "white" if val > threshold else "black"
            if normalize == "none":
                txt = f"{int(raw)}"
            else:
                txt = f"{val:.2f}\n({int(raw)})"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
