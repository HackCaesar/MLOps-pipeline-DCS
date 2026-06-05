"""Debug visualizations: GT, predictions, tile grid, TP/FP/FN overlay, PR curve.

Migrated from ``yolox_eval/src/visualization.py``. Requires opencv or PIL
(picked by ``packages.common.image_io.available_backend()``). When using PIL,
text antialiasing is identical; we draw boxes/text via PIL.ImageDraw.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Tuple

import numpy as np

from apps.evaluation_export.merge import Detection
from apps.evaluation_export.metrics import GTBox
from packages.common.bbox_ops import iou_xyxy
from packages.common.image_io import available_backend

if TYPE_CHECKING:
    pass


_PALETTE = [
    (0, 165, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (128, 128, 0),
]


def _color_for_class(class_id: int) -> Tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def make_blank_canvas(image_size: Tuple[int, int]) -> np.ndarray:
    """``(W, H)`` → grey ``(H, W, 3)`` canvas. Useful when only synthetic data is on hand."""
    w, h = image_size
    return np.full((h, w, 3), 40, dtype=np.uint8)


def draw_boxes(image: np.ndarray, dets: Sequence[Detection],
               classes: Sequence[str], thickness: int = 2,
               label_prefix: str = "") -> np.ndarray:
    img = image.copy()
    for d in dets:
        c = _color_for_class(d.class_id)
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        name = d.class_name or (classes[d.class_id] if 0 <= d.class_id < len(classes) else str(d.class_id))
        label = f"{label_prefix}{name} {d.confidence:.2f}"
        _draw_rect_and_label(img, (x1, y1, x2, y2), c, thickness, label)
    return img


def draw_gt(image: np.ndarray, gts: Sequence[GTBox],
            classes: Sequence[str], thickness: int = 2) -> np.ndarray:
    img = image.copy()
    for g in gts:
        c = (0, 255, 0)
        x1, y1, x2, y2 = (int(v) for v in g.bbox)
        name = g.class_name or (classes[g.class_id] if 0 <= g.class_id < len(classes) else str(g.class_id))
        _draw_rect_and_label(img, (x1, y1, x2, y2), c, thickness, f"GT:{name}")
    return img


def draw_tp_fp_fn(image: np.ndarray, preds: Sequence[Detection],
                  gts: Sequence[GTBox], classes: Sequence[str],
                  iou_threshold: float = 0.5) -> np.ndarray:
    """Green=TP, orange=wrong-class (WC), red=FP, yellow=FN."""
    img = image.copy()
    matched_gt: set = set()
    sorted_preds = sorted(preds, key=lambda d: d.confidence, reverse=True)
    for p in sorted_preds:
        best_iou, best_gi, best_same_class = 0.0, -1, False
        for gi, g in enumerate(gts):
            if gi in matched_gt:
                continue
            iou = iou_xyxy(p.bbox, g.bbox)
            if iou >= iou_threshold and iou > best_iou:
                best_iou, best_gi = iou, gi
                best_same_class = (g.class_id == p.class_id)
        x1, y1, x2, y2 = (int(v) for v in p.bbox)
        if best_gi >= 0 and best_same_class:
            color, tag = (0, 255, 0), "TP"
            matched_gt.add(best_gi)
        elif best_gi >= 0:
            color, tag = (0, 165, 255), "WC"
            matched_gt.add(best_gi)
        else:
            color, tag = (0, 0, 255), "FP"
        _draw_rect_and_label(img, (x1, y1, x2, y2), color, 2, f"{tag}:{p.confidence:.2f}")
    for gi, g in enumerate(gts):
        if gi in matched_gt:
            continue
        x1, y1, x2, y2 = (int(v) for v in g.bbox)
        _draw_rect_and_label(img, (x1, y1, x2, y2), (0, 255, 255), 2, "FN")
    return img


def save_image(image: np.ndarray, path: str | Path) -> None:
    from packages.common.image_io import imwrite
    imwrite(path, image)


def save_pr_curve(report, path: str | Path) -> None:
    """Save the overall PR-curve PNG from a ``MetricsReport``."""
    pr = getattr(report, "pr_curve", None) or {}
    rec = pr.get("recall", [])
    prec = pr.get("precision", [])
    if not rec or not prec:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"mAP@50={report.map50:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title("Precision-Recall (overall)")
    ax.legend()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------- #
# backend-agnostic rect+label rendering
# ---------------------------------------------------------------------- #

def _draw_rect_and_label(img: np.ndarray, box: Tuple[int, int, int, int],
                         color_bgr: Tuple[int, int, int],
                         thickness: int, label: str) -> None:
    """Draw ``box`` and ``label`` on ``img`` in-place. Image is in RGB; we
    convert the BGR palette colour to RGB before drawing."""
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))

    backend = available_backend()
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    if backend == "cv2":
        import cv2
        bgr_view = img  # cv2 expects BGR — we treat array as BGR here
        cv2.rectangle(bgr_view, (x1, y1), (x2, y2), color_bgr, thickness)
        cv2.putText(bgr_view, label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)
        return

    # PIL fallback
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for w_off in range(thickness):
        draw.rectangle((x1 - w_off, y1 - w_off, x2 + w_off, y2 + w_off),
                       outline=color_rgb)
    draw.text((x1, max(0, y1 - 12)), label, fill=color_rgb)
    img[:] = np.asarray(pil)
