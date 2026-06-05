"""Unit tests for apps.evaluation_export.metrics — mAP + confusion matrix.

Adapted from yolox_eval/tests/{test_metrics,test_confusion_matrix}.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apps.evaluation_export.merge import Detection
from apps.evaluation_export.metrics import (
    GTBox,
    build_confusion_matrix,
    compute_metrics,
    save_confusion_heatmap,
)


def _det(image_id, class_id, confidence, bbox) -> Detection:
    return Detection(class_id=class_id, confidence=confidence, bbox=tuple(bbox))


def _gt(image_id, class_id, bbox) -> GTBox:
    return GTBox(image_id=image_id, class_id=class_id, bbox=tuple(bbox))


# ---- compute_metrics --------------------------------------------------

def test_perfect_match_yields_full_metrics() -> None:
    preds = {1: [_det(1, 0, 0.9, (0, 0, 100, 100))]}
    gts   = {1: [_gt(1, 0, (0, 0, 100, 100))]}
    report = compute_metrics(preds, gts, classes=["a", "b"], iou_threshold_confusion=0.5)
    assert report.overall_precision == 1.0
    assert report.overall_recall == 1.0
    assert report.overall_f1 == 1.0
    assert report.map50 == 1.0
    assert report.per_class[0].tp == 1
    assert report.per_class[0].fp == 0
    assert report.per_class[0].fn == 0


def test_false_positive_when_no_gt() -> None:
    preds = {1: [_det(1, 0, 0.9, (0, 0, 100, 100))]}
    gts   = {1: []}
    report = compute_metrics(preds, gts, classes=["a"], iou_threshold_confusion=0.5)
    assert report.per_class[0].fp == 1
    assert report.per_class[0].tp == 0
    assert report.overall_precision == 0.0


def test_false_negative_when_no_pred() -> None:
    preds = {1: []}
    gts   = {1: [_gt(1, 0, (0, 0, 100, 100))]}
    report = compute_metrics(preds, gts, classes=["a"], iou_threshold_confusion=0.5)
    assert report.per_class[0].fn == 1
    assert report.per_class[0].recall == 0.0


def test_wrong_class_counts_as_fp_and_fn() -> None:
    preds = {1: [_det(1, 1, 0.9, (0, 0, 100, 100))]}
    gts   = {1: [_gt(1, 0, (0, 0, 100, 100))]}
    report = compute_metrics(preds, gts, classes=["a", "b"], iou_threshold_confusion=0.5)
    # Pred class 1 doesn't match GT class 0 → FP for class 1, FN for class 0
    assert report.per_class[0].fn == 1
    assert report.per_class[1].fp == 1


def test_multi_image_map_averaged() -> None:
    preds = {
        1: [_det(1, 0, 0.9, (0, 0, 100, 100))],
        2: [_det(2, 0, 0.9, (0, 0, 100, 100))],
    }
    gts = {
        1: [_gt(1, 0, (0, 0, 100, 100))],
        2: [_gt(2, 0, (0, 0, 100, 100))],
    }
    report = compute_metrics(preds, gts, classes=["a"], iou_threshold_confusion=0.5)
    assert report.map50 == 1.0
    assert report.num_gt == 2 and report.num_pred == 2


def test_detection_accuracy_formula() -> None:
    preds = {1: [_det(1, 0, 0.9, (0, 0, 100, 100)),
                  _det(1, 0, 0.8, (1000, 1000, 1100, 1100))]}
    gts = {1: [_gt(1, 0, (0, 0, 100, 100))]}
    report = compute_metrics(preds, gts, classes=["a"])
    # TP=1, FP=1, FN=0 → accuracy = 1/2
    assert report.overall_detection_accuracy == pytest.approx(0.5)


# ---- confusion matrix -------------------------------------------------

def test_confusion_matrix_tp_on_diagonal() -> None:
    preds = {1: [_det(1, 0, 0.9, (0, 0, 100, 100))]}
    gts   = {1: [_gt(1, 0, (0, 0, 100, 100))]}
    cm = build_confusion_matrix(preds, gts, classes=["a"], iou_threshold=0.5)
    assert cm.matrix[0, 0] == 1
    assert cm.matrix.sum() == 1


def test_confusion_matrix_wrong_class_off_diagonal() -> None:
    preds = {1: [_det(1, 1, 0.9, (0, 0, 100, 100))]}
    gts   = {1: [_gt(1, 0, (0, 0, 100, 100))]}
    cm = build_confusion_matrix(preds, gts, classes=["a", "b"], iou_threshold=0.5)
    # GT class 0, pred class 1 → matrix[0, 1] == 1
    assert cm.matrix[0, 1] == 1


def test_confusion_matrix_fp_in_background_row() -> None:
    preds = {1: [_det(1, 0, 0.9, (0, 0, 100, 100))]}
    gts   = {1: []}
    cm = build_confusion_matrix(preds, gts, classes=["a"], iou_threshold=0.5)
    # bg row = last; pred class 0 → matrix[1, 0] == 1
    assert cm.matrix[1, 0] == 1


def test_confusion_matrix_fn_in_background_col() -> None:
    preds = {1: []}
    gts   = {1: [_gt(1, 0, (0, 0, 100, 100))]}
    cm = build_confusion_matrix(preds, gts, classes=["a"], iou_threshold=0.5)
    # GT class 0, no match → matrix[0, 1] == 1 (col bg)
    assert cm.matrix[0, 1] == 1


def test_confusion_matrix_csv_roundtrip(tmp_path: Path) -> None:
    preds = {1: [_det(1, 0, 0.9, (0, 0, 100, 100))]}
    gts   = {1: [_gt(1, 0, (0, 0, 100, 100))]}
    cm = build_confusion_matrix(preds, gts, classes=["a", "b"])
    p = tmp_path / "cm.csv"
    cm.to_csv(p)
    text = p.read_text(encoding="utf-8")
    assert "gt\\pred" in text
    assert "background" in text


def test_save_confusion_heatmap_writes_png(tmp_path: Path) -> None:
    preds = {1: [_det(1, 0, 0.9, (0, 0, 100, 100))]}
    gts   = {1: [_gt(1, 0, (0, 0, 100, 100))]}
    cm = build_confusion_matrix(preds, gts, classes=["a"])
    p = tmp_path / "cm.png"
    save_confusion_heatmap(cm, p)
    assert p.is_file() and p.stat().st_size > 0
