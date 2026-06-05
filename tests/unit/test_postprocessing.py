"""Unit tests for apps.evaluation_export.merge — NMS / Soft-NMS / WBF.

Migrated and adapted from yolox_eval/tests/test_nms.py.
"""
from __future__ import annotations

import pytest

from apps.evaluation_export.merge import (
    Detection,
    apply_postprocessing,
    nms_per_class,
    soft_nms_per_class,
    weighted_boxes_fusion_per_class,
)


def _det(class_id: int, confidence: float, bbox) -> Detection:
    return Detection(class_id=class_id, confidence=confidence, bbox=tuple(bbox))


def test_nms_removes_high_overlap_same_class() -> None:
    dets = [
        _det(0, 0.90, (10, 10, 100, 100)),
        _det(0, 0.85, (15, 15, 105, 105)),   # IoU ≈ 0.78 with first
        _det(0, 0.75, (200, 200, 300, 300)),  # disjoint
    ]
    kept = nms_per_class(dets, iou_threshold=0.5)
    assert len(kept) == 2
    assert sorted([k.confidence for k in kept], reverse=True) == [0.90, 0.75]


def test_nms_does_not_suppress_across_classes() -> None:
    dets = [
        _det(0, 0.90, (10, 10, 100, 100)),
        _det(1, 0.85, (10, 10, 100, 100)),
    ]
    kept = nms_per_class(dets, iou_threshold=0.5)
    assert len(kept) == 2


def test_nms_score_threshold_filters_low_confidence() -> None:
    dets = [_det(0, 0.5, (0, 0, 10, 10)), _det(0, 0.1, (50, 50, 60, 60))]
    kept = nms_per_class(dets, iou_threshold=0.5, score_threshold=0.25)
    assert [d.confidence for d in kept] == [0.5]


def test_nms_max_detections_cap() -> None:
    dets = [_det(0, 1.0 - 0.1 * i, (i * 30, 0, i * 30 + 20, 20)) for i in range(10)]
    kept = nms_per_class(dets, iou_threshold=0.5, max_detections=3)
    assert len(kept) == 3


def test_nms_empty_input() -> None:
    assert nms_per_class([], iou_threshold=0.5) == []


def test_soft_nms_keeps_neighbours_but_decays_score() -> None:
    dets = [
        _det(0, 0.9, (10, 10, 100, 100)),
        _det(0, 0.8, (20, 20, 110, 110)),  # high IoU with first
    ]
    kept = soft_nms_per_class(dets, iou_threshold=0.5, sigma=0.5)
    # Soft-NMS keeps both, decays the second one's score.
    assert len(kept) == 2
    confs = sorted([k.confidence for k in kept], reverse=True)
    assert confs[0] == pytest.approx(0.9)
    assert confs[1] < 0.8


def test_wbf_clusters_near_duplicates() -> None:
    dets = [
        _det(0, 0.90, (10, 10, 100, 100)),
        _det(0, 0.85, (12, 12, 102, 102)),
    ]
    kept = weighted_boxes_fusion_per_class(dets, iou_threshold=0.5)
    assert len(kept) == 1
    fused = kept[0]
    # weighted mean is between the two boxes
    assert 10 <= fused.bbox[0] <= 12
    assert fused.source == "wbf"


def test_apply_postprocessing_dispatches() -> None:
    dets = [_det(0, 0.9, (0, 0, 50, 50)), _det(0, 0.8, (5, 5, 55, 55))]
    for method in ("nms", "soft_nms", "wbf", "class_aware_nms"):
        out = apply_postprocessing(dets, method=method, iou_threshold=0.5)
        assert isinstance(out, list)
    with pytest.raises(ValueError):
        apply_postprocessing(dets, method="bogus")


def test_detection_to_dict_roundtrip() -> None:
    d = _det(2, 0.7, (1.0, 2.0, 3.0, 4.0))
    d.merged_from = [1, 2, 3]
    out = d.to_dict()
    assert out["class_id"] == 2
    assert out["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert out["merged_from"] == [1, 2, 3]
