"""Unit tests for fragment fusion in apps.evaluation_export.merge.

Adapted from yolox_eval/tests/test_box_fusion.py.
"""
from __future__ import annotations

import pytest

from apps.evaluation_export.merge import (
    Detection,
    FusionConfig,
    merge_fragmented_boxes,
)


def _det(class_id: int, confidence: float, bbox) -> Detection:
    return Detection(class_id=class_id, confidence=confidence, bbox=tuple(bbox))


def test_long_object_two_touching_fragments_merge() -> None:
    """A long object spans two crops — each fragment touches at x=100."""
    a = _det(0, 0.8, (0, 50, 100, 100))   # left half
    b = _det(0, 0.7, (100, 50, 200, 100))  # right half, touching
    cfg = FusionConfig(max_distance_px=10, min_iou=0.01, min_ios=0.1,
                       fusion_strategy="union",
                       max_aspect_ratio=200, max_area_ratio=200)
    out, stats = merge_fragmented_boxes([a, b], cfg)
    assert len(out) == 1
    fused = out[0]
    assert fused.bbox == (0, 50, 200, 100)
    assert stats.after_merge == 1
    assert stats.merged_groups == [[0, 1]]


def test_distant_objects_not_merged() -> None:
    a = _det(0, 0.8, (0, 0, 50, 50))
    b = _det(0, 0.7, (500, 500, 550, 550))
    cfg = FusionConfig(max_distance_px=50, min_iou=0.05, min_ios=0.3,
                       max_aspect_ratio=200, max_area_ratio=200)
    out, _ = merge_fragmented_boxes([a, b], cfg)
    assert len(out) == 2


def test_aspect_ratio_rollback() -> None:
    """If merged box becomes too long (e.g. line-like), reject the merge."""
    a = _det(0, 0.8, (0, 100, 50, 110))     # very thin: 50×10
    b = _det(0, 0.7, (60, 100, 1000, 110))  # also thin: 940×10
    cfg = FusionConfig(max_distance_px=20, min_iou=0.0, min_ios=0.0,
                       max_aspect_ratio=20, max_area_ratio=200,
                       fusion_strategy="union")
    out, _ = merge_fragmented_boxes([a, b], cfg)
    # Merged box would have aspect ~100:1 → exceeds 20 → roll back.
    assert len(out) == 2


def test_area_ratio_rollback() -> None:
    """If fused area >> sum of parts (false-positive merge), roll back."""
    a = _det(0, 0.9, (0, 0, 10, 10))         # area 100
    b = _det(0, 0.9, (1000, 1000, 1010, 1010))  # area 100
    cfg = FusionConfig(max_distance_px=10000, min_iou=0.0, min_ios=0.0,
                       max_aspect_ratio=1e6, max_area_ratio=2,
                       fusion_strategy="union")
    out, _ = merge_fragmented_boxes([a, b], cfg)
    # Union box would be ~10M area vs 200 sum → area_ratio = 50000 → roll back.
    assert len(out) == 2


def test_disabled_classes_skipped() -> None:
    a = _det(2, 0.9, (0, 0, 100, 100))
    b = _det(2, 0.8, (50, 0, 150, 100))
    cfg = FusionConfig(max_distance_px=10, min_iou=0.01, min_ios=0.1,
                       max_aspect_ratio=10, max_area_ratio=10,
                       disabled_classes=[2])
    out, stats = merge_fragmented_boxes([a, b], cfg)
    assert len(out) == 2
    assert stats.after_merge == 2


def test_different_classes_not_merged() -> None:
    a = _det(0, 0.9, (0, 0, 100, 100))
    b = _det(1, 0.8, (0, 0, 100, 100))  # same box different class
    cfg = FusionConfig(max_distance_px=10, min_iou=0.1, min_ios=0.1)
    out, _ = merge_fragmented_boxes([a, b], cfg)
    assert len(out) == 2


def test_weighted_strategy() -> None:
    a = _det(0, 1.0, (0, 0, 100, 100))
    b = _det(0, 1.0, (10, 10, 110, 110))
    cfg = FusionConfig(max_distance_px=10, min_iou=0.5, min_ios=0.5,
                       fusion_strategy="weighted",
                       max_aspect_ratio=100, max_area_ratio=100)
    out, _ = merge_fragmented_boxes([a, b], cfg)
    assert len(out) == 1
    # Weighted average between the two boxes
    fused = out[0]
    assert fused.bbox[0] == pytest.approx(5.0, abs=0.5)


def test_fusion_disabled_pass_through() -> None:
    dets = [_det(0, 0.9, (0, 0, 100, 100)), _det(0, 0.8, (10, 10, 110, 110))]
    cfg = FusionConfig(enabled=False, max_distance_px=10, min_iou=0.5, min_ios=0.5)
    out, stats = merge_fragmented_boxes(dets, cfg)
    assert len(out) == 2
    assert stats.after_merge == 2


def test_unknown_strategy_raises() -> None:
    a = _det(0, 0.9, (0, 0, 100, 100))
    b = _det(0, 0.8, (50, 0, 150, 100))
    cfg = FusionConfig(max_distance_px=10, min_iou=0.01, min_ios=0.1,
                       fusion_strategy="bogus",
                       max_aspect_ratio=100, max_area_ratio=100)
    with pytest.raises(ValueError, match="fusion_strategy"):
        merge_fragmented_boxes([a, b], cfg)
