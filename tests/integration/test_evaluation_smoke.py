"""End-to-end evaluation smoke test on a tiny raw dataset + mock predictions.

Verifies the full ``apps.evaluation_export.evaluate.evaluate`` orchestrator:
- consumes a per-split COCO ground-truth file;
- runs the predictor (MockPredictor) over per-scale crops;
- maps detections back to source coords;
- merges via NMS + fragment fusion;
- writes all expected artifacts.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from apps.evaluation_export.evaluate import evaluate
from apps.evaluation_export.inference import MockPredictor
from tests.fixtures.mini_raw_dataset import write_mini_raw_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]

_EVAL_CFG = {
    "enabled": True,
    "full_image_first": True,
    "crop_inference": True,
    "crop_size": 640,
    "stride": 320,
    "include_edge_tiles": True,
    "scales": [
        {"name": "full_source", "mode": "source_full_image"},
        {"name": "original_crops", "mode": "keep_original", "crop": True},
        {"name": "final_640", "mode": "resize", "size": [640, 640], "crop": False},
    ],
    "merge": {
        "method": "class_aware_nms",
        "iou_threshold": 0.5,
        "confidence_threshold": 0.25,
        "max_detections_per_image": 300,
        "fragment_fusion": {
            "enabled": True,
            "max_distance_px": 50,
            "min_iou": 0.05,
            "min_ios": 0.3,
            "strategy": "union",
            "max_aspect_ratio": 30,
            "max_area_ratio": 50,
            "disabled_classes": [],
        },
    },
    "metrics": {
        "iou_thresholds": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        "save_confusion_matrix": True,
        "save_fp_fn_examples": True,
        "save_visualizations": False,
    },
}


def _write_perfect_mock_predictions(raw_dataset_dir: Path, predictions_path: Path) -> None:
    """Build a predictions_tiles.json that mirrors every GT bbox in crop coords.

    Trick: GT bboxes are at known positions (see mini_raw_dataset.py). We can
    inject a "perfect" prediction by placing the bbox in the appropriate
    full_source letterbox crop with the inverse-letterbox transform applied.

    For simpler test, we put the prediction in the ``full_source`` scale (one
    crop per image) using the source bbox in the letterbox coords.
    """
    from packages.common.letterbox import compute_letterbox, source_box_to_letterbox

    coco = json.loads(
        (raw_dataset_dir / "annotations" / "instances_test.json").read_text())
    images = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    images_payload = []
    for img_id, img_meta in images.items():
        t = compute_letterbox(img_meta["width"], img_meta["height"], 640)
        dets = []
        for ann in anns_by_image.get(img_id, []):
            x, y, w, h = ann["bbox"]
            xyxy_src = (x, y, x + w, y + h)
            xyxy_lb = source_box_to_letterbox(xyxy_src, t)
            dets.append({
                "class_id": int(ann["category_id"]),
                "confidence": 0.99,
                "bbox_crop": list(xyxy_lb),
            })
        images_payload.append({
            "image_id": img_id,
            "image_size": [img_meta["width"], img_meta["height"]],
            "tiles": [{
                "scale_name": "full_source",
                "scale_size": [img_meta["width"], img_meta["height"]],
                "crop_offset": [0, 0],
                "crop_size": [img_meta["width"], img_meta["height"]],
                "tile_size": 640,
                "detections": dets,
            }],
        })
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(json.dumps({"images": images_payload}), encoding="utf-8")


def test_evaluation_writes_full_artifact_set(tmp_path: Path) -> None:
    raw = write_mini_raw_dataset(
        tmp_path / "raw",
        images_per_split={"train": 0, "val": 0, "test": 3},
        annotations_per_image=2,
        include_one_background_in_train=False,
    )
    predictions_path = tmp_path / "mock_preds.json"
    _write_perfect_mock_predictions(raw, predictions_path)

    predictor = MockPredictor(predictions_path)
    output_dir = tmp_path / "reports" / "run42"

    result = evaluate(
        predictor=predictor,
        raw_dataset_dir=raw,
        split="test",
        classes=["ships", "helicopters", "airplanes"],
        eval_cfg=_EVAL_CFG,
        output_dir=output_dir,
        backend_name="mock",
        model_path=None,
        predictions_path=str(predictions_path),
    )

    # All expected artifacts written
    for key in ("summary_json", "summary_csv", "per_class_csv",
                "confusion_csv", "confusion_png",
                "false_positives", "false_negatives",
                "predictions_raw_tiles", "predictions_stitched", "predictions_coco"):
        assert key in result.artifacts, f"missing {key}"
        assert result.artifacts[key].is_file(), f"file not written: {result.artifacts[key]}"

    # With perfect predictions on `full_source` scale, mAP should be very high.
    summary = json.loads(result.artifacts["summary_json"].read_text())
    assert summary["metrics"]["overall_precision"] >= 0.9
    assert summary["metrics"]["map50"] >= 0.9


def test_evaluation_no_gt_yields_all_fp(tmp_path: Path) -> None:
    """When predictions exist but GT is empty, everything counts as FP."""
    raw = write_mini_raw_dataset(
        tmp_path / "raw",
        images_per_split={"train": 0, "val": 0, "test": 1},
        annotations_per_image=0,                           # ← no GT
        include_one_background_in_train=False,
    )
    # Inject a single bogus prediction for the one image. With 2560x1440 source
    # and 640x640 letterbox: scale=0.25, pad_x=0, pad_y=140. The letterboxed
    # image occupies y ∈ [140, 500]. Put the bbox INSIDE that band so the
    # inverse-letterbox unmap produces a non-degenerate source bbox.
    predictions = {
        "images": [{
            "image_id": 1, "image_size": [2560, 1440],
            "tiles": [{
                "scale_name": "full_source",
                "scale_size": [2560, 1440],
                "crop_offset": [0, 0],
                "crop_size":  [2560, 1440],
                "tile_size":  640,
                "detections": [{"class_id": 0, "confidence": 0.9,
                                "bbox_crop": [50, 200, 200, 400]}],
            }],
        }],
    }
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps(predictions))

    predictor = MockPredictor(pred_path)
    output_dir = tmp_path / "out"
    result = evaluate(
        predictor=predictor, raw_dataset_dir=raw, split="test",
        classes=["ships", "helicopters", "airplanes"],
        eval_cfg=_EVAL_CFG, output_dir=output_dir, backend_name="mock",
    )
    summary = json.loads(result.artifacts["summary_json"].read_text())
    assert summary["metrics"]["total_tp"] == 0
    assert summary["metrics"]["total_fp"] >= 1


def test_evaluation_cli_subprocess(tmp_path: Path) -> None:
    raw = write_mini_raw_dataset(
        tmp_path / "raw",
        images_per_split={"train": 0, "val": 0, "test": 2},
        include_one_background_in_train=False,
    )
    predictions_path = tmp_path / "preds.json"
    _write_perfect_mock_predictions(raw, predictions_path)

    cfg = {
        "data": {"raw_dataset_dir": str(raw), "class_names_path": "configs/classes.yaml"},
        "evaluation": _EVAL_CFG,
    }
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    output_dir = tmp_path / "reports_cli"
    proc = subprocess.run(
        [sys.executable, "-m", "apps.evaluation_export.cli", "evaluate",
         "--config", str(cfg_path),
         "--backend", "mock",
         "--predictions", str(predictions_path),
         "--split", "test",
         "--output-dir", str(output_dir)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert (output_dir / "metrics" / "summary.json").is_file()
