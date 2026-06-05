"""Tests for apps.evaluation_export.inference (Phase 7+8)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.evaluation_export.inference import (
    CropInput,
    MockPredictor,
    ONNXYOLOXPredictor,
    PyTorchYOLOXPredictor,
    TensorRTYOLOXPredictor,
    build_predictor,
)

# ---- needs_pixels contract --------------------------------------------

def test_mock_predictor_does_not_need_pixels(tmp_path: Path) -> None:
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"images": []}))
    pred = MockPredictor(p)
    assert pred.needs_pixels is False


def test_pytorch_predictor_class_declares_needs_pixels() -> None:
    # Without instantiating (which requires yolox), the class attribute must be True.
    assert PyTorchYOLOXPredictor.needs_pixels is True


def test_stub_predictors_declare_needs_pixels() -> None:
    assert ONNXYOLOXPredictor.needs_pixels is True
    assert TensorRTYOLOXPredictor.needs_pixels is True


# ---- build_predictor dispatch -----------------------------------------

def test_build_predictor_mock(tmp_path: Path) -> None:
    p = tmp_path / "mock.json"
    p.write_text(json.dumps({"images": []}))
    pred = build_predictor("mock", predictions_path=str(p))
    assert isinstance(pred, MockPredictor)


def test_build_predictor_mock_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="predictions_path"):
        build_predictor("mock")


def test_build_predictor_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown predictor backend"):
        build_predictor("bogus")


def test_build_predictor_onnx_requires_onnx_path() -> None:
    """Phase 9: ONNX predictor is real now — needs onnx_path. Without onnxruntime
    we observe ModuleNotFoundError; with it we'd observe FileNotFoundError on
    the fake path. Either is acceptable proof that factory wiring works."""
    try:
        import onnxruntime  # noqa: F401
        with pytest.raises((FileNotFoundError, ValueError)):
            build_predictor("onnx", onnx_path="/no/such/model.onnx", num_classes=4)
    except ImportError:
        with pytest.raises(ModuleNotFoundError):
            build_predictor("onnx", onnx_path="/no/such/model.onnx", num_classes=4)


def test_build_predictor_trt_requires_engine_path() -> None:
    """Phase 9: TRT predictor is real now — needs engine_path."""
    try:
        import pycuda.driver  # noqa: F401
        import tensorrt  # noqa: F401
        with pytest.raises((FileNotFoundError, ValueError)):
            build_predictor("trt", engine_path="/no/such/model.engine", num_classes=4)
    except ImportError:
        with pytest.raises(ModuleNotFoundError):
            build_predictor("trt", engine_path="/no/such/model.engine", num_classes=4)


# ---- PyTorch predictor without yolox installed ------------------------

def test_pytorch_predictor_requires_yolox(tmp_path: Path) -> None:
    """Without yolox installed, constructing PyTorchYOLOXPredictor must raise
    a clear ModuleNotFoundError (not crash with import gibberish).

    With yolox installed, this test is skipped.
    """
    try:
        import yolox  # noqa: F401
        pytest.skip("yolox is installed; this test only relevant without yolox")
    except ImportError:
        pass

    fake_ckpt = tmp_path / "weights.pth"
    fake_ckpt.write_bytes(b"")  # contents irrelevant; we never reach load
    with pytest.raises(ModuleNotFoundError, match="yolox|PyTorch"):
        PyTorchYOLOXPredictor(
            checkpoint=fake_ckpt,
            exp_module="experiments.yolox.class_obj_at_sea",
            num_classes=4,
        )


def test_pytorch_predictor_via_factory_propagates_error(tmp_path: Path) -> None:
    try:
        import yolox  # noqa: F401
        pytest.skip("yolox installed; nothing to test")
    except ImportError:
        pass
    with pytest.raises(ModuleNotFoundError):
        build_predictor("pytorch", checkpoint=tmp_path / "w.pth",
                         exp_module="experiments.yolox.class_obj_at_sea")


# ---- MockPredictor predict_batch behavior -----------------------------

def test_mock_predictor_returns_detections_when_keys_match(tmp_path: Path) -> None:
    payload = {
        "images": [{
            "image_id": 7, "image_size": [100, 100],
            "tiles": [{
                "scale_name": "original", "crop_offset": [0, 0],
                "detections": [{"class_id": 1, "confidence": 0.7,
                                 "bbox_crop": [10, 20, 30, 40]}],
            }],
        }],
    }
    p = tmp_path / "preds.json"
    p.write_text(json.dumps(payload))
    pred = MockPredictor(p)
    ci = CropInput(image_id=7, scale_name="original",
                    scale_size=(100, 100), crop_offset=(0, 0),
                    crop_size=(100, 100), tile_size=640)
    out = pred.predict_batch([ci])
    assert len(out) == 1 and len(out[0]) == 1
    det = out[0][0]
    assert det.class_id == 1
    assert det.bbox_crop == (10.0, 20.0, 30.0, 40.0)


def test_mock_predictor_returns_empty_on_miss(tmp_path: Path) -> None:
    p = tmp_path / "preds.json"
    p.write_text(json.dumps({"images": []}))
    pred = MockPredictor(p)
    ci = CropInput(image_id=99, scale_name="x", scale_size=(1, 1),
                    crop_offset=(0, 0), crop_size=(1, 1), tile_size=1)
    assert pred.predict_batch([ci]) == [[]]


def test_mock_predictor_invalid_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"not_images": []}))
    with pytest.raises(ValueError, match="images"):
        MockPredictor(p)


def test_pytorch_predictor_filters_unknown_kwargs(tmp_path: Path) -> None:
    """build_predictor should not pass arbitrary kwargs to PyTorchYOLOXPredictor."""
    try:
        import yolox  # noqa: F401
        pytest.skip("yolox installed; can't test arg filtering through real predictor easily")
    except ImportError:
        pass
    fake_ckpt = tmp_path / "w.pth"
    fake_ckpt.write_bytes(b"")
    # Pass unsupported kwargs alongside required ones — should not raise TypeError
    # before hitting the yolox import error.
    with pytest.raises(ModuleNotFoundError):
        build_predictor(
            "pytorch",
            checkpoint=fake_ckpt,
            exp_module="experiments.yolox.class_obj_at_sea",
            predictions_path="ignored",   # not a PyTorchYOLOXPredictor kwarg
            random_extra=42,
        )
