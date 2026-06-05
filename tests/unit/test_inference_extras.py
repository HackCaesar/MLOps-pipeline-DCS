"""Phase 9: ONNX and TensorRT predictor graceful-failure + needs_pixels tests.

The actual ``predict_batch`` runs require onnxruntime / tensorrt + a real model;
those are covered by manual smoke testing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apps.evaluation_export.inference import (
    ONNXYOLOXPredictor,
    TensorRTYOLOXPredictor,
    build_predictor,
)


def _onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def _pycuda_available() -> bool:
    try:
        import pycuda.driver  # noqa: F401
        return True
    except ImportError:
        return False


# ---- needs_pixels still True after Phase 9 ----------------------------

def test_onnx_predictor_class_needs_pixels() -> None:
    assert ONNXYOLOXPredictor.needs_pixels is True


def test_tensorrt_predictor_class_needs_pixels() -> None:
    assert TensorRTYOLOXPredictor.needs_pixels is True


# ---- graceful failure when deps absent --------------------------------

@pytest.mark.skipif(_onnxruntime_available(),
                    reason="onnxruntime installed; this test only relevant without")
def test_onnx_predictor_requires_onnxruntime(tmp_path: Path) -> None:
    fake = tmp_path / "model.onnx"
    fake.write_bytes(b"")
    with pytest.raises(ModuleNotFoundError, match="onnxruntime"):
        ONNXYOLOXPredictor(onnx_path=fake, num_classes=4)


@pytest.mark.skipif(_tensorrt_available() and _pycuda_available(),
                    reason="tensorrt+pycuda installed; this test only relevant without")
def test_tensorrt_predictor_requires_tensorrt_or_pycuda(tmp_path: Path) -> None:
    fake = tmp_path / "model.engine"
    fake.write_bytes(b"")
    with pytest.raises(ModuleNotFoundError, match="tensorrt|pycuda"):
        TensorRTYOLOXPredictor(engine_path=fake, num_classes=4)


# ---- factory dispatches and forwards kwargs ---------------------------

@pytest.mark.skipif(_onnxruntime_available(),
                    reason="onnxruntime installed; can't observe lazy-import failure")
def test_factory_onnx_propagates_module_not_found(tmp_path: Path) -> None:
    fake = tmp_path / "x.onnx"
    fake.write_bytes(b"")
    with pytest.raises(ModuleNotFoundError):
        build_predictor("onnx", onnx_path=fake, num_classes=4)


@pytest.mark.skipif(_tensorrt_available() and _pycuda_available(),
                    reason="tensorrt+pycuda installed; can't observe lazy-import failure")
def test_factory_trt_propagates_module_not_found(tmp_path: Path) -> None:
    fake = tmp_path / "x.engine"
    fake.write_bytes(b"")
    with pytest.raises(ModuleNotFoundError):
        build_predictor("trt", engine_path=fake, num_classes=4)


def test_factory_onnx_filters_unknown_kwargs(tmp_path: Path) -> None:
    """build_predictor should silently drop unsupported kwargs for ONNX."""
    if _onnxruntime_available():
        pytest.skip("onnxruntime installed; would actually attempt to load file")
    fake = tmp_path / "x.onnx"
    fake.write_bytes(b"")
    with pytest.raises(ModuleNotFoundError):
        # `predictions_path` is a mock kwarg, not valid for onnx — must be ignored.
        build_predictor("onnx", onnx_path=fake, num_classes=4,
                         predictions_path="ignored", random_extra=42)


def test_factory_trt_filters_unknown_kwargs(tmp_path: Path) -> None:
    if _tensorrt_available() and _pycuda_available():
        pytest.skip("tensorrt+pycuda installed; would actually load engine")
    fake = tmp_path / "x.engine"
    fake.write_bytes(b"")
    with pytest.raises(ModuleNotFoundError):
        build_predictor("trt", engine_path=fake, num_classes=4,
                         predictions_path="ignored", checkpoint="ignored")
