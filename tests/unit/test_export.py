"""Phase 9 tests for apps.evaluation_export.export and CLI export commands.

Most of these test that the export modules import cleanly without onnx/tensorrt
and that the CLI fails gracefully with clear messages when deps are missing.

When onnx (resp. tensorrt) are installed, the actual export functions are NOT
exercised here — that requires a checkpoint + GPU and is covered by manual
smoke testing on a Windows host / Docker.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _yolox_available() -> bool:
    try:
        import yolox  # noqa: F401
        return True
    except ImportError:
        return False


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _onnx_available() -> bool:
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


def _tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


# ---- module imports clean even without deps ---------------------------

def test_export_module_imports_clean() -> None:
    """Importing apps.evaluation_export.export must not require onnx/tensorrt."""
    import importlib
    mod = importlib.import_module("apps.evaluation_export.export")
    assert hasattr(mod, "export_to_onnx")
    assert hasattr(mod, "build_tensorrt_engine")
    assert hasattr(mod, "OnnxExportResult")
    assert hasattr(mod, "TensorRTBuildResult")


# ---- graceful failures without deps -----------------------------------

@pytest.mark.skipif(_torch_available() and _onnx_available(),
                    reason="torch+onnx installed; test only relevant without them")
def test_export_to_onnx_without_deps_raises_clear_error(tmp_path: Path) -> None:
    from apps.evaluation_export.export import export_to_onnx
    fake_ckpt = tmp_path / "weights.pth"
    fake_ckpt.write_bytes(b"")
    with pytest.raises(ModuleNotFoundError, match="torch|onnx"):
        export_to_onnx(
            checkpoint=fake_ckpt,
            exp_module="experiments.yolox.class_obj_at_sea",
            output=tmp_path / "model.onnx",
            num_classes=4,
        )


@pytest.mark.skipif(_tensorrt_available(),
                    reason="tensorrt installed; test only relevant without it")
def test_build_tensorrt_engine_without_deps_raises_clear_error(tmp_path: Path) -> None:
    from apps.evaluation_export.export import build_tensorrt_engine
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_bytes(b"")
    with pytest.raises(ModuleNotFoundError, match="tensorrt"):
        build_tensorrt_engine(
            onnx_path=fake_onnx,
            output=tmp_path / "model.engine",
            precision="fp16",
        )


def test_build_tensorrt_engine_rejects_unknown_precision(tmp_path: Path) -> None:
    if not _tensorrt_available():
        # Without trt, the ModuleNotFoundError fires before the precision check.
        # Skip — precision validation is exercised below if trt is present.
        pytest.skip("tensorrt not installed; precision validation runs after import")
    from apps.evaluation_export.export import build_tensorrt_engine
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_bytes(b"x")
    with pytest.raises(ValueError, match="precision"):
        build_tensorrt_engine(
            onnx_path=fake_onnx,
            output=tmp_path / "x.engine",
            precision="bogus",
        )


def test_build_tensorrt_engine_missing_onnx_raises(tmp_path: Path) -> None:
    if not _tensorrt_available():
        pytest.skip("tensorrt not installed; missing-onnx check runs after import")
    from apps.evaluation_export.export import build_tensorrt_engine
    with pytest.raises(FileNotFoundError):
        build_tensorrt_engine(
            onnx_path=tmp_path / "no.onnx",
            output=tmp_path / "x.engine",
            precision="fp16",
        )


# ---- CLI: --help works without deps -----------------------------------

def test_cli_export_onnx_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "apps.evaluation_export.cli", "export-onnx", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0
    assert "--checkpoint" in proc.stdout
    assert "--output" in proc.stdout
    assert "--exp-module" in proc.stdout
    assert "--opset" in proc.stdout


def test_cli_export_trt_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "apps.evaluation_export.cli", "export-trt", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0
    assert "--onnx" in proc.stdout
    assert "--precision" in proc.stdout
    assert "--calib-images" in proc.stdout
    assert "--workspace-mb" in proc.stdout


def test_cli_evaluate_help_now_lists_onnx_and_engine_flags() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "apps.evaluation_export.cli", "evaluate", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0
    assert "--onnx" in proc.stdout
    assert "--engine" in proc.stdout


# ---- CLI: graceful failure when deps missing --------------------------

@pytest.mark.skipif(_torch_available() and _onnx_available(),
                    reason="torch+onnx installed; CLI would actually run")
def test_cli_export_onnx_without_deps_returns_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "data": {"raw_dataset_dir": str(tmp_path)},
        "evaluation": {"crop_size": 640},
    }))
    fake_ckpt = tmp_path / "w.pth"
    fake_ckpt.write_bytes(b"")
    proc = subprocess.run(
        [sys.executable, "-m", "apps.evaluation_export.cli", "export-onnx",
         "--config", str(cfg_path),
         "--checkpoint", str(fake_ckpt),
         "--output", str(tmp_path / "model.onnx"),
         "--exp-module", "experiments.yolox.class_obj_at_sea"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 1
    out = proc.stdout + proc.stderr
    assert "torch" in out.lower() or "onnx" in out.lower() or "yolox" in out.lower()


@pytest.mark.skipif(_tensorrt_available(),
                    reason="tensorrt installed; CLI would actually run")
def test_cli_export_trt_without_deps_returns_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "data": {"raw_dataset_dir": str(tmp_path)},
        "evaluation": {"crop_size": 640},
    }))
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_bytes(b"")
    proc = subprocess.run(
        [sys.executable, "-m", "apps.evaluation_export.cli", "export-trt",
         "--config", str(cfg_path),
         "--onnx", str(fake_onnx),
         "--output", str(tmp_path / "model.engine"),
         "--precision", "fp16"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 1
    out = proc.stdout + proc.stderr
    assert "tensorrt" in out.lower()
