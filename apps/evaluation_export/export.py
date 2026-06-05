"""Model export: PyTorch (.pth) → ONNX → TensorRT (FP16 / INT8).

Two-step pipeline:

  1. ``export_to_onnx(checkpoint, exp_..., output)`` — load yolox Exp + weights,
     trace through ``torch.onnx.export``, validate with ``onnx.checker``,
     numerically sanity-check PyTorch vs ONNX outputs, emit ``model.onnx`` plus a
     report JSON with versions / hashes / sanity diff.

  2. ``build_tensorrt_engine(onnx, output, precision=..., ...)`` — parse the ONNX
     into a TRT INetworkDefinition, build an engine with FP16 or INT8
     calibration (via ``IInt8EntropyCalibrator2`` over a directory of calibration
     images), emit ``model_<precision>.engine``.

torch/onnx/tensorrt are imported lazily so this module is importable without
them. Each function emits a clear ``ModuleNotFoundError`` with remediation when
a dependency is missing.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from packages.common.logging_utils import get_logger

LOG = get_logger(__name__)


# ---------------------------------------------------------------------- #
# return types
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class OnnxExportResult:
    onnx_path: Path
    report_path: Path
    input_shape: Tuple[int, int, int, int]   # (B, 3, H, W)
    opset: int
    onnx_checker_ok: bool
    max_abs_diff: float
    mean_abs_diff: float
    torch_version: str
    onnx_version: str
    file_size_bytes: int
    sha256: str


@dataclass(frozen=True)
class TensorRTBuildResult:
    engine_path: Path
    report_path: Path
    precision: str                # "fp16" | "int8"
    workspace_mb: int
    input_shape: Tuple[int, int, int, int]
    tensorrt_version: str
    cuda_version: Optional[str]
    file_size_bytes: int
    sha256: str
    calibration_image_count: int = 0
    calibration_cache_used: Optional[Path] = None


# ---------------------------------------------------------------------- #
# ONNX export
# ---------------------------------------------------------------------- #

def export_to_onnx(
    *,
    checkpoint: str | Path,
    exp_file: str | Path | None = None,
    exp_module: str | None = None,
    output: str | Path,
    num_classes: int = 3,
    input_size: Tuple[int, int] = (640, 640),
    opset: int = 11,
    dynamic_axes: bool = False,
    device: str | None = None,
    sanity_atol: float = 1e-3,
) -> OnnxExportResult:
    """Trace a yolox model into ONNX and validate.

    See module docstring for the workflow. Validation steps:
    - ``onnx.checker.check_model`` (raises on schema violations);
    - run the ONNX model in ``onnxruntime`` on a random input and compare to
      PyTorch output with ``np.allclose(atol=sanity_atol)``;
    - report file size + sha256.
    """
    try:
        import torch
    except ImportError as exc:
        raise ModuleNotFoundError(
            "torch is required for ONNX export. "
            "Install: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        ) from exc
    try:
        import onnx
    except ImportError as exc:
        raise ModuleNotFoundError(
            "onnx is required for export. Install: pip install onnx"
        ) from exc
    try:
        import onnxruntime  # noqa: F401  — only needed for sanity check
        have_ort = True
    except ImportError:
        LOG.warning("onnxruntime not installed; ONNX numerical sanity check will be skipped")
        have_ort = False

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if input_size[0] != input_size[1]:
        raise ValueError(f"square input_size required, got {input_size}")

    model, dev = _load_yolox_model(
        checkpoint=checkpoint, exp_file=exp_file, exp_module=exp_module,
        num_classes=num_classes, input_size=input_size, device=device,
    )
    model.eval()

    dummy = torch.zeros(1, 3, input_size[0], input_size[1], device=dev, dtype=torch.float32)
    LOG.info("Tracing model: input=%s, opset=%d, dynamic_axes=%s",
             tuple(dummy.shape), opset, dynamic_axes)

    onnx_kwargs: dict = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output"],
    )
    if dynamic_axes:
        onnx_kwargs["dynamic_axes"] = {
            "images": {0: "batch"},
            "output": {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(model, dummy, str(output), **onnx_kwargs)

    # Schema validation
    onnx_model = onnx.load(str(output))
    checker_ok = True
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as exc:
        LOG.error("onnx.checker rejected the model: %s", exc)
        checker_ok = False

    # Numerical sanity
    max_diff, mean_diff = float("nan"), float("nan")
    if have_ort:
        import numpy as np
        import onnxruntime as ort
        with torch.no_grad():
            ref = model(dummy).detach().cpu().numpy()
        providers = ["CPUExecutionProvider"]
        sess = ort.InferenceSession(str(output), providers=providers)
        ort_out = sess.run(None, {sess.get_inputs()[0].name: dummy.cpu().numpy()})
        # YOLOX exports a single output tensor; some forks split into multiple.
        # Compare element-wise on the first output that matches the PyTorch shape.
        matched = None
        for o in ort_out:
            if o.shape == ref.shape:
                matched = o
                break
        if matched is None:
            LOG.warning("Could not find matching ONNX output shape; sanity check skipped")
        else:
            diff = np.abs(ref - matched)
            max_diff = float(diff.max())
            mean_diff = float(diff.mean())
            if not np.allclose(ref, matched, atol=sanity_atol):
                LOG.warning("PyTorch vs ONNX output diff above atol=%.1e: max=%.4e mean=%.4e",
                            sanity_atol, max_diff, mean_diff)
            else:
                LOG.info("ONNX sanity OK: max_abs_diff=%.2e, mean_abs_diff=%.2e",
                         max_diff, mean_diff)

    # Hashing + report
    sha256 = _sha256_of(output)
    size_bytes = output.stat().st_size
    result = OnnxExportResult(
        onnx_path=output,
        report_path=output.with_suffix(".export_report.json"),
        input_shape=tuple(dummy.shape),
        opset=opset,
        onnx_checker_ok=checker_ok,
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        torch_version=torch.__version__,
        onnx_version=onnx.__version__,
        file_size_bytes=size_bytes,
        sha256=sha256,
    )
    _write_json(result.report_path, _onnx_report_dict(result, checkpoint))
    LOG.info("Wrote ONNX model: %s (%.2f MB, sha256=%s…)",
             output, size_bytes / 1e6, sha256[:12])
    return result


# ---------------------------------------------------------------------- #
# TensorRT engine build
# ---------------------------------------------------------------------- #

def build_tensorrt_engine(
    *,
    onnx_path: str | Path,
    output: str | Path,
    precision: str = "fp16",     # "fp16" | "int8"
    calib_images_dir: str | Path | None = None,
    calib_cache: str | Path | None = None,
    workspace_mb: int = 4096,
    input_size: Tuple[int, int] = (640, 640),
) -> TensorRTBuildResult:
    """Parse ONNX → build TensorRT engine.

    For ``precision="int8"``, requires ``calib_images_dir`` with representative
    images (the calibrator does YOLOX letterbox preprocessing). The calibration
    cache is reused if ``calib_cache`` already exists.
    """
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ModuleNotFoundError(
            "tensorrt is required to build engines. "
            "Install: see https://developer.nvidia.com/tensorrt or use yolox_trainer Docker image."
        ) from exc

    precision = precision.lower()
    if precision not in ("fp16", "int8"):
        raise ValueError(f"precision must be 'fp16' or 'int8', got {precision!r}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    LOG.info("Building TensorRT engine: %s → %s (precision=%s, workspace=%d MB)",
             onnx_path, output, precision, workspace_mb)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            errs = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parsing failed:\n" + "\n".join(str(e) for e in errs))

    config = builder.create_builder_config()
    # workspace memory pool (TRT 8.5+ API; falls back to max_workspace_size on older)
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))
    except AttributeError:
        config.max_workspace_size = workspace_mb * (1 << 20)

    calibration_image_count = 0
    calibration_cache_used: Optional[Path] = None

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            LOG.warning("Platform does NOT report fast FP16; engine will still build but may be slow")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            LOG.warning("Platform does NOT report fast INT8; engine will still build but may be slow")
        config.set_flag(trt.BuilderFlag.INT8)
        if not calib_images_dir and not (calib_cache and Path(calib_cache).is_file()):
            raise ValueError(
                "INT8 requires either --calib-images (dir with sample images) or an "
                "existing --calib-cache file."
            )
        calib_cache_path = Path(calib_cache) if calib_cache else output.with_suffix(".int8.cache")
        calibrator, calibration_image_count = _build_int8_calibrator(
            calib_images_dir=calib_images_dir,
            cache_file=calib_cache_path,
            input_size=input_size,
        )
        config.int8_calibrator = calibrator
        calibration_cache_used = calib_cache_path

    # Build engine
    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize() if engine else None
    if engine_bytes is None:
        raise RuntimeError("TensorRT failed to build engine (returned None)")

    output.write_bytes(bytes(engine_bytes))
    sha256 = _sha256_of(output)
    size_bytes = output.stat().st_size

    cuda_ver: Optional[str] = None
    try:
        cuda_ver = trt.__cuda_version__  # type: ignore[attr-defined]
    except AttributeError:
        pass

    result = TensorRTBuildResult(
        engine_path=output,
        report_path=output.with_suffix(".build_report.json"),
        precision=precision,
        workspace_mb=workspace_mb,
        input_shape=(1, 3, input_size[0], input_size[1]),
        tensorrt_version=str(trt.__version__),
        cuda_version=cuda_ver,
        file_size_bytes=size_bytes,
        sha256=sha256,
        calibration_image_count=calibration_image_count,
        calibration_cache_used=calibration_cache_used,
    )
    _write_json(result.report_path, _trt_report_dict(result, onnx_path))
    LOG.info("Wrote TRT engine: %s (%.2f MB, sha256=%s…)",
             output, size_bytes / 1e6, sha256[:12])
    return result


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #

def _load_yolox_model(
    *,
    checkpoint: str | Path,
    exp_file: str | Path | None,
    exp_module: str | None,
    num_classes: int,
    input_size: Tuple[int, int],
    device: str | None,
) -> Tuple[Any, Any]:
    """Reused by export. Same logic as PyTorchYOLOXPredictor.__init__."""
    import torch
    try:
        from yolox.exp import get_exp
    except ImportError as exc:
        raise ModuleNotFoundError(
            "yolox is required to load the model. Install via "
            "`pip install -v -e /workspace/YOLOX`."
        ) from exc

    if exp_module:
        from importlib import import_module
        mod = import_module(exp_module)
        if not hasattr(mod, "Exp"):
            raise AttributeError(f"Module {exp_module!r} does not define class Exp")
        exp = mod.Exp()
    elif exp_file:
        exp = get_exp(str(exp_file), None)
    else:
        raise ValueError("export requires either exp_file or exp_module")

    exp.num_classes = num_classes
    exp.test_size = (input_size[0], input_size[1])

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = exp.get_model().to(dev).eval()
    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    # weights_only=False: see inference.py PyTorchYOLOXPredictor.__init__ note —
    # YOLOX checkpoints embed numpy scalars in metadata; we trust our own ckpts.
    state = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    weights = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(weights)
    return model, dev


def _build_int8_calibrator(
    *,
    calib_images_dir: str | Path | None,
    cache_file: Path,
    input_size: Tuple[int, int],
):
    """Build an ``IInt8EntropyCalibrator2`` that walks images on disk."""
    import numpy as np
    import tensorrt as trt

    from packages.common.image_io import imread
    from packages.common.letterbox import apply_letterbox

    image_paths: List[Path] = []
    if calib_images_dir:
        d = Path(calib_images_dir)
        if not d.is_dir():
            raise FileNotFoundError(f"calib_images_dir not found: {d}")
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            image_paths.extend(sorted(d.glob(ext)))
        if not image_paths and not cache_file.is_file():
            raise ValueError(f"calib_images_dir is empty and no cache: {d}")

    class _EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self.paths = list(image_paths)
            self.pos = 0
            self.cache = cache_file
            self.input_size = input_size
            # device alloc for one input
            import pycuda.autoinit  # noqa: F401  — initialises CUDA context
            import pycuda.driver as cuda
            self._cuda = cuda
            self.bytes_per_input = 1 * 3 * input_size[0] * input_size[1] * 4  # float32
            self.device_alloc = cuda.mem_alloc(self.bytes_per_input)

        def get_batch_size(self) -> int:  # type: ignore[override]
            return 1

        def get_batch(self, names):  # type: ignore[override]
            if self.pos >= len(self.paths):
                return None
            img = imread(self.paths[self.pos])
            arr, _ = apply_letterbox(img, self.input_size[0], pad_value=114)
            chw = arr.transpose(2, 0, 1).astype(np.float32)
            self._cuda.memcpy_htod(self.device_alloc, chw)
            self.pos += 1
            return [int(self.device_alloc)]

        def read_calibration_cache(self):  # type: ignore[override]
            if self.cache.is_file():
                return self.cache.read_bytes()
            return None

        def write_calibration_cache(self, cache):  # type: ignore[override]
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            self.cache.write_bytes(bytes(cache))

    return _EntropyCalibrator(), len(image_paths)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _onnx_report_dict(r: OnnxExportResult, checkpoint: str | Path) -> dict:
    return {
        "kind": "onnx_export_report",
        "created_at": _utc_now_iso(),
        "source_checkpoint": str(checkpoint),
        "onnx_path": str(r.onnx_path),
        "input_shape": list(r.input_shape),
        "opset": r.opset,
        "onnx_checker_ok": r.onnx_checker_ok,
        "sanity": {
            "max_abs_diff_pytorch_vs_onnx":  r.max_abs_diff,
            "mean_abs_diff_pytorch_vs_onnx": r.mean_abs_diff,
        },
        "versions": {"torch": r.torch_version, "onnx": r.onnx_version},
        "file_size_bytes": r.file_size_bytes,
        "sha256": r.sha256,
    }


def _trt_report_dict(r: TensorRTBuildResult, onnx_path: Path) -> dict:
    return {
        "kind": "tensorrt_build_report",
        "created_at": _utc_now_iso(),
        "source_onnx": str(onnx_path),
        "engine_path": str(r.engine_path),
        "precision": r.precision,
        "workspace_mb": r.workspace_mb,
        "input_shape": list(r.input_shape),
        "versions": {"tensorrt": r.tensorrt_version, "cuda": r.cuda_version},
        "file_size_bytes": r.file_size_bytes,
        "sha256": r.sha256,
        "calibration": {
            "image_count": r.calibration_image_count,
            "cache_used":  str(r.calibration_cache_used) if r.calibration_cache_used else None,
        },
    }


def _utc_now_iso() -> str:
    return (_dt.datetime.now(_dt.timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"))
