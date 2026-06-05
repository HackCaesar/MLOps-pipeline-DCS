"""Predictor abstraction + Mock + PyTorch (Phase 8) + ONNX/TRT stubs (Phase 9).

A Predictor takes a list of ``CropInput`` (preprocessed crops or full source
images carrying letterbox-flag) and returns per-input detections in **input
coords** (i.e. the coordinate system of whatever pixel array was actually fed
to the model). The orchestrator (``evaluate.py``) inverse-maps back to source.

Predictors expose ``needs_pixels`` so the orchestrator knows whether to
materialise pixel arrays (real model = yes, mock = no).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from packages.common.logging_utils import get_logger

if TYPE_CHECKING:
    import torch


LOG = get_logger(__name__)


@dataclass(frozen=True)
class CropDetection:
    """One detection returned by a Predictor, in **input-array coordinates**.

    For tile/crop=false inputs the array IS the tile_size×tile_size patch, so
    the bbox bounds are 0..tile_size. For letterbox=True inputs (full-source),
    the bbox bounds are 0..tile_size in letterbox coords — the orchestrator
    inverts via packages.common.letterbox.letterbox_box_to_source.
    """
    class_id: int
    confidence: float
    bbox_crop: Tuple[float, float, float, float]


@dataclass(frozen=True)
class CropInput:
    """One input to a Predictor.

    ``image`` is HxWxC uint8 in RGB (matches our image_io convention). When
    ``extra.letterbox=True`` the image is the **full source** and the predictor
    is expected to letterbox it to ``tile_size``. Otherwise the image must
    already match ``tile_size × tile_size`` (with any pad applied by caller).
    """
    image_id: int
    scale_name: str
    scale_size: Tuple[int, int]
    crop_offset: Tuple[int, int]
    crop_size: Tuple[int, int]
    tile_size: int
    image: Optional["np.ndarray"] = None
    extra: dict = field(default_factory=dict)


class Predictor(Protocol):
    """Abstract predictor — backend-agnostic."""

    needs_pixels: bool

    def predict_batch(self, inputs: Sequence[CropInput]) -> List[List[CropDetection]]:
        ...

    def close(self) -> None:
        ...


# ---------------------------------------------------------------------- #
# MockPredictor
# ---------------------------------------------------------------------- #

class MockPredictor:
    """Looks up detections by ``(image_id, scale_name, crop_offset)``.

    Used to exercise the eval pipeline without a real model. Accepts the
    ``predictions_tiles.json`` format from ``yolox_eval/mock_generator.py``
    (and an extended variant with ``scale_name``). Does NOT need pixel arrays.
    """

    needs_pixels: bool = False

    def __init__(self, predictions_path: str | Path) -> None:
        self.predictions_path = Path(predictions_path)
        with self.predictions_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "images" not in data:
            raise ValueError(
                f"Predictions file must have an 'images' field: {self.predictions_path}")
        self._lookup: Dict[Tuple[Any, ...], List[CropDetection]] = {}
        self._raw_payload = data
        for img_entry in data["images"]:
            image_id = img_entry["image_id"]
            for tile in img_entry.get("tiles", []):
                key = (image_id, tile.get("scale_name"),
                       tuple(tile.get("crop_offset", (0, 0))))
                dets = [
                    CropDetection(class_id=int(d["class_id"]),
                                  confidence=float(d["confidence"]),
                                  bbox_crop=tuple(d["bbox_crop"]))
                    for d in tile.get("detections", [])
                ]
                self._lookup[key] = dets

    def predict_batch(self, inputs: Sequence[CropInput]) -> List[List[CropDetection]]:
        out: List[List[CropDetection]] = []
        misses = 0
        for inp in inputs:
            key = (inp.image_id, inp.scale_name, tuple(inp.crop_offset))
            dets = self._lookup.get(key)
            if dets is None:
                misses += 1
                dets = self._lookup.get((inp.image_id, None, tuple(inp.crop_offset)), [])
            out.append(list(dets))
        if misses:
            LOG.debug("MockPredictor: %d/%d crops had no pre-baked predictions",
                      misses, len(inputs))
        return out

    def close(self) -> None:
        pass

    @property
    def raw_payload(self) -> dict:
        return self._raw_payload


# ---------------------------------------------------------------------- #
# PyTorchYOLOXPredictor — Phase 8 implementation
# ---------------------------------------------------------------------- #

class PyTorchYOLOXPredictor:
    """YOLOX PyTorch predictor.

    Construction loads the yolox Exp + model + weights. Inference runs in
    batches of ``CropInput``; ``extra.letterbox=True`` triggers source→tile_size
    letterbox; otherwise the image is assumed to already be ``tile_size``.

    Bbox decoding produces detections in **input-array coords** (tile_size
    space) — the orchestrator inverse-maps to source coords.

    Yolox + torch are imported lazily so importing this module never fails.
    """

    needs_pixels: bool = True
    backend_name = "pytorch_yolox"

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        exp_file: str | Path | None = None,
        exp_module: str | None = None,
        num_classes: int = 3,
        input_size: Tuple[int, int] = (640, 640),
        device: Optional[str] = None,
        conf_threshold: float = 0.01,
        nms_threshold: float = 0.65,
        fp16: bool = False,
        batch_size: int = 8,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ModuleNotFoundError(
                "PyTorch is required for PyTorchYOLOXPredictor. "
                "Install via `pip install torch` (or use the yolox_trainer Docker container)."
            ) from exc

        try:
            from yolox.exp import get_exp
        except ImportError as exc:
            raise ModuleNotFoundError(
                "yolox is required for PyTorchYOLOXPredictor. "
                "Install via `pip install -v -e /workspace/YOLOX` or run inside "
                "the yolox_trainer Docker container."
            ) from exc

        self._torch = torch
        self.input_size: Tuple[int, int] = (int(input_size[0]), int(input_size[1]))
        if self.input_size[0] != self.input_size[1]:
            raise ValueError(
                f"PyTorchYOLOXPredictor expects a square input size, got {self.input_size}")
        self.num_classes = int(num_classes)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.batch_size = int(batch_size)
        self.checkpoint = str(checkpoint)

        # Resolve Exp
        if exp_module:
            from importlib import import_module
            mod = import_module(exp_module)
            if not hasattr(mod, "Exp"):
                raise AttributeError(f"Module {exp_module!r} does not define class Exp")
            exp = mod.Exp()
        elif exp_file:
            exp = get_exp(str(exp_file), None)
        else:
            raise ValueError("PyTorchYOLOXPredictor requires either exp_file or exp_module")
        exp.num_classes = self.num_classes
        exp.test_size = self.input_size
        self._exp = exp

        # Device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = bool(fp16) and self.device.type == "cuda"

        # Build + load
        model = exp.get_model().to(self.device).eval()
        ckpt_path = Path(self.checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        # weights_only=False: YOLOX checkpoints carry numpy scalars in metadata
        # (epoch counter, etc.) which PyTorch 2.6+'s default weights_only=True rejects.
        # We only ever load our own training output here, so unpickling is safe.
        state = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        weights = state.get("model", state) if isinstance(state, dict) else state
        model.load_state_dict(weights)
        if self.fp16:
            model = model.half()
        self.model = model
        LOG.info("PyTorchYOLOXPredictor ready: device=%s fp16=%s input=%s ckpt=%s",
                 self.device, self.fp16, self.input_size, ckpt_path)

    # ------------------------------------------------------------------ #

    def predict_batch(self, inputs: Sequence[CropInput]) -> List[List[CropDetection]]:
        if not inputs:
            return []
        from yolox.utils import postprocess

        from packages.common.letterbox import apply_letterbox

        results: List[List[CropDetection]] = [[] for _ in inputs]
        tile_size = self.input_size[0]

        # Preprocess: letterbox where needed, build CHW float tensors.
        # We process in chunks of `batch_size` to bound peak GPU memory.
        for chunk_start in range(0, len(inputs), self.batch_size):
            chunk = list(inputs[chunk_start:chunk_start + self.batch_size])
            tensors: List["torch.Tensor"] = []
            for inp in chunk:
                if inp.image is None:
                    raise ValueError(
                        f"PyTorchYOLOXPredictor needs CropInput.image; got None for "
                        f"image_id={inp.image_id} scale={inp.scale_name}")
                arr = inp.image
                if inp.extra.get("letterbox"):
                    arr, _ = apply_letterbox(arr, tile_size, pad_value=114)
                if arr.shape[:2] != (tile_size, tile_size):
                    raise ValueError(
                        f"Crop pixel array must be {tile_size}×{tile_size}; "
                        f"got {arr.shape[:2]} for image_id={inp.image_id} "
                        f"scale={inp.scale_name} (caller did not pad/extract correctly)")
                # RGB uint8 → CHW float32 (YOLOX standard: 0-255 range, no /255).
                tensor = self._torch.from_numpy(arr).permute(2, 0, 1).float()
                tensors.append(tensor)

            batch = self._torch.stack(tensors).to(self.device)
            if self.fp16:
                batch = batch.half()

            with self._torch.no_grad():
                outputs = self.model(batch)
                outputs = postprocess(
                    outputs,
                    num_classes=self.num_classes,
                    conf_thre=self.conf_threshold,
                    nms_thre=self.nms_threshold,
                    class_agnostic=False,
                )

            for i, out in enumerate(outputs):
                if out is None:
                    continue
                out_np = out.detach().cpu().numpy()
                dets: List[CropDetection] = []
                for row in out_np:
                    # YOLOX postprocess returns: [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
                    x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    obj_conf = float(row[4])
                    cls_conf = float(row[5])
                    cls_id   = int(row[6])
                    dets.append(CropDetection(
                        class_id=cls_id,
                        confidence=obj_conf * cls_conf,
                        bbox_crop=(x1, y1, x2, y2),
                    ))
                results[chunk_start + i] = dets
        return results

    def close(self) -> None:
        # Free GPU memory.
        if hasattr(self, "model"):
            del self.model
        try:
            self._torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------- #
# ONNX predictor (Phase 9)
# ---------------------------------------------------------------------- #

class ONNXYOLOXPredictor:
    """YOLOX inference via onnxruntime.

    Same input/output contract as PyTorchYOLOXPredictor:
    - input arrays must be ``tile_size × tile_size`` (or ``extra.letterbox=True``);
    - returned ``CropDetection.bbox_crop`` is in input-array (letterbox) coords;
    - confidence = obj_conf × cls_conf (YOLOX convention).

    Postprocess (NMS + threshold) is run via ``yolox.utils.postprocess`` on a
    torch tensor wrapped around the ONNX output. This keeps the decoding logic
    identical to the PyTorch path.
    """

    needs_pixels: bool = True
    backend_name = "onnx_yolox"

    def __init__(
        self,
        *,
        onnx_path: str | Path,
        num_classes: int = 3,
        input_size: Tuple[int, int] = (640, 640),
        device: Optional[str] = None,
        conf_threshold: float = 0.01,
        nms_threshold: float = 0.65,
        batch_size: int = 8,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModuleNotFoundError(
                "onnxruntime is required for ONNXYOLOXPredictor. "
                "Install: `pip install onnxruntime-gpu` (CUDA) or `pip install onnxruntime` (CPU)."
            ) from exc

        if input_size[0] != input_size[1]:
            raise ValueError(f"square input_size required, got {input_size}")
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"ONNX file not found: {self.onnx_path}")
        self.input_size: Tuple[int, int] = (int(input_size[0]), int(input_size[1]))
        self.num_classes = int(num_classes)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.batch_size = int(batch_size)

        if providers is None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if (device or "").startswith("cuda") or device is None
                else ["CPUExecutionProvider"]
            )
        self._session = ort.InferenceSession(str(self.onnx_path), providers=list(providers))
        self._input_name = self._session.get_inputs()[0].name
        LOG.info("ONNXYOLOXPredictor ready: providers=%s input=%s onnx=%s",
                 self._session.get_providers(), self.input_size, self.onnx_path)

    def predict_batch(self, inputs: Sequence[CropInput]) -> List[List[CropDetection]]:
        if not inputs:
            return []
        import numpy as np
        import torch
        from yolox.utils import postprocess

        from packages.common.letterbox import apply_letterbox

        tile_size = self.input_size[0]
        results: List[List[CropDetection]] = [[] for _ in inputs]

        for chunk_start in range(0, len(inputs), self.batch_size):
            chunk = list(inputs[chunk_start:chunk_start + self.batch_size])
            tensors: List[np.ndarray] = []
            for inp in chunk:
                if inp.image is None:
                    raise ValueError(
                        f"ONNXYOLOXPredictor needs CropInput.image; got None for "
                        f"image_id={inp.image_id} scale={inp.scale_name}")
                arr = inp.image
                if inp.extra.get("letterbox"):
                    arr, _ = apply_letterbox(arr, tile_size, pad_value=114)
                if arr.shape[:2] != (tile_size, tile_size):
                    raise ValueError(
                        f"Crop pixel array must be {tile_size}×{tile_size}; "
                        f"got {arr.shape[:2]} for image_id={inp.image_id}")
                tensors.append(arr.transpose(2, 0, 1).astype(np.float32))
            batch = np.stack(tensors, axis=0)
            out_list = self._session.run(None, {self._input_name: batch})
            raw = out_list[0]  # primary output; matches PyTorch model.forward

            raw_t = torch.from_numpy(raw)
            decoded = postprocess(
                raw_t, num_classes=self.num_classes,
                conf_thre=self.conf_threshold, nms_thre=self.nms_threshold,
                class_agnostic=False,
            )
            for i, out in enumerate(decoded):
                if out is None:
                    continue
                arr_out = out.detach().cpu().numpy()
                dets: List[CropDetection] = []
                for row in arr_out:
                    x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    obj_conf = float(row[4]); cls_conf = float(row[5]); cls_id = int(row[6])
                    dets.append(CropDetection(class_id=cls_id,
                                              confidence=obj_conf * cls_conf,
                                              bbox_crop=(x1, y1, x2, y2)))
                results[chunk_start + i] = dets
        return results

    def close(self) -> None:
        # onnxruntime sessions release on GC; nothing else to do.
        pass


# ---------------------------------------------------------------------- #
# TensorRT predictor (Phase 9)
# ---------------------------------------------------------------------- #

class TensorRTYOLOXPredictor:
    """YOLOX inference via TensorRT runtime.

    Loads a serialized .engine file built by ``apps.evaluation_export.export.build_tensorrt_engine``.
    Uses a single CUDA stream. Bindings are allocated for the engine's declared
    input/output shapes (we re-allocate if batch size changes).
    """

    needs_pixels: bool = True
    backend_name = "tensorrt_yolox"

    def __init__(
        self,
        *,
        engine_path: str | Path,
        num_classes: int = 3,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.01,
        nms_threshold: float = 0.65,
        batch_size: int = 8,
    ) -> None:
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ModuleNotFoundError(
                "tensorrt is required for TensorRTYOLOXPredictor. "
                "See https://developer.nvidia.com/tensorrt."
            ) from exc
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda_drv  # noqa: F401
        except ImportError as exc:
            raise ModuleNotFoundError(
                "pycuda is required for TensorRTYOLOXPredictor. Install: `pip install pycuda`."
            ) from exc

        if input_size[0] != input_size[1]:
            raise ValueError(f"square input_size required, got {input_size}")
        self.engine_path = Path(engine_path)
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.num_classes = int(num_classes)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.batch_size = int(batch_size)

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with self.engine_path.open("rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {self.engine_path}")
        self._context = self._engine.create_execution_context()
        self._trt = trt
        LOG.info("TensorRTYOLOXPredictor ready: engine=%s input=%s",
                 self.engine_path, self.input_size)

    def predict_batch(self, inputs: Sequence[CropInput]) -> List[List[CropDetection]]:
        if not inputs:
            return []
        import numpy as np
        import pycuda.driver as cuda_drv
        import torch
        from yolox.utils import postprocess

        from packages.common.letterbox import apply_letterbox

        tile_size = self.input_size[0]
        results: List[List[CropDetection]] = [[] for _ in inputs]

        for chunk_start in range(0, len(inputs), self.batch_size):
            chunk = list(inputs[chunk_start:chunk_start + self.batch_size])
            tensors: List[np.ndarray] = []
            for inp in chunk:
                if inp.image is None:
                    raise ValueError(f"TRT needs CropInput.image; got None for "
                                     f"image_id={inp.image_id} scale={inp.scale_name}")
                arr = inp.image
                if inp.extra.get("letterbox"):
                    arr, _ = apply_letterbox(arr, tile_size, pad_value=114)
                if arr.shape[:2] != (tile_size, tile_size):
                    raise ValueError(
                        f"Crop must be {tile_size}×{tile_size}; got {arr.shape[:2]}")
                tensors.append(arr.transpose(2, 0, 1).astype(np.float32))
            batch = np.ascontiguousarray(np.stack(tensors, axis=0))
            batch_size = batch.shape[0]

            # Set dynamic batch (if engine was built with dynamic_axes).
            try:
                self._context.set_input_shape("images", batch.shape)
            except (AttributeError, TypeError):
                pass

            # Allocate device buffers
            d_in = cuda_drv.mem_alloc(batch.nbytes)
            # Output shape from engine
            output_name = self._engine.get_tensor_name(1)
            try:
                out_shape = tuple(self._context.get_tensor_shape(output_name))
            except AttributeError:
                out_shape = tuple(self._engine.get_binding_shape(1))
            if out_shape[0] != batch_size and out_shape[0] in (-1, 0):
                out_shape = (batch_size,) + out_shape[1:]
            out_size = int(np.prod(out_shape) * 4)  # float32
            d_out = cuda_drv.mem_alloc(out_size)

            cuda_drv.memcpy_htod(d_in, batch)
            # TRT v10 uses set_tensor_address + execute_async_v3;
            # earlier versions use execute_v2(bindings=[...]).
            try:
                input_name = self._engine.get_tensor_name(0)
                self._context.set_tensor_address(input_name, int(d_in))
                self._context.set_tensor_address(output_name, int(d_out))
                stream = cuda_drv.Stream()
                self._context.execute_async_v3(stream_handle=stream.handle)
                stream.synchronize()
            except AttributeError:
                bindings = [int(d_in), int(d_out)]
                self._context.execute_v2(bindings=bindings)

            host_out = np.empty(out_shape, dtype=np.float32)
            cuda_drv.memcpy_dtoh(host_out, d_out)

            raw_t = torch.from_numpy(host_out)
            decoded = postprocess(
                raw_t, num_classes=self.num_classes,
                conf_thre=self.conf_threshold, nms_thre=self.nms_threshold,
                class_agnostic=False,
            )
            for i, out in enumerate(decoded):
                if out is None:
                    continue
                arr_out = out.detach().cpu().numpy()
                dets: List[CropDetection] = []
                for row in arr_out:
                    x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    obj_conf = float(row[4]); cls_conf = float(row[5]); cls_id = int(row[6])
                    dets.append(CropDetection(class_id=cls_id,
                                              confidence=obj_conf * cls_conf,
                                              bbox_crop=(x1, y1, x2, y2)))
                results[chunk_start + i] = dets
        return results

    def close(self) -> None:
        # CUDA buffers freed when context/engine release.
        pass


# ---------------------------------------------------------------------- #
# factory
# ---------------------------------------------------------------------- #

def build_predictor(backend: str, **kwargs: Any) -> Predictor:
    """Pick a Predictor by name. ``backend ∈ {"mock", "pytorch", "onnx", "trt"}``."""
    backend = backend.lower()
    if backend == "mock":
        path = kwargs.get("predictions_path")
        if not path:
            raise ValueError("MockPredictor requires predictions_path=…")
        return MockPredictor(path)
    if backend == "pytorch":
        accepted = {"checkpoint", "exp_file", "exp_module", "num_classes",
                    "input_size", "device", "conf_threshold", "nms_threshold",
                    "fp16", "batch_size"}
        return PyTorchYOLOXPredictor(**{k: v for k, v in kwargs.items() if k in accepted})
    if backend == "onnx":
        accepted = {"onnx_path", "num_classes", "input_size", "device",
                    "conf_threshold", "nms_threshold", "batch_size", "providers"}
        return ONNXYOLOXPredictor(**{k: v for k, v in kwargs.items() if k in accepted})
    if backend == "trt":
        accepted = {"engine_path", "num_classes", "input_size",
                    "conf_threshold", "nms_threshold", "batch_size"}
        return TensorRTYOLOXPredictor(**{k: v for k, v in kwargs.items() if k in accepted})
    raise ValueError(f"Unknown predictor backend: {backend!r}")
