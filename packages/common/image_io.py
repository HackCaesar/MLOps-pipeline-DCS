"""Image read/write + light metadata helpers.

Backend resolution at import time:
- prefer ``cv2`` if available (faster for big images);
- fall back to ``PIL`` (Pillow) otherwise.

If neither is installed, top-level ``imread/imwrite`` raise ``ImportError`` with
a clear message. ``get_image_size`` works via Pillow without loading pixels.

Returned arrays are numpy ``uint8`` HxWxC in **RGB** channel order (matches what
YOLOX letterbox + most evaluation visualizations expect).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


_BACKEND: Optional[str] = None


def _resolve_backend() -> str:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        import cv2  # noqa: F401
        _BACKEND = "cv2"
        return _BACKEND
    except ImportError:
        pass
    try:
        import PIL  # noqa: F401
        _BACKEND = "pil"
        return _BACKEND
    except ImportError:
        pass
    _BACKEND = "none"
    return _BACKEND


def available_backend() -> str:
    """Return ``"cv2"``, ``"pil"`` or ``"none"``."""
    return _resolve_backend()


def imread(path: str | Path) -> "np.ndarray":
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    backend = _resolve_backend()
    if backend == "cv2":
        import cv2
        arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError(f"cv2 failed to decode image: {path}")
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if backend == "pil":
        import numpy as np
        from PIL import Image
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"))
    raise ImportError("No image backend available; install either opencv-python-headless or Pillow")


def imwrite(path: str | Path, image: "np.ndarray") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backend = _resolve_backend()
    if backend == "cv2":
        import cv2
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(path), bgr)
        if not ok:
            raise IOError(f"cv2 failed to write image: {path}")
        return path
    if backend == "pil":
        from PIL import Image
        Image.fromarray(image).save(path)
        return path
    raise ImportError("No image backend available; install either opencv-python-headless or Pillow")


def get_image_size(path: str | Path) -> tuple[int, int]:
    """Return ``(width, height)`` without loading pixel data.

    Uses Pillow's lazy load when available (it only reads the header for most
    formats). With cv2-only environments, falls back to a full ``imread``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        from PIL import Image
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except ImportError:
        pass
    arr = imread(path)
    h, w = arr.shape[:2]
    return int(w), int(h)
