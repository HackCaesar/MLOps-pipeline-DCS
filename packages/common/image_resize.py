"""Image resize helpers, decoupled from the chosen backend.

``compute_resize_transform`` returns the geometric transform without touching
pixels — handy for bbox math when you don't want to materialise the resized
image. ``resize_image`` performs the actual pixel resize via the backend chosen
by ``image_io``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class ResizeTransform:
    """Result of ``compute_resize_transform``.

    Use ``scale_x``/``scale_y`` for bbox transforms:

        resized_box = (
            box[0] * scale_x, box[1] * scale_y,
            box[2] * scale_x, box[3] * scale_y,
        )
    """

    src_width: int
    src_height: int
    dst_width: int
    dst_height: int
    scale_x: float
    scale_y: float
    mode: str   # "exact" | "fit_aspect" | "letterbox"


def compute_resize_transform(
    src_w: int, src_h: int, target_w: int, target_h: int,
    mode: str = "exact",
) -> ResizeTransform:
    """Plan a resize and return scale factors.

    Modes:
    - ``exact``: stretch to exactly ``target_w × target_h``. ``scale_x/scale_y`` may differ.
    - ``fit_aspect``: shrink/grow to fit inside ``target_w × target_h`` while preserving
      the source aspect ratio. Output may be smaller than target in one dimension.
      ``scale_x == scale_y``.

    Letterbox (pad to exact size while preserving aspect) is implemented in ``letterbox.py``.
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"src dimensions must be positive, got {src_w}×{src_h}")
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"target dimensions must be positive, got {target_w}×{target_h}")

    if mode == "exact":
        return ResizeTransform(
            src_width=src_w, src_height=src_h,
            dst_width=target_w, dst_height=target_h,
            scale_x=target_w / src_w, scale_y=target_h / src_h,
            mode="exact",
        )

    if mode == "fit_aspect":
        scale = min(target_w / src_w, target_h / src_h)
        dst_w = max(1, int(round(src_w * scale)))
        dst_h = max(1, int(round(src_h * scale)))
        return ResizeTransform(
            src_width=src_w, src_height=src_h,
            dst_width=dst_w, dst_height=dst_h,
            scale_x=scale, scale_y=scale,
            mode="fit_aspect",
        )

    raise ValueError(f"Unknown resize mode: {mode!r}")


def resize_image(image: "np.ndarray", dst_w: int, dst_h: int) -> "np.ndarray":
    """Resize a HxWxC numpy image to ``(dst_h, dst_w)``. Backend chosen by image_io.

    Uses bilinear (cv2 INTER_LINEAR / PIL.BILINEAR) — appropriate for both upsample
    and downsample at the scales the pipeline cares about.
    """
    if dst_w <= 0 or dst_h <= 0:
        raise ValueError(f"dst dimensions must be positive, got {dst_w}×{dst_h}")

    from .image_io import _resolve_backend  # local import to keep numpy lazy
    backend = _resolve_backend()
    if backend == "cv2":
        import cv2
        return cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
    if backend == "pil":
        import numpy as np
        from PIL import Image
        pil = Image.fromarray(image)
        pil = pil.resize((dst_w, dst_h), resample=Image.BILINEAR)
        return np.asarray(pil)
    raise ImportError("No image backend available; install either opencv-python-headless or Pillow")
