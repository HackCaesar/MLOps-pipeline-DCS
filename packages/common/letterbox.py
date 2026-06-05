"""YOLOX-style letterbox: resize-while-preserving-aspect + pad to exact size.

This is the **single source of truth** for full-image preprocessing during
evaluation (the "full source inference" step). The transform
parameters are returned so that predictions can be inverted back into source
image coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import numpy as np


DEFAULT_PAD_VALUE = 114  # YOLOX default; matches official letterbox.


@dataclass(frozen=True)
class LetterboxTransform:
    """Geometric transform of a letterbox preprocess.

    Letterbox places the resized image at offset ``(pad_x, pad_y)`` inside a
    canvas of size ``target_size × target_size``, with the remaining pixels
    filled with ``pad_value``.

    To map a source bbox into letterbox coords:

        x_lb = x * scale + pad_x
        y_lb = y * scale + pad_y

    To map a letterbox detection back into source coords:

        x_src = (x_lb - pad_x) / scale
    """

    src_width: int
    src_height: int
    target_size: int
    scale: float
    new_width: int
    new_height: int
    pad_x: int    # offset of resized image inside the target canvas (left)
    pad_y: int    # ... (top)
    pad_value: int = DEFAULT_PAD_VALUE


def compute_letterbox(
    src_w: int, src_h: int, target_size: int, pad_value: int = DEFAULT_PAD_VALUE,
) -> LetterboxTransform:
    """Plan letterbox geometry. No pixel access."""
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"src dimensions must be positive, got {src_w}×{src_h}")
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    scale = min(target_size / src_w, target_size / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    return LetterboxTransform(
        src_width=src_w, src_height=src_h, target_size=target_size,
        scale=scale, new_width=new_w, new_height=new_h,
        pad_x=pad_x, pad_y=pad_y, pad_value=pad_value,
    )


def apply_letterbox(image: "np.ndarray", target_size: int,
                    pad_value: int = DEFAULT_PAD_VALUE) -> tuple["np.ndarray", LetterboxTransform]:
    """Resize ``image`` (HxWxC) into a letterboxed canvas of ``target_size``.

    Returns ``(canvas, transform)``.
    """
    import numpy as np

    from .image_resize import resize_image

    h, w = image.shape[:2]
    t = compute_letterbox(w, h, target_size, pad_value=pad_value)
    resized = resize_image(image, t.new_width, t.new_height)
    canvas = np.full((target_size, target_size, image.shape[2]) if image.ndim == 3
                     else (target_size, target_size),
                     fill_value=pad_value, dtype=image.dtype)
    canvas[t.pad_y:t.pad_y + t.new_height, t.pad_x:t.pad_x + t.new_width] = resized
    return canvas, t


def letterbox_box_to_source(
    box_xyxy: Sequence[float], t: LetterboxTransform,
) -> tuple[float, float, float, float]:
    """Invert a bbox detected in letterbox coords back into source image coords."""
    x1, y1, x2, y2 = box_xyxy
    return (
        (float(x1) - t.pad_x) / t.scale,
        (float(y1) - t.pad_y) / t.scale,
        (float(x2) - t.pad_x) / t.scale,
        (float(y2) - t.pad_y) / t.scale,
    )


def source_box_to_letterbox(
    box_xyxy: Sequence[float], t: LetterboxTransform,
) -> tuple[float, float, float, float]:
    """Project a source-coords bbox into letterbox coords (useful for tests/viz)."""
    x1, y1, x2, y2 = box_xyxy
    return (
        float(x1) * t.scale + t.pad_x,
        float(y1) * t.scale + t.pad_y,
        float(x2) * t.scale + t.pad_x,
        float(y2) * t.scale + t.pad_y,
    )
