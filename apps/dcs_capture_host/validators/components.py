"""Connected-component utilities for visible-bbox refinement.

These wrap OpenCV's ``connectedComponentsWithStats`` with the project's
filtering conventions and produce the tuple shape the refiner consumes:

``(x, y, width, height, area, centroid)`` where ``centroid`` is a numpy
2-vector ``[cx, cy]`` (same as OpenCV returns).

OpenCV is a hard requirement here; if you call these without ``cv2``
installed you will get an ``ImportError`` at the call site.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None  # type: ignore[assignment]


def component_rows_cols(
    mask: np.ndarray,
    row_threshold: float,
    col_threshold: float,
) -> Optional[List[int]]:
    """Tight bbox around the rows/cols of ``mask`` whose pixel-sum exceeds the
    given thresholds. Returns ``[x_min, y_min, x_max, y_max]`` (half-open in
    both dims) or ``None`` if no row or column is above threshold."""
    rows = np.where(mask.sum(axis=1) > row_threshold)[0]
    cols = np.where(mask.sum(axis=0) > col_threshold)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return [int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1)]


def connected_components(
    mask: np.ndarray,
    min_area_px: int,
) -> List[Tuple[int, int, int, int, int, np.ndarray]]:
    """Run OpenCV 8-connectivity CC and return components with ``area >= min_area_px``.

    Background label ``0`` is skipped.
    """
    labels_count, _labels, stats, centroids = cv2.connectedComponentsWithStats(  # type: ignore[union-attr]
        mask.astype("uint8"), 8
    )
    components: List[Tuple[int, int, int, int, int, np.ndarray]] = []
    for label_index in range(1, labels_count):
        x, y, width, height, area = stats[label_index]
        if int(area) >= min_area_px:
            components.append((int(x), int(y), int(width), int(height), int(area), centroids[label_index]))
    return components
