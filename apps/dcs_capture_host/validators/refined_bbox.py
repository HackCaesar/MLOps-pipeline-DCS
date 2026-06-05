"""``RefinedBBox`` dataclass shared by all refiner modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RefinedBBox:
    bbox_xyxy_px: List[float]
    mask_area_px: int
    method: str
    confidence: str
    reasons: List[str]
    diagnostics: Optional[Dict[str, Any]] = None
