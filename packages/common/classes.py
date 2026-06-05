"""Canonical class contract for the DCS → YOLOX detector.

Single source of truth for the **3** detection classes. Must stay in sync with
``configs/classes.yaml`` (enforced by ``tests/unit/test_classes_contract.py``).

Class mapping vs. the legacy 4-class layout — ``buoys`` is removed because it had
zero annotations in the captured data:

    legacy id  legacy name    canonical id  canonical name
    0          ships      ->  0             ships
    1          buoys      ->  —             (dropped)
    2          helicopters->  1             helicopters
    3          airplanes  ->  2             airplanes
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional

CANONICAL_CATEGORIES: tuple[dict[str, object], ...] = (
    {"id": 0, "name": "ships"},
    {"id": 1, "name": "helicopters"},
    {"id": 2, "name": "airplanes"},
)

NUM_CLASSES: int = len(CANONICAL_CATEGORIES)  # 3
CLASS_NAMES: tuple[str, ...] = tuple(str(c["name"]) for c in CANONICAL_CATEGORIES)

# legacy category_id -> canonical category_id (None = drop the category/annotation).
LEGACY_CATEGORY_ID_REMAP: Mapping[int, Optional[int]] = {
    0: 0,     # ships       -> ships
    1: None,  # buoys       -> dropped (0 annotations in the data)
    2: 1,     # helicopters -> helicopters
    3: 2,     # airplanes   -> airplanes
}

_CANON_BY_ID: dict[int, str] = {int(c["id"]): str(c["name"]) for c in CANONICAL_CATEGORIES}
_ALLOWED_IDS = frozenset(_CANON_BY_ID)


class ClassContractError(ValueError):
    """Raised when a dataset declares/uses categories outside the canonical 3."""


def assert_categories_allowed(categories: Iterable[Mapping[str, object]]) -> None:
    """Fail if a COCO ``categories`` list does not match the canonical 3 (id↔name)."""
    bad = [
        dict(c) for c in categories
        if int(c.get("id", -1)) not in _CANON_BY_ID
        or _CANON_BY_ID[int(c["id"])] != str(c.get("name", ""))
    ]
    if bad:
        raise ClassContractError(
            "categories outside the canonical 3 classes "
            f"{[{'id': i, 'name': n} for i, n in _CANON_BY_ID.items()]}: {bad}. "
            "'buoys' (and any other class) is not allowed — see packages/common/classes.py."
        )


def assert_category_ids_allowed(category_ids: Iterable[int]) -> None:
    """Fail if any annotation references a category id outside the canonical 3."""
    bad = sorted({int(cid) for cid in category_ids} - _ALLOWED_IDS)
    if bad:
        raise ClassContractError(
            f"annotation category ids {bad} are outside the canonical set "
            f"{sorted(_ALLOWED_IDS)} ({', '.join(CLASS_NAMES)})."
        )


def remap_legacy_category_id(legacy_id: int) -> Optional[int]:
    """Map a legacy (4-class) category id to its canonical id, or None to drop."""
    return LEGACY_CATEGORY_ID_REMAP.get(int(legacy_id), None)
