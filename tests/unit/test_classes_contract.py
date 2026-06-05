"""The canonical 3-class contract and its guards.

Keeps ``configs/classes.yaml`` and ``packages/common/classes.py`` in sync and
proves the validation guard rejects ``buoys`` / any foreign category.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from packages.common.classes import (
    CANONICAL_CATEGORIES,
    CLASS_NAMES,
    LEGACY_CATEGORY_ID_REMAP,
    NUM_CLASSES,
    ClassContractError,
    assert_categories_allowed,
    assert_category_ids_allowed,
    remap_legacy_category_id,
)

_CLASSES_YAML = Path(__file__).resolve().parents[2] / "configs" / "classes.yaml"


def test_three_classes() -> None:
    assert NUM_CLASSES == 3
    assert CLASS_NAMES == ("ships", "helicopters", "airplanes")
    assert [c["id"] for c in CANONICAL_CATEGORIES] == [0, 1, 2]


def test_classes_yaml_in_sync_with_code() -> None:
    data = yaml.safe_load(_CLASSES_YAML.read_text(encoding="utf-8"))
    cats = [{"id": c["id"], "name": c["name"]} for c in data["categories"]]
    assert cats == [dict(c) for c in CANONICAL_CATEGORIES]
    assert not any(c["name"] == "buoys" for c in cats)


def test_legacy_remap_mapping() -> None:
    # ships 0->0, buoys 1->drop, helicopters 2->1, airplanes 3->2
    assert LEGACY_CATEGORY_ID_REMAP == {0: 0, 1: None, 2: 1, 3: 2}
    assert remap_legacy_category_id(2) == 1
    assert remap_legacy_category_id(1) is None


def test_assert_categories_allowed_accepts_canonical() -> None:
    assert_categories_allowed(CANONICAL_CATEGORIES)  # no raise


@pytest.mark.parametrize("bad", [
    [{"id": 0, "name": "ships"}, {"id": 1, "name": "buoys"},
     {"id": 2, "name": "helicopters"}, {"id": 3, "name": "airplanes"}],   # legacy 4-class
    [{"id": 0, "name": "airplanes"}],                                      # id↔name mismatch
    [{"id": 5, "name": "submarines"}],                                     # foreign class
])
def test_assert_categories_allowed_rejects(bad: list) -> None:
    with pytest.raises(ClassContractError):
        assert_categories_allowed(bad)


def test_assert_category_ids_allowed() -> None:
    assert_category_ids_allowed([0, 1, 2, 0, 1])      # no raise
    with pytest.raises(ClassContractError):
        assert_category_ids_allowed([0, 1, 3])         # 3 = legacy airplanes id
