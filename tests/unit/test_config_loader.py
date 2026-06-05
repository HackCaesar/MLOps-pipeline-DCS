"""Unit tests for packages.common.config: ${var} interpolation, errors, type preservation."""
from __future__ import annotations

from pathlib import Path

import pytest

from packages.common.config import (
    ConfigError,
    load_config,
    lookup_path,
    resolve_vars,
)


def test_resolves_single_substitution() -> None:
    out = resolve_vars({
        "storage": {"root": "/a", "datasets": "${storage.root}/datasets"},
    })
    assert out["storage"]["datasets"] == "/a/datasets"


def test_resolves_chained_substitution() -> None:
    out = resolve_vars({
        "a": "x",
        "b": "${a}/y",
        "c": "${b}/z",
    })
    assert out["c"] == "x/y/z"


def test_resolves_inside_lists_and_nested_dicts() -> None:
    out = resolve_vars({
        "root": "/r",
        "paths": ["${root}/a", {"deep": "${root}/b"}],
    })
    assert out["paths"][0] == "/r/a"
    assert out["paths"][1]["deep"] == "/r/b"


def test_whole_string_reference_preserves_type() -> None:
    out = resolve_vars({"port": 8765, "alias": "${port}"})
    assert out["alias"] == 8765
    assert isinstance(out["alias"], int)


def test_partial_substitution_yields_string() -> None:
    out = resolve_vars({"port": 8765, "url": "http://host:${port}/api"})
    assert out["url"] == "http://host:8765/api"


def test_circular_reference_raises() -> None:
    with pytest.raises(ConfigError, match="did not converge"):
        resolve_vars({"a": "${b}", "b": "${a}"})


def test_missing_key_raises() -> None:
    with pytest.raises(ConfigError, match="Missing key"):
        resolve_vars({"a": "${does.not.exist}"})


def test_lookup_path_nested() -> None:
    cfg = {"a": {"b": {"c": 42}}}
    assert lookup_path(cfg, "a.b.c") == 42


def test_lookup_path_missing_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Missing key"):
        lookup_path({"a": {"x": 1}}, "a.b")


def test_lookup_path_through_non_mapping_raises() -> None:
    with pytest.raises(KeyError, match="not a mapping"):
        lookup_path({"a": 1}, "a.b")


def test_load_shipped_pipeline_yaml(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_config(repo_root / "configs" / "pipeline.yaml")
    # Locked decisions
    assert cfg["training"]["multiscale_range"] == 0
    assert cfg["training"]["input_size"] == [640, 640]
    assert cfg["training"]["test_size"]  == [640, 640]
    assert cfg["evaluation"]["full_image_first"] is True
    assert cfg["enrichment"]["visible_area_threshold"] == 0.80
    # ${var} got resolved
    assert cfg["storage"]["datasets_dir"] == "/workspace/storage/datasets"
    assert cfg["metadata_store"]["sqlite_path"] == "/workspace/storage/metadata/pipeline.db"


def test_load_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("/no/such/file.yaml")
