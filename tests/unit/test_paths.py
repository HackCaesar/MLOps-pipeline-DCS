"""Unit tests for packages.common.paths."""
from __future__ import annotations

from pathlib import Path

from packages.common.paths import (
    is_windows_absolute,
    project_root,
    resolve_path,
)


def test_project_root_finds_pyproject() -> None:
    root = project_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "apps").is_dir()
    assert (root / "packages" / "common").is_dir()


def test_is_windows_absolute() -> None:
    assert is_windows_absolute("D:/foo/bar")
    assert is_windows_absolute("C:\\Users\\x")
    assert not is_windows_absolute("/etc/foo")
    assert not is_windows_absolute("relative/path")


def test_resolve_path_handles_none_and_null_string() -> None:
    assert resolve_path(None) is None
    assert resolve_path("") is None
    assert resolve_path("null") is None


def test_resolve_path_windows_absolute() -> None:
    assert resolve_path("D:/foo/bar") == Path("D:/foo/bar")


def test_resolve_path_posix_absolute() -> None:
    assert resolve_path("/etc/foo") == Path("/etc/foo")


def test_resolve_path_relative_resolves_against_base(tmp_path: Path) -> None:
    out = resolve_path("subdir/file", base=tmp_path)
    assert out == (tmp_path / "subdir" / "file").resolve()
