"""Unit tests for packages.common.manifest.ManifestWriter / read_manifest."""
from __future__ import annotations

from pathlib import Path

import pytest

from packages.common.manifest import ManifestWriter, count_records, read_manifest


def test_writer_creates_parent_and_writes_records(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "manifest.jsonl"
    with ManifestWriter(path) as mw:
        mw.write({"a": 1})
        mw.write({"b": [1, 2, 3]})
        assert mw.count == 2
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].startswith('{"a"')


def test_append_mode_accumulates(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    with ManifestWriter(path) as mw:
        mw.write({"v": 1})
    with ManifestWriter(path) as mw:
        mw.write({"v": 2})
    records = list(read_manifest(path))
    assert [r["v"] for r in records] == [1, 2]


def test_read_manifest_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    path.write_text('{"a":1}\n\n   \n{"b":2}\n')
    records = list(read_manifest(path))
    assert records == [{"a": 1}, {"b": 2}]


def test_read_manifest_raises_on_malformed(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    path.write_text('{"a":1}\n{this is not json}\n')
    with pytest.raises(ValueError, match="malformed JSON"):
        list(read_manifest(path))


def test_read_manifest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(read_manifest(tmp_path / "no.jsonl"))


def test_count_records(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    assert count_records(p) == 0
    with ManifestWriter(p) as mw:
        for i in range(5):
            mw.write({"i": i})
    assert count_records(p) == 5


def test_write_outside_with_block_raises(tmp_path: Path) -> None:
    mw = ManifestWriter(tmp_path / "m.jsonl")
    with pytest.raises(RuntimeError, match="must be opened first"):
        mw.write({"x": 1})


def test_write_many(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    with ManifestWriter(p) as mw:
        n = mw.write_many([{"i": 0}, {"i": 1}, {"i": 2}])
    assert n == 3 and count_records(p) == 3
