"""JSONL manifests (one JSON object per line) for enrichment + capture artifacts.

Used by:

- ``apps/dataset_enrichment``: ``manifest.jsonl`` (kept records) and
  ``dropped_tiles_manifest.jsonl``;
- ``apps/dcs_capture/adapter``: ``capture_manifest.jsonl``;
- evaluation: ``false_positives.jsonl`` / ``false_negatives.jsonl``.

Format:
- One JSON object per line, no trailing comma;
- UTF-8;
- writes are explicitly fsync-able via the context manager exit.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Iterator, Mapping


class ManifestWriter:
    """Append-only JSONL writer.

    Usage:
        with ManifestWriter(path) as mw:
            mw.write({"key": "value"})
            mw.write({"other": 42})

    Calling ``write`` outside a ``with``-block requires explicit ``open()``/``close()``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh: io.TextIOBase | None = None
        self._count = 0

    def open(self) -> "ManifestWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def write(self, record: Mapping[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("ManifestWriter must be opened first (use 'with' or .open())")
        json.dump(record, self._fh, ensure_ascii=False)
        self._fh.write("\n")
        self._count += 1

    def write_many(self, records: list[Mapping[str, Any]]) -> int:
        for r in records:
            self.write(r)
        return len(records)

    @property
    def count(self) -> int:
        return self._count

    def __enter__(self) -> "ManifestWriter":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_manifest(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield dicts from a JSONL file. Skips blank lines. Raises on malformed JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i}: malformed JSON ({exc})") from exc


def count_records(path: str | Path) -> int:
    """Count records in a JSONL manifest. Returns 0 for a missing file."""
    path = Path(path)
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n
