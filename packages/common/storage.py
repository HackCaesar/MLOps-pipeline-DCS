"""Storage abstraction over a content root.

LocalStorageBackend treats `root_dir` as the URI namespace. Every input is one of:

- a URI relative to root_dir, e.g. ``"datasets/raw/dcs_001"`` or
  ``"local://datasets/raw/dcs_001"``;
- a project-absolute Path that points inside root_dir;
- an absolute Path outside root_dir is allowed for ``put_file`` source / ``get_file`` destination
  (i.e. operations that bridge between an OS path and the storage).

The abstraction is sized for a future S3/MinIO backend with the same interface
but only Local is implemented in the MVP.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable, Protocol

_URI_SCHEME = "local://"
_HASH_CHUNK = 1024 * 1024


class StorageBackend(Protocol):
    """Interface used by every stage that reads/writes files."""

    def make_uri(self, *parts: str) -> str: ...
    def resolve_path(self, uri_or_path: str | os.PathLike[str]) -> Path: ...

    def exists(self, uri_or_path: str | os.PathLike[str]) -> bool: ...
    def list(self, uri_or_path: str | os.PathLike[str]) -> list[str]: ...
    def ensure_dir(self, uri_or_path: str | os.PathLike[str]) -> Path: ...
    def delete(self, uri_or_path: str | os.PathLike[str]) -> None: ...

    def put_file(self, local_path: Path, target: str | os.PathLike[str]) -> Path: ...
    def get_file(self, source: str | os.PathLike[str], local_path: Path) -> Path: ...
    def put_directory(self, local_dir: Path, target_dir: str | os.PathLike[str]) -> Path: ...
    def get_directory(self, source_dir: str | os.PathLike[str], local_dir: Path) -> Path: ...

    def copy_file(self, src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> Path: ...
    def copy_tree(self, src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> Path: ...

    def compute_file_hash(self, uri_or_path: str | os.PathLike[str]) -> str: ...
    def compute_directory_hash(self, uri_or_path: str | os.PathLike[str]) -> str: ...


class LocalStorageBackend:
    """StorageBackend implementation backed by the local filesystem.

    ``root_dir`` is created on construction if missing.
    """

    def __init__(self, root_dir: str | os.PathLike[str]) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    # ---- URI handling ---------------------------------------------------

    def make_uri(self, *parts: str) -> str:
        clean = [str(p).strip("/\\") for p in parts if p not in (None, "")]
        return "/".join(clean)

    def resolve_path(self, uri_or_path: str | os.PathLike[str]) -> Path:
        """Convert a URI (relative or ``local://...``) or Path to an absolute Path.

        Absolute paths are returned as-is. Bare-relative paths are joined to root_dir.
        """
        raw = os.fspath(uri_or_path)
        if raw.startswith(_URI_SCHEME):
            raw = raw[len(_URI_SCHEME):]
        p = Path(raw)
        if p.is_absolute() or (len(raw) > 1 and raw[1] == ":"):
            return Path(raw)
        return (self.root_dir / raw).resolve()

    # ---- existence/listing/deletion -------------------------------------

    def exists(self, uri_or_path: str | os.PathLike[str]) -> bool:
        return self.resolve_path(uri_or_path).exists()

    def list(self, uri_or_path: str | os.PathLike[str]) -> list[str]:
        path = self.resolve_path(uri_or_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        return sorted(child.name for child in path.iterdir())

    def ensure_dir(self, uri_or_path: str | os.PathLike[str]) -> Path:
        path = self.resolve_path(uri_or_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def delete(self, uri_or_path: str | os.PathLike[str]) -> None:
        path = self.resolve_path(uri_or_path)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()

    # ---- file/dir transfers ---------------------------------------------

    def put_file(self, local_path: Path, target: str | os.PathLike[str]) -> Path:
        src = Path(local_path)
        if not src.is_file():
            raise FileNotFoundError(f"Source file does not exist: {src}")
        dst = self.resolve_path(target)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst

    def get_file(self, source: str | os.PathLike[str], local_path: Path) -> Path:
        src = self.resolve_path(source)
        if not src.is_file():
            raise FileNotFoundError(f"Source file does not exist: {src}")
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst

    def put_directory(self, local_dir: Path, target_dir: str | os.PathLike[str]) -> Path:
        return self._copy_tree(Path(local_dir), self.resolve_path(target_dir))

    def get_directory(self, source_dir: str | os.PathLike[str], local_dir: Path) -> Path:
        return self._copy_tree(self.resolve_path(source_dir), Path(local_dir))

    def copy_file(self, src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> Path:
        s = self.resolve_path(src)
        d = self.resolve_path(dst)
        if not s.is_file():
            raise FileNotFoundError(f"Source file does not exist: {s}")
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        return d

    def copy_tree(self, src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> Path:
        return self._copy_tree(self.resolve_path(src), self.resolve_path(dst))

    @staticmethod
    def _copy_tree(src: Path, dst: Path) -> Path:
        if not src.is_dir():
            raise NotADirectoryError(f"Source is not a directory: {src}")
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return dst

    # ---- hashing --------------------------------------------------------

    def compute_file_hash(self, uri_or_path: str | os.PathLike[str]) -> str:
        path = self.resolve_path(uri_or_path)
        if not path.is_file():
            raise FileNotFoundError(f"Not a file: {path}")
        return _sha256_of_file(path)

    def compute_directory_hash(self, uri_or_path: str | os.PathLike[str]) -> str:
        """Deterministic content+path hash of a directory tree.

        Files are walked in sorted relative-path order. For each file:
        ``relpath\\0filehash\\n`` is appended to a running sha256.
        Order-independent across runs; sensitive to renames and content edits.
        """
        path = self.resolve_path(uri_or_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        agg = hashlib.sha256()
        for rel in _iter_files_sorted(path):
            file_hash = _sha256_of_file(path / rel)
            agg.update(rel.encode("utf-8"))
            agg.update(b"\x00")
            agg.update(file_hash.encode("ascii"))
            agg.update(b"\n")
        return agg.hexdigest()


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _iter_files_sorted(root: Path) -> Iterable[str]:
    files: list[str] = []
    for child in root.rglob("*"):
        if child.is_file():
            files.append(str(child.relative_to(root)).replace(os.sep, "/"))
    files.sort()
    return files
