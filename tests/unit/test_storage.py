"""Unit tests for packages.common.storage.LocalStorageBackend."""
from __future__ import annotations

from pathlib import Path

import pytest

from packages.common.storage import LocalStorageBackend


def _make_file(path: Path, content: bytes = b"hello") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_constructor_creates_root_dir(tmp_path: Path) -> None:
    root = tmp_path / "new_root"
    backend = LocalStorageBackend(root)
    assert root.is_dir()
    assert backend.root_dir == root.resolve()


def test_make_uri_joins_parts() -> None:
    backend = LocalStorageBackend("/tmp/whatever")
    assert backend.make_uri("a", "b", "c") == "a/b/c"
    assert backend.make_uri("/a/", "/b/") == "a/b"
    assert backend.make_uri() == ""


def test_resolve_path_relative(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    assert backend.resolve_path("a/b") == (tmp_path / "a" / "b").resolve()


def test_resolve_path_strips_uri_scheme(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    assert backend.resolve_path("local://a/b") == (tmp_path / "a" / "b").resolve()


def test_resolve_path_absolute_kept(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    assert backend.resolve_path("/etc/foo") == Path("/etc/foo")
    assert backend.resolve_path("D:/foo/bar") == Path("D:/foo/bar")


def test_ensure_dir_idempotent(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    a = backend.ensure_dir("a/b/c")
    b = backend.ensure_dir("a/b/c")
    assert a == b
    assert a.is_dir()


def test_put_file_copies(tmp_path: Path) -> None:
    src_root = tmp_path / "src"
    dst_root = tmp_path / "storage"
    src = _make_file(src_root / "x.bin", b"\x00\x01\x02")
    backend = LocalStorageBackend(dst_root)
    target = backend.put_file(src, "subdir/x.bin")
    assert target == (dst_root / "subdir" / "x.bin").resolve()
    assert target.read_bytes() == b"\x00\x01\x02"


def test_get_file_copies_out(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "storage")
    src_inside = backend.ensure_dir("data") / "y.txt"
    src_inside.write_text("ok")
    dst = tmp_path / "out" / "y.txt"
    backend.get_file("data/y.txt", dst)
    assert dst.read_text() == "ok"


def test_put_file_missing_source_raises(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    with pytest.raises(FileNotFoundError):
        backend.put_file(tmp_path / "no_such.bin", "x")


def test_list_returns_sorted_names(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    backend.ensure_dir("d/sub")
    (tmp_path / "d" / "a.txt").write_text("a")
    (tmp_path / "d" / "z.txt").write_text("z")
    assert backend.list("d") == ["a.txt", "sub", "z.txt"]


def test_list_raises_on_non_dir(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        backend.list("file.txt")


def test_exists(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    assert not backend.exists("a/b")
    backend.ensure_dir("a/b")
    assert backend.exists("a/b")


def test_delete_file_and_directory(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    (backend.ensure_dir("a") / "f.txt").write_text("x")
    backend.ensure_dir("b/c")
    backend.delete("a/f.txt")
    assert not (tmp_path / "a" / "f.txt").exists()
    backend.delete("b")
    assert not (tmp_path / "b").exists()
    # Deleting a missing path is a no-op (consistent with idempotent ensure_dir).
    backend.delete("missing")


def test_copy_file_and_tree(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    src = backend.ensure_dir("src")
    (src / "f.txt").write_text("data")
    (src / "nested").mkdir()
    (src / "nested" / "g.txt").write_text("more")

    backend.copy_file("src/f.txt", "dst/f.txt")
    assert (tmp_path / "dst" / "f.txt").read_text() == "data"

    backend.copy_tree("src", "tree_copy")
    assert (tmp_path / "tree_copy" / "f.txt").read_text() == "data"
    assert (tmp_path / "tree_copy" / "nested" / "g.txt").read_text() == "more"


def test_compute_file_hash_deterministic(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    f = backend.ensure_dir("d") / "x.bin"
    f.write_bytes(b"identical content")
    h1 = backend.compute_file_hash("d/x.bin")
    h2 = backend.compute_file_hash("d/x.bin")
    assert h1 == h2 and len(h1) == 64


def test_compute_file_hash_changes_with_content(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    p = tmp_path / "x.bin"
    p.write_bytes(b"A")
    h_a = backend.compute_file_hash("x.bin")
    p.write_bytes(b"B")
    h_b = backend.compute_file_hash("x.bin")
    assert h_a != h_b


def test_compute_directory_hash_stable_across_calls(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    d = backend.ensure_dir("tree")
    (d / "a.txt").write_text("alpha")
    (d / "b.txt").write_text("beta")
    (d / "sub").mkdir()
    (d / "sub" / "c.txt").write_text("gamma")
    h1 = backend.compute_directory_hash("tree")
    h2 = backend.compute_directory_hash("tree")
    assert h1 == h2 and len(h1) == 64


def test_compute_directory_hash_sensitive_to_rename(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    d = backend.ensure_dir("tree")
    (d / "a.txt").write_text("data")
    h1 = backend.compute_directory_hash("tree")
    (d / "a.txt").rename(d / "b.txt")
    h2 = backend.compute_directory_hash("tree")
    assert h1 != h2


def test_put_directory_roundtrip(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "storage")
    src = tmp_path / "src_tree"
    (src / "nested").mkdir(parents=True)
    (src / "a.txt").write_text("A")
    (src / "nested" / "b.txt").write_text("B")
    backend.put_directory(src, "stored_tree")
    assert (tmp_path / "storage" / "stored_tree" / "a.txt").read_text() == "A"
    assert (tmp_path / "storage" / "stored_tree" / "nested" / "b.txt").read_text() == "B"
