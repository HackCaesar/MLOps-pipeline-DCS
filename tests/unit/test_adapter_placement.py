"""Image-placement safety: no silent hardlink->copy fallback, and ephemeral
staging cleanup in the adapter CLI."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from apps.dcs_capture import cli
from apps.dcs_capture.adapter import _hardlink_supported, _place_image, normalize_dataset
from tests.fixtures.mini_dcs_dataset import write_mini_dcs_dataset

# ---- _place_image -------------------------------------------------------

def test_hardlink_success_same_inode(tmp_path: Path) -> None:
    src = tmp_path / "a.png"; src.write_bytes(b"x")
    dst = tmp_path / "b.png"
    if not hasattr(os, "link"):
        pytest.skip("os.link unsupported")
    method = _place_image(src, dst, "hardlink")
    assert method == "hardlink"
    assert os.stat(src).st_ino == os.stat(dst).st_ino


def test_hardlink_failure_raises_by_default(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "a.png"; src.write_bytes(b"x")
    dst = tmp_path / "b.png"

    def _boom(s, d):
        raise OSError("Invalid cross-device link")
    monkeypatch.setattr(os, "link", _boom)

    with pytest.raises(RuntimeError, match="different filesystems"):
        _place_image(src, dst, "hardlink")
    assert not dst.exists()  # no silent duplicate


def test_hardlink_failure_copies_when_allowed(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "a.png"; src.write_bytes(b"x")
    dst = tmp_path / "b.png"
    monkeypatch.setattr(os, "link", lambda s, d: (_ for _ in ()).throw(OSError("EXDEV")))
    method = _place_image(src, dst, "hardlink", allow_copy_fallback=True)
    assert method == "copy_fallback"
    assert dst.read_bytes() == b"x"
    assert os.stat(src).st_ino != os.stat(dst).st_ino


def test_explicit_copy_mode(tmp_path: Path) -> None:
    src = tmp_path / "a.png"; src.write_bytes(b"x")
    dst = tmp_path / "b.png"
    assert _place_image(src, dst, "copy") == "copy"
    assert os.stat(src).st_ino != os.stat(dst).st_ino


# ---- adapter CLI staging cleanup ----------------------------------------

def _make_args(*, source: Path, target_root: Path, cleanup=False, keep=False) -> argparse.Namespace:
    return argparse.Namespace(
        config="ignored", source_dir=str(source), dataset_id="ds",
        name=None, dataset_source="DCS_AutoDataset", dcs_config=None,
        overwrite=False, no_images=False, link_mode="copy",
        allow_copy_fallback=False, cleanup_staging=cleanup, keep_staging=keep,
    )


def _patch_config(monkeypatch, target_root: Path) -> None:
    cfg = {
        "dcs_capture": {"raw_target_dir": str(target_root), "image_placement": "copy"},
        "data": {"split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15}},
    }
    monkeypatch.setattr(cli, "load_config", lambda _p: cfg)


def test_cleanup_staging_removes_source_under_staging(tmp_path: Path, monkeypatch) -> None:
    staging = tmp_path / "_staging" / "dcs"
    source = write_mini_dcs_dataset(staging, num_frames=6)   # -> staging/dataset
    target_root = tmp_path / "raw"
    _patch_config(monkeypatch, target_root)

    rc = cli.cmd_adapter_normalize(_make_args(source=source, target_root=target_root, cleanup=True))
    assert rc == 0
    assert (target_root / "ds" / "metadata" / "classes.json").is_file()
    assert not source.exists(), "staging source should be removed after import"


def test_no_cleanup_by_default(tmp_path: Path, monkeypatch) -> None:
    staging = tmp_path / "_staging" / "dcs"
    source = write_mini_dcs_dataset(staging, num_frames=6)
    target_root = tmp_path / "raw"
    _patch_config(monkeypatch, target_root)

    rc = cli.cmd_adapter_normalize(_make_args(source=source, target_root=target_root, cleanup=False))
    assert rc == 0
    assert source.exists(), "staging must be kept when --cleanup-staging not given"


def test_cleanup_skipped_when_source_not_under_staging(tmp_path: Path, monkeypatch) -> None:
    source = write_mini_dcs_dataset(tmp_path / "plain", num_frames=6)  # no _staging in path
    target_root = tmp_path / "raw"
    _patch_config(monkeypatch, target_root)

    rc = cli.cmd_adapter_normalize(_make_args(source=source, target_root=target_root, cleanup=True))
    assert rc == 0
    assert source.exists(), "must not delete a non-staging source even with --cleanup-staging"


# ---- hardlink self-check + copy default ---------------------------------

def test_hardlink_supported_probe_cleans_up(tmp_path: Path) -> None:
    if not hasattr(os, "link"):
        pytest.skip("os.link unsupported")
    d = tmp_path / "probe_dir"
    assert _hardlink_supported(d) is True            # tmp is one volume
    assert not any(p.name.startswith(".hardlink_probe") for p in d.iterdir())  # no leftovers


def test_normalize_default_is_copy(tmp_path: Path) -> None:
    source = write_mini_dcs_dataset(tmp_path / "src", num_frames=6)
    target = tmp_path / "out" / "ds"
    normalize_dataset(source_dir=source, target_dir=target, dataset_id="ds")  # no image_placement -> copy
    from packages.common.coco_io import load_coco
    for split in ("train", "val", "test"):
        coco = load_coco(target / "annotations" / f"instances_{split}.json")
        for img in coco["images"]:
            src_img = source / "images" / img["file_name"]
            dst_img = target / "images" / split / img["file_name"]
            assert os.stat(src_img).st_ino != os.stat(dst_img).st_ino   # real copy, not a link


def test_hardlink_unsupported_fails_fast(tmp_path: Path, monkeypatch) -> None:
    source = write_mini_dcs_dataset(tmp_path / "src", num_frames=4)
    monkeypatch.setattr(os, "link", lambda s, d: (_ for _ in ()).throw(OSError("EXDEV")))
    with pytest.raises(RuntimeError, match="unsupported"):
        normalize_dataset(source_dir=source, target_dir=tmp_path / "out" / "ds",
                          dataset_id="ds", image_placement="hardlink")


def test_hardlink_unsupported_falls_back_when_allowed(tmp_path: Path, monkeypatch) -> None:
    source = write_mini_dcs_dataset(tmp_path / "src", num_frames=4)
    target = tmp_path / "out" / "ds"
    monkeypatch.setattr(os, "link", lambda s, d: (_ for _ in ()).throw(OSError("EXDEV")))
    normalize_dataset(source_dir=source, target_dir=target, dataset_id="ds",
                      image_placement="hardlink", allow_copy_fallback=True)
    assert (target / "metadata" / "classes.json").is_file()   # placed via copy fallback
