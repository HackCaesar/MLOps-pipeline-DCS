"""Unit tests for scripts.register_dataset.register_dataset."""
from __future__ import annotations

from pathlib import Path

import pytest

from apps.dcs_capture.adapter import normalize_dataset
from packages.common.metadata_store import SQLiteMetadataStore
from scripts.register_dataset import register_dataset
from tests.fixtures.mini_dcs_dataset import write_mini_dcs_dataset


@pytest.fixture
def adapted_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Return (dataset_dir, sqlite_path) for a fresh tiny adapted dataset."""
    source = write_mini_dcs_dataset(tmp_path / "src", num_frames=12)
    dataset_dir = tmp_path / "raw" / "d_reg"
    normalize_dataset(source_dir=source, target_dir=dataset_dir, dataset_id="d_reg")
    return dataset_dir, tmp_path / "pipeline.db"


def test_registration_inserts_dataset_images_annotations(adapted_dataset) -> None:
    dataset_dir, sqlite_path = adapted_dataset
    summary = register_dataset(dataset_dir, sqlite_path, compute_hash=False)

    assert summary["dataset_id"] == "d_reg"
    assert summary["images_registered"] >= 1
    assert summary["annotations_registered"] >= 1
    assert sum(s["num_images"] for s in summary["splits"].values()) == summary["images_registered"]

    with SQLiteMetadataStore(sqlite_path) as store:
        ds = store.get_dataset("d_reg")
        assert ds is not None
        assert ds.num_images == summary["images_registered"]
        for split in ("train", "val", "test"):
            imgs = store.get_images_by_split("d_reg", split)
            assert len(imgs) == summary["splits"].get(split, {}).get("num_images", 0)


def test_content_hash_present_when_computed(adapted_dataset) -> None:
    dataset_dir, sqlite_path = adapted_dataset
    summary = register_dataset(dataset_dir, sqlite_path, compute_hash=True)
    assert summary["content_hash"] is not None
    assert len(summary["content_hash"]) == 64


def test_double_registration_without_replace_raises(adapted_dataset) -> None:
    dataset_dir, sqlite_path = adapted_dataset
    register_dataset(dataset_dir, sqlite_path, compute_hash=False)
    with pytest.raises(RuntimeError, match="already registered"):
        register_dataset(dataset_dir, sqlite_path, compute_hash=False)


def test_replace_drops_and_reinserts(adapted_dataset) -> None:
    dataset_dir, sqlite_path = adapted_dataset
    register_dataset(dataset_dir, sqlite_path, compute_hash=False)
    summary = register_dataset(dataset_dir, sqlite_path,
                               compute_hash=False, replace=True)
    assert summary["replaced"] is True
    # Counts still consistent after replace
    with SQLiteMetadataStore(sqlite_path) as store:
        ds = store.get_dataset("d_reg")
        assert ds is not None
        assert ds.num_images == summary["images_registered"]


def test_images_have_source_frame_id(adapted_dataset) -> None:
    dataset_dir, sqlite_path = adapted_dataset
    register_dataset(dataset_dir, sqlite_path, compute_hash=False)
    with SQLiteMetadataStore(sqlite_path) as store:
        all_images = (
            store.get_images_by_split("d_reg", "train")
            + store.get_images_by_split("d_reg", "val")
            + store.get_images_by_split("d_reg", "test")
        )
        assert all_images
        for img in all_images:
            assert img.source_frame_id == Path(img.file_name).stem


def test_missing_dataset_info_raises(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "no_info"
    dataset_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="dataset_info.json"):
        register_dataset(dataset_dir, tmp_path / "db.sqlite", compute_hash=False)
