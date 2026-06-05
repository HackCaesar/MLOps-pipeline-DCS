"""Content-addressed tile cache build/reuse (Stage 3).

Pure orchestration around ``packages.common.hashing`` — deliberately imports
neither numpy nor mlflow at module load, so the cache logic is unit-testable
without the heavy enrichment stack (the real tiler is injected via ``enrich_fn``,
and the default is imported lazily).

Cache layout (``data_dir`` for YOLOX points at the cache root):

    {cache_root}/{tile_cache_id}/
    ├── annotations/instances_{train,val,test}.json
    ├── images/{train,val,test}/
    ├── tile_manifest.json
    ├── dropped_tiles_manifest.jsonl
    └── cache_meta.json   (schema_version, layout_version, status=complete, hashes…)

Build is atomic: tiles are written to ``{tile_cache_id}.building`` and renamed
into place only after ``cache_meta.json`` (status=complete) is flushed.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from packages.common import hashing
from packages.common.logging_utils import get_logger

LOG = get_logger(__name__)

SCHEMA_VERSION = 1
LAYOUT_VERSION = 1
SPLITS = ("train", "val", "test")


def _tool_version() -> str:
    """Installed package version (for provenance); graceful if not installed."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("dcs-yolox-pipeline")
        except PackageNotFoundError:
            return "0.0.0+unknown"
    except Exception:
        return "0.0.0+unknown"

# enrich_fn(raw_dataset_dir, out_dir, dataset_id, enrichment_cfg) -> result|None
EnrichFn = Callable[[Path, Path, str, Mapping[str, Any]], Any]


def _default_enrich(raw_dataset_dir: Path, out_dir: Path, dataset_id: str,
                    enrichment_cfg: Mapping[str, Any]) -> Any:
    # Lazy import so this module (and its tests) don't require numpy/cv2.
    from apps.dataset_enrichment.enrich import enrich_dataset
    return enrich_dataset(
        raw_dataset_dir=raw_dataset_dir,
        tmp_root=out_dir,
        dataset_id=dataset_id,
        run_id="cache",
        enrichment_cfg=enrichment_cfg,
        write_images=True,
        write_coco=True,
    )


def _count_split_images(d: Path, split: str) -> int:
    p = d / "annotations" / f"instances_{split}.json"
    if not p.is_file():
        return 0
    try:
        return len(json.loads(p.read_text(encoding="utf-8")).get("images", []))
    except json.JSONDecodeError:
        return 0


def cache_is_valid(cache_dir: Path, hashes: Mapping[str, str]) -> bool:
    """A cache is reusable only if its meta is complete and every hash + file matches."""
    meta_p = cache_dir / "cache_meta.json"
    if not meta_p.is_file():
        return False
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if meta.get("status") != "complete":
        return False
    if meta.get("schema_version") != SCHEMA_VERSION:
        return False
    if meta.get("layout_version") != LAYOUT_VERSION:
        return False
    for k in ("dataset_hash", "split_hash", "tile_config_hash"):
        if meta.get(k) != hashes[k]:
            return False
    for s in SPLITS:
        if not (cache_dir / "annotations" / f"instances_{s}.json").is_file():
            return False
        if not (cache_dir / "images" / s).is_dir():
            return False
    return True


def build_or_reuse_cache(
    *,
    raw_dataset_dir: Path,
    dataset_id: str,
    enrichment_cfg: Mapping[str, Any],
    cache_root: Path,
    runs_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    enrich_fn: Optional[EnrichFn] = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Build the tile cache (or reuse an existing valid one). Returns a summary dict."""
    raw_dataset_dir = Path(raw_dataset_dir)
    cache_root = Path(cache_root)
    enrich_fn = enrich_fn or _default_enrich

    hashes = hashing.compute_hashes(raw_dataset_dir, enrichment_cfg, strict=strict)

    # Guard: an empty-input dataset_hash means no annotations AND no images were
    # hashed — the raw dataset is empty/missing. Never build a cache off that.
    if hashes["dataset_hash"] == hashing.EMPTY_SHA256:
        raise ValueError(
            f"Refusing to build a tile cache: the raw dataset at {raw_dataset_dir} "
            "appears empty — no annotations or images were hashed (dataset_hash == "
            "the empty-input SHA-256). Ensure capture/adapter produced a non-empty "
            "raw dataset before running build-cache."
        )

    tile_cache_id = hashes["tile_cache_id"]
    cache_dir = cache_root / tile_cache_id
    building_dir = cache_root / f"{tile_cache_id}.building"

    if cache_is_valid(cache_dir, hashes):
        LOG.info("Tile cache reused: %s", cache_dir)
        status = "reused"
    else:
        # An existing-but-invalid final dir means a corrupt/incomplete cache at
        # this exact key — remove it (logged) before rebuilding.
        if cache_dir.exists():
            LOG.warning("Removing incomplete/stale tile cache at %s (failed validity check)", cache_dir)
            shutil.rmtree(cache_dir, ignore_errors=True)
        # Clean a leftover .building from a previously-crashed build.
        if building_dir.exists():
            LOG.warning("Removing stale .building dir from a prior failed build: %s", building_dir)
            shutil.rmtree(building_dir, ignore_errors=True)

        cache_root.mkdir(parents=True, exist_ok=True)
        building_dir.mkdir(parents=True)
        LOG.info("Building tile cache %s (dataset=%s)", tile_cache_id, dataset_id)
        result = enrich_fn(raw_dataset_dir, building_dir, dataset_id, enrichment_cfg)

        counts = {s: _count_split_images(building_dir, s) for s in SPLITS}
        num_dropped = int(getattr(result, "num_dropped_tiles", 0) or 0)
        num_background = int(getattr(result, "num_background_images", 0) or 0)

        _write_tile_manifest(building_dir, dataset_id, tile_cache_id, enrichment_cfg,
                             counts, num_dropped, num_background)
        _write_cache_meta(building_dir, raw_dataset_dir, dataset_id, tile_cache_id,
                          hashes, enrichment_cfg, counts, num_dropped, num_background)

        # Atomic publish: rename .building -> final.
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        os.replace(building_dir, cache_dir)
        LOG.info("Tile cache created: %s", cache_dir)
        status = "created"

    info = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "tile_cache_id": tile_cache_id,
        "cache_dir": str(cache_dir),
        "status": status,
        "dataset_hash": hashes["dataset_hash"],
        "split_hash": hashes["split_hash"],
        "tile_config_hash": hashes["tile_config_hash"],
        "num_train_tiles": _count_split_images(cache_dir, "train"),
        "num_val_tiles": _count_split_images(cache_dir, "val"),
        "num_test_tiles": _count_split_images(cache_dir, "test"),
    }
    _write_breadcrumb(runs_dir, run_id, info)
    return info


def resolve_cache_dir(raw_dataset_dir: Path, enrichment_cfg: Mapping[str, Any],
                      cache_root: Path, *, strict: bool = False) -> Path:
    """Deterministically locate the cache dir (used by training, no build)."""
    hashes = hashing.compute_hashes(raw_dataset_dir, enrichment_cfg, strict=strict)
    return Path(cache_root) / hashes["tile_cache_id"]


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #

def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_tile_manifest(out_dir: Path, dataset_id: str, tile_cache_id: str,
                         enrichment_cfg: Mapping[str, Any], counts: dict[str, int],
                         num_dropped: int, num_background: int) -> None:
    manifest = {
        "tile_cache_id": tile_cache_id,
        "source_dataset_id": dataset_id,
        "scales": list(enrichment_cfg.get("scales") or []),
        "num_train_tiles": counts["train"],
        "num_val_tiles": counts["val"],
        "num_test_tiles": counts["test"],
        "num_dropped_tiles": num_dropped,
        "num_background_images": num_background,
        "created_at": _utc_now(),
    }
    (out_dir / "tile_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_cache_meta(out_dir: Path, raw_dataset_dir: Path, dataset_id: str,
                      tile_cache_id: str, hashes: Mapping[str, str],
                      enrichment_cfg: Mapping[str, Any], counts: dict[str, int],
                      num_dropped: int, num_background: int) -> None:
    meta = {
        "schema_version": SCHEMA_VERSION,
        "layout_version": LAYOUT_VERSION,
        "status": "complete",
        "tile_cache_id": tile_cache_id,
        "dataset_hash": hashes["dataset_hash"],
        "split_hash": hashes["split_hash"],
        "tile_config_hash": hashes["tile_config_hash"],
        "source_dataset_id": dataset_id,
        "raw_dataset_dir": str(raw_dataset_dir),
        "tile_config": hashing.canonicalize_tile_config(enrichment_cfg),
        "num_train_tiles": counts["train"],
        "num_val_tiles": counts["val"],
        "num_test_tiles": counts["test"],
        "num_dropped_tiles": num_dropped,
        "num_background_images": num_background,
        "created_at": _utc_now(),
        "tool_version": _tool_version(),
        "tiling_algorithm_version": hashing.TILING_ALGORITHM_VERSION,
    }
    # Written LAST inside the .building dir, so a half-built cache never carries
    # status=complete.
    (out_dir / "cache_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_breadcrumb(runs_dir: Optional[Path], run_id: Optional[str],
                      info: Mapping[str, Any]) -> Optional[Path]:
    if runs_dir is None or not run_id:
        return None
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "tile_cache.json"
    p.write_text(json.dumps(dict(info), indent=2, ensure_ascii=False), encoding="utf-8")
    return p
