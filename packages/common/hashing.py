"""Content-addressed hashing for the reusable tile cache (Stage 3).

Three independent hashes key a tile cache so it can be reused across runs:

* ``dataset_hash``     — raw images + annotations content of the source dataset;
* ``split_hash``       — the train/val/test assignment (which frame in which split);
* ``tile_config_hash`` — the enrichment/tiling geometry config (crop/stride/scales/…).

``tile_cache_id = f"{dataset_hash[:12]}__{split_hash[:12]}__{tile_config_hash[:12]}"``

The same functions are used by ``apps.dataset_enrichment`` (to build/reuse the
cache) and by ``apps.yolox_training`` (to locate it) — they must agree, so this
is the single source of truth. All hashing is deterministic: dict keys are
sorted and floats normalized so ``0.8`` and ``0.80`` hash identically.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SPLITS = ("train", "val", "test")
_HASH_LEN = 12

# SHA-256 of no input. ``dataset_hash`` returns this only when NOTHING was hashed
# (no annotations and no images) — i.e. an empty/missing dataset. Callers that
# build a cache must treat this as an error (see tile_cache.build_or_reuse_cache).
EMPTY_SHA256 = hashlib.sha256().hexdigest()  # e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

# Enrichment-config keys that affect tile geometry/content (everything else,
# e.g. ``enabled``/``mode``, is irrelevant to the produced tiles).
_TILE_CONFIG_KEYS = (
    "crop_size",
    "stride",
    "visible_area_threshold",
    "include_edge_tiles",
    "keep_background_images",
    "background_policy",
    "scales",
)

# Bump when the tiling/enrichment ALGORITHM changes (not just its config). Folded
# into tile_config_hash so a code-level change invalidates every existing tile
# cache — otherwise a bugfix in the tiler would silently reuse wrong tiles.
TILING_ALGORITHM_VERSION = 1


def short(h: str, n: int = _HASH_LEN) -> str:
    return h[:n]


# --------------------------------------------------------------------------- #
# canonicalization (deterministic across dict order + float formatting)
# --------------------------------------------------------------------------- #

def _canonicalize(obj: Any) -> Any:
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        # normalize float formatting (0.8 == 0.80) and -0.0 -> 0.0
        return round(obj + 0.0, 6)
    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    return obj


def _canonical_json(obj: Any) -> str:
    return json.dumps(_canonicalize(obj), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# the three hashes
# --------------------------------------------------------------------------- #

def dataset_hash(raw_dataset_dir: str | Path, *, strict: bool = False) -> str:
    """Hash raw annotations (always full bytes) + images.

    Images are hashed by ``(relpath, size)`` by default (fast, stable); pass
    ``strict=True`` to hash image bytes too. Missing files are skipped so the
    function never raises on an incomplete dataset (callers gate on dir existence).
    """
    d = Path(raw_dataset_dir)
    h = hashlib.sha256()
    ann = d / "annotations"
    for s in SPLITS:
        p = ann / f"instances_{s}.json"
        if p.is_file():
            h.update(b"ANN:")
            h.update(s.encode())
            h.update(p.read_bytes())
    images = d / "images"
    if images.is_dir():
        for img in sorted(images.rglob("*")):
            if img.is_file():
                rel = img.relative_to(images).as_posix()
                h.update(b"IMG:")
                h.update(rel.encode("utf-8"))
                if strict:
                    h.update(_file_sha256(img).encode())
                else:
                    h.update(str(img.stat().st_size).encode())
    return h.hexdigest()


def split_hash(raw_dataset_dir: str | Path) -> str:
    """Hash the split assignment only: sorted ``file_name`` lists per split.

    Source of truth = the ``images[]`` lists inside ``annotations/instances_{split}.json``
    (the adapter assigns splits). Independent of bbox edits (covered by dataset_hash).
    """
    d = Path(raw_dataset_dir)
    h = hashlib.sha256()
    ann = d / "annotations"
    for s in SPLITS:
        p = ann / f"instances_{s}.json"
        names: list[str] = []
        if p.is_file():
            coco = json.loads(p.read_text(encoding="utf-8"))
            names = sorted(str(im.get("file_name", "")) for im in coco.get("images", []))
        h.update(b"SPLIT:")
        h.update(s.encode())
        for n in names:
            h.update(b"\0")
            h.update(n.encode("utf-8"))
    return h.hexdigest()


def canonicalize_tile_config(enrichment_cfg: Mapping[str, Any]) -> dict:
    """Return the normalized geometry-relevant subset of the enrichment config."""
    cfg = dict(enrichment_cfg or {})
    subset = {k: cfg[k] for k in _TILE_CONFIG_KEYS if k in cfg}
    return _canonicalize(subset)


def tile_config_hash(enrichment_cfg: Mapping[str, Any]) -> str:
    # Includes the algorithm version so a tiling-CODE change invalidates caches,
    # not only a config change.
    payload = {
        "tiling_algorithm_version": TILING_ALGORITHM_VERSION,
        "config": canonicalize_tile_config(enrichment_cfg),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# composite id
# --------------------------------------------------------------------------- #

def compute_hashes(raw_dataset_dir: str | Path,
                   enrichment_cfg: Mapping[str, Any],
                   *, strict: bool = False) -> dict[str, str]:
    """Return {'dataset_hash','split_hash','tile_config_hash','tile_cache_id'}."""
    ds = dataset_hash(raw_dataset_dir, strict=strict)
    sp = split_hash(raw_dataset_dir)
    tc = tile_config_hash(enrichment_cfg)
    return {
        "dataset_hash": ds,
        "split_hash": sp,
        "tile_config_hash": tc,
        "tile_cache_id": make_tile_cache_id(ds, sp, tc),
    }


def make_tile_cache_id(dataset_hash_: str, split_hash_: str, tile_config_hash_: str) -> str:
    return f"{short(dataset_hash_)}__{short(split_hash_)}__{short(tile_config_hash_)}"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
