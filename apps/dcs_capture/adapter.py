"""Normalize a DCS_AutoDataset plain ``dataset/`` directory into the raw dataset
contract used by the rest of the pipeline.

Input layout (current DCS_AutoDataset output):

    source/
    ├── images/<frame_id>.png             # flat, no train/val/test split
    ├── metadata/<frame_id>.json          # rich per-frame metadata (DCS schema)
    ├── annotations/_annotations.coco.json  # single COCO file
    └── yolo_labels/<frame_id>.txt        # ignored by adapter

Output layout (raw dataset contract):

    storage/datasets/raw/{dataset_id}/
    ├── images/{train,val,test}/<frame_id>.png
    ├── annotations/instances_{train,val,test}.json
    └── metadata/
        ├── dataset_info.json
        ├── classes.json
        ├── capture_manifest.jsonl
        └── dcs_run_config.yaml          (optional, copied if --dcs-config provided)

Split is deterministic by ``sha256(source_frame_id) % 10000`` against the
configured ratios. Train/val/test boundaries are fixed once and reproducible.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from packages.common.coco_io import load_coco, save_coco, split_by_image_ids
from packages.common.logging_utils import get_logger
from packages.common.manifest import ManifestWriter

LOG = get_logger(__name__)

SPLIT_NAMES = ("train", "val", "test")
_SPLIT_BUCKETS = 10_000  # finer than 100 to absorb small ratio drifts


@dataclass(frozen=True)
class AdapterResult:
    dataset_id: str
    target_dir: Path
    num_images: int
    num_annotations: int
    num_classes: int
    splits: dict[str, dict[str, int]]   # split → {num_images, num_annotations}
    warnings: list[str] = field(default_factory=list)


def assign_split(source_frame_id: str, ratios: Mapping[str, float]) -> str:
    """Deterministic split bucket for a frame id.

    ``ratios`` must be a mapping like ``{"train": 0.70, "val": 0.15, "test": 0.15}``.
    Order is fixed (train, val, test); sum-to-one is checked.
    """
    train = float(ratios.get("train", 0.0))
    val   = float(ratios.get("val",   0.0))
    test  = float(ratios.get("test",  0.0))
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0, got {train + val + test}")

    h = int(hashlib.sha256(source_frame_id.encode("utf-8")).hexdigest(), 16) % _SPLIT_BUCKETS
    train_cut = int(round(train * _SPLIT_BUCKETS))
    val_cut   = train_cut + int(round(val * _SPLIT_BUCKETS))
    if h < train_cut:
        return "train"
    if h < val_cut:
        return "val"
    return "test"


def _place_image(src: Path, dst: Path, mode: str, *, allow_copy_fallback: bool = False) -> str:
    """Place a source image into the dataset. Returns the method actually used.

    mode: 'copy' (default) | 'hardlink' | 'move'. A FAILED hardlink raises by
    default — we never silently fall back to a full copy, because that would
    re-introduce a hidden duplicate of the whole dataset. To allow a copy across
    filesystems, pass ``allow_copy_fallback=True`` or use mode 'copy' explicitly.
    """
    if mode == "move":
        shutil.move(str(src), str(dst))
        return "move"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    # default: hardlink — NO silent fallback
    try:
        os.link(src, dst)
    except OSError as exc:
        if allow_copy_fallback:
            shutil.copy2(src, dst)
            return "copy_fallback"
        raise RuntimeError(
            f"Hardlink failed ({src} -> {dst}). Source and target may be on "
            f"different filesystems. Move staging under the same STORAGE_ROOT, "
            f"or rerun with --link-mode copy (or --allow-copy-fallback) explicitly."
        ) from exc
    return "hardlink"


def _hardlink_supported(directory: Path) -> bool:
    """One-shot probe: can we hardlink within ``directory``? (cross-FS / drvfs safe)."""
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / ".hardlink_probe"
    link = directory / ".hardlink_probe_link"
    try:
        probe.write_text("x", encoding="utf-8")
        if link.exists():
            link.unlink()
        os.link(probe, link)
        return True
    except OSError:
        return False
    finally:
        for p in (link, probe):
            try:
                p.unlink()
            except OSError:
                pass


def normalize_dataset(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    dataset_id: str,
    name: str | None = None,
    split_ratios: Mapping[str, float] | None = None,
    dcs_config_path: str | Path | None = None,
    dataset_source: str = "DCS_AutoDataset",
    overwrite: bool = False,
    copy_images: bool = True,
    image_placement: str = "copy",
    allow_copy_fallback: bool = False,
) -> AdapterResult:
    """Convert source DCS dataset → raw dataset contract layout.

    Side effects: writes images, annotations, metadata under ``target_dir``.
    """
    source = Path(source_dir).resolve()
    target = Path(target_dir).resolve()
    ratios = dict(split_ratios) if split_ratios else {"train": 0.70, "val": 0.15, "test": 0.15}

    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset directory not found: {source}")
    coco_path = source / "annotations" / "_annotations.coco.json"
    if not coco_path.is_file():
        raise FileNotFoundError(f"Source COCO not found: {coco_path}")
    src_images_dir = source / "images"
    if not src_images_dir.is_dir():
        raise FileNotFoundError(f"Source images dir not found: {src_images_dir}")
    src_metadata_dir = source / "metadata"

    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target dataset directory already exists: {target}. Pass overwrite=True or remove it."
            )
        shutil.rmtree(target)

    LOG.info("Reading source COCO: %s", coco_path)
    src_coco = load_coco(coco_path)

    # 1) Decide split for each image
    by_split_image_ids: dict[str, list[int]] = {s: [] for s in SPLIT_NAMES}
    split_of_image: dict[int, str] = {}
    for img in src_coco["images"]:
        frame_id = Path(img["file_name"]).stem
        split = assign_split(frame_id, ratios)
        by_split_image_ids[split].append(img["id"])
        split_of_image[img["id"]] = split

    # 2) Split source COCO
    LOG.info("Splitting %d images into train=%d val=%d test=%d",
             len(src_coco["images"]),
             len(by_split_image_ids["train"]),
             len(by_split_image_ids["val"]),
             len(by_split_image_ids["test"]))
    split_cocos = split_by_image_ids(src_coco, by_split_image_ids)

    # 3) Materialize layout
    (target / "images").mkdir(parents=True, exist_ok=True)
    (target / "annotations").mkdir(parents=True, exist_ok=True)
    (target / "metadata").mkdir(parents=True, exist_ok=True)
    for s in SPLIT_NAMES:
        (target / "images" / s).mkdir(parents=True, exist_ok=True)

    # 4) Save split COCO files
    for split_name, coco in split_cocos.items():
        save_coco(coco, target / "annotations" / f"instances_{split_name}.json")

    # 5) Place images. Default is copy (robust); hardlink is opt-in. If hardlink is
    #    requested but unsupported here, fail fast unless copy-fallback is allowed.
    warnings: list[str] = []
    if copy_images and image_placement == "hardlink" and not _hardlink_supported(target / "images"):
        if allow_copy_fallback:
            LOG.warning("Hardlink unsupported at %s — falling back to copy (allow_copy_fallback).",
                        target / "images")
            image_placement = "copy"
        else:
            raise RuntimeError(
                f"Hardlink mode requested but unsupported on the filesystem at "
                f"{target / 'images'} (staging and raw on different volumes, or a "
                f"drvfs/9p mount). Keep all of STORAGE_ROOT on one volume, or rerun "
                f"with --link-mode copy (or --allow-copy-fallback)."
            )
    if copy_images:
        for img in src_coco["images"]:
            file_name = img["file_name"]
            split = split_of_image[img["id"]]
            src_img = src_images_dir / file_name
            dst_img = target / "images" / split / file_name
            if not src_img.is_file():
                warnings.append(f"Source image missing: {src_img} (referenced in COCO)")
                continue
            _place_image(src_img, dst_img, image_placement,
                         allow_copy_fallback=allow_copy_fallback)

    # 6) capture_manifest.jsonl
    manifest_path = target / "metadata" / "capture_manifest.jsonl"
    with ManifestWriter(manifest_path) as mw:
        for img in src_coco["images"]:
            frame_id = Path(img["file_name"]).stem
            split = split_of_image[img["id"]]
            record: dict[str, Any] = {
                "frame_id": frame_id,
                "split": split,
                "source_image": f"images/{split}/{img['file_name']}",
                "image_id_in_split_coco": img["id"],
                "width": img["width"],
                "height": img["height"],
            }
            meta_path = src_metadata_dir / f"{frame_id}.json"
            if meta_path.is_file():
                try:
                    src_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    warnings.append(f"Malformed per-frame metadata: {meta_path}")
                    src_meta = {}
                _augment_capture_record(record, src_meta)
            else:
                record["has_rich_metadata"] = False
            mw.write(record)

    # 7) classes.json
    classes = {"categories": list(src_coco.get("categories", []))}
    (target / "metadata" / "classes.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # 8) optional dcs_run_config.yaml
    if dcs_config_path is not None:
        src_cfg = Path(dcs_config_path)
        if src_cfg.is_file():
            shutil.copy2(src_cfg, target / "metadata" / "dcs_run_config.yaml")
        else:
            warnings.append(f"DCS config path provided but file missing: {src_cfg}")

    # 9) dataset_info.json (last — references split counts)
    info: dict[str, Any] = {
        "dataset_id": dataset_id,
        "name": name or f"DCS dataset {dataset_id}",
        "created_at": _utc_now_iso(),
        "source": dataset_source,
        "num_images": len(src_coco["images"]),
        "num_annotations": len(src_coco["annotations"]),
        "num_classes": len(src_coco.get("categories", [])),
        "split_strategy": "by_source_frame_hash",
        "split_ratios": ratios,
        "splits": {
            s: {
                "num_images": len(split_cocos[s]["images"]),
                "num_annotations": len(split_cocos[s]["annotations"]),
            }
            for s in SPLIT_NAMES
        },
        "annotation_format": "coco",
        "image_format": "png",
        "notes": None,
    }
    (target / "metadata" / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    return AdapterResult(
        dataset_id=dataset_id,
        target_dir=target,
        num_images=info["num_images"],
        num_annotations=info["num_annotations"],
        num_classes=info["num_classes"],
        splits=info["splits"],
        warnings=warnings,
    )


# ---------- helpers -------------------------------------------------------

def _augment_capture_record(record: dict[str, Any], src_meta: Mapping[str, Any]) -> None:
    """Pick the small subset of DCS per-frame metadata that aids leakage tracking + filtering."""
    record["has_rich_metadata"] = True
    scene = src_meta.get("scene", {}) or {}
    objects = src_meta.get("objects", []) or []
    record["scene_id"]      = scene.get("scene_id")
    record["mission_id"]    = scene.get("mission_id")
    record["weather"]       = scene.get("weather")
    record["time_of_day"]   = scene.get("time_of_day")
    record["sea_state"]     = scene.get("sea_state")
    record["scenario_mode"] = scene.get("scenario_mode")
    record["captured_at"]   = src_meta.get("timestamp_utc")
    record["dcs_usable"]    = bool(src_meta.get("usable", False))
    record["num_objects"]   = len(objects)
    record["all_quality_tiers"] = [o.get("quality_tier") for o in objects]


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
