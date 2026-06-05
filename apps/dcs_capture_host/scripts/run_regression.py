"""Regression harness for DCS_AutoDataset.

Replays VisibleBBoxRefiner + DatasetExporter on a small set of pre-captured
PNG + metadata fixtures and compares the produced metadata, YOLO labels and
COCO annotations to a stored golden output. DCS is never launched.

Usage:
    python scripts/run_regression.py                  # check mode (default)
    python scripts/run_regression.py --update-golden  # rewrite golden output
    python scripts/run_regression.py --keep-output DIR  # also keep generated files

Exit code is 0 when current output matches golden under floating tolerance,
1 when any difference is found. Differences are printed to stdout.

This harness intentionally does not touch the runtime config paths used for
DCS: it only reads `generation.allowed_classes`, `pipeline.allowed_quality_tiers_for_training`,
and `bbox_refinement.*` from the YAML.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError as exc:  # pragma: no cover - install hint
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

from exporters.coco_yolo import DatasetExporter
from validators import VisibleBBoxRefiner

FIXTURES_DIR = PROJECT_ROOT / "tests" / "regression" / "fixtures"
GOLDEN_DIR = PROJECT_ROOT / "tests" / "regression" / "golden"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
COCO_FILE_NAME = "_annotations.coco.json"

# Tolerances for floating comparisons. Tight enough to catch real numerical
# drift, loose enough to absorb non-meaningful re-serialisation noise.
FLOAT_ABS_TOL = 1e-6
FLOAT_REL_TOL = 1e-9


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def discover_fixtures() -> List[str]:
    metadata_dir = FIXTURES_DIR / "metadata_input"
    return sorted(p.stem for p in metadata_dir.glob("*.json"))


def values_equivalent(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        return math.isclose(fa, fb, abs_tol=FLOAT_ABS_TOL, rel_tol=FLOAT_REL_TOL)
    return a == b


def _normalize_debug_path(value: str) -> str:
    """Strip the ephemeral tmpdir prefix from a debug-artifact path.

    Refiner writes absolute paths like
    ``C:\\Users\\...\\Temp\\dcs_regression_xxx\\debug_refiner\\ships\\<frame>__<method>__overlay.png``
    The tmpdir component is harness noise; everything from ``debug_refiner/``
    onward encodes the chosen refinement method and is the part that should
    actually be compared against the golden output.
    """
    norm = value.replace("\\", "/")
    marker = "/debug_refiner/"
    idx = norm.find(marker)
    if idx >= 0:
        return norm[idx + 1:]
    if norm.startswith("debug_refiner/"):
        return norm
    return value


def _normalize_for_compare(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _normalize_for_compare(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_normalize_for_compare(v) for v in node]
    if isinstance(node, str) and "debug_refiner" in node.replace("\\", "/"):
        return _normalize_debug_path(node)
    return node


def deep_diff(current: Any, golden: Any, path: str = "") -> List[str]:
    diffs: List[str] = []
    if isinstance(current, dict) and isinstance(golden, dict):
        keys_c, keys_g = set(current.keys()), set(golden.keys())
        for k in sorted(keys_c - keys_g):
            diffs.append(f"{path}.{k}: only in current")
        for k in sorted(keys_g - keys_c):
            diffs.append(f"{path}.{k}: only in golden")
        for k in sorted(keys_c & keys_g):
            diffs.extend(deep_diff(current[k], golden[k], f"{path}.{k}"))
    elif isinstance(current, list) and isinstance(golden, list):
        if len(current) != len(golden):
            diffs.append(f"{path}: list length {len(current)} vs golden {len(golden)}")
        for i, (cv, gv) in enumerate(zip(current, golden)):
            diffs.extend(deep_diff(cv, gv, f"{path}[{i}]"))
    else:
        if not values_equivalent(current, golden):
            diffs.append(f"{path}: {current!r} != golden {golden!r}")
    return diffs


def diff_yolo_text(current: str, golden: str, label: str) -> List[str]:
    a_lines = current.splitlines()
    b_lines = golden.splitlines()
    diffs: List[str] = []
    if len(a_lines) != len(b_lines):
        diffs.append(f"{label}: {len(a_lines)} lines vs golden {len(b_lines)}")
    for i, (ca, cb) in enumerate(zip(a_lines, b_lines)):
        if ca == cb:
            continue
        ta, tb = ca.split(), cb.split()
        if len(ta) != len(tb):
            diffs.append(f"{label} line {i}: token count {len(ta)} vs {len(tb)}")
            continue
        line_bad = False
        for j, (xa, xb) in enumerate(zip(ta, tb)):
            if j == 0:
                if xa != xb:
                    diffs.append(f"{label} line {i} class_id: {xa!r} vs {xb!r}")
                    line_bad = True
            else:
                try:
                    fa, fb = float(xa), float(xb)
                except ValueError:
                    diffs.append(f"{label} line {i} token {j}: {xa!r} vs {xb!r} (not float)")
                    line_bad = True
                    continue
                if not math.isclose(fa, fb, abs_tol=FLOAT_ABS_TOL, rel_tol=FLOAT_REL_TOL):
                    diffs.append(f"{label} line {i} token {j}: {fa!r} vs {fb!r}")
                    line_bad = True
        if not line_bad:
            diffs.append(f"{label} line {i}: format-only diff ({ca!r} vs {cb!r})")
    return diffs


def prepare_image_proxy(work_root: Path) -> Path:
    """Copy fixture PNGs into work_root so that the refiner's debug artifacts
    (written to image_path.parent.parent/debug_refiner/) land inside work_root
    rather than polluting tests/regression/fixtures/.
    """
    proxy_images = work_root / "images"
    proxy_images.mkdir(parents=True, exist_ok=True)
    for png in (FIXTURES_DIR / "images").glob("*.png"):
        shutil.copyfile(png, proxy_images / png.name)
    return proxy_images


def run_pipeline(config: Dict[str, Any], work_root: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
    refiner = VisibleBBoxRefiner(config)
    exporter = DatasetExporter(
        list(config["generation"]["allowed_classes"]),
        allowed_quality_tiers=config["pipeline"].get("allowed_quality_tiers_for_training"),
    )

    metadata_out_dir = work_root / "metadata"
    yolo_out_dir = work_root / "yolo_labels"
    annotations_out_dir = work_root / "annotations"
    for d in (metadata_out_dir, yolo_out_dir, annotations_out_dir):
        d.mkdir(parents=True, exist_ok=True)

    proxy_images = prepare_image_proxy(work_root)

    metadatas: Dict[str, Dict[str, Any]] = {}
    yolos: Dict[str, str] = {}

    for frame_id in discover_fixtures():
        image_path = proxy_images / f"{frame_id}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Fixture image missing: {image_path}")
        meta_in = load_json(FIXTURES_DIR / "metadata_input" / f"{frame_id}.json")
        meta_out = refiner.refine_frame(image_path, meta_in)
        dump_json(metadata_out_dir / f"{frame_id}.json", meta_out)
        yolo_path = yolo_out_dir / f"{frame_id}.txt"
        exporter.write_yolo(meta_out, yolo_path)
        exporter.add_frame(meta_out)
        metadatas[frame_id] = meta_out
        yolos[frame_id] = yolo_path.read_text(encoding="utf-8")

    coco_path = annotations_out_dir / COCO_FILE_NAME
    exporter.write_coco(coco_path)
    coco = load_json(coco_path)
    return metadatas, yolos, coco


def update_golden(metadatas: Dict[str, Dict[str, Any]], yolos: Dict[str, str], coco: Dict[str, Any]) -> None:
    metadata_dir = GOLDEN_DIR / "metadata"
    yolo_dir = GOLDEN_DIR / "yolo_labels"
    coco_dir = GOLDEN_DIR / "annotations"
    for d in (metadata_dir, yolo_dir, coco_dir):
        d.mkdir(parents=True, exist_ok=True)
    for frame_id, payload in metadatas.items():
        dump_json(metadata_dir / f"{frame_id}.json", payload)
    for frame_id, text in yolos.items():
        (yolo_dir / f"{frame_id}.txt").write_text(text, encoding="utf-8")
    dump_json(coco_dir / COCO_FILE_NAME, coco)


def check_against_golden(metadatas: Dict[str, Dict[str, Any]], yolos: Dict[str, str], coco: Dict[str, Any]) -> int:
    failures: List[str] = []

    for frame_id, current in metadatas.items():
        golden_path = GOLDEN_DIR / "metadata" / f"{frame_id}.json"
        if not golden_path.exists():
            failures.append(f"metadata/{frame_id}: golden missing (run --update-golden)")
            continue
        golden = load_json(golden_path)
        failures.extend(
            deep_diff(
                _normalize_for_compare(current),
                _normalize_for_compare(golden),
                path=f"metadata.{frame_id}",
            )
        )

    for frame_id, current_text in yolos.items():
        golden_path = GOLDEN_DIR / "yolo_labels" / f"{frame_id}.txt"
        if not golden_path.exists():
            failures.append(f"yolo/{frame_id}: golden missing (run --update-golden)")
            continue
        golden_text = golden_path.read_text(encoding="utf-8")
        failures.extend(diff_yolo_text(current_text, golden_text, label=f"yolo.{frame_id}"))

    golden_coco_path = GOLDEN_DIR / "annotations" / COCO_FILE_NAME
    if not golden_coco_path.exists():
        failures.append("coco: golden missing (run --update-golden)")
    else:
        golden_coco = load_json(golden_coco_path)
        failures.extend(
            deep_diff(
                _normalize_for_compare(coco),
                _normalize_for_compare(golden_coco),
                path="coco",
            )
        )

    if not failures:
        ann_count = len(coco.get("annotations", []))
        img_count = len(coco.get("images", []))
        print(f"[regression OK] frames={len(metadatas)} coco_images={img_count} coco_annotations={ann_count}")
        return 0

    print(f"[regression FAIL] {len(failures)} difference(s):")
    for line in failures[:200]:
        print(f"  - {line}")
    if len(failures) > 200:
        print(f"  ... and {len(failures) - 200} more")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="DCS_AutoDataset regression harness (no DCS required)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML config")
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help="Overwrite tests/regression/golden/* from current code. Use only after a deliberate change.",
    )
    parser.add_argument(
        "--keep-output",
        help="Optional directory to keep generated metadata/yolo/coco for manual inspection",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    config = load_yaml(config_path)

    fixtures = discover_fixtures()
    if not fixtures:
        raise SystemExit(f"No fixtures found in {FIXTURES_DIR / 'metadata_input'}")
    print(f"[regression] {len(fixtures)} fixture frames: {fixtures}")

    if args.keep_output:
        work_root = Path(args.keep_output).resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        metadatas, yolos, coco = run_pipeline(config, work_root)
        print(f"[regression] generated output kept at {work_root}")
    else:
        with tempfile.TemporaryDirectory(prefix="dcs_regression_") as tmp:
            metadatas, yolos, coco = run_pipeline(config, Path(tmp))
            if args.update_golden:
                update_golden(metadatas, yolos, coco)
                print(f"[regression] golden refreshed in {GOLDEN_DIR}")
                return
            rc = check_against_golden(metadatas, yolos, coco)
            sys.exit(rc)
            return

    if args.update_golden:
        update_golden(metadatas, yolos, coco)
        print(f"[regression] golden refreshed in {GOLDEN_DIR}")
        return

    rc = check_against_golden(metadatas, yolos, coco)
    sys.exit(rc)


if __name__ == "__main__":
    main()
