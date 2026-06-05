"""Per-mission post-capture sanity check.

Runs after one mission's frames have been written. Returns exit code 0 when
the capture for that single mission meets all minimum quality bars, otherwise
a non-zero code that the batch runner uses to **stop early** instead of
plowing through nine more missions that will all fail the same way.

What is checked for one mission's frame batch (default 10 frames):

  * minimum number of usable frames (default ≥ 8);
  * each usable frame has ≥ 1 valid object with a non-empty bbox;
  * no bbox covers more than ``--max-bbox-area-ratio`` of the image;
  * no bbox is smaller than ``--min-bbox-area-px`` pixels;
  * only allowed classes are present (ships, helicopters, airplanes);
  * camera positions inside the mission must be **distinct** — at least
    ``--min-unique-camera-positions`` rounded ``(x, y, z)`` tuples;
  * camera height is at exactly ``fixed_height_m`` ± tolerance;
  * camera pitch is within ± ``--max-pitch-deg``.

A JSON report is written so the runner can append per-mission reports to a
batch-level log.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_CLASSES = {"ships", "helicopters", "airplanes"}


@dataclass
class MissionValidation:
    mission_prefix: str
    frames_seen: int = 0
    usable_frames: int = 0
    invalid_frames: int = 0
    frames_without_bbox: int = 0
    valid_objects: int = 0
    huge_bbox_frames: List[str] = field(default_factory=list)
    tiny_bbox_objects: List[str] = field(default_factory=list)
    foreign_classes: List[str] = field(default_factory=list)
    unique_camera_positions: int = 0
    camera_height_violations: List[str] = field(default_factory=list)
    camera_pitch_violations: List[str] = field(default_factory=list)
    class_counts: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _bbox_area_xyxy(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def validate_mission(
    *,
    dataset_root: Path,
    mission_prefix: str,
    expected_frames: int = 10,
    min_usable_frames: int = 8,
    min_unique_camera_positions: int = 8,
    fixed_height_m: float = 15.0,
    height_tolerance_m: float = 1.0,
    max_pitch_deg: float = 2.0,
    max_bbox_area_ratio: float = 0.60,
    min_bbox_area_px: float = 64.0,
    image_width: int = 2560,
    image_height: int = 1440,
) -> MissionValidation:
    """Walk metadata for one mission's frames and accumulate the report."""
    report = MissionValidation(mission_prefix=mission_prefix)
    metadata_dir = dataset_root / "metadata"
    image_total_px = float(image_width) * float(image_height)

    metadata_files = sorted(metadata_dir.glob(f"{mission_prefix}*.json"))
    report.frames_seen = len(metadata_files)
    if report.frames_seen == 0:
        report.errors.append(f"no metadata files for mission_prefix='{mission_prefix}'")
        return report
    if report.frames_seen != expected_frames:
        report.warnings.append(
            f"frame count mismatch: expected {expected_frames}, found {report.frames_seen}"
        )

    camera_positions: set = set()
    class_counts: Counter[str] = Counter()

    for path in metadata_files:
        try:
            frame = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.errors.append(f"cannot read {path.name}: {exc}")
            continue

        frame_id = str(frame.get("frame_id") or path.stem)

        # Usability
        if not frame.get("usable", False):
            report.invalid_frames += 1
            report.errors.append(
                f"unusable frame {frame_id} reasons={frame.get('invalid_reasons')}"
            )
            continue
        report.usable_frames += 1

        # Camera pose
        camera = frame.get("camera") or {}
        cam_height = float(camera.get("height_asl_m", 0.0))
        if abs(cam_height - fixed_height_m) > height_tolerance_m:
            report.camera_height_violations.append(
                f"{frame_id}: height={cam_height:.2f} m"
            )
            report.errors.append(
                f"camera height out of band ({fixed_height_m}±{height_tolerance_m} m) "
                f"on {frame_id}: {cam_height:.2f}"
            )
        pitch = float(((camera.get("euler_deg") or {}).get("pitch")) or 0.0)
        if abs(pitch) > max_pitch_deg:
            report.camera_pitch_violations.append(
                f"{frame_id}: pitch={pitch:.2f}°"
            )
            report.errors.append(
                f"camera pitch above ±{max_pitch_deg}° on {frame_id}: {pitch:.2f}"
            )

        # Camera position uniqueness — round to the nearest metre so floating
        # noise doesn't masquerade as "new" positions.
        position = camera.get("world_position_m") or {}
        camera_positions.add(
            (round(float(position.get("x", 0.0)), 0),
             round(float(position.get("y", 0.0)), 0),
             round(float(position.get("z", 0.0)), 0))
        )

        # Objects + bboxes
        valid_in_frame = 0
        for obj in frame.get("objects", []):
            class_name = obj.get("class")
            if class_name not in ALLOWED_CLASSES:
                report.foreign_classes.append(f"{frame_id}: {class_name}")
                report.errors.append(f"foreign class '{class_name}' on {frame_id}")
                continue
            if not obj.get("validation", {}).get("valid", False):
                continue
            bbox = obj.get("projection", {}).get("bbox_xyxy_px")
            if not bbox or len(bbox) != 4:
                continue
            area = _bbox_area_xyxy(bbox)
            if area < min_bbox_area_px:
                report.tiny_bbox_objects.append(f"{frame_id}: area={area:.1f}px²")
                report.warnings.append(
                    f"tiny bbox ({area:.1f}px²) on {frame_id} class={class_name}"
                )
                continue
            ratio = area / image_total_px if image_total_px > 0 else 0.0
            if ratio > max_bbox_area_ratio:
                report.huge_bbox_frames.append(
                    f"{frame_id}: ratio={ratio:.2%}"
                )
                report.errors.append(
                    f"bbox covers {ratio:.2%} of image (>{max_bbox_area_ratio:.0%}) "
                    f"on {frame_id} class={class_name}"
                )
                continue
            class_counts[class_name] += 1
            valid_in_frame += 1
            report.valid_objects += 1

        if valid_in_frame == 0:
            report.frames_without_bbox += 1
            report.errors.append(f"no valid bbox on usable frame {frame_id}")

    report.unique_camera_positions = len(camera_positions)
    if report.unique_camera_positions < min_unique_camera_positions:
        report.errors.append(
            f"only {report.unique_camera_positions} unique camera positions; "
            f"expected ≥ {min_unique_camera_positions}"
        )

    if report.usable_frames < min_usable_frames:
        report.errors.append(
            f"only {report.usable_frames}/{report.frames_seen} usable; "
            f"expected ≥ {min_usable_frames}"
        )

    report.class_counts = dict(class_counts)
    return report


def write_report(report: MissionValidation, output: Optional[Path]) -> None:
    payload = {
        "mission_prefix": report.mission_prefix,
        "ok": report.ok,
        "frames_seen": report.frames_seen,
        "usable_frames": report.usable_frames,
        "invalid_frames": report.invalid_frames,
        "frames_without_bbox": report.frames_without_bbox,
        "valid_objects": report.valid_objects,
        "huge_bbox_frames": report.huge_bbox_frames,
        "tiny_bbox_objects": report.tiny_bbox_objects,
        "foreign_classes": report.foreign_classes,
        "unique_camera_positions": report.unique_camera_positions,
        "camera_height_violations": report.camera_height_violations,
        "camera_pitch_violations": report.camera_pitch_violations,
        "class_counts": report.class_counts,
        "errors": report.errors,
        "warnings": report.warnings,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default=str(PROJECT_ROOT / "dataset"))
    p.add_argument("--mission-prefix", required=True,
                   help="Prefix shared by all frames of one mission, e.g. "
                        "caucasus_static_neutral_03_spring_morning_haze_helos")
    p.add_argument("--expected-frames", type=int, default=10)
    p.add_argument("--min-usable-frames", type=int, default=8)
    p.add_argument("--min-unique-camera-positions", type=int, default=8)
    p.add_argument("--fixed-height-m", type=float, default=15.0)
    p.add_argument("--height-tolerance-m", type=float, default=1.0)
    p.add_argument("--max-pitch-deg", type=float, default=2.0)
    # Cargo ships at the 320 m end of the orbit naturally fill ~50 % of
    # a 2560×1440 frame when shot from the 15 m ship-camera angle. 0.60 leaves
    # enough headroom to keep flagging truly broken framings (target dominates
    # the entire canvas).
    p.add_argument("--max-bbox-area-ratio", type=float, default=0.60)
    p.add_argument("--min-bbox-area-px", type=float, default=64.0)
    p.add_argument("--image-width", type=int, default=2560)
    p.add_argument("--image-height", type=int, default=1440)
    p.add_argument("--report", default=None,
                   help="Optional path for the JSON report.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = validate_mission(
        dataset_root=Path(args.dataset_root),
        mission_prefix=args.mission_prefix,
        expected_frames=args.expected_frames,
        min_usable_frames=args.min_usable_frames,
        min_unique_camera_positions=args.min_unique_camera_positions,
        fixed_height_m=args.fixed_height_m,
        height_tolerance_m=args.height_tolerance_m,
        max_pitch_deg=args.max_pitch_deg,
        max_bbox_area_ratio=args.max_bbox_area_ratio,
        min_bbox_area_px=args.min_bbox_area_px,
        image_width=args.image_width,
        image_height=args.image_height,
    )
    write_report(report, Path(args.report) if args.report else None)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
