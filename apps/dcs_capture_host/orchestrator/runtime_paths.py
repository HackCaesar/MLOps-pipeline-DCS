"""Path resolution helpers and the ``RuntimePaths`` dataclass.

This module owns the single source of truth for ``PROJECT_ROOT`` used across
the orchestrator package and materialises every filesystem location the
pipeline touches from a parsed YAML config.

Nothing here talks to DCS or Lua directly; it only translates config strings
into ``Path`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def is_windows_absolute(raw_path: str) -> bool:
    return len(raw_path) > 1 and raw_path[1] == ":"


def resolve_config_path(raw_path: Optional[str]) -> Optional[Path]:
    if raw_path in (None, "", "null"):
        return None
    if is_windows_absolute(raw_path):
        return Path(raw_path)
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    output_root: Path
    images_dir: Path
    metadata_dir: Path
    overlays_dir: Path
    annotations_dir: Path
    yolo_dir: Path
    logs_dir: Path
    project_export_lua: Path
    project_hook_lua: Path
    model_dims_path: Path
    scripts_dir: Path
    hooks_dir: Path
    main_export_lua: Path
    installed_export_lua: Path
    installed_hook_lua: Path
    request_file: Path
    snapshot_dir: Path
    screenshot_ack_file: Path
    pause_ack_file: Path
    camera_request_file: Path
    camera_ack_file: Path
    screenshots_dir: Path
    dcs_exe: Path
    steam_exe: Path
    default_mission_path: Optional[Path]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RuntimePaths":
        output_root = resolve_config_path(config["paths"]["output_root"])
        logs_dir = resolve_config_path(config["paths"]["logs_dir"])
        scripts_dir = resolve_config_path(config["paths"]["scripts_dir"])
        if output_root is None or logs_dir is None or scripts_dir is None:
            raise ValueError("Output/logs/scripts paths must be configured")

        hooks_dir = scripts_dir / "Hooks"
        request_file = resolve_config_path(config["integration"]["request_file"])
        snapshot_dir = resolve_config_path(config["integration"]["snapshot_dir"])
        screenshot_ack_file = resolve_config_path(config["integration"]["screenshot_ack_file"])
        camera_request_file = resolve_config_path(config["integration"]["camera_request_file"])
        camera_ack_file = resolve_config_path(config["integration"]["camera_ack_file"])
        screenshots_dir = resolve_config_path(config["paths"]["screenshots_dir"])
        pause_ack_file = resolve_config_path(config["integration"].get("pause_ack_file"))

        if request_file is None or snapshot_dir is None or screenshot_ack_file is None:
            raise ValueError("Capture integration paths must be configured")
        if camera_request_file is None or camera_ack_file is None or screenshots_dir is None:
            raise ValueError("Camera/screenshot integration paths must be configured")
        if pause_ack_file is None:
            pause_ack_file = request_file.parent / "cv_pause_ack.json"

        return cls(
            project_root=PROJECT_ROOT,
            output_root=output_root,
            images_dir=output_root / "images",
            metadata_dir=output_root / "metadata",
            overlays_dir=output_root / "debug_overlays",
            annotations_dir=output_root / "annotations",
            yolo_dir=output_root / "yolo_labels",
            logs_dir=logs_dir,
            project_export_lua=PROJECT_ROOT / "dcs_export" / "Export.lua",
            project_hook_lua=PROJECT_ROOT / "dcs_export" / "Hooks.lua",
            model_dims_path=PROJECT_ROOT / "projection" / "model_dims.json",
            scripts_dir=scripts_dir,
            hooks_dir=hooks_dir,
            main_export_lua=scripts_dir / "Export.lua",
            installed_export_lua=scripts_dir / "cv_dataset_export.lua",
            installed_hook_lua=hooks_dir / "cv_dataset_hook.lua",
            request_file=request_file,
            snapshot_dir=snapshot_dir,
            screenshot_ack_file=screenshot_ack_file,
            pause_ack_file=pause_ack_file,
            camera_request_file=camera_request_file,
            camera_ack_file=camera_ack_file,
            screenshots_dir=screenshots_dir,
            dcs_exe=resolve_config_path(config["paths"]["dcs_exe"]),
            steam_exe=resolve_config_path(config["paths"]["steam_exe"]),
            default_mission_path=resolve_config_path(config["paths"].get("default_mission_path")),
        )

    def snapshot_path(self, frame_token: str) -> Path:
        return self.snapshot_dir / f"snapshot_{frame_token}.json"

    def frame_image_path(self, frame_id: str, image_format: str) -> Path:
        return self.images_dir / f"{frame_id}.{image_format}"

    def frame_metadata_path(self, frame_id: str) -> Path:
        return self.metadata_dir / f"{frame_id}.json"

    def overlay_path(self, frame_id: str) -> Path:
        return self.overlays_dir / f"{frame_id}_overlay.png"

    def screenshot_name(self, frame_id: str, frame_token: str) -> str:
        return f"{frame_id}_{frame_token}.png"

    def source_screenshot_path(self, frame_id: str, frame_token: str) -> Path:
        return self.screenshots_dir / self.screenshot_name(frame_id, frame_token)
