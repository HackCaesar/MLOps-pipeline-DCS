"""Minimal end-to-end DCS capture pipeline.

MVP flow:
1. Install `Export.lua` and `Hooks.lua` into Saved Games.
2. Optionally launch DCS with a mission.
3. Request a probe snapshot without screenshot.
4. Plan an experimental camera pose at exactly 15 m ASL.
5. Send camera request via `LoSetCameraPosition` path.
6. Request final `snapshot + screenshot` using a shared frame token.
7. Project fallback cuboids into 2D, save `frame.png + frame.json`, YOLO, COCO.

Limits of this MVP are explicit:
- `LoSetCameraPosition` is experimental and needs validation on the target DCS build.
- Sync is stronger than a plain latest-frame approach because it is token-based, but it is
  not yet a proven same-render-tick guarantee.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure the project root is on sys.path when this file is invoked directly
# (``python orchestrator/main.py``) so that sibling packages can be imported.
_PROJECT_ROOT_FOR_SYS_PATH = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_FOR_SYS_PATH) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_SYS_PATH))

from exporters.coco_yolo import DatasetExporter
from validators import VisibleBBoxRefiner

from orchestrator.camera_planner import CameraPlanner
from orchestrator.dcs_exchange import DCSExchange
from orchestrator.json_io import atomic_write_json, load_json, load_yaml_config, utc_now_iso
from orchestrator.metadata_builder import MetadataBuilder
from orchestrator.overlay import render_overlay
from orchestrator.runtime_paths import PROJECT_ROOT, RuntimePaths, resolve_config_path

LOGGER = logging.getLogger("dcs_autodataset")


class PipelineOrchestrator:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.config = load_yaml_config(self.config_path)
        self.paths = RuntimePaths.from_config(self.config)
        self.runtime_cfg = self.config.get("runtime", {})
        self.allowed_classes = list(self.config["generation"]["allowed_classes"])
        self.exporter = DatasetExporter(
            self.allowed_classes,
            allowed_quality_tiers=self.config["pipeline"].get("allowed_quality_tiers_for_training"),
        )
        self.bbox_refiner = VisibleBBoxRefiner(self.config)
        self.model_dims = {
            key.lower(): value
            for key, value in load_json(self.paths.model_dims_path).items()
        }
        self.exchange = DCSExchange(
            paths=self.paths,
            runtime_cfg=self.runtime_cfg,
            image_format=self.config["export"]["image_format"],
            logger=LOGGER,
        )
        self.metadata_builder = MetadataBuilder(
            config=self.config,
            allowed_classes=self.allowed_classes,
            model_dims=self.model_dims,
            logger=LOGGER,
        )
        self.camera_planner = CameraPlanner(
            config=self.config,
            allowed_classes=self.allowed_classes,
            metadata_builder=self.metadata_builder,
        )
        self._configure_logging()

    def _configure_logging(self) -> None:
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        if logging.getLogger().handlers:
            return
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.paths.logs_dir / "pipeline.log", encoding="utf-8"),
            ],
        )

    def prepare_runtime_dirs(self) -> None:
        for path in [
            self.paths.output_root,
            self.paths.images_dir,
            self.paths.metadata_dir,
            self.paths.overlays_dir,
            self.paths.annotations_dir,
            self.paths.yolo_dir,
            self.paths.logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        if os.name == "nt":
            for path in [
                self.paths.scripts_dir,
                self.paths.hooks_dir,
                self.paths.snapshot_dir,
                self.paths.request_file.parent,
                self.paths.screenshot_ack_file.parent,
                self.paths.pause_ack_file.parent,
                self.paths.camera_request_file.parent,
                self.paths.camera_ack_file.parent,
                self.paths.screenshots_dir,
            ]:
                path.mkdir(parents=True, exist_ok=True)

    def install_lua_scripts(self) -> None:
        shutil.copy2(self.paths.project_export_lua, self.paths.installed_export_lua)
        shutil.copy2(self.paths.project_hook_lua, self.paths.installed_hook_lua)

        include_line = 'local lfs=require("lfs"); dofile(lfs.writedir().."Scripts/cv_dataset_export.lua")\n'
        if self.paths.main_export_lua.exists():
            content = self.paths.main_export_lua.read_text(encoding="utf-8", errors="ignore")
            if "cv_dataset_export.lua" not in content:
                with self.paths.main_export_lua.open("a", encoding="utf-8") as handle:
                    if not content.endswith("\n"):
                        handle.write("\n")
                    handle.write(include_line)
        else:
            self.paths.main_export_lua.write_text(include_line, encoding="utf-8")

        LOGGER.info("Lua scripts installed into %s", self.paths.scripts_dir)

    def launch_dcs(self, mission_path: Optional[Path]) -> subprocess.Popen[Any]:
        if os.name != "nt":
            raise RuntimeError("Real DCS launch is supported only from native Windows Python")
        if not self.paths.dcs_exe.exists():
            raise FileNotFoundError(f"DCS executable not found: {self.paths.dcs_exe}")

        if self.runtime_cfg.get("kill_existing_dcs_before_launch", False):
            self._kill_existing_dcs_processes()

        cmd = [str(self.paths.dcs_exe)]
        if mission_path is not None:
            if not mission_path.exists():
                raise FileNotFoundError(f"Mission not found: {mission_path}")
            launch_style = self.runtime_cfg.get("mission_launch_style", "force_start")
            if launch_style == "force_start":
                cmd.extend(["--force_start", str(mission_path)])
            else:
                cmd.append(str(mission_path))

        LOGGER.info("Launching DCS: %s", " ".join(cmd))
        process = subprocess.Popen(cmd, cwd=self.paths.dcs_exe.parent)
        time.sleep(float(self.runtime_cfg.get("launch_wait_s", 20.0)))
        return process

    def _kill_existing_dcs_processes(self) -> None:
        if os.name != "nt":
            return
        LOGGER.info("Closing any existing DCS.exe process before mission launch")
        subprocess.run(
            ["taskkill", "/IM", "DCS.exe", "/F", "/T"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        time.sleep(float(self.runtime_cfg.get("dcs_restart_wait_s", 8.0)))

    def close_dcs(self, process: Optional[subprocess.Popen[Any]]) -> None:
        if self.runtime_cfg.get("kill_dcs_on_exit", False):
            self._kill_existing_dcs_processes()
            return
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=20)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def clear_artifacts(self, frame_id: str, frame_token: str) -> None:
        """Clear both DCS exchange files and frame output files (image, metadata,
        YOLO, overlay, camera request/ack) for ``frame_id``/``frame_token``."""
        self.exchange.clear_capture_exchange(frame_id, frame_token)
        for path in [
            self.paths.camera_request_file,
            self.paths.camera_ack_file,
            self.paths.frame_image_path(frame_id, self.config["export"]["image_format"]),
            self.paths.frame_metadata_path(frame_id),
            self.paths.yolo_dir / f"{frame_id}.txt",
            self.paths.overlay_path(frame_id),
        ]:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                LOGGER.warning("Could not delete stale artifact: %s", path)

    def _probe_scene_and_move_camera(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        probe_token = uuid.uuid4().hex[:8]
        probe_frame_id = f"probe_{probe_token}"
        self.clear_artifacts(probe_frame_id, probe_token)
        self.exchange.request_capture(probe_frame_id, probe_token, capture_screenshot=False)
        probe_snapshot = self.exchange.wait_for_snapshot(probe_token)
        pose_request, primary_target_summary = self.camera_planner.plan_camera_pose(probe_snapshot)
        if pose_request is None:
            LOGGER.warning("No allowed target objects found during probe snapshot")
            return None, None

        camera_request_token = uuid.uuid4().hex[:8]
        self.exchange.request_camera_pose(camera_request_token, pose_request)
        camera_settle_delay_s = float(self.runtime_cfg.get("camera_settle_delay_s", 0.5))
        try:
            camera_ack = self.exchange.wait_for_camera_ack(camera_request_token)
            camera_ack["camera_request_token"] = camera_request_token
            camera_ack["camera_ack_time"] = utc_now_iso()
            camera_ack["camera_settle_delay_s"] = camera_settle_delay_s
            if camera_ack.get("status") == "applied" and camera_settle_delay_s > 0.0:
                time.sleep(camera_settle_delay_s)
        except Exception as exc:
            LOGGER.warning("Camera request failed before ack: %s", exc)
            camera_ack = {"request_token": camera_request_token, "status": "error", "error": str(exc)}
        return primary_target_summary, camera_ack

    def capture_one_frame(
        self,
        frame_id: Optional[str],
        mission_path: Optional[Path],
        skip_camera_move: bool,
    ) -> Dict[str, Any]:
        frame_id = frame_id or f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        frame_token = uuid.uuid4().hex[:8]
        self.clear_artifacts(frame_id, frame_token)

        primary_target_summary = None
        camera_ack = None
        if self.runtime_cfg.get("probe_before_capture", True) and not skip_camera_move:
            primary_target_summary, camera_ack = self._probe_scene_and_move_camera()
            if camera_ack is not None and camera_ack.get("status") != "applied":
                LOGGER.warning("Camera control not confirmed: %s", camera_ack)

        self.exchange.clear_capture_exchange(frame_id, frame_token)
        pause_ack, capture_requested_epoch_s, _capture_requested_at_utc = self.exchange.request_capture(frame_id, frame_token, capture_screenshot=True)
        snapshot = self.exchange.wait_for_snapshot(frame_token)
        image_path, screenshot_ack, screenshot_diagnostics = self.exchange.wait_for_screenshot(
            frame_id,
            frame_token,
            requested_epoch_s=capture_requested_epoch_s,
        )

        frame_metadata = self.metadata_builder.build_frame_metadata(
            frame_id=frame_id,
            snapshot=snapshot,
            screenshot_ack=screenshot_ack,
            mission_path=mission_path,
            primary_target=primary_target_summary,
            camera_ack=camera_ack,
            pause_ack=pause_ack,
            screenshot_diagnostics=screenshot_diagnostics,
        )
        frame_metadata = self.bbox_refiner.refine_frame(image_path, frame_metadata)
        atomic_write_json(self.paths.frame_metadata_path(frame_id), frame_metadata)
        self.exporter.add_frame(frame_metadata)
        self.exporter.write_yolo(frame_metadata, self.paths.yolo_dir / f"{frame_id}.txt")
        self.exporter.write_coco(self.paths.annotations_dir / "_annotations.coco.json")
        if self.config["pipeline"]["save_debug_overlays"]:
            render_overlay(
                image_path=image_path,
                frame_metadata=frame_metadata,
                overlay_path=self.paths.overlay_path(frame_metadata["frame_id"]),
                logger=LOGGER,
            )

        LOGGER.info(
            "Frame captured: %s | usable=%s | valid_objects=%d",
            frame_id,
            frame_metadata["usable"],
            sum(1 for obj in frame_metadata["objects"] if obj["validation"]["valid"]),
        )
        return frame_metadata

    def dry_run_summary(self) -> Dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "output_root": str(self.paths.output_root),
            "strict_sync": self.config["pipeline"]["strict_sync"],
            "pause_first_capture": bool(self.runtime_cfg.get("pause_first_capture", True)),
            "fixed_camera_height_m": self.config["camera"]["fixed_height_m"],
            "allowed_classes": self.allowed_classes,
            "frames_per_scene": self.config["pipeline"]["frames_per_scene"],
            "max_scenes": self.config["pipeline"]["max_scenes"],
        }

    def run(
        self,
        dry_run: bool = False,
        attach: bool = False,
        mission_path: Optional[Path] = None,
        frame_id: Optional[str] = None,
        frames: int = 1,
        install_only: bool = False,
        skip_install: bool = False,
        skip_camera_move: bool = False,
    ) -> None:
        self.prepare_runtime_dirs()
        if dry_run:
            print(json.dumps(self.dry_run_summary(), indent=2, ensure_ascii=False))
            return

        if not skip_install:
            self.install_lua_scripts()
        if install_only:
            LOGGER.info("Lua scripts installed. Exiting because --install-only was requested.")
            return

        launched_process: Optional[subprocess.Popen[Any]] = None
        resolved_mission = mission_path or self.paths.default_mission_path
        if not attach:
            launched_process = self.launch_dcs(resolved_mission)

        try:
            results = []
            for index in range(max(1, frames)):
                current_frame_id = frame_id
                if frames > 1:
                    prefix = frame_id or f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    current_frame_id = f"{prefix}_{index + 1:04d}"
                frame_metadata = self.capture_one_frame(
                    frame_id=current_frame_id,
                    mission_path=resolved_mission,
                    skip_camera_move=skip_camera_move,
                )
                results.append(
                    {
                        "frame_id": frame_metadata["frame_id"],
                        "usable": frame_metadata["usable"],
                        "valid_objects": frame_metadata.get("object_stats", {}).get("valid_objects"),
                        "image": str(self.paths.frame_image_path(frame_metadata["frame_id"], self.config["export"]["image_format"])),
                        "metadata": str(self.paths.frame_metadata_path(frame_metadata["frame_id"])),
                        "yolo": str(self.paths.yolo_dir / f"{frame_metadata['frame_id']}.txt"),
                        "coco": str(self.paths.annotations_dir / "_annotations.coco.json"),
                    }
                )
            print(json.dumps(results[0] if frames == 1 else results, indent=2, ensure_ascii=False))
        finally:
            if launched_process is not None and self.runtime_cfg.get("close_dcs_on_exit", False):
                self.close_dcs(launched_process)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DCS synthetic dataset MVP orchestrator")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config and create runtime folders")
    parser.add_argument("--attach", action="store_true", help="Do not launch DCS, attach to already running mission")
    parser.add_argument("--mission", help="Path to .miz mission file for auto-launch")
    parser.add_argument("--frame-id", help="Optional explicit frame id")
    parser.add_argument("--frames", type=int, default=1, help="Number of frames to capture in one DCS session")
    parser.add_argument("--install-only", action="store_true", help="Only install Lua scripts into Saved Games")
    parser.add_argument("--skip-install", action="store_true", help="Skip Lua installation step")
    parser.add_argument("--skip-camera-move", action="store_true", help="Skip experimental LoSetCameraPosition camera move")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    orchestrator = PipelineOrchestrator(Path(args.config))
    mission_path = resolve_config_path(args.mission) if args.mission else None
    orchestrator.run(
        dry_run=args.dry_run,
        attach=args.attach,
        mission_path=mission_path,
        frame_id=args.frame_id,
        frames=args.frames,
        install_only=args.install_only,
        skip_install=args.skip_install,
        skip_camera_move=args.skip_camera_move,
    )


if __name__ == "__main__":
    main()
