"""Pure TUI view-model tests — must NOT import rich (rich is lazy in render/run_live)."""
from __future__ import annotations

import json
from pathlib import Path

from apps.pipeline import tui
from apps.pipeline.tui import (
    STAGES_ORDER,
    ProgressLine,
    build_view,
    fmt_eta,
    latest_run_id,
    tail_events,
)


def _ev(stage: str, status: str, **kw) -> dict:
    return {"event_schema_version": 1, "stage": stage, "status": status, **kw}


def _states(vm) -> dict[str, str]:
    return {r.name: r.state for r in vm.stages}


def test_stage_sweep_pending_active_done() -> None:
    events = [_ev(s, "success") for s in STAGES_ORDER[:6]]
    status = {"run_id": "r", "status": "running", "current_stage": "build_or_reuse_tile_cache"}
    vm = build_view(status, events)
    by = _states(vm)
    assert by["validate_config"] == "done"
    assert by["build_or_reuse_tile_cache"] == "active"
    assert by["train_yolox"] == "pending"
    assert vm.active_stage == "build_or_reuse_tile_cache"


def test_failed_stage_marks_failed_downstream_pending() -> None:
    status = {"run_id": "r", "status": "failed", "current_stage": "train_yolox"}
    vm = build_view(status, [_ev("train_yolox", "failed")])
    by = _states(vm)
    assert by["train_yolox"] == "failed" and by["evaluate_model"] == "pending"
    assert vm.overall_status == "failed"


def test_failed_without_explicit_event() -> None:
    # run died on the active stage; only overall=failed signals it
    vm = build_view({"run_id": "r", "status": "failed", "current_stage": "train_yolox"}, [])
    assert _states(vm)["train_yolox"] == "failed"


def test_train_progress_remap_and_eta() -> None:
    status = {
        "run_id": "r", "status": "running", "current_stage": "train_yolox",
        "progress": {"epoch": 4, "max_epoch": 10, "iter": 54, "max_iter": 112},
        "last_metrics": {"loss": 3.21, "lr": 0.0008},
    }
    events = [_ev("train", "running", message="Epoch 4/10", payload={"eta_seconds": 440.0})]
    vm = build_view(status, events)
    assert vm.active_stage == "train_yolox"
    assert _states(vm)["train_yolox"] == "active"
    assert vm.progress is not None and vm.progress.eta_seconds == 440.0
    assert vm.progress.format() == "Epoch 4/10 | Iter 54/112 | loss=3.21 | lr=0.0008 | ETA 00:07:20"


def test_progress_drops_missing_segments_and_eta_format() -> None:
    assert ProgressLine(epoch=1, max_epoch=2).format() == "Epoch 1/2"
    assert ProgressLine().format() == ""
    assert fmt_eta(440) == "00:07:20"
    assert fmt_eta(None) == "?"


def test_terminal_success_all_done() -> None:
    events = [_ev(s, "success") for s in STAGES_ORDER]
    status = {"run_id": "r", "status": "success", "current_stage": "update_pipeline_run_status"}
    vm = build_view(status, events)
    assert all(r.state == "done" for r in vm.stages)
    assert vm.overall_status == "success"


def test_last_events_labels_and_oneshot(capsys) -> None:
    events = [_ev("run", "running", message="run started"),
              _ev("train", "running", message="Epoch 1/10")]
    vm = build_view({"run_id": "r", "status": "running", "current_stage": "train_yolox",
                     "dataset_id": "ds", "tile_cache_id": "abc",
                     "model": {"name": "YOLOX-S", "num_classes": 3}}, events, last_n=5)
    assert len(vm.last_events) == 2
    tui.print_oneshot(vm, ascii=True)
    out = capsys.readouterr().out
    assert "Run:        r" in out
    assert "- train: Epoch 1/10" in out          # "train" event relabelled
    assert "[>] train" in out                     # active glyph (ascii) + label


def test_tolerant_tail_and_read(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "events.jsonl").write_text(
        json.dumps(_ev("run", "running")) + "\n"
        + json.dumps(_ev("validate_config", "success")) + "\n"
        + '{"stage": "train", "status": "ru',   # torn final line
        encoding="utf-8")
    evs = tail_events(run, 10)
    assert len(evs) == 2 and evs[-1]["stage"] == "validate_config"
    (run / "status.json").write_text("{ not json", encoding="utf-8")   # half-written
    vm = build_view(tui.read_status(run), evs)
    assert vm.run_id == "?"                       # tolerant: no crash on bad status.json


def test_latest_run_id(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    (runs / "a").mkdir(parents=True)
    (runs / "b").mkdir(parents=True)
    (runs / "a" / "status.json").write_text('{"run_id":"a","updated_at":"2026-06-01T10:00:00Z"}')
    (runs / "b" / "status.json").write_text('{"run_id":"b","updated_at":"2026-06-02T10:00:00Z"}')
    assert latest_run_id(runs) == "b"
    assert latest_run_id(tmp_path / "nope") is None
