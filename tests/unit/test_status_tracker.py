"""StatusTracker file-state behavior (status.json + events.jsonl), SQLite-free.

(The SQLite mirror is lazy/optional; these tests run without _sqlite3.)
"""
from __future__ import annotations

import json
from pathlib import Path

from packages.common.status_tracker import ModelInfo, StatusTracker


def _tr(tmp_path: Path, run_id="r1", **kw) -> StatusTracker:
    return StatusTracker(run_id, tmp_path / "runs", db_path=None, stage_total=kw.pop("stage_total", 3), **kw)


def _status(tmp_path: Path, run_id="r1") -> dict:
    return json.loads((tmp_path / "runs" / run_id / "status.json").read_text())


def _events(tmp_path: Path, run_id="r1") -> list[dict]:
    text = (tmp_path / "runs" / run_id / "events.jsonl").read_text().strip()
    return [json.loads(l) for l in text.splitlines()] if text else []


def test_init_run_seeds_status_event_and_snapshot(tmp_path: Path) -> None:
    tr = _tr(tmp_path)
    tr.init_run(config={"pipeline": {"run_id": "r1"}, "storage": {"x": "y"}},
                dataset_id="ds",
                model=ModelInfo(name="YOLOX-S", num_classes=3, input_size=[640, 640], pretrained=None))
    run_dir = tmp_path / "runs" / "r1"
    assert (run_dir / "config_snapshot.yaml").is_file()
    s = _status(tmp_path)
    assert s["run_id"] == "r1" and s["status"] == "running" and s["dataset_id"] == "ds"
    assert s["model"]["name"] == "YOLOX-S" and s["model"]["num_classes"] == 3
    assert s["progress"]["stage_total"] == 3
    assert s["updated_at"].endswith("Z")
    ev = _events(tmp_path)
    assert len(ev) == 1 and ev[0]["event_schema_version"] == 1
    assert ev[0]["stage"] == "run" and ev[0]["status"] == "running"
    assert ev[0]["created_at"].endswith("Z")


def test_training_progress_does_not_clobber_run_all_fields(tmp_path: Path) -> None:
    tr = _tr(tmp_path)
    tr.init_run(dataset_id="ds")
    tr.set_tile_cache("abc__def__ghi")
    tr.set_current_stage("train", stage_index=4, stage_total=10)
    tr.emit_stage("train", "running", current=4, total=10, message="Epoch 1/10")
    tr.record_training_progress(epoch=1, max_epoch=10, iter=5, max_iter=112, loss=3.21, lr=0.0008)

    s = _status(tmp_path)
    assert s["tile_cache_id"] == "abc__def__ghi"      # NOT clobbered by progress write
    assert s["dataset_id"] == "ds"                    # NOT clobbered
    assert s["progress"]["stage_total"] == 10         # preserved from set_current_stage
    assert s["progress"]["epoch"] == 1 and s["progress"]["remaining_epochs"] == 9
    assert s["progress"]["iter"] == 5 and s["progress"]["max_iter"] == 112
    assert s["last_metrics"] == {"loss": 3.21, "lr": 0.0008}


def test_progress_only_writes_available_fields(tmp_path: Path) -> None:
    tr = _tr(tmp_path)
    tr.init_run()
    tr.record_training_progress(epoch=2, max_epoch=10)   # no loss/lr available
    s = _status(tmp_path)
    assert s["progress"]["epoch"] == 2
    assert s["last_metrics"] == {"loss": None, "lr": None}   # seed untouched, nothing fabricated


def test_events_are_append_only(tmp_path: Path) -> None:
    tr = _tr(tmp_path)
    tr.init_run()
    tr.emit_stage("validate_raw_dataset", "running", current=0, total=3)
    tr.emit_stage("validate_raw_dataset", "success", current=1, total=3)
    ev = _events(tmp_path)
    assert [e["stage"] for e in ev] == ["run", "validate_raw_dataset", "validate_raw_dataset"]
    assert [e["status"] for e in ev] == ["running", "running", "success"]


def test_finish_success_and_failed(tmp_path: Path) -> None:
    tr = _tr(tmp_path)
    tr.init_run()
    tr.finish(status="success")
    assert _status(tmp_path)["status"] == "success"
    assert _events(tmp_path)[-1] == {**_events(tmp_path)[-1], "stage": "finalize", "status": "success"}

    tr2 = _tr(tmp_path, run_id="r2")
    tr2.init_run()
    tr2.finish(status="failed", error="train exited 2")
    s2 = _status(tmp_path, "r2")
    assert s2["status"] == "failed"
    last = _events(tmp_path, "r2")[-1]
    assert last["stage"] == "finalize" and last["status"] == "failed"
    assert last["payload"]["error"] == "train exited 2"


def test_should_log_iter_throttle(tmp_path: Path) -> None:
    tr = _tr(tmp_path, train_log_interval_iters=10, train_log_interval_seconds=5.0)
    assert tr.should_log_iter(10, last_log_time=100.0, now=100.1) is True    # iter stride hit
    assert tr.should_log_iter(7, last_log_time=100.0, now=100.1) is False    # neither
    assert tr.should_log_iter(7, last_log_time=100.0, now=106.0) is True     # 5s floor passed


def test_init_run_writes_provenance_manifest(tmp_path: Path) -> None:
    import hashlib
    tr = _tr(tmp_path)
    cfg = {"pipeline": {"run_id": "r1"}, "enrichment": {"crop_size": 640}}
    tr.init_run(config=cfg, dataset_id="ds")
    man = json.loads((tmp_path / "runs" / "r1" / "run_manifest.json").read_text())
    assert man["manifest_schema_version"] == 1
    assert man["run_id"] == "r1"
    assert man["python_version"] and man["created_at"].endswith("Z")
    # provenance keys present (values may be None when a probe is unavailable)
    for k in ("git_sha", "torch_version", "cuda_version", "yolox_version",
              "package_version", "tiling_algorithm_version"):
        assert k in man
    expected = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert man["config_sha256"] == expected
