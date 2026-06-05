"""Pure retention-planning tests (no SQLite / no filesystem state)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from apps.pipeline.gc import (
    CacheInfo,
    GCItem,
    GCPlan,
    RetentionPolicy,
    RunInfo,
    StagingInfo,
    apply_plan,
    plan_retention,
)

NOW = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)


def _iso(days_ago: int) -> str:
    return (NOW - dt.timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _runs(n: int) -> list[RunInfo]:
    # r0 = oldest ... r{n-1} = newest
    return [RunInfo(run_id=f"r{i}", started_at=_iso(n - i), tile_cache_id=f"c{i}") for i in range(n)]


def _deleted(plan: GCPlan, kind: str) -> set[str]:
    return {i.id for i in plan.delete if i.kind == kind}


def test_keep_last_n() -> None:
    pol = RetentionPolicy(runs_keep_last=3, runs_keep_best=0, runs_max_age_days=None,
                          tile_caches_keep_last_per_dataset=0)
    plan = plan_retention(_runs(10), [], [], pol, now=NOW)
    assert _deleted(plan, "run") == {f"r{i}" for i in range(7)}   # keep r7,r8,r9
    assert plan.kept_runs == 3


def test_keep_best_by_metric() -> None:
    pol = RetentionPolicy(runs_keep_last=1, runs_keep_best=2, runs_max_age_days=None,
                          tile_caches_keep_last_per_dataset=0)
    runs = _runs(5)
    runs[0].best_metric = 0.9   # oldest but best
    runs[1].best_metric = 0.8
    runs[2].best_metric = 0.1
    kept = {r.run_id for r in runs} - _deleted(plan_retention(runs, [], [], pol, now=NOW), "run")
    assert {"r4", "r0", "r1"} <= kept       # last + best-2


def test_max_age_keeps_recent() -> None:
    pol = RetentionPolicy(runs_keep_last=0, runs_keep_best=0, runs_max_age_days=30,
                          tile_caches_keep_last_per_dataset=0)
    runs = [RunInfo("old", _iso(100)), RunInfo("recent", _iso(5))]
    assert _deleted(plan_retention(runs, [], [], pol, now=NOW), "run") == {"old"}


def test_orphan_caches_pruned_referenced_kept() -> None:
    pol = RetentionPolicy(runs_keep_last=1, runs_keep_best=0, runs_max_age_days=None,
                          tile_caches_keep_last_per_dataset=0)
    runs = _runs(2)                                   # keep r1 -> references c1
    caches = [CacheInfo("c0", "ds"), CacheInfo("c1", "ds"), CacheInfo("corphan", "ds")]
    dc = _deleted(plan_retention(runs, caches, [], pol, now=NOW), "tile_cache")
    assert "c1" not in dc                              # referenced by a kept run
    assert {"c0", "corphan"} <= dc                     # unreferenced, keep_last_per_dataset=0


def test_keep_last_per_dataset() -> None:
    pol = RetentionPolicy(runs_keep_last=0, runs_keep_best=0, runs_max_age_days=None,
                          tile_caches_keep_last_per_dataset=2)
    caches = [CacheInfo(f"c{i}", "ds", created_at=_iso(10 - i)) for i in range(5)]  # c4 newest
    assert _deleted(plan_retention([], caches, [], pol, now=NOW), "tile_cache") == {"c0", "c1", "c2"}


def test_staging_ttl() -> None:
    pol = RetentionPolicy(staging_max_age_hours=24)
    staging = [
        StagingInfo(Path("/x/_staging/old"), (NOW - dt.timedelta(hours=48)).timestamp()),
        StagingInfo(Path("/x/_staging/fresh"), (NOW - dt.timedelta(hours=1)).timestamp()),
    ]
    assert _deleted(plan_retention([], [], staging, pol, now=NOW), "staging") == {"old"}


def test_apply_refuses_datasets_raw(tmp_path: Path) -> None:
    raw = tmp_path / "datasets" / "raw" / "ds"
    raw.mkdir(parents=True)
    plan = GCPlan(delete=[GCItem("run", "x", str(raw), "should be refused")])
    res = apply_plan(plan, dry_run=False)
    assert raw.exists()                                # NEVER delete raw
    assert any("refused" in s for s in res["skipped"])


def test_apply_dry_run_deletes_nothing(tmp_path: Path) -> None:
    d = tmp_path / "runs" / "r1"
    d.mkdir(parents=True)
    (d / "f.txt").write_text("x")
    plan = GCPlan(delete=[GCItem("run", "r1", str(d), "old")])
    res = apply_plan(plan, dry_run=True)
    assert d.exists() and res["deleted"] == 0 and res["freed_bytes_estimate"] >= 1
