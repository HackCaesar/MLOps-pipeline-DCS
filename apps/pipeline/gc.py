"""Retention / garbage collection for ``MLOps_storage``.

The planning core (``plan_retention``) is **pure** — it takes plain records and a
policy and returns a deletion plan, so it is unit-tested without SQLite or a
filesystem. The CLI adapter (``collect_state`` + ``apply_plan``) wires it to the
SQLite registry and the real directories.

**Safety:** GC only ever considers ``runs/``, ``cache/tiles/`` and
``datasets/_staging/``. It NEVER touches ``datasets/raw`` (immutable, valuable),
and ``apply_plan`` refuses to delete anything under a ``raw`` path. Default is a
dry-run; deletion requires an explicit ``--apply``.
"""
from __future__ import annotations

import datetime as _dt
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from packages.common.logging_utils import get_logger

LOG = get_logger(__name__)


@dataclass
class RetentionPolicy:
    runs_keep_last: int = 20
    runs_keep_best: int = 5                 # top-K runs by primary metric
    runs_max_age_days: Optional[int] = 90   # None => age is not a keep reason
    tile_caches_keep_last_per_dataset: int = 3
    staging_max_age_hours: int = 24

    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, Any]]) -> "RetentionPolicy":
        c = dict(cfg or {})
        runs = dict(c.get("runs") or {})
        caches = dict(c.get("tile_caches") or {})
        staging = dict(c.get("staging") or {})
        d = cls()
        return cls(
            runs_keep_last=int(runs.get("keep_last", d.runs_keep_last)),
            runs_keep_best=int(runs.get("keep_best", d.runs_keep_best)),
            runs_max_age_days=(None if runs.get("max_age_days", d.runs_max_age_days) is None
                               else int(runs.get("max_age_days", d.runs_max_age_days))),
            tile_caches_keep_last_per_dataset=int(
                caches.get("keep_last_per_dataset", d.tile_caches_keep_last_per_dataset)),
            staging_max_age_hours=int(staging.get("max_age_hours", d.staging_max_age_hours)),
        )


@dataclass
class RunInfo:
    run_id: str
    started_at: Optional[str]
    tile_cache_id: Optional[str] = None
    best_metric: Optional[float] = None
    path: Optional[Path] = None


@dataclass
class CacheInfo:
    tile_cache_id: str
    dataset_id: str
    created_at: Optional[str] = None
    path: Optional[Path] = None


@dataclass
class StagingInfo:
    path: Path
    mtime_epoch: float


@dataclass
class GCItem:
    kind: str            # "run" | "tile_cache" | "staging"
    id: str
    path: Optional[str]
    reason: str


@dataclass
class GCPlan:
    delete: list[GCItem] = field(default_factory=list)
    kept_runs: int = 0
    kept_caches: int = 0

    def to_dict(self) -> dict:
        return {
            "delete": [vars(i) for i in self.delete],
            "kept_runs": self.kept_runs,
            "kept_caches": self.kept_caches,
            "delete_counts": {
                k: sum(1 for i in self.delete if i.kind == k)
                for k in ("run", "tile_cache", "staging")
            },
        }


def _parse_iso(s: Optional[str]) -> Optional[_dt.datetime]:
    if not s:
        return None
    try:
        return _dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)
    except ValueError:
        try:
            d = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
        except ValueError:
            return None


def plan_retention(runs: list[RunInfo], caches: list[CacheInfo],
                   staging: list[StagingInfo], policy: RetentionPolicy,
                   *, now: _dt.datetime) -> GCPlan:
    """Pure: decide which runs / tile caches / staging dirs to delete."""
    # --- runs: keep = last-N  ∪  best-K  ∪  (within max-age) ---
    by_recent = sorted(runs, key=lambda r: r.started_at or "", reverse=True)
    keep: set[str] = {r.run_id for r in by_recent[:max(0, policy.runs_keep_last)]}

    with_metric = sorted((r for r in runs if r.best_metric is not None),
                         key=lambda r: r.best_metric, reverse=True)  # type: ignore[arg-type]
    keep |= {r.run_id for r in with_metric[:max(0, policy.runs_keep_best)]}

    if policy.runs_max_age_days is not None:
        cutoff = now - _dt.timedelta(days=policy.runs_max_age_days)
        for r in runs:
            dt = _parse_iso(r.started_at)
            if dt is not None and dt >= cutoff:
                keep.add(r.run_id)

    plan = GCPlan()
    for r in runs:
        if r.run_id not in keep:
            plan.delete.append(GCItem("run", r.run_id, str(r.path) if r.path else None,
                                      "not in keep-last / keep-best / max-age window"))
    plan.kept_runs = len(runs) - sum(1 for i in plan.delete if i.kind == "run")

    # --- tile caches: keep those referenced by a kept run; of the rest, keep
    #     last-K per dataset (orphan/old caches get pruned) ---
    referenced = {r.tile_cache_id for r in runs if r.run_id in keep and r.tile_cache_id}
    keep_caches: set[str] = set(referenced)
    by_ds: dict[str, list[CacheInfo]] = defaultdict(list)
    for c in caches:
        if c.tile_cache_id not in referenced:
            by_ds[c.dataset_id].append(c)
    for _ds, lst in by_ds.items():
        lst.sort(key=lambda c: c.created_at or "", reverse=True)
        for c in lst[:max(0, policy.tile_caches_keep_last_per_dataset)]:
            keep_caches.add(c.tile_cache_id)
    for c in caches:
        if c.tile_cache_id not in keep_caches:
            reason = ("orphan: no kept run references it"
                      if c.tile_cache_id not in referenced else "superseded")
            plan.delete.append(GCItem("tile_cache", c.tile_cache_id,
                                      str(c.path) if c.path else None, reason))
    plan.kept_caches = len(caches) - sum(1 for i in plan.delete if i.kind == "tile_cache")

    # --- staging: TTL ---
    cutoff_epoch = now.timestamp() - policy.staging_max_age_hours * 3600
    for s in staging:
        if s.mtime_epoch < cutoff_epoch:
            plan.delete.append(GCItem("staging", s.path.name, str(s.path),
                                      f"older than {policy.staging_max_age_hours}h"))
    return plan


def _is_under_raw(path: Path) -> bool:
    parts = path.resolve().parts
    return "raw" in parts and "datasets" in parts


def apply_plan(plan: GCPlan, *, dry_run: bool = True) -> dict:
    """Delete planned dirs (unless dry_run). Refuses anything under datasets/raw."""
    freed = 0
    deleted = 0
    skipped: list[str] = []
    for item in plan.delete:
        if not item.path:
            continue
        p = Path(item.path)
        if _is_under_raw(p):
            skipped.append(f"{item.path} (refused: under datasets/raw)")
            continue
        if not p.exists():
            continue
        size = _dir_size(p)
        if dry_run:
            freed += size
            continue
        try:
            shutil.rmtree(p)
            freed += size
            deleted += 1
        except OSError as exc:
            skipped.append(f"{item.path} ({exc})")
    return {"dry_run": dry_run, "deleted": deleted,
            "freed_bytes_estimate": freed, "skipped": skipped}


def _dir_size(p: Path) -> int:
    total = 0
    if p.is_file():
        return p.stat().st_size
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


# --------------------------------------------------------------------------- #
# SQLite + filesystem adapter (used by the CLI; not needed by the pure tests)
# --------------------------------------------------------------------------- #

def collect_state(store, *, runs_dir: Optional[Path], cache_tiles_dir: Optional[Path],
                  staging_dir: Optional[Path]) -> tuple[list[RunInfo], list[CacheInfo], list[StagingInfo]]:
    runs: list[RunInfo] = []
    for row in store.list_runs():
        best = None
        try:
            metrics = [r.primary_metric_value for r in store.get_evaluation_reports(row.run_id)
                       if r.primary_metric_value is not None]
            best = max(metrics) if metrics else None
        except Exception:
            best = None
        path = (runs_dir / row.run_id) if runs_dir else None
        runs.append(RunInfo(run_id=row.run_id, started_at=row.started_at,
                            tile_cache_id=row.tile_cache_id, best_metric=best,
                            path=path if (path and path.exists()) else None))

    caches: list[CacheInfo] = []
    for row in store.list_tile_caches():
        path = Path(row.storage_path) if row.storage_path else (
            (cache_tiles_dir / row.tile_cache_id) if cache_tiles_dir else None)
        caches.append(CacheInfo(tile_cache_id=row.tile_cache_id, dataset_id=row.dataset_id,
                                created_at=row.created_at,
                                path=path if (path and path.exists()) else None))

    staging: list[StagingInfo] = []
    if staging_dir and staging_dir.is_dir():
        for child in staging_dir.iterdir():
            if child.is_dir():
                staging.append(StagingInfo(path=child, mtime_epoch=child.stat().st_mtime))
    return runs, caches, staging
