"""Terminal monitoring for a pipeline run (Stage 5).

Two layers:

* **PURE** (stdlib only — unit-tested without rich): read ``status.json`` /
  ``events.jsonl`` / ``run_manifest.json``, build a :class:`ViewModel`, format the
  training-progress line, and a plain-text one-shot renderer.
* **RICH** (``rich`` imported lazily): render the ViewModel + a live refresh loop.

It only ever READS the files written by ``packages.common.status_tracker``; it
never mutates run state. ``status`` works without ``rich`` installed.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# Mirror of ``apps.pipeline.run_all.STAGES_ORDER`` — duplicated on purpose so this
# (pure, rich-free) module does NOT import run_all, which pulls sqlite via
# metadata_store. Kept in sync by tests/unit/test_pipeline_tui.py (sqlite-gated).
STAGES_ORDER: tuple[str, ...] = (
    "validate_config",
    "init_metadata_db",
    "create_pipeline_run",
    "dcs_capture",
    "validate_raw_dataset",
    "register_dataset",
    "build_or_reuse_tile_cache",
    "train_yolox",
    "evaluate_model",
    "export_onnx",
    "export_trt_fp16",
    "export_trt_int8",
    "register_artifacts",
    "update_pipeline_run_status",
)

# Display labels (match docs/tui.md). Stages without an entry render as-is.
STAGE_LABELS: dict[str, str] = {
    "train_yolox": "train",
    "evaluate_model": "evaluate",
    "update_pipeline_run_status": "finalize",
}

# Some events carry a progress/lifecycle alias in their ``stage`` field
# (train.py emits stage="train"); map them back to the canonical stage.
_EVENT_STAGE_ALIASES = {"train": "train_yolox"}

_GLYPHS = {"done": "[✓]", "active": "[▶]", "pending": "[ ]", "failed": "[✗]"}
_GLYPHS_ASCII = {"done": "[x]", "active": "[>]", "pending": "[ ]", "failed": "[!]"}
_TERMINAL = {"success", "failed"}


# --------------------------------------------------------------------------- #
# data classes
# --------------------------------------------------------------------------- #

@dataclass
class StageRow:
    name: str            # canonical STAGES_ORDER name
    label: str           # display label
    state: str           # done | active | pending | failed


@dataclass
class ProgressLine:
    epoch: Optional[int] = None
    max_epoch: Optional[int] = None
    iter: Optional[int] = None
    max_iter: Optional[int] = None
    loss: Optional[float] = None
    lr: Optional[float] = None
    eta_seconds: Optional[float] = None

    def format(self) -> str:
        parts: list[str] = []
        if self.epoch is not None and self.max_epoch is not None:
            parts.append(f"Epoch {self.epoch}/{self.max_epoch}")
        if self.iter is not None and self.max_iter is not None:
            parts.append(f"Iter {self.iter}/{self.max_iter}")
        if self.loss is not None:
            parts.append(f"loss={self.loss:.2f}")
        if self.lr is not None:
            parts.append(f"lr={self.lr:.4g}")
        if self.eta_seconds is not None:
            parts.append(f"ETA {fmt_eta(self.eta_seconds)}")
        return " | ".join(parts)


@dataclass
class ViewModel:
    run_id: str
    overall_status: str
    dataset_id: Optional[str] = None
    tile_cache: str = "-"
    model_name: Optional[str] = None
    num_classes: Optional[int] = None
    device: Optional[str] = None
    stages: list[StageRow] = field(default_factory=list)
    active_stage: Optional[str] = None
    progress: Optional[ProgressLine] = None
    last_events: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# pure readers (tolerant of concurrent writers / torn lines)
# --------------------------------------------------------------------------- #

def read_status(run_dir: str | Path) -> dict:
    p = Path(run_dir) / "status.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def read_manifest(run_dir: str | Path) -> Optional[dict]:
    p = Path(run_dir) / "run_manifest.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def tail_events(run_dir: str | Path, n: int = 8) -> list[dict]:
    p = Path(run_dir) / "events.jsonl"
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return []
    out: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))   # skip a torn final line rather than crash
        except json.JSONDecodeError:
            continue
    return out[-n:] if n and n > 0 else out


def latest_run_id(runs_dir: str | Path) -> Optional[str]:
    d = Path(runs_dir)
    if not d.is_dir():
        return None
    cands: list[tuple[str, float, str]] = []
    for child in d.iterdir():
        if child.is_dir() and (child / "status.json").is_file():
            updated = str(read_status(child).get("updated_at") or "")
            try:
                mtime = child.stat().st_mtime
            except OSError:
                mtime = 0.0
            cands.append((updated, mtime, child.name))
    if not cands:
        return None
    cands.sort(key=lambda c: (c[0], c[1], c[2]))
    return cands[-1][2]


def fmt_eta(secs) -> str:
    try:
        s = max(0, int(round(float(secs))))
    except (TypeError, ValueError):
        return "?"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# --------------------------------------------------------------------------- #
# pure view-model builder
# --------------------------------------------------------------------------- #

def _canonical_stage(stage: str) -> Optional[str]:
    s = _EVENT_STAGE_ALIASES.get(stage, stage)
    return s if s in STAGES_ORDER else None


def build_view(status: Optional[dict], events: Optional[list], stages: Sequence[str] = STAGES_ORDER,
               *, last_n: int = 8, manifest: Optional[dict] = None) -> ViewModel:
    status = status or {}
    events = events or []
    overall = status.get("status", "running")
    current = status.get("current_stage")

    latest_by_stage: dict[str, str] = {}
    for e in events:
        cs = _canonical_stage(str(e.get("stage", "")))
        st = e.get("status")
        if cs is not None and st:
            latest_by_stage[cs] = st

    current_index = stages.index(current) if current in stages else None

    rows: list[StageRow] = []
    for i, name in enumerate(stages):
        ev = latest_by_stage.get(name)
        if ev == "failed":
            state = "failed"
        elif ev == "success":
            state = "done"
        elif name == current:
            state = ("failed" if overall == "failed"
                     else "done" if overall == "success" else "active")
        elif ev == "running" and overall == "running":
            state = "active"   # a stale 'running' progress event must not override a finished run
        elif current_index is not None and i < current_index:
            state = "done"                       # passed, even with no terminal event
        else:
            state = "pending"
        rows.append(StageRow(name, STAGE_LABELS.get(name, name), state))

    active_stage = current if current in stages else None

    progress: Optional[ProgressLine] = None
    if active_stage == "train_yolox":
        pr = status.get("progress") or {}
        lm = status.get("last_metrics") or {}
        eta = None
        for e in reversed(events):
            if _canonical_stage(str(e.get("stage", ""))) == "train_yolox":
                val = (e.get("payload") or {}).get("eta_seconds")
                if val is not None:
                    eta = val
                    break
        progress = ProgressLine(
            epoch=pr.get("epoch"), max_epoch=pr.get("max_epoch"),
            iter=pr.get("iter"), max_iter=pr.get("max_iter"),
            loss=lm.get("loss"), lr=lm.get("lr"), eta_seconds=eta,
        )

    model = status.get("model") or {}
    device = None
    if manifest:
        device = ("CUDA" if manifest.get("cuda_version")
                  else "CPU" if manifest.get("torch_version") else None)

    return ViewModel(
        run_id=str(status.get("run_id", "?")),
        overall_status=overall,
        dataset_id=status.get("dataset_id"),
        tile_cache=status.get("tile_cache_id") or "-",
        model_name=model.get("name"),
        num_classes=model.get("num_classes"),
        device=device,
        stages=rows,
        active_stage=active_stage,
        progress=progress,
        last_events=list(events)[-last_n:] if last_n and last_n > 0 else list(events),
    )


# --------------------------------------------------------------------------- #
# plain-text one-shot (NO rich) + ascii detection
# --------------------------------------------------------------------------- #

def want_ascii(stream=None) -> bool:
    """True if the stream can't encode the unicode glyphs (e.g. cp1251 on Windows)."""
    import sys
    stream = stream or sys.stdout
    enc = getattr(stream, "encoding", None) or "ascii"
    try:
        "✓▶✗".encode(enc)
        return False
    except (UnicodeEncodeError, LookupError):
        return True


def _event_line(e: dict) -> str:
    cs = _EVENT_STAGE_ALIASES.get(str(e.get("stage", "")), str(e.get("stage", "")))
    label = STAGE_LABELS.get(cs, cs)
    msg = e.get("message") or e.get("status") or ""
    return f"- {label}: {msg}"


def render_lines(vm: ViewModel, *, ascii: bool = False) -> list[str]:
    """The full one-shot view as a list of plain text lines (pure, testable)."""
    g = _GLYPHS_ASCII if ascii else _GLYPHS
    lines = [
        "DCS → YOLOX Pipeline" if not ascii else "DCS -> YOLOX Pipeline",
        "",
        f"Run:        {vm.run_id}",
        f"Dataset:    {vm.dataset_id or '-'}",
        f"Tile cache: {vm.tile_cache}",
        f"Model:      {vm.model_name or '-'}   "
        f"Classes: {vm.num_classes if vm.num_classes is not None else '-'}   "
        f"Device: {vm.device or '?'}",
        f"Status:     {vm.overall_status}",
        "",
    ]
    for row in vm.stages:
        line = f"{g[row.state]} {row.label}"
        if row.name == "train_yolox" and row.state == "active" and vm.progress:
            pf = vm.progress.format()
            if pf:
                line += f"    {pf}"
        lines.append(line)
    if vm.last_events:
        lines += ["", "Last events:"] + [_event_line(e) for e in vm.last_events]
    return lines


def print_oneshot(vm: ViewModel, *, ascii: bool = False, stream=None) -> None:
    import sys
    out = stream or sys.stdout
    print("\n".join(render_lines(vm, ascii=ascii)), file=out)


# --------------------------------------------------------------------------- #
# rich layer (lazy imports)
# --------------------------------------------------------------------------- #

def render(vm: ViewModel, *, ascii: bool = False):
    """Build a rich renderable for the ViewModel (rich imported lazily)."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    g = _GLYPHS_ASCII if ascii else _GLYPHS
    style = {"done": "green", "active": "yellow", "pending": "dim", "failed": "red"}

    header = Text()
    header.append(f"Run:        {vm.run_id}\n")
    header.append(f"Dataset:    {vm.dataset_id or '-'}\n")
    header.append(f"Tile cache: {vm.tile_cache}\n")
    header.append(f"Model:      {vm.model_name or '-'}   "
                  f"Classes: {vm.num_classes if vm.num_classes is not None else '-'}   "
                  f"Device: {vm.device or '?'}\n")
    header.append("Status:     ")
    header.append(vm.overall_status,
                  style=("green" if vm.overall_status == "success"
                         else "red" if vm.overall_status == "failed" else "yellow"))

    checklist = Text()
    for row in vm.stages:
        checklist.append(f"{g[row.state]} ", style=style.get(row.state, ""))
        line = row.label
        if row.name == "train_yolox" and row.state == "active" and vm.progress:
            pf = vm.progress.format()
            if pf:
                line += f"    {pf}"
        checklist.append(line + "\n", style=style.get(row.state, ""))

    blocks = [header, Text("\n"), checklist]
    if vm.last_events:
        ev = Text("\nLast events:\n", style="bold")
        for e in vm.last_events:
            ev.append(_event_line(e) + "\n", style="dim")
        blocks.append(ev)

    title = "DCS → YOLOX Pipeline" if not ascii else "DCS -> YOLOX Pipeline"
    return Panel(Group(*blocks), title=title, border_style="cyan")


def run_live(run_dir: str | Path, *, interval: float = 0.7, stop_event=None,
             ascii: bool = False, stages: Sequence[str] = STAGES_ORDER) -> str:
    """Live-refresh the TUI until the run reaches a terminal status (or Ctrl-C).

    Falls back to a single one-shot snapshot if ``rich`` is not installed.
    Returns the last observed overall status.
    """
    run_dir = Path(run_dir)

    def _snapshot() -> ViewModel:
        return build_view(read_status(run_dir), tail_events(run_dir), stages,
                          manifest=read_manifest(run_dir))

    try:
        from rich.console import Console
        from rich.live import Live
    except ImportError:
        print("rich is not installed; showing a one-shot snapshot. "
              "For the live view: pip install '.[tui]'")
        print_oneshot(_snapshot(), ascii=ascii)
        return read_status(run_dir).get("status", "unknown")

    last = "running"
    try:
        with Live(console=Console(), refresh_per_second=4, screen=False) as live:
            while True:
                vm = _snapshot()
                last = vm.overall_status
                live.update(render(vm, ascii=ascii))
                if last in _TERMINAL:
                    break
                if stop_event is not None and stop_event.is_set():
                    vm = _snapshot()                       # worker done — final paint
                    last = vm.overall_status
                    live.update(render(vm, ascii=ascii))
                    break
                time.sleep(interval)
    except KeyboardInterrupt:
        print("detached (run continues in the background).")
    return last
