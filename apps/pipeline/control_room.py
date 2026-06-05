"""Pipeline Control Room — a Textual live TUI for a DCS→YOLOX run.

This is the rich/interactive view behind ``apps.pipeline.cli tui`` (and ``run
--watch``). It is a pure OBSERVER: every ``interval`` seconds it re-reads the
files written by :mod:`packages.common.status_tracker`
(``status.json`` / ``events.jsonl`` / ``run_manifest.json``) via the readers in
:mod:`apps.pipeline.tui`, builds the shared :class:`~apps.pipeline.tui.ViewModel`,
and repaints. It never mutates run state.

Honesty over decoration: we only show data we actually have. Per-iter loss/lr,
epoch/iter, ETA, stage states and the event stream come from the run files; GPU
util/mem/temp come from ``nvidia-smi`` on the host *if present* (training runs on
the host GPU via Docker), otherwise they read ``n/a``. Throughput is measured
from successive iter readings. Metrics we do not collect are simply omitted.

Importing this module requires ``textual`` (in the ``tui`` extra); callers should
fall back to :func:`apps.pipeline.tui.run_live` when the import fails.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Log, ProgressBar, Sparkline, Static

from apps.pipeline import tui

_TOPO_ICON = {"done": "●", "active": "◉", "pending": "○", "failed": "✖"}
_STATE_LABEL = {"done": "✓ done", "active": "◉ running", "pending": "○ queued", "failed": "✗ failed"}
_STATE_STYLE = {"done": "bold green", "active": "bold cyan", "pending": "dim", "failed": "bold red"}


def nvidia_smi() -> Optional[dict]:
    """Real host GPU telemetry via ``nvidia-smi`` — ``None`` when unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0, check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None
    try:
        util, mem_used, mem_total, temp = (x.strip() for x in line[0].split(","))
        return {"util": float(util), "mem_used_gb": float(mem_used) / 1024.0,
                "mem_total_gb": float(mem_total) / 1024.0, "temp": float(temp)}
    except (ValueError, IndexError):
        return None


def _micro_bar(pct: float, width: int = 14) -> Text:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    bar = "█" * filled + "░" * (width - filled)
    style = "green" if pct >= 100 else "cyan" if pct >= 66 else "yellow" if pct >= 33 else "white"
    return Text(bar, style=style)


def _train_fraction(p) -> Optional[float]:
    """Fraction (0..1) of the train stage done, from epoch/iter — None if unknown."""
    if not p or not all(v is not None for v in (p.epoch, p.max_epoch, p.iter, p.max_iter)) \
            or not p.max_epoch or not p.max_iter:
        return None
    done = (p.epoch - 1) * p.max_iter + p.iter
    return max(0.0, min(1.0, done / (p.max_epoch * p.max_iter)))


class ControlRoomApp(App[None]):
    TITLE = "DCS → YOLOX · Pipeline Control Room"
    SUB_TITLE = "build-cache · train · evaluate · export"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("a", "toggle_autoscroll", "Autoscroll"),
    ]

    CSS = """
    Screen { layout: vertical; }
    Header { dock: top; }
    Footer { dock: bottom; }
    #mission { height: 8; margin: 1 1 0 1; border: round $primary; padding: 1 2; }
    #overall { height: 1; margin: 0 2; }
    #body { height: 1fr; margin: 1 1; }
    #left { width: 62%; padding-right: 1; }
    #right { width: 38%; }
    #topology { height: 8; border: round $accent; padding: 1; }
    #stages { height: 18; border: round $accent; }
    #events { height: 1fr; border: round $success; padding: 0 1; }
    #metrics { height: 15; border: round $accent; padding: 1; }
    #notes { height: 1fr; border: round $warning; padding: 1; }
    .chart-title { height: 1; margin-top: 1; text-style: bold; }
    Sparkline { height: 1; width: 100%; }
    """

    def __init__(self, run_dir: str | Path, *, interval: float = 0.7,
                 exit_on_terminal: bool = False) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.interval = max(0.2, float(interval))
        self.exit_on_terminal = exit_on_terminal
        self.vm: Optional[tui.ViewModel] = None
        self.loss_hist: deque[float] = deque(maxlen=160)
        self.gpu_hist: deque[float] = deque(maxlen=160)
        self.thr_hist: deque[float] = deque(maxlen=160)
        self._events_written = 0
        self._last_iter: Optional[int] = None
        self._last_iter_ts: Optional[float] = None
        self._last_throughput: float = 0.0
        self._gpu: Optional[dict] = None
        self._terminal_seen_at: Optional[float] = None

    # ---- layout ---------------------------------------------------------- #

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="mission")
        yield ProgressBar(total=100, show_eta=False, id="overall")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Static(id="topology")
                yield DataTable(id="stages")
                yield Log(id="events", highlight=True, max_lines=1000, auto_scroll=True)
            with Vertical(id="right"):
                yield Static(id="metrics")
                yield Static("loss", classes="chart-title")
                yield Sparkline([], id="loss_chart")
                yield Static("gpu utilization", classes="chart-title")
                yield Sparkline([], id="gpu_chart")
                yield Static("throughput (it/s)", classes="chart-title")
                yield Sparkline([], id="thr_chart")
                yield Static(id="notes")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#stages", DataTable)
        table.zebra_stripes = True
        table.cursor_type = "row"
        table.add_columns("stage", "state", "%", "progress", "detail")
        self.query_one("#events", Log).write_line(
            f"control room attached to {self.run_dir.name} (poll {self.interval:g}s)")
        self.set_interval(self.interval, self.poll)
        self.poll()

    # ---- polling --------------------------------------------------------- #

    def poll(self) -> None:
        status = tui.read_status(self.run_dir)
        events = tui.tail_events(self.run_dir, n=0)        # all events (full stage-state inference)
        manifest = tui.read_manifest(self.run_dir)
        self.vm = tui.build_view(status, events, manifest=manifest, last_n=0)
        self._gpu = nvidia_smi()
        self._update_histories(status)
        self._stream_events(events)
        self._refresh_ui()

        if self.vm.overall_status in tui._TERMINAL:
            if self._terminal_seen_at is None:
                self._terminal_seen_at = time.monotonic()
                self.query_one("#events", Log).write_line(
                    f"run reached terminal status: {self.vm.overall_status}")
            if self.exit_on_terminal and (time.monotonic() - self._terminal_seen_at) > 2.0:
                self.exit()

    def _update_histories(self, status: dict) -> None:
        lm = status.get("last_metrics") or {}
        loss = lm.get("loss")
        if isinstance(loss, (int, float)):
            self.loss_hist.append(float(loss))

        pr = status.get("progress") or {}
        it = pr.get("iter")
        now = time.monotonic()
        if isinstance(it, int) and self._last_iter is not None and self._last_iter_ts is not None:
            d_it, d_t = it - self._last_iter, now - self._last_iter_ts
            if d_it > 0 and d_t > 0:
                self._last_throughput = d_it / d_t
                self.thr_hist.append(self._last_throughput)
        if isinstance(it, int):
            self._last_iter, self._last_iter_ts = it, now

        if self._gpu is not None:
            self.gpu_hist.append(self._gpu["util"])

    def _stream_events(self, events: list[dict]) -> None:
        log = self.query_one("#events", Log)
        for e in events[self._events_written:]:
            ts = str(e.get("created_at", ""))[-9:-1] or "--:--:--"
            stage = tui.STAGE_LABELS.get(
                tui._EVENT_STAGE_ALIASES.get(str(e.get("stage", "")), str(e.get("stage", ""))),
                str(e.get("stage", "")))
            st = str(e.get("status", ""))
            msg = e.get("message") or ""
            tag = {"failed": "ERROR", "success": "DONE ", "running": "INFO "}.get(st, st.upper()[:5])
            log.write_line(f"[{ts}] {tag} {stage}: {msg}")
        self._events_written = len(events)

    # ---- rendering ------------------------------------------------------- #

    def _refresh_ui(self) -> None:
        vm = self.vm
        assert vm is not None
        self.query_one("#mission", Static).update(self._render_mission())
        self.query_one("#overall", ProgressBar).update(progress=self._overall_pct())
        self.query_one("#topology", Static).update(self._render_topology())
        self._refresh_stage_table()
        self.query_one("#metrics", Static).update(self._render_metrics())
        self.query_one("#notes", Static).update(self._render_notes())
        self.query_one("#loss_chart", Sparkline).data = list(self.loss_hist)
        self.query_one("#gpu_chart", Sparkline).data = list(self.gpu_hist)
        self.query_one("#thr_chart", Sparkline).data = list(self.thr_hist)

    def _overall_pct(self) -> float:
        vm = self.vm
        if not vm or not vm.stages:
            return 0.0
        total = len(vm.stages)
        done = sum(1 for s in vm.stages if s.state == "done")
        frac = 0.0
        if vm.active_stage == "train_yolox":
            tf = _train_fraction(vm.progress)
            if tf is not None:
                frac = tf
        return min(100.0, (done + frac) / total * 100.0)

    def _status_color(self) -> str:
        return {"success": "green", "failed": "red"}.get(self.vm.overall_status, "yellow")

    def _render_mission(self) -> Panel:
        vm = self.vm
        manifest = tui.read_manifest(self.run_dir) or {}
        torch_v = manifest.get("torch_version") or "?"
        cuda_v = manifest.get("cuda_version")
        # Device: prefer the train stage's recorded value (status.model.device); the
        # manifest's cuda_version is null in the orchestrator (it never imports torch).
        dev = (self.vm and getattr(self.vm, "device", None)) or None
        status = tui.read_status(self.run_dir).get("model") or {}
        dev = status.get("device") or dev or ("cuda" if cuda_v else "?")
        eta = "-"
        if vm.progress and vm.progress.eta_seconds is not None:
            eta = tui.fmt_eta(vm.progress.eta_seconds)
        active = tui.STAGE_LABELS.get(vm.active_stage or "", vm.active_stage or "-")
        text = Text()
        text.append("run:    ", style="bold"); text.append(f"{vm.run_id}\n")
        text.append("data:   ", style="bold")
        text.append(f"{vm.dataset_id or '-'}   ")
        text.append("model: ", style="bold")
        text.append(f"{vm.model_name or '-'} · {vm.num_classes if vm.num_classes is not None else '-'} cls · ")
        text.append(f"{dev}", style="green" if str(dev).lower().startswith("cuda") else "yellow")
        text.append(f"   torch {torch_v}\n")
        text.append("status: ", style="bold")
        text.append(vm.overall_status, style=f"bold {self._status_color()}")
        text.append("   stage: ", style="bold"); text.append(f"{active}")
        text.append("   overall: ", style="bold"); text.append(f"{self._overall_pct():.0f}%")
        text.append("   ETA: ", style="bold"); text.append(eta)
        text.append("\ncache:  ", style="bold"); text.append(f"{vm.tile_cache}", style="dim")
        return Panel(text, title="Mission Control", border_style="cyan")

    def _render_topology(self) -> Panel:
        vm = self.vm
        syms = []
        for s in vm.stages:
            icon = _TOPO_ICON.get(s.state, "○")
            sty = _STATE_STYLE.get(s.state, "")
            syms.append(f"[{sty}]{icon}[/] {s.label}")
        half = (len(syms) + 1) // 2
        line1 = "  →  ".join(syms[:half])
        line2 = "  →  ".join(syms[half:])
        legend = "\n[dim]● done   ◉ running   ○ queued   ✖ failed[/dim]"
        return Panel(f"{line1}\n{line2}{legend}", title="Pipeline topology")

    def _refresh_stage_table(self) -> None:
        vm = self.vm
        table = self.query_one("#stages", DataTable)
        table.clear(columns=False)
        last_msg = self._last_message_per_stage()
        rows = []
        for s in vm.stages:
            pct = self._stage_pct(s)
            if s.name == "train_yolox" and s.state == "active" and vm.progress:
                detail = vm.progress.format()
            else:
                detail = last_msg.get(s.name, "")
            rows.append((
                s.label,
                Text(_STATE_LABEL.get(s.state, s.state), style=_STATE_STYLE.get(s.state, "")),
                f"{pct:4.0f}",
                _micro_bar(pct),
                detail,
            ))
        table.add_rows(rows)

    def _stage_pct(self, row) -> float:
        if row.state == "done":
            return 100.0
        if row.state == "active" and row.name == "train_yolox":
            tf = _train_fraction(self.vm.progress)
            return (tf * 100.0) if tf is not None else 0.0
        return 0.0

    def _last_message_per_stage(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for e in tui.tail_events(self.run_dir, n=0):
            cs = tui._EVENT_STAGE_ALIASES.get(str(e.get("stage", "")), str(e.get("stage", "")))
            if cs in tui.STAGES_ORDER and e.get("message"):
                out[cs] = str(e.get("message"))
        return out

    def _render_metrics(self) -> Panel:
        vm = self.vm
        g = Table.grid(expand=True)
        g.add_column(ratio=2); g.add_column(justify="right", ratio=1); g.add_column(ratio=2)
        p = vm.progress
        if p and p.epoch is not None:
            g.add_row("epoch", f"{p.epoch}/{p.max_epoch}", self._dim("training"))
            g.add_row("iter", f"{p.iter}/{p.max_iter}", self._dim("in epoch"))
            if p.loss is not None:
                g.add_row("loss", f"{p.loss:.4f}", self._loss_note())
            if p.lr is not None:
                g.add_row("lr", f"{p.lr:.3e}", self._dim("schedule"))
        else:
            g.add_row("epoch", "-", self._dim("not training"))
        g.add_row("throughput", f"{self._last_throughput:.1f}/s" if self._last_throughput else "-",
                  self._dim("iters, measured"))
        if self._gpu is not None:
            g.add_row("gpu_util", f"{self._gpu['util']:.0f}%", self._gpu_note())
            g.add_row("gpu_mem", f"{self._gpu['mem_used_gb']:.1f}/{self._gpu['mem_total_gb']:.1f} GB", self._dim("host"))
            g.add_row("gpu_temp", f"{self._gpu['temp']:.0f} °C",
                      "[red]hot[/red]" if self._gpu['temp'] > 80 else self._dim("ok"))
        else:
            g.add_row("gpu", "n/a", self._dim("nvidia-smi not found"))
        return Panel(g, title="Live metrics")

    @staticmethod
    def _dim(s: str) -> str:
        return f"[dim]{s}[/dim]"

    def _loss_note(self) -> str:
        h = list(self.loss_hist)
        if len(h) < 5:
            return "[dim]warming[/dim]"
        return "[green]down[/green]" if h[-1] < min(h[-5:-1]) else "[yellow]noisy[/yellow]"

    def _gpu_note(self) -> str:
        if self._gpu is None:
            return ""
        u = self._gpu["util"]
        if self.vm and self.vm.active_stage == "train_yolox" and u < 50:
            return "[yellow]underfed[/yellow]"
        return "[green]busy[/green]" if u > 80 else self._dim("normal")

    def _render_notes(self) -> Panel:
        vm = self.vm
        t = Table.grid(expand=True); t.add_column(ratio=1)
        failed = [s.label for s in vm.stages if s.state == "failed"]
        if vm.overall_status == "failed" or failed:
            t.add_row("[b red]run degraded[/b red]")
            for f in failed:
                t.add_row(f"• stage failed: [red]{f}[/red]")
            t.add_row("[dim]inspect the events log / stage stderr[/dim]")
        elif vm.overall_status == "success":
            t.add_row("[green]pipeline finished[/green]")
            t.add_row("[dim]checkpoints + onnx under artifacts/<run_id>[/dim]")
        else:
            t.add_row("[green]healthy[/green]")
            if self._gpu is None:
                t.add_row("[dim]gpu telemetry off (no nvidia-smi on host)[/dim]")
            else:
                t.add_row(self._dim("monitoring train metrics"))
        return Panel(t, title="Notes", border_style=self._status_color())

    # ---- actions --------------------------------------------------------- #

    def action_toggle_autoscroll(self) -> None:
        log = self.query_one("#events", Log)
        log.auto_scroll = not log.auto_scroll


def run_control_room(run_dir: str | Path, *, interval: float = 0.7,
                     exit_on_terminal: bool = False) -> None:
    """Launch the Control Room TUI for ``run_dir`` (blocks until the user quits)."""
    ControlRoomApp(run_dir, interval=interval, exit_on_terminal=exit_on_terminal).run()


def run_control_room_with_worker(run_dir: str | Path, worker, *, interval: float = 0.7) -> None:
    """``run --watch`` helper: run ``worker()`` (the pipeline) in a daemon thread
    and show the Control Room until the run reaches a terminal status."""
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    ControlRoomApp(run_dir, interval=interval, exit_on_terminal=True).run()
    t.join(timeout=10)
