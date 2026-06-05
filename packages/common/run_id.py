"""Generate deterministic per-day run_id "YYYY_MM_DD_NNN"."""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from typing import Callable, Optional

_RUN_ID_RE = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_(\d{3})$")


def generate_run_id(
    runs_dir: Optional[Path] = None,
    now: Callable[[], _dt.datetime] = _dt.datetime.utcnow,
) -> str:
    """Return next available "YYYY_MM_DD_NNN" id.

    Counter NNN is derived by scanning `runs_dir` for existing run_id-named
    subdirectories matching today's date. If runs_dir is None or doesn't
    exist, NNN starts at 001.
    """
    today = now().strftime("%Y_%m_%d")
    if runs_dir is None or not Path(runs_dir).exists():
        return f"{today}_001"

    used = set()
    for child in Path(runs_dir).iterdir():
        m = _RUN_ID_RE.match(child.name)
        if m and child.is_dir() and f"{m.group(1)}_{m.group(2)}_{m.group(3)}" == today:
            used.add(int(m.group(4)))

    n = 1
    while n in used:
        n += 1
    if n > 999:
        raise RuntimeError(f"Too many runs today ({today}); rotate runs_dir")
    return f"{today}_{n:03d}"


def is_valid_run_id(s: str) -> bool:
    return bool(_RUN_ID_RE.fullmatch(s))


def parse_run_id(s: str) -> tuple[_dt.date, int]:
    m = _RUN_ID_RE.fullmatch(s)
    if not m:
        raise ValueError(f"Not a valid run_id: {s!r}")
    return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))), int(m.group(4))
