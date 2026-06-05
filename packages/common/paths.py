"""Path helpers: project root discovery + cross-platform path resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def project_root() -> Path:
    """Pipeline source root (directory containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "pyproject.toml").exists() and (ancestor / "apps").exists():
            return ancestor
    return here.parents[2]


def is_windows_absolute(raw: str) -> bool:
    return len(raw) > 1 and raw[1] == ":"


def resolve_path(raw: str | None, base: Path | None = None) -> Path | None:
    """Resolve a config path to an absolute Path.

    - Returns None for None/empty/"null".
    - Honors Windows-absolute (`D:/...`), POSIX-absolute (`/...`), and relative.
    - Relative paths are resolved against `base` (default: project_root()).
    """
    if raw in (None, "", "null"):
        return None
    if is_windows_absolute(raw) or raw.startswith("/"):
        return Path(raw)
    base = base or project_root()
    return (base / raw).resolve()


def storage_root_from_config(config: Mapping[str, Any]) -> Path:
    """Return storage.root_dir or fall back to project_root()/storage."""
    storage = config.get("storage", {})
    root = storage.get("root_dir")
    resolved = resolve_path(root) if isinstance(root, str) else None
    if resolved is not None:
        return resolved
    return project_root().parent / "storage"
