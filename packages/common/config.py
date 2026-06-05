"""YAML config loader with ${section.key} interpolation."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import yaml

_VAR_PATTERN = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_.]*)\}")


class ConfigError(Exception):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve ${section.key} references inside string values.

    Resolver is iterative: stops when no substitution is made in a pass or when
    a circular reference is detected (>50 passes).
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")
    return resolve_vars(raw)


def resolve_vars(config: Mapping[str, Any], max_passes: int = 50) -> dict[str, Any]:
    """Repeatedly substitute ${path.to.key} references until fixpoint."""
    result: Any = dict(config)
    for _ in range(max_passes):
        new, changed = _substitute_pass(result, result)
        if not changed:
            return new
        result = new
    raise ConfigError("Variable interpolation did not converge (possible circular reference)")


def lookup_path(config: Mapping[str, Any], dotted_path: str) -> Any:
    """Look up `a.b.c` in nested dict. Raises KeyError if missing."""
    parts = dotted_path.split(".")
    node: Any = config
    for part in parts:
        if not isinstance(node, Mapping):
            raise KeyError(f"Cannot descend into {part!r} (parent is not a mapping)")
        if part not in node:
            raise KeyError(f"Missing key in config: {dotted_path}")
        node = node[part]
    return node


def _substitute_pass(node: Any, root: Mapping[str, Any]) -> tuple[Any, bool]:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        changed_any = False
        for k, v in node.items():
            new_v, c = _substitute_pass(v, root)
            out[k] = new_v
            changed_any = changed_any or c
        return out, changed_any
    if isinstance(node, list):
        new_list = []
        changed_any = False
        for item in node:
            new_item, c = _substitute_pass(item, root)
            new_list.append(new_item)
            changed_any = changed_any or c
        return new_list, changed_any
    if isinstance(node, str):
        return _substitute_string(node, root)
    return node, False


def _substitute_string(s: str, root: Mapping[str, Any]) -> tuple[Any, bool]:
    if "${" not in s:
        return s, False

    match = _VAR_PATTERN.fullmatch(s)
    if match is not None:
        # Whole-string reference: preserve original type (int/list/dict/None).
        try:
            value = lookup_path(root, match.group(1))
        except KeyError as exc:
            raise ConfigError(str(exc)) from None
        if isinstance(value, str) and "${" in value:
            return value, True
        return value, True

    def _repl(m: re.Match[str]) -> str:
        try:
            value = lookup_path(root, m.group(1))
        except KeyError as exc:
            raise ConfigError(str(exc)) from None
        if value is None:
            return ""
        return str(value)

    new_s = _VAR_PATTERN.sub(_repl, s)
    return new_s, new_s != s
