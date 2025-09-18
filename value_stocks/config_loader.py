"""Configuration loading utilities without external YAML deps."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class AppConfig:
    """Container for configuration values."""

    raw: Dict[str, Any]

    @property
    def execution(self) -> Dict[str, Any]:
        return self.raw.get("execution", {})

    @property
    def data_source(self) -> Dict[str, Any]:
        return self.raw.get("data_source", {})

    @property
    def universe(self) -> Dict[str, Any]:
        return self.raw.get("universe", {})

    @property
    def filters(self) -> Dict[str, Any]:
        return self.raw.get("filters", {})

    @property
    def scores(self) -> Dict[str, Any]:
        return self.raw.get("scores", {})


def load_config(path: str | Path) -> AppConfig:
    """Load configuration from a small YAML subset."""
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    raw = _parse_simple_yaml(lines)
    return AppConfig(raw=raw)


def _parse_simple_yaml(lines: List[str]) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(0, root)]
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.lstrip().partition(":")
        key = key.strip()
        value = value.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value == "":
            new_dict: Dict[str, Any] = {}
            current[key] = new_dict
            stack.append((indent + 2, new_dict))
            continue
        current[key] = _convert_scalar(value)
    return root


def _convert_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except Exception:  # pragma: no cover
            return value
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"\'')
