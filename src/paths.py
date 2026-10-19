"""Canonical project-relative path handling shared by runtime components."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(value: str | Path) -> Path:
    """Resolve relative paths from the repository root, not the caller's CWD."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
