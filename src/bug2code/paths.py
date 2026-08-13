"""Filesystem anchors. Every other module resolves paths through here."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DOCS_DIR: Path = PROJECT_ROOT / "docs"


def resolve(path: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing and return it."""
    p = resolve(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
