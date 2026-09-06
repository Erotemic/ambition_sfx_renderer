"""Path helpers."""

from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    """Return the tool repository root."""
    # ambition_sfx_renderer/paths.py -> tool root
    return Path(__file__).resolve().parents[1]


def sounds_root() -> Path:
    return package_root() / "sounds"


def output_root() -> Path:
    return package_root() / "output"


def resolve_path(value: str | Path, *, base_dir: Path | None = None) -> Path:
    """Resolve a path from YAML.

    Resolution order:
        1. absolute path as written
        2. relative to the YAML directory
        3. relative to the tool repository root
    """
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    if base_dir is not None:
        candidate = (base_dir / p).resolve()
        if candidate.exists():
            return candidate
    return (package_root() / p).resolve()
