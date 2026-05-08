"""YAML schema normalization for SFXIR v1."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ambition_sfx_renderer.errors import SchemaError
from ambition_sfx_renderer.paths import sounds_root

SUPPORTED_SCHEMA = "ambition.sfxir.v1"
SCORE_DIRS = ("active", "examples", "archive")


@dataclass(frozen=True)
class CueSpec:
    path: Path
    raw: dict[str, Any]

    @property
    def cue_id(self) -> str:
        return str(self.raw["id"])

    @property
    def duration_seconds(self) -> float:
        if "duration_seconds" in self.raw:
            return float(self.raw["duration_seconds"])
        return float(self.raw.get("duration_ms", 100.0)) / 1000.0

    @property
    def sample_rate(self) -> int:
        return int(self.raw.get("render", {}).get("sample_rate", 48000))

    @property
    def channels(self) -> int:
        return int(self.raw.get("render", {}).get("channels", 2))

    @property
    def seed(self) -> int | None:
        seed = self.raw.get("render", {}).get("seed")
        return None if seed is None else int(seed)

    @property
    def layers(self) -> list[dict[str, Any]]:
        return list(self.raw.get("layers", []))

    @property
    def postprocess(self) -> list[dict[str, Any]]:
        return list(self.raw.get("postprocess", []))


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise SchemaError(f"expected YAML mapping in {path}")
    return data


def validate(data: dict[str, Any], path: Path) -> None:
    schema = data.get("schema")
    if schema != SUPPORTED_SCHEMA:
        raise SchemaError(f"{path}: expected schema {SUPPORTED_SCHEMA!r}, got {schema!r}")
    cue_id = data.get("id")
    if not cue_id or not isinstance(cue_id, str):
        raise SchemaError(f"{path}: field 'id' must be a non-empty string")
    if "duration_ms" not in data and "duration_seconds" not in data:
        raise SchemaError(f"{path}: one of duration_ms or duration_seconds is required")
    layers = data.get("layers")
    if not isinstance(layers, list) or not layers:
        raise SchemaError(f"{path}: field 'layers' must be a non-empty list")
    for idx, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise SchemaError(f"{path}: layer {idx} must be a mapping")
        if not layer.get("kind"):
            raise SchemaError(f"{path}: layer {idx} is missing kind")
        if not layer.get("name"):
            layer["name"] = f"layer_{idx:02d}"


def load_cue(path: Path) -> CueSpec:
    path = Path(path).resolve()
    data = load_yaml(path)
    validate(data, path)
    return CueSpec(path=path, raw=data)


def find_cue(cue: str, *, root: Path | None = None) -> Path | None:
    """Find a cue by id/name or explicit YAML path."""
    root = root or sounds_root()
    p = Path(cue)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p.resolve()
    candidates: list[Path] = []
    for sub in SCORE_DIRS:
        candidates.extend(
            [
                root / sub / f"{cue}.sfx.yaml",
                root / sub / f"{cue}.yaml",
            ]
        )
    # Also allow an unsorted sounds/<cue>.sfx.yaml layout.
    candidates.extend([root / f"{cue}.sfx.yaml", root / f"{cue}.yaml"])
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def iter_cue_files(root: Path | None = None, *, group: str = "active") -> list[Path]:
    root = root or sounds_root()
    d = root / group
    if not d.exists():
        d = root
    paths = {p.resolve() for p in [*d.glob("*.sfx.yaml"), *d.glob("*.yaml")] if p.is_file()}
    return sorted(paths)
