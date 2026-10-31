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
    def loop(self) -> dict[str, float] | None:
        """A seamless loop (``loop:`` in the YAML), or None.

        ``duration_ms`` is then the loop's length, the part that plays and
        repeats. The layers are laid out on a longer render: ``preroll_ms``
        before the loop (layers settle in), ``crossfade_ms`` after it (folded
        into its head so its end runs on into its start) and ``postroll_ms``
        after that (layer releases, never heard). Layer ``start_ms`` counts
        from the start of that render.
        """
        raw = self.raw.get("loop")
        if raw is None:
            return None
        return {
            "preroll_ms": float(raw.get("preroll_ms", 0.0)),
            "crossfade_ms": float(raw["crossfade_ms"]),
            "postroll_ms": float(raw.get("postroll_ms", 0.0)),
            "normalize_peak_db": None if raw.get("normalize_peak_db") is None else float(raw["normalize_peak_db"]),
        }

    @property
    def render_seconds(self) -> float:
        """Length of the rendered timeline: the cue, or a loop's whole render."""
        loop = self.loop
        if loop is None:
            return self.duration_seconds
        return self.duration_seconds + (loop["preroll_ms"] + loop["crossfade_ms"] + loop["postroll_ms"]) / 1000.0

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
    loop = data.get("loop")
    if loop is not None:
        if not isinstance(loop, dict) or "crossfade_ms" not in loop:
            raise SchemaError(f"{path}: 'loop' must be a mapping with crossfade_ms")
        unknown = set(loop) - {"preroll_ms", "crossfade_ms", "postroll_ms", "normalize_peak_db"}
        if unknown:
            raise SchemaError(f"{path}: 'loop' has unknown keys {sorted(unknown)}")
        length_ms = float(data.get("duration_ms", float(data.get("duration_seconds", 0.0)) * 1000.0))
        crossfade = float(loop["crossfade_ms"])
        if not 0.0 < crossfade <= length_ms:
            raise SchemaError(f"{path}: loop.crossfade_ms must be > 0 and <= the loop's duration")
        for key in ("preroll_ms", "postroll_ms"):
            if float(loop.get(key, 0.0)) < 0.0:
                raise SchemaError(f"{path}: loop.{key} must be >= 0")
    layers = data.get("layers")
    if not isinstance(layers, list) or not layers:
        raise SchemaError(f"{path}: field 'layers' must be a non-empty list")
    for idx, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise SchemaError(f"{path}: layer {idx} must be a mapping")
        if "filter" in layer:
            raise SchemaError(f"{path}: layer {idx} uses unsupported filter; use an effects chain")
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
    """Find a cue by explicit YAML path, canonical namespaced filename, or YAML id.

    Canonical project cues should use namespaced filenames such as
    ``sounds/active/player.blink.sfx.yaml`` with matching ``id: player.blink``.
    """
    root = root or sounds_root()
    p = Path(cue)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p.resolve()

    candidates: list[Path] = []
    for sub in SCORE_DIRS:
        candidates.extend([root / sub / f"{cue}.sfx.yaml", root / sub / f"{cue}.yaml"])
    candidates.extend([root / f"{cue}.sfx.yaml", root / f"{cue}.yaml"])
    for c in candidates:
        if c.exists():
            return c.resolve()

    for group in SCORE_DIRS:
        for c in iter_cue_files(root, group=group):
            try:
                spec = load_cue(c)
            except Exception:
                continue
            if spec.cue_id == cue:
                return c.resolve()
    return None


def iter_cue_files(root: Path | None = None, *, group: str = "active") -> list[Path]:
    """Return cue YAML files for a group.

    Discovery is recursive so future actor folders such as
    ``sounds/active/player/jump.sfx.yaml`` work without another CLI change.
    """
    root = root or sounds_root()
    d = root / group
    if not d.exists():
        d = root
    paths = {p.resolve() for p in [*d.rglob("*.sfx.yaml"), *d.rglob("*.yaml")] if p.is_file()}
    seen: dict[str, Path] = {}
    for path in sorted(paths):
        cue_id = load_cue(path).cue_id
        if cue_id in seen:
            raise SchemaError(f"duplicate cue ID {cue_id!r}: {seen[cue_id]} and {path}")
        seen[cue_id] = path
    return sorted(paths)
