"""Pedalboard effect-chain integration.

This adapter only builds actual Pedalboard plugins.  Core renderer effects such
as ``soft_clip`` and ``tone_safety`` are intentionally handled in
``ambition_sfx_renderer.effects``.  If they reach this module anyway, ignore
them instead of raising: that keeps old YAML / old caller paths from crashing,
while the normal ordered router still applies those effects correctly.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ambition_sfx_renderer.audio import ensure_chans_first
from ambition_sfx_renderer.errors import BackendUnavailableError
from ambition_sfx_renderer.paths import resolve_path


def _import_pedalboard():
    try:
        import pedalboard as pb
    except Exception as ex:  # pragma: no cover - depends on local environment
        raise BackendUnavailableError(
            "pedalboard is required for configured effects. Install with `uv pip install pedalboard`."
        ) from ex
    return pb


def _is_core_effect_name(effect: str) -> bool:
    # Keep this duplicated instead of importing effects.CORE_EFFECTS to avoid a
    # circular import when the core router imports this adapter for a chunk.
    return effect in {
        "normalize_peak", "normalize", "clip", "hard_clip", "soft_clip", "saturate",
        "gain", "highpass", "highpass_filter", "hp", "lowpass", "lowpass_filter", "lp",
        "band_reduce", "deharsh", "notch_reduce", "tone_safety", "dc_block", "fade_edges",
    }


def _build_plugin(spec: dict[str, Any], context: dict[str, Any]):
    pb = _import_pedalboard()
    effect = str(spec.get("effect") or spec.get("type") or "").lower().strip()
    if _is_core_effect_name(effect):
        return None
    if effect in {"compressor", "compress"}:
        kwargs = {
            "threshold_db": float(spec.get("threshold_db", -18.0)),
            "ratio": float(spec.get("ratio", 2.5)),
            "attack_ms": float(spec.get("attack_ms", 3.0)),
            "release_ms": float(spec.get("release_ms", 80.0)),
        }
        return pb.Compressor(**kwargs)
    if effect in {"limiter", "limit"}:
        kwargs = {
            "threshold_db": float(spec.get("threshold_db", -2.0)),
            "release_ms": float(spec.get("release_ms", 50.0)),
        }
        return pb.Limiter(**kwargs)
    if effect in {"reverb"}:
        return pb.Reverb(
            room_size=float(spec.get("room_size", 0.10)),
            damping=float(spec.get("damping", 0.65)),
            wet_level=float(spec.get("wet_level", 0.05)),
            dry_level=float(spec.get("dry_level", 1.0)),
            width=float(spec.get("width", 1.0)),
            freeze_mode=float(spec.get("freeze_mode", 0.0)),
        )
    if effect in {"chorus"}:
        plugin = pb.Chorus()
        for key in ("rate_hz", "depth", "centre_delay_ms", "feedback", "mix"):
            if key in spec:
                setattr(plugin, key, float(spec[key]))
        return plugin
    if effect in {"phaser"}:
        plugin = pb.Phaser()
        for key in ("rate_hz", "depth", "centre_frequency_hz", "feedback", "mix"):
            if key in spec:
                setattr(plugin, key, float(spec[key]))
        return plugin
    if effect in {"distortion", "distort"}:
        return pb.Distortion(drive_db=float(spec.get("drive_db", 12.0)))
    if effect in {"delay"}:
        return pb.Delay(
            delay_seconds=float(spec.get("delay_seconds", spec.get("delay_ms", 90.0) / 1000.0)),
            feedback=float(spec.get("feedback", 0.15)),
            mix=float(spec.get("mix", 0.18)),
        )
    if effect in {"bitcrush", "bitcrusher"}:
        return pb.Bitcrush(bit_depth=int(spec.get("bit_depth", 8)))
    if effect in {"pitch_shift", "pitchshift"}:
        return pb.PitchShift(semitones=float(spec.get("semitones", 0.0)))
    if effect in {"vst3", "vst", "audio_unit", "au", "plugin"}:
        plugin_path = resolve_path(spec["path"], base_dir=context.get("base_dir"))
        plugin = pb.load_plugin(str(plugin_path))
        for key, value in dict(spec.get("parameters") or {}).items():
            setattr(plugin, key, value)
        return plugin
    raise ValueError(
        f"unknown pedalboard effect: {effect!r}. "
        "Core effects such as soft_clip/tone_safety should be applied through "
        "ambition_sfx_renderer.effects.apply_effects."
    )


def apply_pedalboard(audio: np.ndarray, sample_rate: int, effects: list[dict[str, Any]], context: dict[str, Any]) -> np.ndarray:
    pb = _import_pedalboard()
    plugins = []
    for spec in effects or []:
        plugin = _build_plugin(spec, context)
        if plugin is not None:
            plugins.append(plugin)
    if not plugins:
        return ensure_chans_first(audio)
    board = pb.Pedalboard(plugins)
    return ensure_chans_first(board(ensure_chans_first(audio), int(sample_rate))).astype(np.float32)
