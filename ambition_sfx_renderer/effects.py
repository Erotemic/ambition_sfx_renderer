"""Effect-chain orchestration."""
from __future__ import annotations

from typing import Any

import numpy as np

from ambition_sfx_renderer.audio import hard_clip, peak_normalize

CORE_EFFECTS = {"normalize_peak", "normalize", "clip", "hard_clip"}


def split_core_effects(effects: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pedalboard_effects: list[dict[str, Any]] = []
    core_effects: list[dict[str, Any]] = []
    for spec in effects or []:
        name = str(spec.get("effect") or spec.get("type") or "").lower()
        if name in CORE_EFFECTS:
            core_effects.append(spec)
        else:
            pedalboard_effects.append(spec)
    return pedalboard_effects, core_effects


def apply_effects(audio: np.ndarray, sample_rate: int, effects: list[dict[str, Any]], context: dict[str, Any]) -> np.ndarray:
    pedalboard_effects, core_effects = split_core_effects(effects)
    out = audio
    if pedalboard_effects:
        from ambition_sfx_renderer_gpl.pedalboard_fx import apply_pedalboard

        out = apply_pedalboard(out, sample_rate, pedalboard_effects, context)
    for spec in core_effects:
        name = str(spec.get("effect") or spec.get("type")).lower()
        if name in {"normalize_peak", "normalize"}:
            out = peak_normalize(out, float(spec.get("target_db", -3.0)))
        elif name in {"clip", "hard_clip"}:
            out = hard_clip(out, float(spec.get("limit", 1.0)))
        else:  # pragma: no cover - guarded by split
            raise ValueError(f"unknown core effect: {name}")
    return out.astype(np.float32)
