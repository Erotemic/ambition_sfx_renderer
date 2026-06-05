"""Effect-chain orchestration with built-in corrective de-harsh processing.

The renderer has two effect families:

* core NumPy/SciPy effects implemented in this package; and
* Pedalboard effects implemented in ``ambition_sfx_renderer_gpl``.

Keep the YAML authoring surface simple by allowing both families in the same
``effects`` / ``postprocess`` list.  The important detail is that effect order
must be preserved.  Earlier versions split the list into all-core then
all-Pedalboard, which was both sonically wrong and could route core-only names
such as ``soft_clip`` into the Pedalboard adapter when a lower-level caller used
that adapter directly.  This module now walks the list in order and only sends
contiguous non-core chunks to Pedalboard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ambition_sfx_renderer.audio import (
    apply_gain,
    ensure_chans_first,
    hard_clip,
)


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def peak_normalize(
    audio: np.ndarray, target_db: float = -3.0, *, only_if_louder: bool = False
) -> np.ndarray:
    """Normalize to a target peak with a local compatibility implementation.

    Older installed trees export ``audio.peak_normalize(audio, target_db)`` but
    not the newer ``only_if_louder`` keyword.  Keep this module self-contained
    so effect-chain behavior is stable across overlay versions.
    """
    x = ensure_chans_first(audio).copy()
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 1e-12:
        return x.astype(np.float32)
    target = _db_to_amp(target_db)
    if only_if_louder and peak <= target:
        return x.astype(np.float32)
    return (x * (target / peak)).astype(np.float32)


def soft_clip(audio: np.ndarray, drive: float = 1.2, mix: float = 1.0) -> np.ndarray:
    """Soft saturate audio with a tanh waveshaper.

    This helper intentionally lives in ``effects.py`` instead of assuming a
    matching ``audio.soft_clip`` exists.  Some installed trees have the newer
    effect router but an older ``audio.py`` from before ``soft_clip`` was added,
    which caused import-time failures before any command could run.
    """
    x = ensure_chans_first(audio).astype(np.float32)
    drive = max(0.001, float(drive))
    wet = np.tanh(x * drive) / np.tanh(drive)
    mix = float(np.clip(mix, 0.0, 1.0))
    return (x * (1.0 - mix) + wet * mix).astype(np.float32)


CORE_EFFECTS = {
    "normalize_peak",
    "normalize",
    "clip",
    "hard_clip",
    "soft_clip",
    "saturate",
    "gain",
    "highpass",
    "highpass_filter",
    "hp",
    "lowpass",
    "lowpass_filter",
    "lp",
    "bandpass",
    "bp",
    "band_reduce",
    "deharsh",
    "notch_reduce",
    "tone_safety",
    "dc_block",
    "fade_edges",
}


def _effect_name(spec: dict[str, Any]) -> str:
    return str(spec.get("effect") or spec.get("type") or "").lower().strip()


def is_core_effect(spec: dict[str, Any] | str) -> bool:
    """Return True if an effect spec/name is handled by the core renderer."""
    if isinstance(spec, str):
        name = spec.lower().strip()
    else:
        name = _effect_name(spec)
    return name in CORE_EFFECTS


def split_core_effects(
    effects: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify effects without applying them.

    This helper is kept for compatibility with callers/tests, but the main
    ``apply_effects`` path no longer uses it because it reorders effects.
    """
    pedalboard_effects: list[dict[str, Any]] = []
    core_effects: list[dict[str, Any]] = []
    for spec in effects or []:
        if is_core_effect(spec):
            core_effects.append(spec)
        else:
            pedalboard_effects.append(spec)
    return pedalboard_effects, core_effects


def _sos_filter(audio: np.ndarray, sos: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import sosfiltfilt

        return sosfiltfilt(sos, audio, axis=1).astype(np.float32)
    except Exception:
        from scipy.signal import sosfilt

        return sosfilt(sos, audio, axis=1).astype(np.float32)


def _butter(
    audio: np.ndarray, sample_rate: int, *, kind: str, cutoff_hz: float, order: int = 2
) -> np.ndarray:
    from scipy.signal import butter

    nyq = float(sample_rate) * 0.5
    cutoff = float(np.clip(cutoff_hz, 5.0, nyq * 0.98))
    sos = butter(int(order), cutoff / nyq, btype=kind, output="sos")
    return _sos_filter(ensure_chans_first(audio), sos)


def _bandpass_filter(
    audio: np.ndarray, sample_rate: int, *, center_hz: float, q: float = 1.0, order: int = 2
) -> np.ndarray:
    from scipy.signal import butter

    audio = ensure_chans_first(audio)
    nyq = float(sample_rate) * 0.5
    center = float(np.clip(center_hz, 40.0, nyq * 0.90))
    q = max(0.2, float(q))
    width = center / q
    low = max(20.0, center - width * 0.5)
    high = min(nyq * 0.95, center + width * 0.5)
    if high <= low:
        return audio.astype(np.float32)
    sos = butter(int(order), [low / nyq, high / nyq], btype="bandpass", output="sos")
    return _sos_filter(audio, sos).astype(np.float32)


def _band_reduce(
    audio: np.ndarray, sample_rate: int, *, center_hz: float, amount: float, q: float = 1.1
) -> np.ndarray:
    from scipy.signal import butter

    audio = ensure_chans_first(audio)
    nyq = float(sample_rate) * 0.5
    center = float(np.clip(center_hz, 80.0, nyq * 0.90))
    q = max(0.2, float(q))
    width = center / q
    low = max(40.0, center - width * 0.5)
    high = min(nyq * 0.95, center + width * 0.5)
    if high <= low:
        return audio.astype(np.float32)
    sos = butter(2, [low / nyq, high / nyq], btype="bandpass", output="sos")
    band = _sos_filter(audio, sos)
    amount = float(np.clip(amount, 0.0, 1.0))
    return (audio - band * amount).astype(np.float32)


def _fade_edges(
    audio: np.ndarray, sample_rate: int, *, in_ms: float = 0.5, out_ms: float = 2.0
) -> np.ndarray:
    audio = ensure_chans_first(audio).copy()
    n = audio.shape[1]
    ni = max(0, int(round(float(in_ms) * 0.001 * sample_rate)))
    no = max(0, int(round(float(out_ms) * 0.001 * sample_rate)))
    if ni > 0:
        x = np.linspace(0.0, 1.0, min(ni, n), endpoint=True, dtype=np.float32)
        audio[:, : x.size] *= x[None, :]
    if no > 0:
        x = np.linspace(1.0, 0.0, min(no, n), endpoint=True, dtype=np.float32)
        audio[:, -x.size :] *= x[None, :]
    return audio.astype(np.float32)


def _tone_safety(
    audio: np.ndarray, sample_rate: int, spec: dict[str, Any], context: dict[str, Any]
) -> np.ndarray:
    out = ensure_chans_first(audio)
    if spec.get("enabled", True) is False:
        return out.astype(np.float32)
    highpass_hz = float(spec.get("highpass_hz", 30.0))
    if highpass_hz > 0:
        out = _butter(out, sample_rate, kind="highpass", cutoff_hz=highpass_hz, order=2)
    bands = spec.get("bands") or [
        {
            "center_hz": spec.get("deharsh_hz", 3200.0),
            "amount": spec.get("deharsh_amount", 0.20),
            "q": spec.get("deharsh_q", 0.9),
        }
    ]
    for band in bands:
        out = _band_reduce(
            out,
            sample_rate,
            center_hz=float(band.get("center_hz", 3200.0)),
            amount=float(band.get("amount", 0.18)),
            q=float(band.get("q", 1.0)),
        )
    lowpass_hz = float(spec.get("lowpass_hz", 8800.0))
    if lowpass_hz > 0:
        out = _butter(out, sample_rate, kind="lowpass", cutoff_hz=lowpass_hz, order=2)
    if spec.get("soft_clip", True):
        out = soft_clip(
            out, drive=float(spec.get("drive", 1.08)), mix=float(spec.get("clip_mix", 0.55))
        )
    if spec.get("fade_edges", True):
        out = _fade_edges(
            out,
            sample_rate,
            in_ms=float(spec.get("fade_in_ms", 0.2)),
            out_ms=float(spec.get("fade_out_ms", 1.5)),
        )
    target_db = spec.get("target_peak_db", context.get("final_peak_db", -6.0))
    if target_db is not None:
        out = peak_normalize(
            out, float(target_db), only_if_louder=bool(spec.get("only_if_louder", True))
        )
    return out.astype(np.float32)


def _apply_core_effect(
    out: np.ndarray, sample_rate: int, spec: dict[str, Any], context: dict[str, Any]
) -> np.ndarray:
    name = _effect_name(spec)
    if name in {"normalize_peak", "normalize"}:
        return peak_normalize(
            out,
            float(spec.get("target_db", -3.0)),
            only_if_louder=bool(spec.get("only_if_louder", False)),
        )
    if name in {"clip", "hard_clip"}:
        return hard_clip(out, float(spec.get("limit", 1.0)))
    if name in {"soft_clip", "saturate"}:
        return soft_clip(out, drive=float(spec.get("drive", 1.2)), mix=float(spec.get("mix", 1.0)))
    if name == "gain":
        return apply_gain(out, gain_db=spec.get("gain_db"), gain=spec.get("gain"))
    if name in {"highpass", "highpass_filter", "hp", "dc_block"}:
        return _butter(
            out,
            sample_rate,
            kind="highpass",
            cutoff_hz=float(spec.get("cutoff_hz", spec.get("hz", 30.0))),
            order=int(spec.get("order", 2)),
        )
    if name in {"lowpass", "lowpass_filter", "lp"}:
        return _butter(
            out,
            sample_rate,
            kind="lowpass",
            cutoff_hz=float(spec.get("cutoff_hz", spec.get("hz", 9000.0))),
            order=int(spec.get("order", 2)),
        )
    if name in {"bandpass", "bp"}:
        return _bandpass_filter(
            out,
            sample_rate,
            center_hz=float(spec.get("center_hz", spec.get("hz", 1200.0))),
            q=float(spec.get("q", 1.0)),
            order=int(spec.get("order", 2)),
        )
    if name in {"band_reduce", "deharsh", "notch_reduce"}:
        return _band_reduce(
            out,
            sample_rate,
            center_hz=float(spec.get("center_hz", spec.get("hz", 3200.0))),
            amount=float(spec.get("amount", 0.20)),
            q=float(spec.get("q", 1.0)),
        )
    if name == "fade_edges":
        return _fade_edges(
            out,
            sample_rate,
            in_ms=float(spec.get("in_ms", 0.5)),
            out_ms=float(spec.get("out_ms", 2.0)),
        )
    if name == "tone_safety":
        return _tone_safety(out, sample_rate, spec, context)
    raise ValueError(f"unknown core effect: {name}")


def _apply_pedalboard_chunk(
    out: np.ndarray, sample_rate: int, chunk: list[dict[str, Any]], context: dict[str, Any]
) -> np.ndarray:
    if not chunk:
        return ensure_chans_first(out)
    from ambition_sfx_renderer_gpl.pedalboard_fx import apply_pedalboard

    return apply_pedalboard(out, sample_rate, chunk, context)


def apply_effects(
    audio: np.ndarray, sample_rate: int, effects: list[dict[str, Any]], context: dict[str, Any]
) -> np.ndarray:
    """Apply mixed core/Pedalboard effects while preserving YAML order."""
    out = ensure_chans_first(audio)
    pedalboard_chunk: list[dict[str, Any]] = []
    for spec in effects or []:
        if is_core_effect(spec):
            out = _apply_pedalboard_chunk(out, sample_rate, pedalboard_chunk, context)
            pedalboard_chunk = []
            out = _apply_core_effect(out, sample_rate, spec, context)
        else:
            pedalboard_chunk.append(spec)
    out = _apply_pedalboard_chunk(out, sample_rate, pedalboard_chunk, context)
    return ensure_chans_first(out).astype(np.float32)
