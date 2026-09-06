"""Deterministic NumPy oscillator backend for layered musical SFX.

This is intentionally a small sound-design primitive rather than a fallback for
missing heavyweight plugins.  It supports pitch automation, harmonic stacks,
subtle stereo detune, vibrato, and several band-limited-enough-for-SFX waveform
families.  Cues compose multiple short layers around it; no single oscillator is
expected to carry an entire authored sound.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ambition_sfx_renderer.audio import ms_to_samples


def _frequency_curve(value: Any, n: int) -> np.ndarray:
    if isinstance(value, dict):
        start = float(value.get("start", value.get("value", 440.0)))
        end = float(value.get("end", start))
        curve = str(value.get("curve", "linear")).lower()
        x = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
        if curve in {"exp", "exponential"} and start > 0.0 and end > 0.0:
            return start * np.power(end / start, x)
        if curve in {"smooth", "smoothstep"}:
            x = x * x * (3.0 - 2.0 * x)
        elif curve in {"ease_in", "quadratic"}:
            x = x * x
        elif curve in {"ease_out"}:
            x = 1.0 - np.square(1.0 - x)
        return start + (end - start) * x
    return np.full(n, float(value), dtype=np.float64)


def _waveform(phase: np.ndarray, kind: str, pulse_width: float) -> np.ndarray:
    kind = kind.lower()
    if kind in {"sine", "sin"}:
        return np.sin(phase)
    if kind in {"triangle", "tri"}:
        return (2.0 / np.pi) * np.arcsin(np.sin(phase))
    if kind in {"square", "sq"}:
        return np.where(np.sin(phase) >= 0.0, 1.0, -1.0)
    cycle = np.mod(phase / (2.0 * np.pi), 1.0)
    if kind in {"pulse"}:
        return np.where(cycle < pulse_width, 1.0, -1.0)
    if kind in {"saw", "sawtooth"}:
        return 2.0 * cycle - 1.0
    if kind in {"soft_saw", "rounded_saw"}:
        # A short harmonic series is much less brittle than an ideal saw while
        # retaining enough edge for a game transformation cue.
        out = np.zeros_like(phase)
        for harmonic in range(1, 8):
            out += np.sin(phase * harmonic) / harmonic
        return out * (2.0 / np.pi)
    if kind in {"bell", "chime"}:
        return (
            np.sin(phase)
            + 0.42 * np.sin(phase * 2.01 + 0.3)
            + 0.19 * np.sin(phase * 3.96 + 1.1)
            + 0.08 * np.sin(phase * 6.13 + 0.8)
        ) / 1.69
    raise ValueError(
        f"unknown synth waveform {kind!r}; expected sine, triangle, square, "
        "pulse, saw, soft_saw, or bell"
    )


def _db_to_gain(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def render_synth_layer(layer: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    sample_rate = int(context["sample_rate"])
    channels = int(context["channels"])
    duration_ms = float(
        layer.get("duration_ms", float(context.get("duration_seconds", 0.1)) * 1000.0)
    )
    n = max(1, ms_to_samples(duration_ms, sample_rate))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)

    frequency_spec = layer.get("frequency_hz", layer.get("frequency", layer.get("freq", 440.0)))
    base_frequency = _frequency_curve(frequency_spec, n)

    vibrato = dict(layer.get("vibrato", {}) or {})
    vibrato_rate = float(vibrato.get("rate_hz", layer.get("vibrato_rate_hz", 0.0)))
    vibrato_depth = float(vibrato.get("depth_cents", layer.get("vibrato_depth_cents", 0.0)))
    vibrato_delay = float(vibrato.get("delay_ms", 0.0)) * 0.001
    if vibrato_rate > 0.0 and vibrato_depth != 0.0:
        fade = np.clip((t - vibrato_delay) / 0.025, 0.0, 1.0)
        cents = np.sin(2.0 * np.pi * vibrato_rate * t) * vibrato_depth * fade
        base_frequency = base_frequency * np.power(2.0, cents / 1200.0)

    tremolo = dict(layer.get("tremolo", {}) or {})
    tremolo_rate = float(tremolo.get("rate_hz", 0.0))
    tremolo_depth = float(np.clip(tremolo.get("depth", 0.0), 0.0, 1.0))
    amplitude_mod = np.ones(n, dtype=np.float64)
    if tremolo_rate > 0.0 and tremolo_depth > 0.0:
        phase = float(tremolo.get("phase", 0.0))
        lfo = 0.5 + 0.5 * np.sin(2.0 * np.pi * tremolo_rate * t + phase)
        amplitude_mod = (1.0 - tremolo_depth) + tremolo_depth * lfo

    harmonics = list(layer.get("harmonics", []) or [])
    if not harmonics:
        harmonics = [{"ratio": 1.0, "gain_db": 0.0}]

    waveform = str(layer.get("waveform", "sine"))
    pulse_width = float(np.clip(layer.get("pulse_width", 0.5), 0.05, 0.95))
    phase_offset = float(layer.get("phase", 0.0))
    stereo_spread_cents = float(layer.get("stereo_spread_cents", 0.0))
    stereo_phase = float(layer.get("stereo_phase", 0.18))

    seed = int(layer.get("seed", context.get("seed") or 0))
    rng = np.random.default_rng(seed)
    noise_amount = float(np.clip(layer.get("noise_amount", 0.0), 0.0, 1.0))

    output = np.zeros((channels, n), dtype=np.float64)
    for channel in range(channels):
        if channels <= 1:
            detune_cents = 0.0
            channel_phase = 0.0
        else:
            side = -1.0 + 2.0 * channel / float(channels - 1)
            detune_cents = side * stereo_spread_cents * 0.5
            channel_phase = side * stereo_phase
        detune = 2.0 ** (detune_cents / 1200.0)
        channel_mix = np.zeros(n, dtype=np.float64)
        for harmonic in harmonics:
            ratio = float(harmonic.get("ratio", harmonic.get("multiple", 1.0)))
            harmonic_detune = 2.0 ** (float(harmonic.get("detune_cents", 0.0)) / 1200.0)
            gain = _db_to_gain(float(harmonic.get("gain_db", 0.0))) * float(
                harmonic.get("gain", 1.0)
            )
            frequency = base_frequency * ratio * detune * harmonic_detune
            phase = phase_offset + channel_phase + 2.0 * np.pi * np.cumsum(frequency) / sample_rate
            channel_mix += gain * _waveform(phase, waveform, pulse_width)
        if noise_amount > 0.0:
            channel_mix = (
                channel_mix * (1.0 - noise_amount)
                + rng.normal(0.0, 0.35, n) * noise_amount
            )
        output[channel] = channel_mix * amplitude_mod

    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 1e-9:
        output /= peak
    return output.astype(np.float32)
