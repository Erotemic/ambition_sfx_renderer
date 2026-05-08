"""Procedural noise / foley burst backend.

This backend exists for non-tonal sounds such as footsteps, scuffs, dirt
impacts, short debris, cloth puffs, and other cues that should *not* sound like
an oscillator or a pyfxr UI beep.

It intentionally uses only NumPy/SciPy and outputs a raw buffer; normal layer
processing still applies gain, pan, envelope, and effects from the YAML.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ambition_sfx_renderer.audio import ms_to_samples, stereoize


def _rng_for(layer: dict[str, Any], context: dict[str, Any]) -> np.random.Generator:
    seed = layer.get("seed", context.get("seed"))
    if seed is None:
        # Keep deterministic-ish across a run, but callers should normally set
        # a seed in render.seed or per layer for reproducible assets.
        seed = 0
    return np.random.default_rng(int(seed))


def _white(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 1.0, int(n)).astype(np.float32)


def _brown(n: int, rng: np.random.Generator) -> np.ndarray:
    x = _white(n, rng)
    y = np.cumsum(x).astype(np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-9:
        y /= peak
    return y.astype(np.float32)


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return approximate pink noise using Paul Kellet-style filtering."""
    white = _white(n, rng)
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
    out = np.empty(int(n), dtype=np.float32)
    for i, x in enumerate(white):
        b0 = 0.99886 * b0 + x * 0.0555179
        b1 = 0.99332 * b1 + x * 0.0750759
        b2 = 0.96900 * b2 + x * 0.1538520
        b3 = 0.86650 * b3 + x * 0.3104856
        b4 = 0.55000 * b4 + x * 0.5329522
        b5 = -0.7616 * b5 - x * 0.0168980
        y = b0 + b1 + b2 + b3 + b4 + b5 + b6 + x * 0.5362
        b6 = x * 0.115926
        out[i] = y * 0.11
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1e-9:
        out /= peak
    return out.astype(np.float32)


def _colored_noise(color: str, n: int, rng: np.random.Generator) -> np.ndarray:
    color = str(color or "white").lower()
    if color in {"white", "bright"}:
        return _white(n, rng)
    if color in {"pink", "soft"}:
        return _pink(n, rng)
    if color in {"brown", "brownian", "red", "dark"}:
        return _brown(n, rng)
    raise ValueError(f"unknown noise color {color!r}; expected white, pink, or brown")


def _grain_train(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    count: int,
    decay_ms: float,
    spread_ms: float | None = None,
    start_ms: float = 0.0,
) -> np.ndarray:
    """Sparse random clicks with short exponential decays.

    Useful as the "grit" layer of a footstep. This is deliberately noisy and
    non-periodic, so it reads as dirt/gravel/cloth instead of a pitched beep.
    """
    out = np.zeros(int(n), dtype=np.float32)
    count = max(1, int(count))
    decay = max(1, ms_to_samples(float(decay_ms), sample_rate))
    start = min(max(0, ms_to_samples(float(start_ms), sample_rate)), max(0, n - 1))
    if spread_ms is None:
        spread = max(1, n - start)
    else:
        spread = max(1, ms_to_samples(float(spread_ms), sample_rate))
    positions = start + rng.integers(0, max(1, min(spread, max(1, n - start))), size=count)
    kernel = np.exp(-np.arange(decay, dtype=np.float32) / max(1.0, decay * 0.35)).astype(np.float32)
    for pos in positions:
        amp = float(rng.uniform(0.35, 1.0)) * (1.0 if rng.random() > 0.5 else -1.0)
        end = min(n, int(pos) + decay)
        out[int(pos):end] += amp * kernel[: end - int(pos)]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1e-9:
        out /= peak
    return out.astype(np.float32)


def _thud(n: int, sample_rate: int, rng: np.random.Generator, color: str) -> np.ndarray:
    base = _colored_noise(color, n, rng)
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    decay_seconds = max(0.010, float(n) / sample_rate * 0.42)
    env = np.exp(-t / decay_seconds).astype(np.float32)
    impulse = _grain_train(n, sample_rate, rng, count=2, decay_ms=12.0, spread_ms=8.0)
    out = base * env * 0.85 + impulse * 0.35
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1e-9:
        out /= peak
    return out.astype(np.float32)


def _scrape(n: int, sample_rate: int, rng: np.random.Generator, color: str) -> np.ndarray:
    base = _colored_noise(color, n, rng)
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    # A quick brush that rises immediately and dies without a clean periodic envelope.
    env = np.minimum(1.0, t / 0.12) * np.exp(-3.8 * t)
    grains = _grain_train(n, sample_rate, rng, count=9, decay_ms=5.0, spread_ms=float(n) / sample_rate * 1000.0)
    out = base * env * 0.55 + grains * 0.55
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1e-9:
        out /= peak
    return out.astype(np.float32)


def render_noise_layer(layer: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    sample_rate = int(context["sample_rate"])
    channels = int(context["channels"])
    duration_ms = float(layer.get("duration_ms", float(context.get("duration_seconds", 0.1)) * 1000.0))
    n = max(1, ms_to_samples(duration_ms, sample_rate))
    rng = _rng_for(layer, context)
    mode = str(layer.get("mode", layer.get("texture", "burst"))).lower()
    color = str(layer.get("color", "pink")).lower()

    if mode in {"burst", "noise", "plain"}:
        mono = _colored_noise(color, n, rng)
    elif mode in {"grains", "grain", "impulses", "grit"}:
        mono = _grain_train(
            n,
            sample_rate,
            rng,
            count=int(layer.get("grain_count", layer.get("impulse_count", 6))),
            decay_ms=float(layer.get("grain_decay_ms", layer.get("decay_ms", 7.0))),
            spread_ms=layer.get("spread_ms"),
            start_ms=float(layer.get("grain_start_ms", 0.0)),
        )
        # Blend in a little continuous noise so it does not sound like isolated UI clicks.
        mono = mono * 0.75 + _colored_noise(color, n, rng) * 0.25
    elif mode in {"thud", "impact", "dirt_thud"}:
        mono = _thud(n, sample_rate, rng, color)
    elif mode in {"scrape", "scuff", "brush"}:
        mono = _scrape(n, sample_rate, rng, color)
    else:
        raise ValueError(f"unknown noise mode {mode!r}; expected burst, grains, thud, or scrape")

    return stereoize(mono[None, :].astype(np.float32), channels=channels)
