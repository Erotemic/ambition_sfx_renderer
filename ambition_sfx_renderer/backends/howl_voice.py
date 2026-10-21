"""A howl: one voice that is nearly a tone, and is still a throat.

The `howl` noise mode. A wolf's howl is a near-pure tone on a long contour
(Jon's spectrogram, 2026-10-21: f0 ~375 Hz, a step up, a peak at ~620 Hz,
a long fall with a step; the 2nd harmonic brightest, the 3rd faint, almost
no noise). A pure tone on that contour is a siren (Jon: "it sounds like a
siren, it doesn't have the timbre of a wolf"). What a throat adds, and this
renders:

- harmonics shaped by FORMANTS that move with the mouth (`mouth`, 0 shut
  to 1 open): the "oo-OW-oo" of a howl is its timbre changing, not only its
  pitch;
- a voice that is never a dial: pitch JITTER and amplitude SHIMMER, each a
  smoothed random walk (no fixed rate: a fixed flutter is an engine, see
  `creature_voice`), and a slow DRIFT of a few cents;
- BREATH through the same formants, pulsed with the voice (aspiration rides
  the glottis), louder where the voice is quiet;
- a ROUGH ONSET: a subharmonic and more breath in the first `onset_ms`.

Layer keys (curves are `[[fraction, value], ...]` over the layer):

    mode: howl
    seed: 821
    f0_hz: [[0, 375], [0.3, 620], [1, 440]]
    amp: [[0, 0], [0.05, 0.8], [0.3, 1], [1, 0]]
    mouth: [[0, 0.2], [0.28, 1.0], [0.9, 0.6], [1, 0.1]]
    harmonics: 10
    tilt_db_per_octave: -6
    formants:                  # [F with the mouth shut, F open, bandwidth, gain]
      - [780, 1150, 520, 1.0]
      - [1650, 2100, 700, 0.35]
      - [2900, 3300, 900, 0.12]
    floor: 0.04                # what a harmonic keeps far from every formant
    jitter: 0.006              # pitch, a fraction, at ~jitter_hz
    jitter_hz: 28
    drift_cents: 10
    shimmer: 0.10              # amplitude, a fraction, at ~shimmer_hz
    shimmer_hz: 22
    breath: 0.06
    onset_ms: 180
    onset_rough: 0.35          # the subharmonic's level at the start
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal


def _curve(n: int, points) -> np.ndarray:
    xs, ys = zip(*sorted((float(a), float(b)) for a, b in points))
    return np.interp(np.linspace(0.0, 1.0, n, endpoint=False), xs, ys)


def _log_curve(n: int, points) -> np.ndarray:
    xs, ys = zip(*sorted((float(a), float(b)) for a, b in points))
    return np.exp(np.interp(np.linspace(0.0, 1.0, n, endpoint=False), xs, np.log(ys)))


def _walk(rng: np.random.Generator, n: int, rate_hz: float, sr: int) -> np.ndarray:
    """A random walk smoothed to `rate_hz`, unit spread."""
    k = max(2, int(n * rate_hz / sr) + 2)
    pts = rng.standard_normal(k)
    t = np.linspace(0, k - 1, n)
    # Cubic-ish smoothing: interpolate, then a short moving average.
    x = np.interp(t, np.arange(k), pts)
    win = max(1, int(sr / rate_hz / 2))
    return np.convolve(x, np.ones(win) / win, mode="same")


def _resonance(f: np.ndarray, centre: np.ndarray, bandwidth: float) -> np.ndarray:
    return 1.0 / (1.0 + np.square((f - centre) / (0.5 * bandwidth)))


def render_howl(n: int, sr: int, layer: dict[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(int(layer.get("seed", 0)))
    f0 = _log_curve(n, layer["f0_hz"])
    amp = _curve(n, layer.get("amp", [[0, 1], [1, 1]]))
    mouth = np.clip(_curve(n, layer.get("mouth", [[0, 0.5], [1, 0.5]])), 0.0, 1.0)
    # The voice is never a dial: a slow drift and a fast jitter of its pitch,
    # a shimmer of its loudness. All random walks, drawn in this order.
    drift = _walk(rng, n, 1.5, sr) * float(layer.get("drift_cents", 10.0))
    jitter = _walk(rng, n, float(layer.get("jitter_hz", 28.0)), sr) * float(layer.get("jitter", 0.006))
    shimmer = _walk(rng, n, float(layer.get("shimmer_hz", 22.0)), sr) * float(layer.get("shimmer", 0.10))
    f0 = f0 * np.power(2.0, drift / 1200.0) * (1.0 + jitter)
    phase = 2.0 * np.pi * np.cumsum(f0) / sr

    formants = [tuple(float(v) for v in f) for f in layer.get("formants", [[780, 1150, 520, 1.0], [1650, 2100, 700, 0.35]])]
    centres = [(shut + (opened - shut) * mouth, bw, g) for shut, opened, bw, g in formants]
    tilt = float(layer.get("tilt_db_per_octave", -6.0))
    floor = float(layer.get("floor", 0.04))
    voice = np.zeros(n)
    for k in range(1, int(layer.get("harmonics", 10)) + 1):
        fk = k * f0
        shape = floor + sum(g * _resonance(fk, c, bw) for c, bw, g in centres)
        gain = shape * 10.0 ** (tilt * np.log2(k) / 20.0)
        # Above the band the ear gives a howl, nothing: no harmonic past 6 kHz.
        gain = np.where(fk < 6000.0, gain, 0.0)
        voice += gain * np.sin(k * phase + rng.uniform(0.0, 2.0 * np.pi))
    voice /= max(1e-9, float(np.max(np.abs(voice))))

    # A rough onset: a subharmonic (period doubling) and more breath, fading.
    onset = max(1, int(float(layer.get("onset_ms", 180.0)) * 0.001 * sr))
    rough = np.zeros(n)
    rough[:onset] = np.linspace(1.0, 0.0, onset) ** 2
    sub = np.sin(0.5 * phase + 0.3) * float(layer.get("onset_rough", 0.35)) * rough

    # Breath through the same formants, pulsed with the glottis.
    noise = rng.standard_normal(n)
    breath = np.zeros(n)
    for c, bw, g in centres:
        mid = float(np.median(c))
        lo, hi = max(40.0, mid - bw), min(0.45 * sr, mid + bw)
        b, a = signal.butter(2, [lo / (0.5 * sr), hi / (0.5 * sr)], btype="band")
        breath += g * signal.lfilter(b, a, noise)
    breath /= max(1e-9, float(np.max(np.abs(breath))))
    pulse = 0.55 + 0.45 * np.cos(phase)
    breath_level = float(layer.get("breath", 0.06)) * (1.0 + 2.5 * rough)

    out = (voice + sub + breath * pulse * breath_level) * amp * (1.0 + shimmer)
    return out.astype(np.float32)
