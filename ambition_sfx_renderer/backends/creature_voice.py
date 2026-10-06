"""A creature's voice: an irregular, breathy glottis through a long throat.

The `creature` noise mode. Ported unchanged from the audition scripts the
T-rex's voice was chosen from (the game repo's
`untracked/sfx-candidates/trex_roar2..5`, Jon's picks 2026-10-06), so a recipe
renders the sample he heard, bit for bit before the final level.

⛔ SIZE IS IN THE THROAT, NOT THE PITCH. The first T-rex roar (the `roar` mode)
voiced a 46-96 Hz pulse train with a regular 21-28 Hz flutter; Jon: "It sounds
like a lawnmower". Here a big animal is formants packed close together
(`spacing_hz` ~240-310: a vocal tract metres long) over a voice at 100-500 Hz
that is irregular (jitter, shimmer, period doubling) and mostly breath, and
every wobble is a smoothed random walk, never a fixed rate.

One layer renders the whole sound: its voices draw from ONE random stream, in
order, and the finish (saturation, loudness shape, a short room, the stop)
acts on their sum, as the auditions did.

Layer keys (fractions are of that voice's own duration; curves are
`[[fraction, value], ...]`):

    mode: creature
    seed: 51
    voices:
      - start_ms: 0
        duration_ms: 2000
        level: 1.0             # after the voice is peak-normalized
        f0_hz: [[0, 120], [0.25, 210], [1, 160]]
        jaw: [[0, 0.1], [0.2, 1.0], [1, 0.7]]        # 0 closed .. 1 open
        amp: [[0, 0], [0.08, 0.7], [1, 0.85]]
        doubling: [[0, 0.2], [1, 0.6]]               # subharmonic growl
        spacing_hz: 290        # formant dispersion: smaller is bigger
        f1_hz: [260, 620]      # F1 with the jaw closed, open
        breath: 0.55
        jitter: 0.06
        tilt_hz: 1800
        flutter: 0.10
        formants: 8
        breaks: [[0.32, 0.07, 1.32]]   # optional: (at, half-width, pitch ratio)
    sub:                       # optional: a sine swell under the voices
      start_ms: 0
      duration_ms: 2000
      level: 0.35
      freq_hz: [[0, 48], [0.3, 62], [1, 50]]
      amp: [[0, 0], [0.15, 1], [1, 0.8]]
    finish:
      shape_db: [[0, 0], [0.74, 0], [0.87, -45], [1, -70]]  # over the layer
      saturate: 1.5
      room_s: 0.45
      room_mix: 0.22
      room_seed: 7
      cut_ms: 25
      peak_db: -6
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal


def _smooth_walk(rng: np.random.Generator, n: int, rate_hz: float, depth: float, sr: int) -> np.ndarray:
    """A random walk smoothed to `rate_hz`: wander with no fixed period."""
    k = max(2, int(n * rate_hz / sr) + 2)
    pts = rng.standard_normal(k)
    x = np.interp(np.linspace(0, k - 1, n), np.arange(k), pts)
    return depth * x


def _contour(n: int, points) -> np.ndarray:
    xs, ys = zip(*[(float(a), float(b)) for a, b in points])
    return np.interp(np.linspace(0, 1, n), xs, ys)


def _glottal(rng, f0, sr, jitter=0.06, shimmer=0.35, doubling=0.0, open_q=0.6):
    """An irregular glottal pulse train following f0 (Hz, per sample): each
    period jittered, each height shimmered, and with `doubling` every other
    pulse weakened and delayed (period doubling, the strained growl)."""
    n = len(f0)
    out = np.zeros(n)
    t = 0.0
    k = 0
    while True:
        i = int(t)
        if i >= n:
            break
        period = sr / max(f0[i], 20.0) * (1.0 + jitter * rng.standard_normal())
        amp = 1.0 + shimmer * rng.standard_normal()
        d = doubling[i] if np.ndim(doubling) else doubling
        if k % 2:
            amp *= 1.0 - 0.8 * d
            period *= 1.0 + 0.25 * d
        L = max(4, int(period * open_q))
        if i + L < n:
            ph = np.linspace(0, 1, L)
            pulse = np.where(ph < 0.7, 0.5 * (1 - np.cos(np.pi * ph / 0.7)), np.cos(np.pi * (ph - 0.7) / 0.6))
            out[i:i + L] += amp * pulse
        t += max(period, 8.0)
        k += 1
    return np.diff(out, prepend=0.0)


def _formant_filter(x, centers, bws, gains, sr, hop=256):
    n = len(x)
    y = np.zeros(n)
    for c, bw, g in zip(centers, bws, gains):
        out = np.zeros(n)
        zi = np.zeros(2)
        for s in range(0, n, hop):
            e = min(n, s + hop)
            f = float(np.mean(c[s:e]))
            r = np.exp(-np.pi * bw / sr)
            th = 2 * np.pi * f / sr
            b = [1 - r, 0, 0]
            a = [1, -2 * r * np.cos(th), r * r]
            out[s:e], zi = signal.lfilter(b, a, x[s:e], zi=zi)
        y += g * out
    return y


def _with_breaks(f0_points, breaks):
    """The pitch curve with each voice BREAK: the register jumps up by `ratio`
    around `at` (half-width `width`), then falls back."""
    n = 400
    xs = np.linspace(0, 1, n)
    base = np.interp(xs, *zip(*[(float(a), float(b)) for a, b in f0_points]))
    for at, width, ratio in breaks:
        bump = np.clip(1 - np.abs(xs - float(at)) / float(width), 0, 1) ** 0.35
        base *= 1 + (float(ratio) - 1) * bump
    return list(zip(xs, base))


def _voice(rng, spec: dict[str, Any], sr: int) -> np.ndarray:
    dur = float(spec["duration_ms"]) / 1000.0
    n = int(sr * dur)
    f0_points = spec["f0_hz"]
    if spec.get("breaks"):
        f0_points = _with_breaks(f0_points, spec["breaks"])
    spacing = float(spec.get("spacing_hz", 300.0))
    f1 = spec.get("f1_hz", (260, 620))
    f0 = _contour(n, f0_points) * np.exp(_smooth_walk(rng, n, 7.0, float(spec.get("flutter", 0.10)), sr))
    jaw = _contour(n, spec["jaw"])
    amp = _contour(n, spec["amp"])
    dbl = _contour(n, spec.get("doubling", [[0, 0.0], [1, 0.0]]))
    src = _glottal(rng, f0, sr, jitter=float(spec.get("jitter", 0.06)), doubling=dbl)
    src /= np.max(np.abs(src)) + 1e-9
    noise = rng.standard_normal(n)
    rough = 1.0 + _smooth_walk(rng, n, 40.0, 0.5, sr)
    breath = float(spec.get("breath", 0.55))
    exc = (1 - breath) * src + breath * noise * 0.35 * np.clip(rough, 0.2, None)
    b, a = signal.butter(1, float(spec.get("tilt_hz", 1800.0)) / (sr / 2), "low")
    exc = signal.lfilter(b, a, exc)
    centers, bws, gains = [], [], []
    for j in range(int(spec.get("formants", 8))):
        base = float(f1[0]) + j * spacing
        opened = float(f1[1]) + j * spacing * 1.12
        centers.append(base + (opened - base) * jaw)
        bws.append(90 + 35 * j)
        gains.append(1.0 / (1 + 0.45 * j))
    return _formant_filter(exc, centers, bws, gains, sr) * amp


def _sub(spec: dict[str, Any], sr: int) -> np.ndarray:
    n = int(sr * float(spec["duration_ms"]) / 1000.0)
    f = _contour(n, spec["freq_hz"])
    ph = 2 * np.pi * np.cumsum(f) / sr
    return np.sin(ph) * _contour(n, spec["amp"])


def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.max(np.abs(x)) + 1e-9)


def _room(x, sr, seconds, mix, seed):
    rng = np.random.default_rng(int(seed))
    m = int(sr * seconds)
    ir = rng.standard_normal(m) * np.exp(-np.linspace(0, 9.0, m))
    b, a = signal.butter(2, 3200 / (sr / 2), "low")
    ir = signal.lfilter(b, a, ir)
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = signal.fftconvolve(x, ir)[: len(x)]
    return (1 - mix) * x + mix * wet * 2.5


def _finish(y: np.ndarray, sr: int, spec: dict[str, Any]) -> np.ndarray:
    n = len(y)
    b, a = signal.butter(2, 28 / (sr / 2), "high")
    y = signal.lfilter(b, a, y)
    y = np.tanh(float(spec.get("saturate", 1.5)) * y / (np.max(np.abs(y)) + 1e-9))
    xs, ds = zip(*[(float(p), float(d)) for p, d in spec.get("shape_db", [[0, 0], [1, 0]])])
    y *= 10 ** (np.interp(np.linspace(0, 1, n), xs, ds) / 20)
    y = _room(y, sr, float(spec.get("room_s", 0.45)), float(spec.get("room_mix", 0.22)),
              int(spec.get("room_seed", 7)))
    c = int(sr * float(spec.get("cut_ms", 25)) / 1000)
    if c > 0:
        y[-c:] *= np.linspace(1, 0, c) ** 2
    fade_in = int(sr * 0.004)
    y[:fade_in] *= np.linspace(0, 1, fade_in)
    return y / (np.max(np.abs(y)) + 1e-12) * 10 ** (float(spec.get("peak_db", -6.0)) / 20)


def render_creature(n: int, sample_rate: int, layer: dict[str, Any]) -> np.ndarray:
    """The whole creature sound of `n` samples, mono."""
    sr = int(sample_rate)
    rng = np.random.default_rng(int(layer.get("seed", 0)))
    voices = list(layer.get("voices") or [])
    if not voices:
        raise ValueError("creature mode needs at least one entry in `voices`")
    rendered = [(_norm(_voice(rng, v, sr)) * float(v.get("level", 1.0)), float(v.get("start_ms", 0.0))) for v in voices]
    sub = layer.get("sub")
    if sub:
        rendered.append((_sub(sub, sr) * float(sub.get("level", 0.35)), float(sub.get("start_ms", 0.0))))
    # The auditions' mix: as long as the longest part, each placed at its
    # start and cut to that length.
    body = max(len(p) for p, _ in rendered)
    mix = np.zeros(body)
    for part, start_ms in rendered:
        s = int(sr * start_ms / 1000.0)
        e = min(body, s + len(part))
        if e > s:
            mix[s:e] += part[: e - s]
    y = np.zeros(int(n))
    y[: min(len(y), body)] = mix[: len(y)]
    return _finish(y, sr, dict(layer.get("finish") or {}))
