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
from scipy import signal

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
        out[int(pos) : end] += amp * kernel[: end - int(pos)]
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
    grains = _grain_train(
        n, sample_rate, rng, count=9, decay_ms=5.0, spread_ms=float(n) / sample_rate * 1000.0
    )
    out = base * env * 0.55 + grains * 0.55
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1e-9:
        out /= peak
    return out.astype(np.float32)



def _normalized(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-9:
        audio = audio / peak
    return audio.astype(np.float32)


def _filter_mono(
    audio: np.ndarray,
    sample_rate: int,
    *,
    kind: str,
    hz: float,
    q: float = 0.8,
) -> np.ndarray:
    nyquist = sample_rate * 0.5
    hz = float(np.clip(hz, 20.0, nyquist * 0.92))
    if kind == "lowpass":
        sos = signal.butter(3, hz / nyquist, btype="lowpass", output="sos")
    elif kind == "highpass":
        sos = signal.butter(3, hz / nyquist, btype="highpass", output="sos")
    elif kind == "bandpass":
        bandwidth = max(80.0, hz / max(0.2, float(q)))
        low = max(20.0, hz - bandwidth * 0.5)
        high = min(nyquist * 0.95, hz + bandwidth * 0.5)
        if high <= low + 10.0:
            high = min(nyquist * 0.95, low + 10.0)
        sos = signal.butter(2, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    else:  # pragma: no cover - private callers use the three modes above
        raise ValueError(kind)
    return signal.sosfilt(sos, audio).astype(np.float32)


def _air_sweep(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    start_hz: float,
    end_hz: float,
    q: float,
) -> np.ndarray:
    """Broad, descending blade wake rather than a static filtered hiss."""
    source = _white(n, rng)
    out = np.zeros(n, dtype=np.float32)
    pieces = 12
    centers = np.geomspace(max(80.0, start_hz), max(80.0, end_hz), pieces)
    width = max(64, int(np.ceil(n / (pieces - 2))))
    for idx, center in enumerate(centers):
        midpoint = int(round(idx * (n - 1) / max(1, pieces - 1)))
        begin = max(0, midpoint - width)
        end = min(n, midpoint + width)
        if end <= begin:
            continue
        filtered = _filter_mono(source, sample_rate, kind="bandpass", hz=float(center), q=q)
        window = np.hanning(max(2, (end - begin) * 2))[end - begin :]
        if idx < pieces // 2:
            window = np.hanning(max(2, (end - begin) * 2))[: end - begin]
        else:
            window = np.hanning(max(2, (end - begin) * 2))[end - begin :]
        out[begin:end] += filtered[begin:end] * window.astype(np.float32)
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    arc = np.sin(np.pi * np.clip((t - 0.015) / 0.985, 0.0, 1.0)) ** 0.72
    # A tiny pressure edge in the middle makes the swing read as a moving blade,
    # not merely wind. It remains noise, so there is no UI-beep pitch.
    edge_center = int(n * 0.46)
    edge_len = max(1, int(sample_rate * 0.010))
    edge = np.zeros(n, dtype=np.float32)
    edge_end = min(n, edge_center + edge_len)
    edge[edge_center:edge_end] = _white(edge_end - edge_center, rng) * np.linspace(
        1.0, 0.0, edge_end - edge_center, dtype=np.float32
    )
    edge = _filter_mono(edge, sample_rate, kind="bandpass", hz=2900.0, q=1.4)
    return _normalized(out * arc + edge * 0.22)


def _moving_band_noise(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    start_hz: float,
    end_hz: float,
    q: float,
    color: str = "pink",
    pieces: int = 10,
) -> np.ndarray:
    """Noise whose strongest band slides over time.

    A moving noisy formant reads as material deformation without becoming a
    clean oscillator sweep.  It is useful for viscous squelches and compact
    scrape/crack bodies.
    """
    source = _colored_noise(color, n, rng)
    out = np.zeros(n, dtype=np.float32)
    pieces = max(4, int(pieces))
    centers = np.geomspace(max(35.0, start_hz), max(35.0, end_hz), pieces)
    half_width = max(32, int(np.ceil(n / max(2, pieces - 2))))
    for idx, center in enumerate(centers):
        midpoint = int(round(idx * (n - 1) / max(1, pieces - 1)))
        begin = max(0, midpoint - half_width)
        end = min(n, midpoint + half_width)
        if end <= begin:
            continue
        filtered = _filter_mono(source, sample_rate, kind="bandpass", hz=float(center), q=q)
        window = np.hanning(max(2, end - begin)).astype(np.float32)
        out[begin:end] += filtered[begin:end] * window
    return _normalized(out)


def _viscous_texture(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    cutoff_hz: float,
    pulse_hz: float,
) -> np.ndarray:
    """Irregular low-passed, softly saturated motion for wet/squishy matter."""
    source = _filter_mono(_brown(n, rng), sample_rate, kind="lowpass", hz=cutoff_hz)
    modulation = np.abs(_white(n, rng))
    modulation = _filter_mono(
        modulation,
        sample_rate,
        kind="lowpass",
        hz=max(4.0, pulse_hz),
    )
    modulation -= float(np.min(modulation))
    peak = float(np.max(modulation)) if modulation.size else 0.0
    if peak > 1e-9:
        modulation /= peak
    modulation = 0.30 + 0.95 * modulation
    return _normalized(np.tanh(source * modulation * 2.8))


def _smooth_random_curve(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    smooth_ms: float,
) -> np.ndarray:
    """Slow irregular control curve without exposing a periodic oscillator."""
    raw = np.abs(_white(n, rng))
    width = max(3, ms_to_samples(float(smooth_ms), sample_rate))
    if width % 2 == 0:
        width += 1
    kernel = np.hanning(width).astype(np.float32)
    kernel /= max(1e-9, float(np.sum(kernel)))
    curve = signal.fftconvolve(raw, kernel, mode="same").astype(np.float32)
    curve -= float(np.min(curve))
    peak = float(np.max(curve)) if curve.size else 0.0
    if peak > 1e-9:
        curve /= peak
    return curve


def _windowed_dynamic_noise(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    kind: str,
    min_hz: float,
    max_hz: float,
    color: str,
    q: float = 0.45,
    segments: int = 18,
) -> np.ndarray:
    """Continuously changing filtered noise for soft material motion.

    The overlap-add windows prevent the stationary resonance that made earlier
    flesh attempts read like a struck box.  Each window gets a different broad
    cutoff/formant, producing irregular sliding and folding instead of a thud.
    """
    source = _colored_noise(color, n, rng)
    out = np.zeros(n, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    segments = max(6, int(segments))
    hop = max(16, int(np.ceil(n / max(1, segments - 1))))
    radius = max(64, hop * 2)
    controls = _smooth_random_curve(segments, sample_rate, rng, smooth_ms=0.12)
    controls = 0.12 + 0.88 * controls
    low = max(25.0, float(min_hz))
    high = max(low + 10.0, float(max_hz))
    ratio = high / low

    for idx in range(segments):
        midpoint = int(round(idx * (n - 1) / max(1, segments - 1)))
        begin = max(0, midpoint - radius)
        finish = min(n, midpoint + radius)
        if finish <= begin:
            continue
        hz = low * ratio ** float(controls[idx])
        filtered = _filter_mono(source, sample_rate, kind=kind, hz=hz, q=q)
        local_n = finish - begin
        window = np.hanning(local_n + 2)[1:-1].astype(np.float32)
        window = np.sqrt(np.maximum(window, 0.0)).astype(np.float32)
        out[begin:finish] += filtered[begin:finish] * window
        weight[begin:finish] += window

    valid = weight > 1e-5
    out[valid] /= weight[valid]
    return _normalized(out)


def _rounded_event(
    t: np.ndarray,
    *,
    center_s: float,
    width_s: float,
    power: float = 1.0,
) -> np.ndarray:
    """Smooth isolated control pulse with no impact-like edge."""
    half = max(1e-6, float(width_s) * 0.5)
    phase = np.clip(1.0 - np.abs(t - float(center_s)) / half, 0.0, 1.0)
    return np.power(np.sin(phase * np.pi * 0.5), float(power)).astype(np.float32)


def _gel_compression_squish(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    deep: bool,
) -> np.ndarray:
    """Exaggerated wet compression: the explicit audible ``squish`` gesture.

    Earlier versions hid wetness inside broadband motion.  This version gives
    the material two or three large, rounded compression folds.  Each fold has
    a low viscous body, a moving wet formant, and irregular lip-like flutter.
    There is deliberately no sample-zero transient or decaying thud.
    """
    t = np.arange(n, dtype=np.float32) / float(sample_rate)

    if deep:
        events = (
            (0.042, 0.078, 0.96),
            (0.122, 0.116, 1.00),
            (0.214, 0.138, 0.82),
        )
    else:
        events = (
            (0.038, 0.068, 1.00),
            (0.104, 0.098, 0.92),
        )

    folds = np.zeros(n, dtype=np.float32)
    for center, width, amp in events:
        folds += float(amp) * _rounded_event(
            t, center_s=center, width_s=width, power=0.72
        )
    folds = np.clip(folds, 0.0, 1.55)

    # Dense low wet mass.  The broad cutoff wanders continuously so this does
    # not resemble a resonant cardboard body.
    low = _windowed_dynamic_noise(
        n,
        sample_rate,
        rng,
        kind="lowpass",
        min_hz=115.0 if deep else 155.0,
        max_hz=720.0 if deep else 930.0,
        color="pink",
        segments=30 if deep else 24,
    )
    low = np.tanh(low * 2.35)

    # The audible squelch: a broad noisy formant repeatedly compresses down in
    # pitch during each fold.  A small rough oscillator reinforces the gesture
    # so the listener cannot mistake it for generic filtered noise.
    wet_formant = np.zeros(n, dtype=np.float32)
    wet_voice = np.zeros(n, dtype=np.float32)
    for idx, (center, width, amp) in enumerate(events):
        begin_s = max(0.0, center - width * 0.52)
        finish_s = center + width * 0.58
        begin = min(n, max(0, int(round(begin_s * sample_rate))))
        finish = min(n, max(begin + 1, int(round(finish_s * sample_rate))))
        local_n = finish - begin
        if local_n <= 8:
            continue
        x = np.linspace(0.0, 1.0, local_n, endpoint=False, dtype=np.float32)
        local_t = np.arange(local_n, dtype=np.float32) / float(sample_rate)

        source = _pink(local_n, rng)
        upper_hz = (1280.0 if deep else 1620.0) * float(rng.uniform(0.88, 1.08))
        lower_hz = (245.0 if deep else 330.0) * float(rng.uniform(0.88, 1.12))
        upper = _filter_mono(source, sample_rate, kind="bandpass", hz=upper_hz, q=0.78)
        lower = _filter_mono(source, sample_rate, kind="bandpass", hz=lower_hz, q=0.62)
        glide = np.power(np.clip(x, 0.0, 1.0), 0.58)
        packet = upper * (1.0 - glide) + lower * glide

        # Rounded fold envelope, with a brief re-expansion after maximum squeeze.
        env = np.sin(np.pi * np.clip(x, 0.0, 1.0)) ** 0.58
        env *= 0.78 + 0.22 * np.sin(np.pi * np.clip(x, 0.0, 1.0))
        wet_formant[begin:finish] += packet * env * float(amp)

        f0 = (510.0 if deep else 690.0) * float(rng.uniform(0.90, 1.08))
        f1 = (112.0 if deep else 155.0) * float(rng.uniform(0.90, 1.12))
        freq = f0 * (f1 / f0) ** np.power(x, 0.66)
        jitter = _smooth_random_curve(local_n, sample_rate, rng, smooth_ms=3.8)
        freq *= 0.91 + 0.18 * jitter
        phase = 2.0 * np.pi * np.cumsum(freq) / float(sample_rate)
        flutter = 0.62 + 0.38 * np.sin(
            2.0 * np.pi * (18.0 + idx * 3.7) * local_t + float(rng.uniform(0, np.pi * 2))
        )
        voice = (
            np.sin(phase)
            + 0.23 * np.sin(2.04 * phase + 0.7)
            + 0.09 * np.sin(3.11 * phase + 1.4)
        )
        wet_voice[begin:finish] += voice * flutter * env * float(amp)

    # Sticky high-frequency lip/rub texture.  It is amplitude-gated by the big
    # compression folds, making it read as wet surfaces sliding over each other.
    lips = _windowed_dynamic_noise(
        n,
        sample_rate,
        rng,
        kind="bandpass",
        min_hz=760.0 if deep else 920.0,
        max_hz=2600.0 if deep else 3300.0,
        color="pink",
        q=0.68,
        segments=24 if deep else 20,
    )
    flutter = 0.34 + 0.66 * np.power(
        _smooth_random_curve(n, sample_rate, rng, smooth_ms=4.5), 1.35
    )

    mixed = (
        low * folds * 0.86
        + wet_formant * 0.92
        + wet_voice * folds * (0.34 if deep else 0.29)
        + lips * folds * flutter * 0.56
    )
    mixed = np.tanh(mixed * (1.42 if deep else 1.34))
    mixed = _filter_mono(
        mixed,
        sample_rate,
        kind="lowpass",
        hz=3350.0 if deep else 3900.0,
    )
    return _normalized(mixed)


def _vacuum_slurp_release(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    deep: bool,
) -> np.ndarray:
    """A deliberately obvious suction/slurp followed by a wet ``glup``.

    The slurp is kept as a distinct late gesture instead of being blended into
    the main shear.  It combines noisy airflow, a rough descending suction
    formant, liquid flutter, and a rounded terminal bubble collapse.
    """
    out = np.zeros(n, dtype=np.float32)
    delay_s = 0.244 if deep else 0.145
    start = min(n, max(0, int(round(delay_s * sample_rate))))
    if start >= n:
        return out

    local_n = n - start
    t = np.arange(local_n, dtype=np.float32) / float(sample_rate)
    duration = max(1e-6, local_n / float(sample_rate))
    x = np.clip(t / duration, 0.0, 1.0)

    attack = np.clip(t / (0.018 if deep else 0.014), 0.0, 1.0)
    release = np.power(np.maximum(0.0, 1.0 - x), 0.48 if deep else 0.58)
    slurp_env = attack * release

    # Broad suction airflow descending into a lower cavity resonance.
    air = _moving_band_noise(
        local_n,
        sample_rate,
        rng,
        start_hz=2350.0 if deep else 2850.0,
        end_hz=280.0 if deep else 390.0,
        q=0.76,
        color="pink",
        pieces=24 if deep else 20,
    )
    cavity = _windowed_dynamic_noise(
        local_n,
        sample_rate,
        rng,
        kind="bandpass",
        min_hz=190.0 if deep else 260.0,
        max_hz=1050.0 if deep else 1320.0,
        color="pink",
        q=0.52,
        segments=18,
    )

    # Rough descending suction voice.  Strong enough to communicate "slurp",
    # but noise-jittered and inharmonic enough to avoid a clean cartoon whistle.
    f0 = 760.0 if deep else 980.0
    f1 = 105.0 if deep else 145.0
    freq = f0 * (f1 / f0) ** np.power(x, 0.70)
    jitter = _smooth_random_curve(local_n, sample_rate, rng, smooth_ms=3.2)
    freq *= 0.88 + 0.24 * jitter
    phase = 2.0 * np.pi * np.cumsum(freq) / float(sample_rate)
    voice = (
        np.sin(phase)
        + 0.21 * np.sin(2.09 * phase + 0.45)
        + 0.08 * np.sin(3.23 * phase + 1.31)
    )

    # Audible liquid gulp modulation.  The frequency itself wanders so the
    # pulsation reads as liquid being drawn through a narrowing cavity.
    wobble_rate = (16.0 if deep else 20.0) + (10.0 if deep else 13.0) * x
    wobble_phase = 2.0 * np.pi * np.cumsum(wobble_rate) / float(sample_rate)
    gulp = 0.28 + 0.72 * np.power((np.sin(wobble_phase) + 1.0) * 0.5, 1.85)
    gulp *= 0.65 + 0.35 * _smooth_random_curve(
        local_n, sample_rate, rng, smooth_ms=5.5
    )

    slurp = (
        air * 0.64
        + cavity * 0.78
        + voice * (0.38 if deep else 0.33)
    ) * slurp_env * gulp

    # Terminal ``glup``: a low rounded bubble collapse, clearly separated from
    # the suction sweep.  There is still no click or hard impact transient.
    glup_center = duration * (0.77 if deep else 0.73)
    glup_width = min(duration * 0.34, 0.105 if deep else 0.082)
    glup_env = _rounded_event(
        t, center_s=glup_center, width_s=glup_width, power=0.60
    )
    glup_x = np.clip(
        (t - (glup_center - glup_width * 0.5)) / max(1e-6, glup_width),
        0.0,
        1.0,
    )
    gf0 = 255.0 if deep else 335.0
    gf1 = 58.0 if deep else 78.0
    gfreq = gf0 * (gf1 / gf0) ** np.power(glup_x, 0.60)
    gphase = 2.0 * np.pi * np.cumsum(gfreq) / float(sample_rate)
    glup_voice = (
        np.sin(gphase)
        + 0.27 * np.sin(2.17 * gphase + 0.8)
    )
    glup_noise = _filter_mono(
        _pink(local_n, rng),
        sample_rate,
        kind="lowpass",
        hz=520.0 if deep else 690.0,
    )
    glup = (glup_voice * 0.72 + glup_noise * 0.82) * glup_env

    # Two or three small rounded liquid bubbles after the main gulp.
    bubbles = np.zeros(local_n, dtype=np.float32)
    bubble_specs = (
        ((0.84, 0.045, 210.0, 74.0), (0.93, 0.035, 160.0, 62.0))
        if deep
        else ((0.86, 0.038, 245.0, 88.0),)
    )
    for frac, width, bf0, bf1 in bubble_specs:
        center = duration * frac
        env = _rounded_event(t, center_s=center, width_s=width, power=0.52)
        bx = np.clip((t - (center - width * 0.5)) / max(1e-6, width), 0.0, 1.0)
        bfreq = bf0 * (bf1 / bf0) ** bx
        bphase = 2.0 * np.pi * np.cumsum(bfreq) / float(sample_rate)
        bubbles += np.sin(bphase) * env

    out[start:] = np.tanh(
        (slurp * 1.22 + glup * 1.02 + bubbles * 0.30) * (1.32 if deep else 1.25)
    )
    return _normalized(out)


def _blade_wet_shear(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    deep: bool,
) -> np.ndarray:
    """Subordinate cutting texture connecting squish to slurp."""
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    duration = max(1e-6, n / float(sample_rate))
    x = np.clip(t / duration, 0.0, 1.0)
    shear = _moving_band_noise(
        n,
        sample_rate,
        rng,
        start_hz=3100.0 if deep else 3900.0,
        end_hz=540.0 if deep else 720.0,
        q=0.82,
        color="pink",
        pieces=22,
    )
    body = _windowed_dynamic_noise(
        n,
        sample_rate,
        rng,
        kind="bandpass",
        min_hz=430.0 if deep else 570.0,
        max_hz=1750.0 if deep else 2250.0,
        color="pink",
        q=0.72,
        segments=20,
    )
    attack = np.clip(t / 0.014, 0.0, 1.0)
    release = np.power(np.maximum(0.0, 1.0 - x), 0.80)
    lumps = 0.22 + 0.78 * _smooth_random_curve(
        n, sample_rate, rng, smooth_ms=8.0
    )
    return _normalized((shear * 0.62 + body * 0.54) * attack * release * lumps)


def _ballistic_gel_cut(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    deep: bool,
) -> np.ndarray:
    """An explicit three-part goo cut: SQUISH, shear, then SLURP.

    The cue is intentionally less realistic-subtle than the previous attempt.
    Its job is to communicate the requested material instantly during repeated
    gameplay: large wet compression folds dominate the first half, and a loud
    vacuum-like pull-free dominates the second half.
    """
    t = np.arange(n, dtype=np.float32) / float(sample_rate)

    squish = _gel_compression_squish(n, sample_rate, rng, deep=deep)
    shear = _blade_wet_shear(n, sample_rate, rng, deep=deep)
    slurp = _vacuum_slurp_release(n, sample_rate, rng, deep=deep)

    # Tiny rounded membrane opening.  This is a smear, not a hit.
    membrane = _filter_mono(
        _pink(n, rng),
        sample_rate,
        kind="bandpass",
        hz=1450.0 if deep else 1850.0,
        q=0.68,
    )
    membrane_env = np.clip(t / 0.009, 0.0, 1.0) * np.exp(
        -t / (0.046 if deep else 0.036)
    )
    membrane *= membrane_env

    if deep:
        mixed = squish * 1.02 + shear * 0.30 + slurp * 1.30 + membrane * 0.14
    else:
        mixed = squish * 1.05 + shear * 0.31 + slurp * 1.34 + membrane * 0.15

    # Preserve the large internal level changes.  Heavy compression would hide
    # the distinction between the compression squish and suction release.
    mixed = np.tanh(mixed * (1.14 if deep else 1.10))
    mixed = _filter_mono(
        mixed,
        sample_rate,
        kind="highpass",
        hz=38.0 if deep else 48.0,
    )
    mixed = _filter_mono(
        mixed,
        sample_rate,
        kind="lowpass",
        hz=3900.0 if deep else 4550.0,
    )
    return _normalized(mixed)



def _foley_packet(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    start_s: float,
    duration_s: float,
    color: str,
    filter_kind: str,
    filter_hz: float,
    q: float = 0.45,
    attack_s: float = 0.006,
    release_power: float = 1.15,
) -> np.ndarray:
    """Broad, soft-edged filtered-noise packet for close-mic Foley.

    This deliberately avoids click impulses and resonant oscillators.  The
    raised attack and rounded release make each packet behave like material
    folding, tearing, or collapsing rather than an object being struck.
    """
    out = np.zeros(n, dtype=np.float32)
    start = int(round(max(0.0, start_s) * sample_rate))
    length = max(1, int(round(max(0.001, duration_s) * sample_rate)))
    finish = min(n, start + length)
    if start >= n or finish <= start:
        return out
    local_n = finish - start
    local_t = np.arange(local_n, dtype=np.float32) / float(sample_rate)
    x = np.clip(local_t / max(1e-6, duration_s), 0.0, 1.0)
    attack = np.sin(
        np.clip(local_t / max(1e-6, attack_s), 0.0, 1.0) * np.pi * 0.5
    ) ** 1.4
    release = np.sin(np.clip(1.0 - x, 0.0, 1.0) * np.pi * 0.5) ** float(
        release_power
    )
    source = _colored_noise(color, local_n, rng)
    source = _filter_mono(
        source,
        sample_rate,
        kind=filter_kind,
        hz=float(filter_hz),
        q=float(q),
    )
    out[start:finish] = source * attack * release
    return out



def _subaudio_square_edges(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    rate_points_hz: tuple[float, ...],
    jitter: float = 0.18,
    click_decay_ms: float = 2.4,
) -> np.ndarray:
    """Square-wave edge clicks below normal pitch perception.

    This follows the useful part of the supplied squishy-bass workflow: drive a
    resonant filter with the *edges* of an extremely low square wave.  At about
    8--28 Hz the source is heard as discrete sticky clicks rather than a note.
    The pulse rate wanders independently from the later filter motion.
    """
    out = np.zeros(int(n), dtype=np.float32)
    if n <= 0:
        return out

    controls_x = np.linspace(0.0, 1.0, len(rate_points_hz), dtype=np.float32)
    pos = float(rng.uniform(0.006, 0.014) * sample_rate)
    sign = 1.0
    decay_n = max(3, ms_to_samples(float(click_decay_ms), sample_rate))
    kt = np.arange(decay_n, dtype=np.float32)
    kernel = np.exp(-kt / max(1.0, decay_n * 0.31)).astype(np.float32)
    # A tiny opposite-polarity tail approximates a differentiated square edge
    # and avoids a one-sided DC-like tick.
    kernel -= 0.42 * np.exp(-kt / max(1.0, decay_n * 0.72)).astype(np.float32)

    while pos < n:
        progress = float(np.clip(pos / max(1.0, n - 1), 0.0, 1.0))
        rate = float(np.interp(progress, controls_x, rate_points_hz))
        rate *= float(rng.uniform(1.0 - jitter, 1.0 + jitter))
        rate = max(4.0, rate)
        # Both edges of the conceptual square wave are audible.
        interval = sample_rate / (rate * 2.0)
        index = int(round(pos))
        finish = min(n, index + decay_n)
        amp = float(rng.uniform(0.62, 1.0)) * sign
        out[index:finish] += amp * kernel[: finish - index]
        sign *= -1.0
        pos += interval * float(rng.uniform(0.88, 1.15))

    return out.astype(np.float32)


def _click_resonator_motion(
    source: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    center_points_hz: tuple[float, ...],
    q: float,
    segments: int,
    delay_taps_ms: tuple[tuple[float, float], ...] = (),
) -> np.ndarray:
    """Independently move a resonant band-pass over sparse edge clicks.

    The cutoff follows a deliberately non-monotonic control path with random
    perturbation.  That independence between pulse timing and filter motion is
    what creates the rubbery/squelchy pattern described in the transcript,
    while avoiding the clean one-way sweep that made the previous cue sci-fi.
    """
    n = int(source.size)
    out = np.zeros(n, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    segments = max(6, int(segments))
    controls_x = np.linspace(0.0, 1.0, len(center_points_hz), dtype=np.float32)
    hop = max(32, int(np.ceil(n / max(1, segments - 1))))
    radius = max(96, hop * 2)

    for idx in range(segments):
        progress = idx / max(1, segments - 1)
        center = float(np.interp(progress, controls_x, center_points_hz))
        center *= float(rng.uniform(0.86, 1.16))
        filtered = _filter_mono(
            source,
            sample_rate,
            kind="bandpass",
            hz=center,
            q=float(q) * float(rng.uniform(0.88, 1.12)),
        )
        midpoint = int(round(progress * max(0, n - 1)))
        begin = max(0, midpoint - radius)
        finish = min(n, midpoint + radius)
        if finish <= begin:
            continue
        local_n = finish - begin
        window = np.hanning(local_n + 2)[1:-1].astype(np.float32)
        window = np.sqrt(np.maximum(window, 0.0)).astype(np.float32)
        out[begin:finish] += filtered[begin:finish] * window
        weight[begin:finish] += window

    valid = weight > 1e-5
    out[valid] /= weight[valid]

    # A few quiet, non-feedback micro-delays make each click smear like sticky
    # material.  These are short taps, not a flanger or pitched comb effect.
    wet = out.copy()
    for delay_ms, gain in delay_taps_ms:
        delay_n = max(1, ms_to_samples(float(delay_ms), sample_rate))
        if delay_n < n:
            wet[delay_n:] += out[:-delay_n] * float(gain)

    return _normalized(np.tanh(wet * 1.42))




def _soft_gate(
    t: np.ndarray,
    *,
    start_s: float,
    finish_s: float,
    attack_s: float,
    release_s: float,
    power: float = 1.0,
) -> np.ndarray:
    """Rounded time window used for continuous biological Foley layers."""
    attack = np.sin(
        np.clip((t - float(start_s)) / max(1e-6, float(attack_s)), 0.0, 1.0)
        * np.pi
        * 0.5
    ) ** float(power)
    release = np.sin(
        np.clip((float(finish_s) - t) / max(1e-6, float(release_s)), 0.0, 1.0)
        * np.pi
        * 0.5
    ) ** float(power)
    return (attack * release).astype(np.float32)


def _wet_bubble_field(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    start_s: float,
    finish_s: float,
    count: int,
    min_hz: float,
    max_hz: float,
    gain: float,
) -> np.ndarray:
    """Dense, irregular micro-bubbles for blood and tissue-fluid gurgle.

    Individual bubbles are intentionally short, heavily damped, randomly
    phased, and mixed with noisy skins.  A large randomized cloud reads as
    liquid movement instead of a pitched sci-fi chirp.
    """
    out = np.zeros(n, dtype=np.float32)
    start_s = max(0.0, float(start_s))
    finish_s = max(start_s + 0.001, float(finish_s))
    for _ in range(max(1, int(count))):
        event_start_s = float(rng.uniform(start_s, finish_s))
        event_start = int(round(event_start_s * sample_rate))
        if event_start >= n:
            continue
        duration_s = float(rng.uniform(0.010, 0.038))
        length = min(n - event_start, max(4, int(round(duration_s * sample_rate))))
        local_t = np.arange(length, dtype=np.float32) / float(sample_rate)
        x = np.clip(local_t / max(1e-6, duration_s), 0.0, 1.0)
        f0 = float(rng.uniform(min_hz, max_hz))
        # Real liquid bubbles often rise slightly in frequency as they collapse.
        # Keep the excursion small and random so the cloud has no coherent tune.
        ratio = float(rng.uniform(1.03, 1.24))
        freq = f0 * ratio ** np.power(x, float(rng.uniform(0.55, 1.25)))
        phase = float(rng.uniform(0.0, np.pi * 2.0)) + 2.0 * np.pi * np.cumsum(freq) / float(sample_rate)
        env = np.exp(-local_t / float(rng.uniform(0.0045, 0.014))).astype(np.float32)
        env *= np.sin(np.clip(local_t / float(rng.uniform(0.0010, 0.0035)), 0.0, 1.0) * np.pi * 0.5)
        tone = np.sin(phase).astype(np.float32)
        skin = _filter_mono(
            _pink(length, rng),
            sample_rate,
            kind="bandpass",
            hz=f0 * float(rng.uniform(1.2, 2.4)),
            q=float(rng.uniform(0.28, 0.55)),
        )
        event = tone * env * float(rng.uniform(0.30, 0.68)) + skin * env * float(rng.uniform(0.16, 0.38))
        out[event_start : event_start + length] += event * float(rng.uniform(0.35, 1.0))

    out = _filter_mono(out, sample_rate, kind="highpass", hz=95.0)
    out = _filter_mono(out, sample_rate, kind="lowpass", hz=2400.0)
    return (_normalized(np.tanh(out * 1.6)) * float(gain)).astype(np.float32)


def _blood_slurry(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    deep: bool,
) -> np.ndarray:
    """Broad wet slosh with separate heavy fluid surges and surface splatter."""
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    duration = max(1e-6, n / float(sample_rate))

    low = _windowed_dynamic_noise(
        n,
        sample_rate,
        rng,
        kind="lowpass",
        min_hz=110.0 if deep else 155.0,
        max_hz=1050.0 if deep else 1320.0,
        color="pink",
        q=0.32,
        segments=34 if deep else 24,
    )
    low2 = _filter_mono(_brown(n, rng), sample_rate, kind="lowpass", hz=520.0 if deep else 690.0)
    motion = _smooth_random_curve(n, sample_rate, rng, smooth_ms=5.4 if deep else 3.8)
    motion = 0.16 + 1.12 * np.power(motion, 1.8)

    surge = np.zeros(n, dtype=np.float32)
    events = (
        ((0.048, 0.120, 0.82), (0.152, 0.178, 1.00), (0.278, 0.150, 0.76))
        if deep
        else ((0.035, 0.074, 1.00), (0.105, 0.070, 0.58))
    )
    for center_s, width_s, amp in events:
        surge += float(amp) * _rounded_event(
            t,
            center_s=min(center_s, duration * 0.82),
            width_s=min(width_s, duration * 0.62),
            power=0.54,
        )
    surge = np.clip(surge, 0.0, 1.65)

    # Thin surface liquid gives the slosh a wet top without turning into hiss.
    surface = _filter_mono(_pink(n, rng), sample_rate, kind="bandpass", hz=1750.0 if deep else 2150.0, q=0.36)
    surface_motion = _smooth_random_curve(n, sample_rate, rng, smooth_ms=2.2)
    surface *= 0.12 + 0.92 * np.power(surface_motion, 2.2)

    slurry = np.tanh((low * 0.92 + low2 * 0.58) * motion * surge * 2.25)
    slurry += surface * surge * (0.20 if deep else 0.16)
    slurry = _filter_mono(slurry, sample_rate, kind="highpass", hz=46.0 if deep else 72.0)
    slurry = _filter_mono(slurry, sample_rate, kind="lowpass", hz=3300.0 if deep else 3900.0)
    return _normalized(slurry)


def _muscle_and_vein_tears(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    deep: bool,
) -> np.ndarray:
    """Fibrous muscle ripping plus smaller slick vessel snaps."""
    out = np.zeros(n, dtype=np.float32)
    duration = n / float(sample_rate)
    count = 28 if deep else 11
    earliest = 0.045 if deep else 0.027
    latest = min(duration * 0.77, 0.292 if deep else 0.143)
    starts = np.sort(rng.uniform(earliest, max(earliest + 0.001, latest), count))

    for idx, start_s in enumerate(starts):
        is_vein = (idx % 4 == 0) or (rng.random() < 0.20)
        if is_vein:
            event = _foley_packet(
                n,
                sample_rate,
                rng,
                start_s=float(start_s),
                duration_s=float(rng.uniform(0.005, 0.013)),
                color="white",
                filter_kind="bandpass",
                filter_hz=float(rng.uniform(1250.0, 2900.0)),
                q=float(rng.uniform(0.32, 0.62)),
                attack_s=float(rng.uniform(0.0012, 0.0032)),
                release_power=float(rng.uniform(1.65, 2.45)),
            )
            amp = float(rng.uniform(0.055, 0.13))
        else:
            event = _foley_packet(
                n,
                sample_rate,
                rng,
                start_s=float(start_s),
                duration_s=float(rng.uniform(0.012, 0.034 if deep else 0.024)),
                color="pink",
                filter_kind="bandpass",
                filter_hz=float(rng.uniform(720.0, 2650.0)),
                q=float(rng.uniform(0.25, 0.48)),
                attack_s=float(rng.uniform(0.0028, 0.0080)),
                release_power=float(rng.uniform(1.25, 1.95)),
            )
            amp = float(rng.uniform(0.085, 0.20 if deep else 0.15))
        out += event * amp

    # Smear each tear very slightly through wet material; no feedback or comb tone.
    wet = out.copy()
    for delay_ms, gain in ((2.4, 0.21), (5.8, 0.12), (10.7, 0.055)):
        delay_n = ms_to_samples(delay_ms, sample_rate)
        if 0 < delay_n < n:
            wet[delay_n:] += out[:-delay_n] * gain
    wet = _filter_mono(wet, sample_rate, kind="highpass", hz=420.0)
    wet = _filter_mono(wet, sample_rate, kind="lowpass", hz=4700.0)
    return _normalized(np.tanh(wet * 2.0))


def _brain_pulp(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    deep: bool,
) -> np.ndarray:
    """Soft collapsing pulp that fills the gap between squish and liquid slosh."""
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    duration = max(1e-6, n / float(sample_rate))
    pulp = _windowed_dynamic_noise(
        n,
        sample_rate,
        rng,
        kind="lowpass",
        min_hz=180.0 if deep else 260.0,
        max_hz=1450.0 if deep else 1800.0,
        color="pink",
        q=0.28,
        segments=32 if deep else 22,
    )
    pockets = np.zeros(n, dtype=np.float32)
    events = (
        ((0.074, 0.105, 0.82), (0.174, 0.138, 1.00), (0.285, 0.110, 0.72))
        if deep
        else ((0.052, 0.082, 1.00), (0.119, 0.070, 0.52))
    )
    for center_s, width_s, amp in events:
        pockets += float(amp) * _rounded_event(
            t,
            center_s=min(center_s, duration * 0.84),
            width_s=min(width_s, duration * 0.64),
            power=0.47,
        )
    flutter = _smooth_random_curve(n, sample_rate, rng, smooth_ms=4.6 if deep else 3.1)
    pulp = np.tanh(pulp * pockets * (0.26 + 1.06 * np.power(flutter, 1.35)) * 2.5)
    return _normalized(_filter_mono(pulp, sample_rate, kind="lowpass", hz=2250.0))


def _wet_suction_release(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    deep: bool,
) -> np.ndarray:
    """Close-mic mouth-like slurp and sticky blade withdrawal.

    This is deliberately not a pitch sweep.  It uses several broad stationary
    mouth/formant regions, independently fluttered noise, lip smacks, low fluid
    pressure, and a cloud of short bubbles.  The amplitude gesture builds under
    suction and then collapses abruptly, which is what makes the release read as
    a slurp rather than generic filtered noise.
    """
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    duration = max(1e-6, n / float(sample_rate))
    start_s = min(0.205 if deep else 0.098, duration * (0.56 if deep else 0.47))
    finish_s = min(0.392 if deep else 0.211, duration * 0.992)

    # Two overlapping suction pulls.  Each builds toward the end and drops
    # quickly, mimicking sticky material releasing around the blade.
    pull_shape = np.zeros(n, dtype=np.float32)
    pulls = (
        ((0.216, 0.118, 1.00), (0.307, 0.095, 0.84))
        if deep
        else ((0.102, 0.073, 1.00), (0.160, 0.048, 0.63))
    )
    for pull_start_s, pull_duration_s, amp in pulls:
        local_start = int(round(pull_start_s * sample_rate))
        local_finish = min(n, int(round((pull_start_s + pull_duration_s) * sample_rate)))
        if local_finish <= local_start or local_start >= n:
            continue
        local_n = local_finish - local_start
        x = np.linspace(0.0, 1.0, local_n, endpoint=False, dtype=np.float32)
        # Slow pressure build followed by a fast wet release.
        env = np.power(np.clip(x / 0.76, 0.0, 1.0), 0.58)
        tail = np.sin(np.clip((1.0 - x) / 0.24, 0.0, 1.0) * np.pi * 0.5) ** 1.8
        env *= tail
        pull_shape[local_start:local_finish] += float(amp) * env
    pull_shape = np.clip(pull_shape, 0.0, 1.45)

    source = _pink(n, rng)
    pressure = _filter_mono(source, sample_rate, kind="lowpass", hz=520.0 if deep else 720.0)
    mouth_low = _filter_mono(source, sample_rate, kind="bandpass", hz=410.0 if deep else 520.0, q=0.38)
    mouth_mid = _filter_mono(source, sample_rate, kind="bandpass", hz=890.0 if deep else 1120.0, q=0.42)
    mouth_high = _filter_mono(source, sample_rate, kind="bandpass", hz=1680.0 if deep else 2050.0, q=0.35)

    # Independent noisy flutter in each mouth region avoids a clean vocoder or
    # spacecraft character while preserving the recognizable wet oral slurp.
    flutter_a = 0.18 + 1.02 * np.power(_smooth_random_curve(n, sample_rate, rng, smooth_ms=4.8 if deep else 3.4), 1.55)
    flutter_b = 0.14 + 1.08 * np.power(_smooth_random_curve(n, sample_rate, rng, smooth_ms=2.7 if deep else 2.0), 1.85)
    flutter_c = 0.10 + 0.92 * np.power(_smooth_random_curve(n, sample_rate, rng, smooth_ms=1.7 if deep else 1.3), 2.15)
    mouth = pressure * 0.70
    mouth += mouth_low * flutter_a * 0.76
    mouth += mouth_mid * flutter_b * 0.58
    mouth += mouth_high * flutter_c * 0.31

    # Irregular lip smacks / wet seals.  Filtered noise pulses, not dry clicks.
    smacks = np.zeros(n, dtype=np.float32)
    smack_count = 10 if deep else 5
    for _ in range(smack_count):
        smack_start = float(rng.uniform(start_s, max(start_s + 0.002, finish_s - 0.012)))
        smacks += _foley_packet(
            n,
            sample_rate,
            rng,
            start_s=smack_start,
            duration_s=float(rng.uniform(0.008, 0.021 if deep else 0.015)),
            color="pink",
            filter_kind="bandpass",
            filter_hz=float(rng.uniform(560.0, 1850.0 if deep else 2350.0)),
            q=float(rng.uniform(0.30, 0.52)),
            attack_s=float(rng.uniform(0.0018, 0.0045)),
            release_power=float(rng.uniform(1.45, 2.20)),
        ) * float(rng.uniform(0.10, 0.23))

    bubbles = _wet_bubble_field(
        n,
        sample_rate,
        rng,
        start_s=start_s + (0.010 if deep else 0.006),
        finish_s=max(start_s + 0.025, finish_s - 0.004),
        count=30 if deep else 12,
        min_hz=110.0 if deep else 170.0,
        max_hz=780.0 if deep else 1120.0,
        gain=1.0,
    )

    # A low-rate click/filter trace gives sticky articulation, but is kept low
    # and broad enough that it cannot become the identity of the cue again.
    edges = _subaudio_square_edges(
        n,
        sample_rate,
        rng,
        rate_points_hz=(16.0, 27.0, 13.0, 20.0, 8.0) if deep else (21.0, 33.0, 17.0, 10.0),
        jitter=0.32,
        click_decay_ms=2.6 if deep else 1.9,
    )
    sticky = _click_resonator_motion(
        edges,
        sample_rate,
        rng,
        center_points_hz=(650.0, 300.0, 980.0, 390.0, 590.0) if deep else (930.0, 430.0, 1320.0, 560.0),
        q=0.60,
        segments=22 if deep else 17,
        delay_taps_ms=((4.4, 0.15), (9.8, 0.06)) if deep else ((3.0, 0.12), (6.6, 0.05)),
    )

    gate = _soft_gate(
        t,
        start_s=start_s,
        finish_s=finish_s,
        attack_s=0.022 if deep else 0.014,
        release_s=0.018 if deep else 0.011,
        power=0.54,
    )
    mixed = mouth * pull_shape * (0.92 if deep else 0.84)
    mixed += smacks * gate * (0.90 if deep else 0.82)
    mixed += bubbles * gate * (0.88 if deep else 0.70)
    mixed += sticky * pull_shape * (0.25 if deep else 0.20)
    mixed = np.tanh(mixed * 2.05)
    mixed = _filter_mono(mixed, sample_rate, kind="highpass", hz=52.0 if deep else 82.0)
    mixed = _filter_mono(mixed, sample_rate, kind="lowpass", hz=3600.0 if deep else 4200.0)
    return _normalized(mixed)



def _light_juice_slurp_tail(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """A late, close-mic liquid suction for the light flesh contact.

    The front of the light hit should remain a restrained wet nick.  This tail
    begins after that contact and becomes the dominant gesture: irregular
    aspiration through fluid, small juice bubbles, soft wet seals, and a final
    sticky release.  All tonal regions are broad and stationary so the result
    reads as Foley rather than a pitched sci-fi sweep.
    """
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    duration = max(1e-6, n / float(sample_rate))
    start_s = min(0.112, duration * 0.40)
    finish_s = min(0.278, duration * 0.985)

    # Two overlapping inward pulls.  Both rise in pressure, briefly flutter,
    # and drop quickly as the wet seal breaks around the withdrawing blade.
    pull_shape = np.zeros(n, dtype=np.float32)
    pulls = ((0.116, 0.100, 0.82), (0.174, 0.096, 1.00))
    for pull_start_s, pull_duration_s, amp in pulls:
        local_start = int(round(pull_start_s * sample_rate))
        local_finish = min(n, int(round((pull_start_s + pull_duration_s) * sample_rate)))
        if local_finish <= local_start or local_start >= n:
            continue
        x = np.linspace(0.0, 1.0, local_finish - local_start, endpoint=False, dtype=np.float32)
        rise = np.power(np.clip(x / 0.73, 0.0, 1.0), 0.48)
        release = np.sin(np.clip((1.0 - x) / 0.27, 0.0, 1.0) * np.pi * 0.5) ** 1.55
        flutter = 0.78 + 0.22 * np.sin(2.0 * np.pi * (3.2 * x + 0.18 * np.sin(2.0 * np.pi * 1.7 * x)))
        pull_shape[local_start:local_finish] += float(amp) * rise * release * flutter
    pull_shape = np.clip(pull_shape, 0.0, 1.55)

    # Broad aspiration and mouth-cavity noise.  The independent random motion
    # supplies wet articulation without any frequency glide or pitched carrier.
    source = _pink(n, rng) * 0.72 + _white(n, rng) * 0.28
    fluid_low = _filter_mono(source, sample_rate, kind="lowpass", hz=760.0)
    mouth_a = _filter_mono(source, sample_rate, kind="bandpass", hz=570.0, q=0.34)
    mouth_b = _filter_mono(source, sample_rate, kind="bandpass", hz=1180.0, q=0.38)
    mouth_c = _filter_mono(source, sample_rate, kind="bandpass", hz=2260.0, q=0.30)
    flutter_a = 0.17 + 1.00 * np.power(_smooth_random_curve(n, sample_rate, rng, smooth_ms=3.9), 1.55)
    flutter_b = 0.12 + 1.05 * np.power(_smooth_random_curve(n, sample_rate, rng, smooth_ms=2.1), 1.90)
    flutter_c = 0.08 + 0.88 * np.power(_smooth_random_curve(n, sample_rate, rng, smooth_ms=1.25), 2.25)
    aspiration = fluid_low * 0.76
    aspiration += mouth_a * flutter_a * 0.78
    aspiration += mouth_b * flutter_b * 0.61
    aspiration += mouth_c * flutter_c * 0.34

    # Dense little liquid pockets make the tail sound like juice moving through
    # a narrow opening rather than dry inhalation.
    bubbles = _wet_bubble_field(
        n,
        sample_rate,
        rng,
        start_s=start_s + 0.010,
        finish_s=max(start_s + 0.030, finish_s - 0.006),
        count=21,
        min_hz=155.0,
        max_hz=1180.0,
        gain=1.0,
    )

    # Small wet seals and one final sticky lip release.  These are filtered noise
    # packets, not clicks, so the tail stays soft and fleshy.
    seals = np.zeros(n, dtype=np.float32)
    for smack_start, duration_s, amp, center_hz in (
        (0.142, 0.017, 0.18, 980.0),
        (0.196, 0.021, 0.23, 720.0),
        (0.252, 0.024, 0.31, 560.0),
    ):
        seals += _foley_packet(
            n,
            sample_rate,
            rng,
            start_s=smack_start,
            duration_s=duration_s,
            color="pink",
            filter_kind="bandpass",
            filter_hz=center_hz,
            q=0.36,
            attack_s=0.0035,
            release_power=1.55,
        ) * amp

    # A sparse click-filter trace adds sticky articulation at the release, but
    # remains far below the aspiration and liquid layers.
    edges = _subaudio_square_edges(
        n,
        sample_rate,
        rng,
        rate_points_hz=(18.0, 29.0, 15.0, 23.0, 10.0),
        jitter=0.37,
        click_decay_ms=1.8,
    )
    sticky = _click_resonator_motion(
        edges,
        sample_rate,
        rng,
        center_points_hz=(860.0, 420.0, 1320.0, 520.0, 740.0),
        q=0.48,
        segments=19,
        delay_taps_ms=((3.2, 0.11), (7.4, 0.045)),
    )

    gate = _soft_gate(
        t,
        start_s=start_s,
        finish_s=finish_s,
        attack_s=0.020,
        release_s=0.012,
        power=0.50,
    )
    mixed = aspiration * pull_shape * 1.04
    mixed += bubbles * gate * 0.92
    mixed += seals * gate * 1.00
    mixed += sticky * pull_shape * 0.14
    mixed = np.tanh(mixed * 2.20)
    mixed = _filter_mono(mixed, sample_rate, kind="highpass", hz=78.0)
    mixed = _filter_mono(mixed, sample_rate, kind="lowpass", hz=4300.0)
    return _normalized(mixed)

def _oscillating_wet_splatter(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    deep: bool,
) -> np.ndarray:
    """Irregular wet splatter whose pulse rate and filter motion drift apart.

    This retains the useful oscillating click/filter gesture from the attributed
    squish technique, but breaks it into short biological splash bursts.  Two
    out-of-sync low-rate edge trains, low-Q resonators, broadband wet skins, and
    scattered droplets prevent the layer from becoming a clean synth wobble.
    """
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    duration = max(1e-6, n / float(sample_rate))

    burst_gate = np.zeros(n, dtype=np.float32)
    bursts = (
        ((0.060, 0.075, 0.78), (0.158, 0.105, 1.00), (0.286, 0.088, 0.72))
        if deep
        else ((0.043, 0.052, 1.00), (0.118, 0.055, 0.64))
    )
    for center_s, width_s, amp in bursts:
        burst_gate += float(amp) * _rounded_event(
            t,
            center_s=min(center_s, duration * 0.86),
            width_s=min(width_s, duration * 0.48),
            power=0.46,
        )
    burst_gate = np.clip(burst_gate, 0.0, 1.55)

    edges_a = _subaudio_square_edges(
        n,
        sample_rate,
        rng,
        rate_points_hz=(13.0, 27.0, 17.0, 31.0, 11.0) if deep else (18.0, 34.0, 21.0, 14.0),
        jitter=0.38,
        click_decay_ms=2.1 if deep else 1.6,
    )
    edges_b = _subaudio_square_edges(
        n,
        sample_rate,
        rng,
        rate_points_hz=(21.0, 12.0, 29.0, 15.0, 24.0) if deep else (29.0, 17.0, 37.0, 20.0),
        jitter=0.46,
        click_decay_ms=1.5 if deep else 1.2,
    )
    wobble_a = _click_resonator_motion(
        edges_a,
        sample_rate,
        rng,
        center_points_hz=(980.0, 430.0, 1640.0, 610.0, 1180.0) if deep else (1420.0, 620.0, 2160.0, 880.0),
        q=0.54 if deep else 0.50,
        segments=25 if deep else 19,
        delay_taps_ms=((2.8, 0.16), (6.4, 0.07)) if deep else ((2.1, 0.13), (4.7, 0.055)),
    )
    wobble_b = _click_resonator_motion(
        edges_b,
        sample_rate,
        rng,
        center_points_hz=(520.0, 1320.0, 370.0, 1880.0, 720.0) if deep else (810.0, 1880.0, 520.0, 2520.0),
        q=0.42,
        segments=21 if deep else 16,
        delay_taps_ms=((3.7, 0.11), (8.9, 0.045)) if deep else ((2.6, 0.09), (5.8, 0.04)),
    )

    # Broadband liquid skin follows the splatter bursts while its texture moves
    # independently from either edge train.  This is what makes the oscillation
    # feel wet rather than like a dry gated synthesizer.
    skin = _windowed_dynamic_noise(
        n,
        sample_rate,
        rng,
        kind="bandpass",
        min_hz=620.0 if deep else 880.0,
        max_hz=3300.0 if deep else 4100.0,
        color="pink",
        q=0.28,
        segments=31 if deep else 23,
    )
    skin_flutter = _smooth_random_curve(n, sample_rate, rng, smooth_ms=1.9 if deep else 1.4)
    skin *= 0.10 + 0.90 * np.power(skin_flutter, 2.0)

    droplets = np.zeros(n, dtype=np.float32)
    count = 22 if deep else 9
    latest = min(duration * 0.86, 0.345 if deep else 0.178)
    for _ in range(count):
        start_s = float(rng.uniform(0.030 if deep else 0.024, max(0.032, latest)))
        droplets += _foley_packet(
            n,
            sample_rate,
            rng,
            start_s=start_s,
            duration_s=float(rng.uniform(0.0045, 0.015 if deep else 0.011)),
            color="pink",
            filter_kind="bandpass",
            filter_hz=float(rng.uniform(780.0 if deep else 1050.0, 3900.0 if deep else 4700.0)),
            q=float(rng.uniform(0.22, 0.46)),
            attack_s=float(rng.uniform(0.0008, 0.0026)),
            release_power=float(rng.uniform(1.35, 2.25)),
        ) * float(rng.uniform(0.035, 0.105 if deep else 0.080))

    body = (wobble_a * 0.66 + wobble_b * 0.42) * burst_gate
    body += skin * burst_gate * (0.42 if deep else 0.36)
    body += droplets
    body = np.tanh(body * (1.72 if deep else 1.58))
    body = _filter_mono(body, sample_rate, kind="highpass", hz=190.0 if deep else 260.0)
    body = _filter_mono(body, sample_rate, kind="lowpass", hz=4300.0 if deep else 5000.0)
    return _normalized(body)

def _flesh_cut_components(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    deep: bool,
) -> dict[str, np.ndarray]:
    """Build material-specific stems for an exaggerated biological slash."""
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    duration = max(1e-6, n / float(sample_rate))

    # Click/filter squish remains useful, but is now only one component among
    # literal fluid, muscle, vessel, pulp, and suction gestures.
    compression = np.zeros(n, dtype=np.float32)
    events = (
        ((0.056, 0.096, 0.72), (0.142, 0.150, 1.00), (0.246, 0.150, 0.84))
        if deep
        else ((0.039, 0.067, 1.00), (0.099, 0.057, 0.52))
    )
    for center_s, width_s, amp in events:
        compression += float(amp) * _rounded_event(
            t,
            center_s=min(center_s, duration * 0.80),
            width_s=min(width_s, duration * 0.58),
            power=0.60,
        )
    compression = np.clip(compression, 0.0, 1.55)

    edges = _subaudio_square_edges(
        n,
        sample_rate,
        rng,
        rate_points_hz=(11.0, 23.0, 14.0, 27.0, 9.0) if deep else (16.0, 31.0, 19.0, 12.0),
        jitter=0.24,
        click_decay_ms=2.9 if deep else 2.2,
    )
    squish = _click_resonator_motion(
        edges,
        sample_rate,
        rng,
        center_points_hz=(620.0, 270.0, 940.0, 350.0, 590.0) if deep else (980.0, 430.0, 1280.0, 570.0),
        q=0.88 if deep else 0.82,
        segments=24 if deep else 18,
        delay_taps_ms=((4.5, 0.18), (10.2, 0.08)) if deep else ((3.1, 0.15), (7.0, 0.06)),
    ) * compression

    blood = _blood_slurry(n, sample_rate, rng, deep=deep)
    muscle = _muscle_and_vein_tears(n, sample_rate, rng, deep=deep)
    pulp = _brain_pulp(n, sample_rate, rng, deep=deep)
    suction = (
        _wet_suction_release(n, sample_rate, rng, deep=True)
        if deep
        else _light_juice_slurp_tail(n, sample_rate, rng)
    )
    splatter = _oscillating_wet_splatter(n, sample_rate, rng, deep=deep)
    bubbles = _wet_bubble_field(
        n,
        sample_rate,
        rng,
        start_s=0.038 if deep else 0.028,
        finish_s=min(0.330 if deep else 0.170, duration * 0.91),
        count=30 if deep else 11,
        min_hz=145.0 if deep else 220.0,
        max_hz=1050.0 if deep else 1450.0,
        gain=1.0,
    )

    # Soft wet entry and a narrow blade shear make this an actual cut rather
    # than a hand squeezing a bag of liquid.
    entry = _foley_packet(
        n,
        sample_rate,
        rng,
        start_s=0.003,
        duration_s=0.043 if deep else 0.029,
        color="pink",
        filter_kind="bandpass",
        filter_hz=1550.0 if deep else 2050.0,
        q=0.32,
        attack_s=0.0065 if deep else 0.0045,
        release_power=1.55,
    )
    blade_gate = _soft_gate(
        t,
        start_s=0.018 if deep else 0.014,
        finish_s=min(0.292 if deep else 0.147, duration * 0.82),
        attack_s=0.024 if deep else 0.014,
        release_s=0.047 if deep else 0.024,
        power=0.72,
    )
    blade = _windowed_dynamic_noise(
        n,
        sample_rate,
        rng,
        kind="bandpass",
        min_hz=820.0 if deep else 1180.0,
        max_hz=3600.0 if deep else 4400.0,
        color="pink",
        q=0.35,
        segments=30 if deep else 22,
    )
    blade_flutter = _smooth_random_curve(n, sample_rate, rng, smooth_ms=2.6)
    blade = blade * blade_gate * (0.10 + 0.95 * np.power(blade_flutter, 2.1))

    return {
        "squish": _normalized(squish),
        "blood": _normalized(blood),
        "muscle": _normalized(muscle),
        "pulp": _normalized(pulp),
        "suction": _normalized(suction),
        "splatter": _normalized(splatter),
        "bubbles": _normalized(bubbles),
        "entry": _normalized(entry),
        "blade": _normalized(blade),
    }


def _smoothed_activity_envelope(
    audio: np.ndarray,
    sample_rate: int,
    *,
    cutoff_hz: float,
) -> np.ndarray:
    """Return a normalized control envelope for stem-aware dynamic mixing."""
    rectified = np.abs(np.asarray(audio, dtype=np.float32))
    nyquist = sample_rate * 0.5
    cutoff = float(np.clip(cutoff_hz, 4.0, nyquist * 0.25))
    sos = signal.butter(2, cutoff / nyquist, btype="lowpass", output="sos")
    envelope = signal.sosfilt(sos, rectified).astype(np.float32)
    peak = float(np.max(envelope)) if envelope.size else 0.0
    if peak > 1e-9:
        envelope /= peak
    return np.sqrt(np.clip(envelope, 0.0, 1.0)).astype(np.float32)


def _delay_mono(audio: np.ndarray, samples: int) -> np.ndarray:
    """Delay one mono stem without wrapping its tail back to the beginning."""
    audio = np.asarray(audio, dtype=np.float32)
    samples = max(0, int(samples))
    if samples <= 0:
        return audio.copy()
    out = np.zeros_like(audio)
    if samples < audio.size:
        out[samples:] = audio[:-samples]
    return out


def _biological_flesh_cut(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    deep: bool,
) -> np.ndarray:
    """Polished wet tissue cut with a centered body and lateral splatter.

    The biological mass, blade travel, and suction stay near mono for force and
    compatibility.  Only the brighter droplets and oscillating splatter receive
    a few milliseconds of asymmetric delay.  Stem-aware ducking briefly clears
    blade and tear noise around each splatter pulse so the wet motion reads as a
    sequence of physical events instead of one flat wall of broadband noise.
    """
    stems = _flesh_cut_components(n, sample_rate, rng, deep=deep)

    splatter_activity = _smoothed_activity_envelope(
        stems["splatter"],
        sample_rate,
        cutoff_hz=48.0 if deep else 66.0,
    )
    suction_activity = _smoothed_activity_envelope(
        stems["suction"],
        sample_rate,
        cutoff_hz=34.0 if deep else 46.0,
    )

    # Preserve the cutting edge, but briefly make room for each wet splatter
    # pulse and the final pull-free.  This is intentionally modest: the cue must
    # still read as a blade cut, not a bag of liquid being squeezed.
    blade = stems["blade"] * (
        1.0 - splatter_activity * (0.34 if deep else 0.28)
    )
    muscle = stems["muscle"] * (
        1.0 - splatter_activity * (0.14 if deep else 0.10)
    )
    bubbles = stems["bubbles"] * (
        0.82 + splatter_activity * (0.30 if deep else 0.24)
    )

    # Keep the weight and gross wet motion in the center.  A low-passed portion
    # of the oscillating splatter remains here so the sound does not disappear
    # or phase out when folded to mono.
    splatter_body = _filter_mono(
        stems["splatter"],
        sample_rate,
        kind="lowpass",
        hz=1680.0 if deep else 2050.0,
    )
    splatter_bright = _filter_mono(
        stems["splatter"],
        sample_rate,
        kind="highpass",
        hz=760.0 if deep else 980.0,
    )
    droplets_bright = _filter_mono(
        bubbles,
        sample_rate,
        kind="highpass",
        hz=520.0 if deep else 720.0,
    )

    if deep:
        center = (
            stems["blood"] * 1.06
            + stems["suction"] * 1.08
            + splatter_body * 0.31
            + stems["pulp"] * 0.75
            + muscle * 0.58
            + bubbles * 0.52
            + stems["squish"] * 0.20
            + blade * 0.47
            + stems["entry"] * 0.19
        )
        side_gain = 0.46
        side_delay_a = ms_to_samples(1.4, sample_rate)
        side_delay_b = ms_to_samples(3.1, sample_rate)
        drive = 1.37
        highpass_hz = 43.0
        lowpass_hz = 4350.0
        fade_in_s = 0.007
        fade_out_s = 0.025
    else:
        # The light hit is a restrained wet nick followed by a deliberately
        # exposed juice-slurp tail.  Keep the initial tissue mass and splatter
        # sparse so the late suction, not the contact transient, defines it.
        center = (
            stems["blood"] * 0.61
            + stems["suction"] * 1.24
            + splatter_body * 0.18
            + stems["pulp"] * 0.31
            + muscle * 0.36
            + bubbles * 0.23
            + stems["squish"] * 0.15
            + blade * 0.44
            + stems["entry"] * 0.13
        )
        side_gain = 0.27
        side_delay_a = ms_to_samples(0.9, sample_rate)
        side_delay_b = ms_to_samples(1.9, sample_rate)
        drive = 1.19
        highpass_hz = 74.0
        lowpass_hz = 4650.0
        fade_in_s = 0.0055
        fade_out_s = 0.022

    # The side image contains only upper wet detail.  Cross-delayed versions
    # create a compact close-mic spread rather than a large reverb-like width.
    wet_detail = splatter_bright * 0.94 + droplets_bright * 0.58
    wet_detail *= 0.72 + 0.42 * splatter_activity
    left_wet = wet_detail + _delay_mono(wet_detail, side_delay_b) * 0.22
    right_wet = _delay_mono(wet_detail, side_delay_a) * 0.92
    right_wet += _delay_mono(droplets_bright, side_delay_b) * 0.24

    # Let the final suction pull narrow toward the center; this makes the blade
    # withdrawal feel attached to the target rather than smeared across space.
    side_narrow = 1.0 - suction_activity * (0.34 if deep else 0.28)
    left = center + left_wet * side_narrow * side_gain
    right = center + right_wet * side_narrow * side_gain

    stereo = np.stack([left, right], axis=0).astype(np.float32)
    stereo = np.tanh(stereo * drive)
    for channel in range(stereo.shape[0]):
        stereo[channel] = _filter_mono(
            stereo[channel], sample_rate, kind="highpass", hz=highpass_hz
        )
        stereo[channel] = _filter_mono(
            stereo[channel], sample_rate, kind="lowpass", hz=lowpass_hz
        )

    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    duration = max(1e-6, n / float(sample_rate))
    fade_in = np.sin(np.clip(t / fade_in_s, 0.0, 1.0) * np.pi * 0.5) ** 1.25
    fade_out = np.sin(np.clip((duration - t) / fade_out_s, 0.0, 1.0) * np.pi * 0.5) ** 1.35
    stereo *= (fade_in * fade_out)[None, :]
    return _normalized(stereo)

def _light_flesh_squish_cut(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    preset: str = "balanced",
) -> np.ndarray:
    """Light wet tissue slice; ``preset`` is retained for API compatibility."""
    del preset
    return _biological_flesh_cut(n, sample_rate, rng, deep=False)


def _deep_flesh_squish_cut(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    preset: str = "balanced",
) -> np.ndarray:
    """Deep wet tissue slash with blood slosh, tearing, pulp, and suction."""
    del preset
    return _biological_flesh_cut(n, sample_rate, rng, deep=True)

def _flesh_impact(
    n: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    deep: bool,
) -> np.ndarray:
    if deep:
        return _deep_flesh_squish_cut(n, sample_rate, rng)
    return _light_flesh_squish_cut(n, sample_rate, rng)

def _wet_impact(n: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Compatibility name for the light flesh-contact recipe."""
    return _flesh_impact(n, sample_rate, rng, deep=False)


def _robot_crunch(n: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    body = _filter_mono(_brown(n, rng), sample_rate, kind="lowpass", hz=560.0)
    body *= np.exp(-t / 0.055)
    crunch = _grain_train(
        n,
        sample_rate,
        rng,
        count=34,
        decay_ms=4.0,
        spread_ms=min(78.0, n / sample_rate * 1000.0),
        start_ms=2.0,
    )
    crunch = _filter_mono(crunch, sample_rate, kind="bandpass", hz=2050.0, q=0.68)
    shell = np.zeros(n, dtype=np.float32)
    for freq, decay, amp in ((470.0, 0.055, 0.50), (815.0, 0.040, 0.34), (1370.0, 0.028, 0.20)):
        phase = float(rng.uniform(0.0, np.pi * 2.0))
        shell += amp * np.sin(2.0 * np.pi * freq * t + phase) * np.exp(-t / decay)
    snap = _grain_train(n, sample_rate, rng, count=5, decay_ms=10.0, spread_ms=18.0)
    return _normalized(body * 0.78 + crunch * 0.62 + shell * 0.34 + snap * 0.22)


def _metal_chink(n: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Short, bright blade-on-thin-metal contact with almost no bell tail."""
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    strike = _grain_train(n, sample_rate, rng, count=4, decay_ms=4.5, spread_ms=3.5)
    strike = _filter_mono(strike, sample_rate, kind="bandpass", hz=3200.0, q=0.82)
    scrape = _moving_band_noise(
        n,
        sample_rate,
        rng,
        start_hz=6500.0,
        end_hz=2200.0,
        q=0.95,
        color="white",
        pieces=8,
    )
    scrape *= np.exp(-t / 0.025)
    ring = np.zeros(n, dtype=np.float32)
    for freq, decay, amp in (
        (2380.0, 0.072, 0.54),
        (3670.0, 0.052, 0.39),
        (5210.0, 0.035, 0.25),
        (6980.0, 0.024, 0.13),
    ):
        phase = float(rng.uniform(0.0, np.pi * 2.0))
        ring += amp * np.sin(2.0 * np.pi * freq * t + phase) * np.exp(-t / decay)
    return _normalized(np.tanh((strike * 0.78 + scrape * 0.34 + ring) * 1.15))


def _metal_gong(n: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Low, broad, inharmonic impact for heavy or large metal bodies."""
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    body = _filter_mono(_brown(n, rng), sample_rate, kind="lowpass", hz=420.0)
    body *= np.exp(-t / 0.090)
    strike = _grain_train(n, sample_rate, rng, count=3, decay_ms=11.0, spread_ms=5.0)
    strike = _filter_mono(strike, sample_rate, kind="bandpass", hz=760.0, q=0.62)
    ring = np.zeros(n, dtype=np.float32)
    for freq, decay, amp, wobble in (
        (178.0, 0.48, 0.58, 2.1),
        (267.0, 0.39, 0.46, 2.7),
        (401.0, 0.31, 0.35, 3.2),
        (615.0, 0.23, 0.25, 4.1),
        (925.0, 0.16, 0.14, 5.0),
    ):
        phase0 = float(rng.uniform(0.0, np.pi * 2.0))
        phase = 2.0 * np.pi * freq * t + 0.018 * np.sin(2.0 * np.pi * wobble * t) + phase0
        ring += amp * np.sin(phase) * np.exp(-t / decay)
    grit = _moving_band_noise(
        n,
        sample_rate,
        rng,
        start_hz=1450.0,
        end_hz=420.0,
        q=0.60,
        color="pink",
        pieces=9,
    )
    grit *= np.exp(-t / 0.075)
    return _normalized(np.tanh((body * 0.70 + strike * 0.62 + ring + grit * 0.24) * 1.08))


def _metal_hit(n: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Compatibility name for the compact high metal contact."""
    return _metal_chink(n, sample_rate, rng)


def _pogo_impact(n: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Low rubber thud followed by a separate damped spring rebound."""
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    thud = _filter_mono(_brown(n, rng), sample_rate, kind="lowpass", hz=300.0)
    thud *= np.exp(-t / 0.068)
    strike = _grain_train(n, sample_rate, rng, count=3, decay_ms=9.0, spread_ms=6.0)
    strike = _filter_mono(strike, sample_rate, kind="lowpass", hz=680.0)

    # Keep the spring below the old toy-like register and make it a distinct
    # rebound after the ground contact, not one long exposed chirp.
    delay_s = 0.024
    local_t = np.maximum(0.0, t - delay_s)
    rise_end = 0.060
    fall_end = 0.155
    rise = np.clip(local_t / rise_end, 0.0, 1.0)
    f_rise = 76.0 * (154.0 / 76.0) ** rise
    fall = np.clip((local_t - rise_end) / max(0.001, fall_end - rise_end), 0.0, 1.0)
    f_fall = 154.0 * (96.0 / 154.0) ** fall
    freq = np.where(local_t <= rise_end, f_rise, f_fall)
    phase = 2.0 * np.pi * np.cumsum(freq) / float(sample_rate)
    spring_env = (local_t > 0.0).astype(np.float32)
    spring_env *= np.minimum(1.0, local_t / 0.006) * np.exp(-local_t / 0.105)
    spring = (
        np.sin(phase)
        + 0.14 * np.sin(2.0 * phase + 0.4)
        + 0.04 * np.sin(3.0 * phase + 1.1)
    ) * spring_env

    rubber = _viscous_texture(n, sample_rate, rng, cutoff_hz=360.0, pulse_hz=13.0)
    rubber *= np.minimum(1.0, t / 0.004) * np.exp(-t / 0.090)
    return _normalized(
        np.tanh((thud * 1.00 + strike * 0.34 + spring * 0.48 + rubber * 0.48) * 1.22)
    )

def _blade_impact(n: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Neutral, compact blade bite used only as a safety fallback."""
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    body = _filter_mono(_brown(n, rng), sample_rate, kind="lowpass", hz=520.0)
    body *= np.exp(-t / 0.046)
    crack = _filter_mono(_white(n, rng), sample_rate, kind="bandpass", hz=2250.0, q=0.68)
    crack *= np.exp(-t / 0.020)
    bite = _grain_train(
        n,
        sample_rate,
        rng,
        count=18,
        decay_ms=4.2,
        spread_ms=min(42.0, n / sample_rate * 1000.0),
        start_ms=1.0,
    )
    bite = _filter_mono(bite, sample_rate, kind="bandpass", hz=1280.0, q=0.58)
    drag = _moving_band_noise(
        n,
        sample_rate,
        rng,
        start_hz=1850.0,
        end_hz=520.0,
        q=0.62,
        color="pink",
        pieces=9,
    )
    drag *= np.exp(-t / 0.058)
    return _normalized(np.tanh((body * 0.73 + crack * 0.67 + bite * 0.58 + drag * 0.40) * 1.35))

def render_noise_layer(layer: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    sample_rate = int(context["sample_rate"])
    channels = int(context["channels"])
    duration_ms = float(
        layer.get("duration_ms", float(context.get("duration_seconds", 0.1)) * 1000.0)
    )
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
    elif mode in {"air_sweep", "blade_air", "whoosh"}:
        mono = _air_sweep(
            n,
            sample_rate,
            rng,
            start_hz=float(layer.get("start_hz", 6200.0)),
            end_hz=float(layer.get("end_hz", 850.0)),
            q=float(layer.get("q", 0.82)),
        )
    elif mode in {"wet_impact", "flesh", "flesh_hit", "flesh_light"}:
        mono = _flesh_impact(n, sample_rate, rng, deep=False)
    elif mode in {"flesh_deep", "deep_flesh", "flesh_heavy"}:
        mono = _flesh_impact(n, sample_rate, rng, deep=True)
    elif mode in {"robot_crunch", "machine_crunch", "robot_hit"}:
        mono = _robot_crunch(n, sample_rate, rng)
    elif mode in {"metal_hit", "metal_ching", "metal", "metal_chink"}:
        mono = _metal_chink(n, sample_rate, rng)
    elif mode in {"metal_gong", "metal_heavy", "heavy_metal"}:
        mono = _metal_gong(n, sample_rate, rng)
    elif mode in {"pogo_impact", "spring_impact", "bounce"}:
        mono = _pogo_impact(n, sample_rate, rng)
    elif mode in {"blade_impact", "slash_impact"}:
        mono = _blade_impact(n, sample_rate, rng)
    else:
        raise ValueError(
            f"unknown noise mode {mode!r}; expected burst, grains, thud, scrape, "
            "air_sweep, flesh_light, flesh_deep, robot_crunch, metal_chink, "
            "metal_gong, pogo_impact, or blade_impact"
        )

    audio = np.asarray(mono, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]
    return stereoize(audio, channels=channels)
