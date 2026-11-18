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
    harmonic_db:               # optional: a level curve (dB) for a harmonic,
      "2": [[0, -35], [0.6, -35], [0.9, -7]]   # added to its formant shape
    breath_curve: [[0, 1], [1, 1]]  # optional: scales `breath` over the layer

A reference howl (2026-10-21, Jon's pick: freesound 810171 by mokasza,
CC BY 4.0, an AI howl) is near a pure tone: its 2nd harmonic is 32-40 dB
under f0 while it holds, and grows to -7 dB as it falls. What makes it a
wolf there is the fine motion: pitch jitter of ~14 cents RMS that is FAST
(15-120 Hz), not a slow wobble (Jon of a slow one: "it just wobbles"), and
a shimmer of ~11% at 8-40 Hz.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


def _curve(n: int, points) -> np.ndarray:
    xs, ys = zip(*sorted((float(a), float(b)) for a, b in points))
    return np.interp(np.linspace(0.0, 1.0, n, endpoint=False), xs, ys)


def _log_curve(n: int, points) -> np.ndarray:
    xs, ys = zip(*sorted((float(a), float(b)) for a, b in points))
    return np.exp(np.interp(np.linspace(0.0, 1.0, n, endpoint=False), xs, np.log(ys)))


def _walk(rng: np.random.Generator, n: int, rate_hz: float, sr: int) -> np.ndarray:
    """Smooth interpolated random values with a centered mean."""
    k = max(2, int(n * rate_hz / sr) + 2)
    pts = rng.standard_normal(k)
    t = np.linspace(0, k - 1, n)
    # Cubic-ish smoothing: interpolate, then a short moving average.
    x = np.interp(t, np.arange(k), pts)
    # A wider window would reduce every sample with zero padding.
    win = max(1, min(n, int(sr / rate_hz / 2)))
    return uniform_filter1d(x, size=win, mode="constant", cval=0.0)


def _resonance(f: np.ndarray, centre: np.ndarray, bandwidth: float) -> np.ndarray:
    return 1.0 / (1.0 + np.square((f - centre) / (0.5 * bandwidth)))


def render_howl(n: int, sr: int, layer: dict[str, Any]) -> np.ndarray:
    if "howl" in layer:
        return render_described_howl(n, sr, layer)
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
    harmonic_db = {str(key): value for key, value in (layer.get("harmonic_db") or {}).items()}
    voice = np.zeros(n)
    for k in range(1, int(layer.get("harmonics", 10)) + 1):
        fk = k * f0
        shape = floor + sum(g * _resonance(fk, c, bw) for c, bw, g in centres)
        gain = shape * 10.0 ** (tilt * np.log2(k) / 20.0)
        if str(k) in harmonic_db:
            gain = gain * 10.0 ** (_curve(n, harmonic_db[str(k)]) / 20.0)
        # Above the band the ear gives a howl, nothing: no harmonic past 6 kHz.
        gain = np.where(fk < 6000.0, gain, 0.0)
        voice += gain * np.sin(k * phase + rng.uniform(0.0, 2.0 * np.pi))
    voice /= max(1e-9, float(np.max(np.abs(voice))))

    # A rough onset: a subharmonic (period doubling) and more breath, fading.
    onset = max(1, min(n, int(float(layer.get("onset_ms", 180.0)) * 0.001 * sr)))
    rough = np.zeros(n)
    rough[:onset] = np.linspace(1.0, 0.0, onset) ** 2
    sub = np.sin(0.5 * phase + 0.3) * float(layer.get("onset_rough", 0.35)) * rough

    # Breath through the same formants, pulsed with the glottis.
    noise = rng.standard_normal(n)
    breath = np.zeros(n)
    for c, bw, g in centres:
        mid = float(np.median(c))
        lo, hi = max(40.0, mid - bw), min(0.45 * sr, mid + bw)
        # 4th order: a breath that is loud (a yelp) stays in its band.
        sos = signal.butter(4, [lo / (0.5 * sr), hi / (0.5 * sr)], btype="band", output="sos")
        breath += g * signal.sosfilt(sos, noise)
    breath /= max(1e-9, float(np.max(np.abs(breath))))
    pulse = 0.55 + 0.45 * np.cos(phase)
    breath_level = float(layer.get("breath", 0.06)) * (1.0 + 2.5 * rough)
    if layer.get("breath_curve"):
        breath_level = breath_level * _curve(n, layer["breath_curve"])

    out = (voice + sub + breath * pulse * breath_level) * amp * (1.0 + shimmer)
    return out.astype(np.float32)


# ── A howl from its description ──────────────────────────────────────────────
#
# A howl, said in a dozen numbers a person can read. The shape is the one
# every wolf's howl has (Jon's spectrogram and the reference, 2026-10-21):
#
#   yelp  -> a short rough start in two voices, a fifth apart, noisy;
#   rise  -> up onto the held note;
#   hold  -> the held note, near a pure tone, sagging a little;
#   fall  -> the pitch falls, the voice darkens (its 2nd harmonic grows)
#            and it grows quieter;
#   droop -> the voice gives out.
#
#     mode: howl
#     seed: 851
#     howl:
#       pitch_hz: 650            # the held note
#       yelp_s: 0.5              # 0 for no yelp
#       yelp_low: 0.66           # the yelp's second voice, as a part of the pitch
#       yelp_noise_db: -14       # the yelp's breath noise, against the voice
#       rise_s: 0.12
#       hold_s: 1.8
#       sag_semitones: 0.8       # down over the hold
#       fall_s: 1.5
#       fall_semitones: 3.6
#       fall_db: -25             # quieter by the end of the fall
#       droop_s: 0.3
#       droop_semitones: 1.6
#       bright_db: -9            # the 2nd harmonic while it holds (on top of
#       dark_db: 15              #   the voice's tilt), and after it darkens
#       jitter_cents: 15         # RMS, fast: never a slow wobble
#       shimmer: 0.12
#       breath: 0.05


def _semis(hz: float, semitones: float) -> float:
    return hz * 2.0 ** (semitones / 12.0)


def _at(segments, t: float) -> float:
    """Piecewise straight: `segments` is [(t, value), ...] in seconds."""
    ts, vs = zip(*segments)
    return float(np.interp(t, ts, vs))


def described_howl_seconds(h: dict[str, Any]) -> float:
    """How long the described howl is, from its first sound to its end."""
    yelp = float(h.get("yelp_s", 0.5))
    return yelp + float(h.get("rise_s", 0.12)) + float(h.get("hold_s", 1.8)) + float(h.get("fall_s", 1.5)) + float(h.get("droop_s", 0.3))


def render_described_howl(n: int, sr: int, layer: dict[str, Any]) -> np.ndarray:
    h = dict(layer["howl"])
    pitch = float(h.get("pitch_hz", 650.0))
    k = pitch / 650.0
    yelp = float(h.get("yelp_s", 0.5))
    rise, hold = float(h.get("rise_s", 0.12)), float(h.get("hold_s", 1.8))
    fall, droop = float(h.get("fall_s", 1.5)), float(h.get("droop_s", 0.3))
    seed = int(layer.get("seed", 0))
    out = np.zeros(n)

    def place(voice: dict[str, Any], start_s: float, dur_s: float, gain_db: float = 0.0) -> None:
        i = int(start_s * sr)
        m = max(1, min(n - i, int(dur_s * sr)))
        if i >= n:
            return
        out[i:i + m] += render_howl(m, sr, voice).astype(np.float64) * 10.0 ** (gain_db / 20.0)

    # The held voice: from the end of the yelp to the end.
    t0 = max(0.0, yelp - 0.05)
    end = t0 + rise + hold + fall + droop
    peak_t, sag_t, fall_t = t0 + rise, t0 + rise + hold, t0 + rise + hold + fall
    sagged = _semis(pitch, -float(h.get("sag_semitones", 0.8)))
    fallen = _semis(sagged, -float(h.get("fall_semitones", 3.6)))
    drooped = _semis(fallen, -float(h.get("droop_semitones", 1.6)))
    pitch_path = [(t0, pitch * (0.92 if yelp > 0 else 0.83)), (peak_t, pitch), (sag_t, sagged),
                  (sag_t + 0.35 * fall, _semis(sagged, -0.25 * float(h.get("fall_semitones", 3.6)))), (fall_t, fallen), (end, drooped)]
    fall_db = float(h.get("fall_db", -25.0))
    loud_path = [(t0, -6.0), (t0 + 0.1, 0.0), (sag_t, -1.0), (sag_t + 0.2 * fall, -11.0 * fall_db / -25.0), (fall_t, fall_db),
                 (fall_t + 0.6 * droop, fall_db - 14.0), (end, -80.0)]
    bright, dark = float(h.get("bright_db", -9.0)), float(h.get("dark_db", 15.0))
    span = end - t0
    frac = lambda t: round((t - t0) / span, 5)
    pts = 40
    times = [t0 + span * j / (pts - 1) for j in range(pts)]
    held = {
        "seed": seed,
        "f0_hz": [[frac(t), round(_at(pitch_path, t), 2)] for t in sorted({*times, *(t for t, _ in pitch_path)})],
        "amp": [[frac(t), round(10 ** (_at(loud_path, t) / 20.0), 5)] for t in sorted({*times, *(t for t, _ in loud_path)})],
        "mouth": [[0, 0.5], [1, 0.5]], "harmonics": 5, "tilt_db_per_octave": -14.0,
        # One resonance on the held note: it is near a pure tone.
        "formants": [[pitch, pitch, 600.0 * k, 1.0]], "floor": 0.0006,
        "harmonic_db": {"2": [[0, bright], [frac(sag_t), bright], [frac(sag_t + 0.35 * fall), dark], [1, dark]],
                        "3": [[0, -18.0], [1, -18.0]]},
        # Calibrated: `jitter_cents` 15 measures ~15 cents RMS on the held note.
        "jitter": float(h.get("jitter_cents", 15.0)) / 1283.0, "jitter_hz": 90, "drift_cents": 0.0,
        "shimmer": float(h.get("shimmer", 0.12)), "shimmer_hz": 24,
        "breath": float(h.get("breath", 0.05)),
        "breath_curve": [[0, 1], [frac(sag_t + 0.2 * fall), 1], [frac(sag_t + 0.5 * fall), 3], [1, 5]],
        "onset_ms": 1, "onset_rough": 0.0,
    }
    place(held, t0, span)

    if yelp > 0:
        rough = {"jitter": 0.02, "jitter_hz": 90, "drift_cents": 3, "shimmer": 0.25, "shimmer_hz": 30,
                 "mouth": [[0, 0.6], [1, 0.6]], "harmonics": 8, "tilt_db_per_octave": -9.0,
                 "formants": [[800 * k, 800 * k, 420 * k, 1.0]], "amp": [[0, 0], [0.12, 1], [0.8, 0.9], [1, 0]]}
        # The first voice: on the pitch, near pure, mostly breath.
        hi_s = 0.64 * yelp
        place(dict(rough, seed=seed + 1, f0_hz=[[0, 0.86 * pitch], [0.3, 1.015 * pitch], [1, pitch]], floor=0.0005,
                   harmonic_db={"2": [[0, -30], [1, -30]], "3": [[0, -30], [1, -30]]}, breath=0.25,
                   onset_ms=hi_s * 600.0, onset_rough=0.5), 0.0, hi_s)
        # The second voice, under it, rich, gliding up into the held note.
        low = float(h.get("yelp_low", 0.66))
        place(dict(rough, seed=seed + 2, f0_hz=[[0, low * pitch], [0.6, 0.985 * low * pitch], [1, 0.91 * pitch]], floor=0.01,
                   harmonic_db={"2": [[0, 6], [1, 6]], "3": [[0, 6], [1, 6]]}, breath=0.12,
                   onset_ms=0.6 * yelp * 600.0, onset_rough=0.5), 0.46 * yelp, 0.6 * yelp, gain_db=-2.0)
        # Its breath: a blob of noise under ~1.2 kHz (for this pitch).
        rng = np.random.default_rng(seed + 3)
        i, m = int(0.02 * yelp * sr), int(0.76 * yelp * sr)
        m = max(1, min(n - i, m))
        sos = signal.butter(4, [max(40.0, 330.0 * k) / (0.5 * sr), min(0.45 * sr, 1250.0 * k) / (0.5 * sr)], btype="band", output="sos")
        air = signal.sosfilt(sos, rng.standard_normal(m))
        air /= max(1e-9, float(np.max(np.abs(air))))
        env = np.minimum(1.0, np.minimum(np.arange(m) / max(1, 0.05 * m), (m - np.arange(m)) / max(1, 0.4 * m)))
        out[i:i + m] += air * env * 10.0 ** (float(h.get("yelp_noise_db", -14.0)) / 20.0) * 0.5
    return out.astype(np.float32)
