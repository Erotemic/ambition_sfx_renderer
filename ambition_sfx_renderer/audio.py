"""Small audio-buffer utilities.

Internal convention: NumPy arrays are always channel-first, i.e. `(channels,
samples)`, float32, roughly in [-1, 1]. This matches DawDreamer and Pedalboard.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def db_to_amp(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def amp_to_db(amp: float, *, floor_db: float = -120.0) -> float:
    amp = float(abs(amp))
    if amp <= 0:
        return floor_db
    return max(floor_db, 20.0 * math.log10(amp))


def ensure_chans_first(audio: Any) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 2:
        # soundfile returns (samples, channels); DawDreamer/Pedalboard use (channels, samples).
        if arr.shape[0] > arr.shape[1] and arr.shape[1] <= 8:
            arr = arr.T
    else:
        raise ValueError(f"audio must be 1D or 2D, got shape={arr.shape!r}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def stereoize(audio: np.ndarray, channels: int = 2) -> np.ndarray:
    audio = ensure_chans_first(audio)
    if audio.shape[0] == channels:
        return audio
    if channels == 1:
        return np.mean(audio, axis=0, keepdims=True).astype(np.float32)
    if audio.shape[0] == 1 and channels == 2:
        return np.repeat(audio, 2, axis=0).astype(np.float32)
    if audio.shape[0] > channels:
        return audio[:channels]
    # Pad additional channels with silence.
    out = np.zeros((channels, audio.shape[1]), dtype=np.float32)
    out[: audio.shape[0], :] = audio
    return out


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    return max(1, int(round(float(seconds) * int(sample_rate))))


def ms_to_samples(ms: float, sample_rate: int) -> int:
    return max(0, int(round(float(ms) * 0.001 * int(sample_rate))))


def fit_length(audio: np.ndarray, n_samples: int) -> np.ndarray:
    audio = ensure_chans_first(audio)
    n_samples = int(n_samples)
    if audio.shape[1] == n_samples:
        return audio
    out = np.zeros((audio.shape[0], n_samples), dtype=np.float32)
    n = min(audio.shape[1], n_samples)
    out[:, :n] = audio[:, :n]
    return out


def mix_into(base: np.ndarray, clip: np.ndarray, offset_samples: int) -> None:
    clip = ensure_chans_first(clip)
    offset = max(0, int(offset_samples))
    if offset >= base.shape[1]:
        return
    n = min(clip.shape[1], base.shape[1] - offset)
    chans = min(base.shape[0], clip.shape[0])
    base[:chans, offset : offset + n] += clip[:chans, :n]


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    audio = ensure_chans_first(audio)
    src_sr = int(src_sr)
    dst_sr = int(dst_sr)
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    try:
        from scipy.signal import resample_poly
    except Exception as ex:  # pragma: no cover - scipy is a declared dependency
        raise RuntimeError("scipy is required for resampling") from ex
    gcd = math.gcd(src_sr, dst_sr)
    up = dst_sr // gcd
    down = src_sr // gcd
    return resample_poly(audio, up, down, axis=1).astype(np.float32)


def curve_values(start: float, end: float, n: int, curve: str = "linear") -> np.ndarray:
    n = max(1, int(n))
    start = float(start)
    end = float(end)
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    curve = str(curve or "linear").lower()
    if curve in {"linear", "lin"}:
        y = x
    elif curve in {"smooth", "smoothstep"}:
        y = x * x * (3.0 - 2.0 * x)
    elif curve in {"exp", "exponential"}:
        # Exponential interpolation needs positive endpoints; fall back gracefully.
        if start > 0 and end > 0:
            return (start * (end / start) ** x).astype(np.float32)
        y = x * x
    elif curve in {"log", "logarithmic"}:
        y = np.sqrt(x)
    else:
        raise ValueError(f"unknown curve: {curve!r}")
    return (start + (end - start) * y).astype(np.float32)


def envelope(length: int, sample_rate: int, spec: dict[str, Any] | None) -> np.ndarray:
    if not spec:
        return np.ones(int(length), dtype=np.float32)
    n = int(length)
    env = np.ones(n, dtype=np.float32)
    attack = ms_to_samples(spec.get("attack_ms", 0.0), sample_rate)
    hold = ms_to_samples(spec.get("hold_ms", 0.0), sample_rate)
    release = ms_to_samples(spec.get("release_ms", 0.0), sample_rate)
    # If attack+hold+release is longer than the clip, reduce hold/release first.
    if attack + hold + release > n:
        overflow = attack + hold + release - n
        take = min(hold, overflow)
        hold -= take
        overflow -= take
        release = max(0, release - overflow)
    if attack > 0:
        x = np.linspace(0.0, 1.0, attack, endpoint=False, dtype=np.float32)
        env[:attack] *= x * x * (3.0 - 2.0 * x)
    if release > 0:
        x = np.linspace(1.0, 0.0, release, endpoint=True, dtype=np.float32)
        env[-release:] *= x * x * (3.0 - 2.0 * x)
    return env


def apply_envelope(audio: np.ndarray, sample_rate: int, spec: dict[str, Any] | None) -> np.ndarray:
    audio = ensure_chans_first(audio).copy()
    env = envelope(audio.shape[1], sample_rate, spec)
    audio *= env[None, :]
    return audio


def apply_gain(audio: np.ndarray, gain_db: float | None = None, gain: float | None = None) -> np.ndarray:
    audio = ensure_chans_first(audio).copy()
    if gain is not None:
        audio *= float(gain)
    if gain_db is not None:
        audio *= db_to_amp(float(gain_db))
    return audio


def apply_pan(audio: np.ndarray, pan: float | None = None, channels: int = 2) -> np.ndarray:
    audio = stereoize(audio, channels=channels).copy()
    if channels != 2 or pan is None:
        return audio
    pan = float(np.clip(pan, -1.0, 1.0))
    # Equal-power pan: -1 left, 0 center, +1 right.
    angle = (pan + 1.0) * math.pi / 4.0
    left = math.cos(angle) * math.sqrt(2.0)
    right = math.sin(angle) * math.sqrt(2.0)
    audio[0] *= left
    audio[1] *= right
    return audio


def peak_normalize(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    audio = ensure_chans_first(audio).copy()
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-12:
        return audio
    target = db_to_amp(target_db)
    return (audio * (target / peak)).astype(np.float32)


def hard_clip(audio: np.ndarray, limit: float = 1.0) -> np.ndarray:
    return np.clip(ensure_chans_first(audio), -float(limit), float(limit)).astype(np.float32)


def audit_stats(audio: np.ndarray) -> dict[str, float]:
    audio = ensure_chans_first(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    return {
        "peak_db": amp_to_db(peak),
        "rms_db": amp_to_db(rms),
        "peak_linear": peak,
        "rms_linear": rms,
    }
