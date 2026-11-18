from __future__ import annotations

import numpy as np
import pytest

from ambition_sfx_renderer.backends.howl_voice import render_howl

SR = 48_000
LAYER = {
    "mode": "howl",
    "seed": 5,
    "f0_hz": [[0, 400], [0.4, 600], [1, 450]],
    "amp": [[0, 0], [0.1, 1], [0.9, 1], [1, 0]],
    "mouth": [[0, 0.2], [0.4, 1.0], [1, 0.3]],
    "formants": [[780, 1150, 520, 1.0], [1650, 2100, 600, 0.22], [2900, 3300, 800, 0.05]],
    "floor": 0.008,
    "tilt_db_per_octave": -10,
}


def test_a_howl_is_deterministic_and_bounded() -> None:
    first = render_howl(SR, SR, LAYER)
    assert np.array_equal(first, render_howl(SR, SR, LAYER))
    assert np.all(np.isfinite(first)) and float(np.max(np.abs(first))) < 4.0


def test_at_its_peak_the_second_harmonic_is_the_brightest_line() -> None:
    # A wolf's howl (Jon's spectrogram, 2026-10-21): the 2nd harmonic over
    # f0, the 3rd faint. The open mouth puts the first formant on the 2nd.
    audio = render_howl(SR, SR, LAYER)
    seg = audio[int(0.38 * SR) : int(0.38 * SR) + 8192] * np.hanning(8192)
    spec = np.abs(np.fft.rfft(seg))
    freq = np.fft.rfftfreq(8192, 1 / SR)

    def level(hz: float) -> float:
        return float(spec[(freq > hz - 30) & (freq < hz + 30)].max())

    band = (freq > 450) & (freq < 750)
    f0 = float(freq[band][np.argmax(spec[band])])
    assert level(2 * f0) > level(f0) > level(3 * f0)
    assert 20 * np.log10(level(2 * f0) / level(3 * f0)) > 10.0


def test_a_short_howl_is_as_long_as_it_is_asked() -> None:
    # A slow drift's smoothing window is longer than this layer.
    assert render_howl(500, SR, LAYER).shape == (500,)


DESCRIBED = {
    "mode": "howl",
    "seed": 9,
    "howl": {"pitch_hz": 650, "yelp_s": 0.4, "rise_s": 0.12, "hold_s": 1.2, "sag_semitones": 0.8, "fall_s": 1.2,
             "fall_semitones": 3.6, "droop_s": 0.3, "bright_db": -9, "dark_db": 15},
}


def _pitch_and_h2(audio: np.ndarray, t: float) -> tuple[float, float]:
    n = 8192
    i = int(t * SR)
    spec = np.abs(np.fft.rfft(audio[i : i + n] * np.hanning(n)))
    freq = np.fft.rfftfreq(n, 1 / SR)
    band = (freq > 300) & (freq < 900)
    f0 = float(freq[band][np.argmax(spec[band])])
    level = lambda hz: float(spec[(freq > hz - 25) & (freq < hz + 25)].max())  # noqa: E731
    return f0, 20 * np.log10(level(2 * f0) / level(f0))


def test_a_described_howl_holds_its_pitch_then_falls_and_darkens() -> None:
    from ambition_sfx_renderer.backends.howl_voice import described_howl_seconds

    seconds = described_howl_seconds(DESCRIBED["howl"])
    audio = render_howl(int(seconds * SR), SR, DESCRIBED)
    assert np.array_equal(audio, render_howl(int(seconds * SR), SR, DESCRIBED))
    # In the hold: on its pitch (it sags by the end), near a pure tone.
    f0, h2 = _pitch_and_h2(audio, 0.4 + 0.3)
    assert abs(1200 * np.log2(f0 / 650.0)) < 60 and h2 < -25
    # Late in the fall: about 0.8 + 3.6 semitones down, and the 2nd harmonic
    # has grown.
    f0_late, h2_late = _pitch_and_h2(audio, 0.4 + 0.12 + 1.2 + 1.0)
    assert -5.2 < 12 * np.log2(f0_late / 650.0) < -3.0
    assert h2_late > h2 + 15


@pytest.mark.parametrize("n, rate, sr", [(1024, 5, 1000), (1024, 8, 1000), (31, 0.01, 1000), (1, 30, 48000)])
def test_walk_smoothing_is_a_centered_zero_padded_mean(n, rate, sr):
    from ambition_sfx_renderer.backends.howl_voice import _walk

    rng = np.random.default_rng(27)
    count = max(2, int(n * rate / sr) + 2)
    values = rng.standard_normal(count)
    interpolated = np.interp(np.linspace(0, count - 1, n), np.arange(count), values)
    window = max(1, min(n, int(sr / rate / 2)))
    reference = np.convolve(interpolated, np.ones(window) / window, mode="same")
    np.testing.assert_allclose(_walk(np.random.default_rng(27), n, rate, sr), reference,
                               rtol=1e-12, atol=1e-12)
