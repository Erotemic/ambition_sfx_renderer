from __future__ import annotations

import numpy as np

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
