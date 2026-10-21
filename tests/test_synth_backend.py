from __future__ import annotations

import numpy as np

from ambition_sfx_renderer.backends.synth_backend import render_synth_layer


def _context() -> dict:
    return {
        "sample_rate": 48_000,
        "channels": 2,
        "duration_seconds": 0.1,
        "seed": 123,
    }


def test_synth_layer_is_deterministic_stereo_and_bounded() -> None:
    layer = {
        "kind": "synth_tone",
        "duration_ms": 100,
        "waveform": "bell",
        "frequency_hz": {"start": 220.0, "end": 880.0, "curve": "exp"},
        "stereo_spread_cents": 11.0,
        "harmonics": [
            {"ratio": 1.0, "gain_db": 0.0},
            {"ratio": 1.5, "gain_db": -12.0},
        ],
    }
    first = render_synth_layer(layer, _context())
    second = render_synth_layer(layer, _context())
    assert first.shape == (2, 4_800)
    assert np.array_equal(first, second)
    assert float(np.max(np.abs(first))) <= 1.0
    assert not np.array_equal(first[0], first[1])


def test_synth_layer_supports_scalar_frequency() -> None:
    audio = render_synth_layer(
        {"kind": "synth_tone", "duration_ms": 20, "frequency_hz": 440.0},
        _context(),
    )
    assert audio.shape == (2, 960)
    assert float(np.sqrt(np.mean(np.square(audio)))) > 0.1


def test_a_pitch_curve_of_points_glides_through_each_point() -> None:
    # Up to 600 Hz at a quarter of the layer, then down to 450 Hz: the
    # contour of a howl. Measured by zero crossings in three windows.
    layer = {
        "kind": "synth_tone",
        "duration_ms": 2000,
        "frequency_hz": {"points": [[0.0, 330.0], [0.25, 600.0], [1.0, 450.0]]},
        "amplitude_points": [[0.0, 0.2], [0.25, 1.0], [1.0, 0.5]],
    }
    ctx = dict(_context(), channels=1)
    audio = render_synth_layer(layer, ctx)[0]

    def hz(u: float) -> float:
        i = int(u * len(audio))
        w = audio[i : i + 2400]
        return float(np.count_nonzero(np.diff(np.signbit(w)))) / 2.0 / 0.05

    assert abs(hz(0.25) - 600.0) < 20.0
    assert abs(hz(0.97) - 455.0) < 20.0
    assert hz(0.0) < hz(0.12) < hz(0.25)
