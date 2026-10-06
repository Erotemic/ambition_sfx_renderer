"""The `creature` noise mode: one seeded voice renders the same samples twice,
fills the layer, and is not silence. Its sound is Jon's (the T-rex auditions);
these hold only that a recipe renders it the same every time."""

import numpy as np

from ambition_sfx_renderer.backends.creature_voice import render_creature

LAYER = {
    "seed": 51,
    "voices": [
        {
            "duration_ms": 400,
            "f0_hz": [[0, 120], [0.5, 200], [1, 160]],
            "jaw": [[0, 0.1], [0.3, 1], [1, 0.7]],
            "amp": [[0, 0], [0.1, 1], [1, 0.8]],
            "spacing_hz": 290,
            "breaks": [[0.5, 0.1, 1.3]],
        }
    ],
    "sub": {"duration_ms": 400, "level": 0.35, "freq_hz": [[0, 48], [1, 50]], "amp": [[0, 0], [0.2, 1], [1, 0.8]]},
    "finish": {"shape_db": [[0, 0], [0.6, 0], [0.8, -45], [1, -70]], "peak_db": -6},
}


def test_a_seeded_voice_renders_the_same_twice():
    a = render_creature(24000, 48000, LAYER)
    b = render_creature(24000, 48000, LAYER)
    assert a.shape == (24000,)
    assert np.array_equal(a, b)


def test_it_peaks_where_its_finish_says_and_is_not_silence():
    x = render_creature(24000, 48000, LAYER)
    assert abs(20 * np.log10(np.max(np.abs(x))) - (-6.0)) < 1e-3
    assert np.sqrt(np.mean(x[2400:12000] ** 2)) > 1e-3


def test_another_seed_is_another_take():
    other = dict(LAYER, seed=52)
    assert not np.array_equal(render_creature(24000, 48000, LAYER), render_creature(24000, 48000, other))
