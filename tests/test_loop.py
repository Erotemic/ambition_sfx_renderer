"""A `loop:` cue renders a seamless loop: its own render, with the tail folded into the head."""

import numpy as np
import pytest

from ambition_sfx_renderer.errors import SchemaError
from ambition_sfx_renderer.render import render_cue
from ambition_sfx_renderer.schema import load_cue

SR = 8000
LAYERS = """layers:
  - name: bed
    kind: noise_burst
    color: pink
    seed: 7
    gain_db: -6
    start_ms: 0
    duration_ms: 1300
  - name: blob
    kind: noise_burst
    color: white
    seed: 8
    gain_db: -3
    start_ms: 900
    duration_ms: 120
"""
HEAD = f"""schema: ambition.sfxir.v1
render: {{sample_rate: {SR}, channels: 2, seed: 3, safety_profile: none}}
"""


def cue(tmp_path, name, body):
    path = tmp_path / f"{name}.sfx.yaml"
    path.write_text(HEAD + f"id: fixture.{name}\n" + body + LAYERS)
    return load_cue(path)


def test_a_loop_is_its_render_with_the_tail_folded_into_the_head(tmp_path):
    """Pre-roll 200, loop 800, crossfade 200, post-roll 100 ms. The same
    layers rendered plainly over all 1300 ms are the render the loop is cut
    from, so the loop must be exactly that render's [200, 1000) ms with its
    first 200 ms crossfaded against [1000, 1200) ms."""
    looped, report = render_cue(cue(tmp_path, "loop", "duration_ms: 800\n"
                                    "loop: {preroll_ms: 200, crossfade_ms: 200, postroll_ms: 100}\n"))
    whole, _ = render_cue(cue(tmp_path, "whole", "duration_ms: 1300\n"))
    pre, length, fade = 200 * SR // 1000, 800 * SR // 1000, 200 * SR // 1000

    assert looped.shape == (2, length)
    assert report["duration_seconds"] == pytest.approx(0.8)
    assert report["loop"]["crossfade_ms"] == 200
    # After the crossfade the loop is the render, sample for sample.
    np.testing.assert_array_equal(looped[:, fade:], whole[:, pre + fade : pre + length])
    # Its first sample is the render's next one after its last: the wrap runs on.
    np.testing.assert_allclose(looped[:, 0], whole[:, pre + length], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(looped[:, -1], whole[:, pre + length - 1])


def test_a_loop_can_be_brought_to_a_peak_after_the_fold(tmp_path):
    looped, _ = render_cue(cue(tmp_path, "peak", "duration_ms: 800\n"
                               "loop: {preroll_ms: 200, crossfade_ms: 200, normalize_peak_db: -18}\n"))
    assert 20 * np.log10(np.abs(looped).max()) == pytest.approx(-18.0, abs=1e-4)


@pytest.mark.parametrize("loop", ["{preroll_ms: 100}", "{crossfade_ms: 0}", "{crossfade_ms: 900}",
                                  "{crossfade_ms: 100, preroll_ms: -1}", "{crossfade_ms: 100, wrap: true}"])
def test_a_malformed_loop_is_refused(tmp_path, loop):
    with pytest.raises(SchemaError):
        cue(tmp_path, "bad", f"duration_ms: 800\nloop: {loop}\n")
