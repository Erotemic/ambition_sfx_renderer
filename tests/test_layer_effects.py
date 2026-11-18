import numpy as np

from ambition_sfx_renderer.layers import render_layer


def test_a_layer_effect_chain_attenuates_frequencies_above_its_cutoff():
    context = {"sample_rate": 8000, "channels": 1, "duration_seconds": 0.1, "seed": 2}
    layer = {"kind": "synth_tone", "duration_ms": 100, "waveform": "sine", "frequency_hz": 3000}
    dry = render_layer(layer, context)
    filtered = render_layer(dict(layer, effects=[{"effect": "lowpass", "cutoff_hz": 500}]), context)
    assert np.sqrt(np.mean(filtered[:, 100:] ** 2)) < np.sqrt(np.mean(dry[:, 100:] ** 2)) * 0.1
