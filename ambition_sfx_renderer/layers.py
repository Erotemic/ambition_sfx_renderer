"""Layer dispatch and common layer processing."""

from __future__ import annotations

from typing import Any

import numpy as np

from ambition_sfx_renderer.audio import apply_envelope, apply_gain, apply_pan, fit_length, stereoize
from ambition_sfx_renderer.effects import apply_effects


def render_layer_source(layer: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    kind = str(layer.get("kind", "")).lower()
    if kind == "sample":
        from ambition_sfx_renderer.backends.sample_backend import render_sample_layer

        return render_sample_layer(layer, context)
    if kind == "pyfxr":
        from ambition_sfx_renderer.backends.pyfxr_backend import render_pyfxr_layer

        return render_pyfxr_layer(layer, context)
    if kind in {"noise", "noise_burst", "foley_noise", "grain_noise"}:
        from ambition_sfx_renderer.backends.noise_backend import render_noise_layer

        return render_noise_layer(layer, context)
    if kind in {"dawdreamer_faust", "faust"}:
        from ambition_sfx_renderer_gpl.dawdreamer_backend import render_faust_layer

        return render_faust_layer(layer, context)
    if kind in {"dawdreamer_plugin", "plugin", "vst", "vst3", "audio_unit", "au"}:
        from ambition_sfx_renderer_gpl.dawdreamer_backend import render_plugin_layer

        return render_plugin_layer(layer, context)
    if kind in {"pyo", "pyo_patch"}:
        from ambition_sfx_renderer_gpl.pyo_backend import render_pyo_patch_layer

        return render_pyo_patch_layer(layer, context)
    raise NotImplementedError(
        f"Layer kind {kind!r} is not implemented. TODO: add a backend module and dispatch here."
    )


def render_layer(layer: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    audio = render_layer_source(layer, context)
    sample_rate = int(context["sample_rate"])
    channels = int(context["channels"])
    if "duration_ms" in layer:
        wanted = int(round(float(layer["duration_ms"]) * 0.001 * sample_rate))
        audio = fit_length(audio, wanted)
    audio = stereoize(audio, channels=channels)
    audio = apply_envelope(audio, sample_rate, layer.get("envelope"))
    audio = apply_gain(audio, gain_db=layer.get("gain_db"), gain=layer.get("gain"))
    audio = apply_pan(audio, pan=layer.get("pan"), channels=channels)
    if layer.get("effects"):
        audio = apply_effects(audio, sample_rate, list(layer.get("effects") or []), context)
    return audio.astype(np.float32)
