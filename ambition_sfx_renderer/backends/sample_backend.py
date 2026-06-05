"""Sample-file layer backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from ambition_sfx_renderer.audio import fit_length, resample_audio, stereoize
from ambition_sfx_renderer.io import read_audio
from ambition_sfx_renderer.paths import resolve_path


def render_sample_layer(layer: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    path_value = layer.get("path") or layer.get("file")
    if not path_value:
        raise ValueError(f"sample layer {layer.get('name')} requires path")
    path = resolve_path(path_value, base_dir=context["base_dir"])
    audio, sr = read_audio(path, target_sample_rate=context["sample_rate"])
    audio = stereoize(audio, channels=context["channels"])
    pitch = float(layer.get("pitch", 1.0))
    if pitch <= 0:
        raise ValueError("sample pitch must be positive")
    if abs(pitch - 1.0) > 1e-6:
        # Pitch via playback-rate change: resample length inversely with pitch.
        target_len = max(1, int(round(audio.shape[1] / pitch)))
        # resample_poly by ratio target_len / old_len using scipy's Fourier resampler
        from scipy.signal import resample

        audio = resample(audio, target_len, axis=1).astype(np.float32)
    if "duration_ms" in layer:
        audio = fit_length(
            audio, int(round(float(layer["duration_ms"]) * 0.001 * context["sample_rate"]))
        )
    return audio.astype(np.float32)
