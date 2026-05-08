"""pyfxr backend for fast game-SFX seed layers."""
from __future__ import annotations

import random
from typing import Any

import numpy as np

from ambition_sfx_renderer.audio import resample_audio, stereoize
from ambition_sfx_renderer.errors import BackendUnavailableError

PRESETS = {
    "pickup",
    "laser",
    "explosion",
    "powerup",
    "hurt",
    "jump",
    "select",
}


def _import_pyfxr():
    try:
        import pyfxr
    except Exception as ex:  # pragma: no cover - depends on local environment
        raise BackendUnavailableError(
            "pyfxr is required for kind: pyfxr layers. Install with `uv pip install pyfxr`."
        ) from ex
    return pyfxr


def _soundbuffer_to_numpy(buf: Any) -> tuple[np.ndarray, int]:
    sample_rate = int(getattr(buf, "sample_rate", 44100))
    channels = int(getattr(buf, "channels", 1))
    # pyfxr SoundBuffer supports the buffer protocol and stores 16-bit samples.
    arr = np.frombuffer(memoryview(buf), dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        arr = arr.reshape((-1, channels)).T
    else:
        arr = arr[None, :]
    return arr.astype(np.float32), sample_rate


def render_pyfxr_layer(layer: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    pyfxr = _import_pyfxr()
    seed = layer.get("seed", context.get("seed"))
    if seed is not None:
        random.seed(int(seed))
    preset = layer.get("preset")
    params = dict(layer.get("params") or {})
    if preset:
        preset = str(preset)
        if preset not in PRESETS:
            raise ValueError(f"unknown pyfxr preset {preset!r}; expected one of {sorted(PRESETS)}")
        sfx = getattr(pyfxr, preset)()
        # Allow parameter overrides on the returned SFX object where possible.
        for key, value in params.items():
            if key == "wave_type" and isinstance(value, str):
                value = getattr(pyfxr.WaveType, value.upper())
            setattr(sfx, key, value)
    else:
        if "wave_type" in params and isinstance(params["wave_type"], str):
            params["wave_type"] = getattr(pyfxr.WaveType, params["wave_type"].upper())
        sfx = pyfxr.SFX(**params)
    buf = sfx.build() if hasattr(sfx, "build") else sfx
    audio, sr = _soundbuffer_to_numpy(buf)
    if sr != context["sample_rate"]:
        audio = resample_audio(audio, sr, context["sample_rate"])
    return stereoize(audio, channels=context["channels"])
