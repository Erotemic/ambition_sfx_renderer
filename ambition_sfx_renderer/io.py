"""Audio file I/O."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ambition_sfx_renderer.audio import ensure_chans_first


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write an audio file.

    Tries soundfile first. For OGG/Vorbis, falls back to pedalboard.io when
    libsndfile lacks the encoder.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = ensure_chans_first(audio)
    data = audio.T  # file libraries expect samples x channels
    suffix = path.suffix.lower()
    try:
        if suffix == ".ogg":
            sf.write(path, data, sample_rate, format="OGG", subtype="VORBIS")
        else:
            sf.write(path, data, sample_rate)
        return
    except Exception as sf_ex:
        if suffix != ".ogg":
            raise
        try:
            from pedalboard.io import AudioFile

            with AudioFile(str(path), "w", sample_rate, audio.shape[0], quality=0.75) as file:
                file.write(audio)
            return
        except Exception as pb_ex:  # pragma: no cover - depends on local codecs
            raise RuntimeError(f"failed to write {path} with soundfile or pedalboard") from pb_ex


def read_audio(path: Path, *, target_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    audio = ensure_chans_first(data)
    if target_sample_rate and int(sr) != int(target_sample_rate):
        from ambition_sfx_renderer.audio import resample_audio

        audio = resample_audio(audio, int(sr), int(target_sample_rate))
        sr = int(target_sample_rate)
    return audio, int(sr)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf8")
