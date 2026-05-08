"""Cue rendering orchestration."""
from __future__ import annotations

import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ambition_sfx_renderer import __version__
from ambition_sfx_renderer.audio import audit_stats, mix_into, ms_to_samples, seconds_to_samples
from ambition_sfx_renderer.effects import apply_effects
from ambition_sfx_renderer.io import write_audio, write_json
from ambition_sfx_renderer.layers import render_layer
from ambition_sfx_renderer.schema import CueSpec, load_cue


def cue_hash(path: Path) -> str:
    data = path.read_bytes()
    h = hashlib.sha256()
    h.update(data)
    h.update(__version__.encode())
    return h.hexdigest()[:12]


def render_cue(spec: CueSpec) -> tuple[np.ndarray, dict[str, Any]]:
    raw = spec.raw
    sample_rate = spec.sample_rate
    channels = spec.channels
    duration_seconds = spec.duration_seconds
    n_samples = seconds_to_samples(duration_seconds, sample_rate)
    mix = np.zeros((channels, n_samples), dtype=np.float32)
    context: dict[str, Any] = {
        "cue_id": spec.cue_id,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": duration_seconds,
        "seed": spec.seed,
        "base_dir": spec.path.parent,
        "yaml_path": spec.path,
        "buffer_size": int(raw.get("render", {}).get("buffer_size", 128)),
    }
    layer_reports: list[dict[str, Any]] = []
    for layer in spec.layers:
        start_ms = float(layer.get("start_ms", 0.0))
        offset = ms_to_samples(start_ms, sample_rate)
        clip = render_layer(layer, context)
        mix_into(mix, clip, offset)
        layer_reports.append(
            {
                "name": layer.get("name"),
                "kind": layer.get("kind"),
                "start_ms": start_ms,
                "duration_seconds": clip.shape[1] / sample_rate,
            }
        )
    if spec.postprocess:
        mix = apply_effects(mix, sample_rate, spec.postprocess, context)
    stats = audit_stats(mix)
    report: dict[str, Any] = {
        "id": spec.cue_id,
        "schema": raw.get("schema"),
        "renderer_version": __version__,
        "source": str(spec.path),
        "hash": cue_hash(spec.path),
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": duration_seconds,
        "layers": layer_reports,
        **stats,
    }
    return mix, report


def render_file(path: Path, *, out_root: Path, write_wav: bool = True, write_ogg: bool = True) -> dict[str, Any]:
    spec = load_cue(path)
    audio, report = render_cue(spec)
    out_dir = Path(out_root) / spec.cue_id
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    if write_wav:
        wav_path = out_dir / f"{spec.cue_id}.wav"
        write_audio(wav_path, audio, spec.sample_rate)
        outputs["wav"] = str(wav_path)
    if write_ogg:
        ogg_path = out_dir / f"{spec.cue_id}.ogg"
        write_audio(ogg_path, audio, spec.sample_rate)
        outputs["ogg"] = str(ogg_path)
    report["outputs"] = outputs
    manifest_path = out_dir / f"{spec.cue_id}.render.json"
    write_json(manifest_path, report)
    _write_regen(out_dir / "regen.sh", path, out_root)
    return report


def _write_regen(path: Path, cue_path: Path, out_root: Path) -> None:
    cmd = [sys.executable, "-m", "ambition_sfx_renderer", "render", str(cue_path), "--outdir", str(out_root), "--force"]
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(shlex.quote(x) for x in cmd) + "\n", encoding="utf8")
    path.chmod(0o755)
