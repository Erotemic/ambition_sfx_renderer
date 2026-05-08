"""Cue rendering orchestration."""
from __future__ import annotations

import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ambition_sfx_renderer import __version__
from ambition_sfx_renderer.audio import audit_stats, mix_into, ms_to_samples, seconds_to_samples
from ambition_sfx_renderer.effects import apply_effects
from ambition_sfx_renderer.errors import SfxRenderError
from ambition_sfx_renderer.io import write_audio, write_json
from ambition_sfx_renderer.layers import render_layer
from ambition_sfx_renderer.schema import CueSpec, load_cue

DEFAULT_WAV_MAX_SECONDS = 0.300
FormatPolicy = Literal["auto", "both", "wav", "ogg"]


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


def select_output_formats(
    spec: CueSpec,
    *,
    format_policy: str = "auto",
    no_wav: bool = False,
    no_ogg: bool = False,
    wav_max_seconds: float = DEFAULT_WAV_MAX_SECONDS,
) -> tuple[bool, bool, str]:
    """Resolve output formats for a cue.

    The default ``auto`` policy writes WAV for short cues and OGG for longer
    cues. The boundary is strict: durations greater than ``wav_max_seconds``
    use OGG, otherwise WAV. The default boundary is 0.300 seconds.
    """
    policy = str(format_policy or "auto").lower().strip()
    if policy not in {"auto", "both", "wav", "ogg"}:
        raise SfxRenderError(f"unknown format policy {format_policy!r}")
    if policy == "auto":
        write_wav = spec.duration_seconds <= float(wav_max_seconds)
        write_ogg = spec.duration_seconds > float(wav_max_seconds)
        resolved = "auto:wav" if write_wav else "auto:ogg"
    elif policy == "both":
        write_wav = True
        write_ogg = True
        resolved = "both"
    elif policy == "wav":
        write_wav = True
        write_ogg = False
        resolved = "wav"
    elif policy == "ogg":
        write_wav = False
        write_ogg = True
        resolved = "ogg"
    else:  # pragma: no cover
        raise AssertionError(policy)
    if no_wav:
        write_wav = False
    if no_ogg:
        write_ogg = False
    if not write_wav and not write_ogg:
        raise SfxRenderError(
            "format selection produced no audio files; remove --no-wav/--no-ogg "
            "or choose --format-policy wav|ogg|both"
        )
    return write_wav, write_ogg, resolved


def expected_output_paths(
    spec: CueSpec,
    out_root: Path,
    *,
    format_policy: str = "auto",
    no_wav: bool = False,
    no_ogg: bool = False,
    wav_max_seconds: float = DEFAULT_WAV_MAX_SECONDS,
) -> dict[str, Path]:
    """Return canonical output paths for a cue spec."""
    write_wav, write_ogg, _ = select_output_formats(
        spec,
        format_policy=format_policy,
        no_wav=no_wav,
        no_ogg=no_ogg,
        wav_max_seconds=wav_max_seconds,
    )
    out_dir = Path(out_root) / spec.cue_id
    outputs: dict[str, Path] = {}
    if write_wav:
        outputs["wav"] = out_dir / f"{spec.cue_id}.wav"
    if write_ogg:
        outputs["ogg"] = out_dir / f"{spec.cue_id}.ogg"
    outputs["manifest"] = out_dir / f"{spec.cue_id}.render.json"
    return outputs


def is_render_current(
    path: Path,
    *,
    out_root: Path,
    format_policy: str = "auto",
    no_wav: bool = False,
    no_ogg: bool = False,
    wav_max_seconds: float = DEFAULT_WAV_MAX_SECONDS,
) -> tuple[bool, dict[str, Any] | None]:
    """Return true when the output manifest and requested audio files are current."""
    spec = load_cue(path)
    expected = expected_output_paths(
        spec,
        out_root,
        format_policy=format_policy,
        no_wav=no_wav,
        no_ogg=no_ogg,
        wav_max_seconds=wav_max_seconds,
    )
    manifest_path = expected["manifest"]
    if not manifest_path.exists():
        return False, None
    for kind, out_path in expected.items():
        if kind != "manifest" and not out_path.exists():
            return False, None
    try:
        report = json.loads(manifest_path.read_text(encoding="utf8"))
    except Exception:
        return False, None
    if report.get("hash") != cue_hash(Path(path)):
        return False, report
    if report.get("renderer_version") != __version__:
        return False, report
    return True, report


def render_file(
    path: Path,
    *,
    out_root: Path,
    format_policy: str = "auto",
    no_wav: bool = False,
    no_ogg: bool = False,
    wav_max_seconds: float = DEFAULT_WAV_MAX_SECONDS,
    force: bool = True,
) -> dict[str, Any]:
    spec = load_cue(path)
    write_wav, write_ogg, resolved_policy = select_output_formats(
        spec,
        format_policy=format_policy,
        no_wav=no_wav,
        no_ogg=no_ogg,
        wav_max_seconds=wav_max_seconds,
    )
    if not force:
        current, report = is_render_current(
            path,
            out_root=out_root,
            format_policy=format_policy,
            no_wav=no_wav,
            no_ogg=no_ogg,
            wav_max_seconds=wav_max_seconds,
        )
        if current and report is not None:
            report = dict(report)
            report["skipped"] = True
            report.setdefault("resolved_format_policy", resolved_policy)
            return report
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
    report["format_policy"] = format_policy
    report["resolved_format_policy"] = resolved_policy
    report["wav_max_seconds"] = float(wav_max_seconds)
    report["skipped"] = False
    manifest_path = out_dir / f"{spec.cue_id}.render.json"
    write_json(manifest_path, report)
    _write_regen(
        out_dir / "regen.sh",
        path,
        out_root,
        format_policy=format_policy,
        no_wav=no_wav,
        no_ogg=no_ogg,
        wav_max_seconds=wav_max_seconds,
    )
    return report


def _write_regen(
    path: Path,
    cue_path: Path,
    out_root: Path,
    *,
    format_policy: str,
    no_wav: bool,
    no_ogg: bool,
    wav_max_seconds: float,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "ambition_sfx_renderer",
        "render",
        str(cue_path),
        "--outdir",
        str(out_root),
        "--format-policy",
        str(format_policy),
        "--wav-max-seconds",
        str(wav_max_seconds),
        "--force",
    ]
    if no_wav:
        cmd.append("--no-wav")
    if no_ogg:
        cmd.append("--no-ogg")
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + " ".join(shlex.quote(x) for x in cmd)
        + "\n",
        encoding="utf8",
    )
    path.chmod(0o755)
