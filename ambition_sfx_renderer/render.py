"""Cue rendering orchestration."""
from __future__ import annotations

import hashlib
import json
import os
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
DEFAULT_SAFETY_PROFILE = "game_sfx_v2"
FormatPolicy = Literal["auto", "both", "wav", "ogg"]


def cue_hash(path: Path) -> str:
    data = path.read_bytes()
    h = hashlib.sha256()
    h.update(data)
    h.update(__version__.encode())
    return h.hexdigest()[:12]


def default_final_peak_db(cue_id: str, duration_seconds: float) -> float:
    """Cue-family peak ceilings used by the default tone-safety pass."""
    cue = cue_id.lower()
    if ".loop" in cue or cue.endswith("_loop") or duration_seconds > 0.75:
        return -10.0
    if cue.startswith("ui."):
        return -10.0
    if "footstep" in cue or cue.endswith(".land") or "wall_slide" in cue:
        return -11.0
    if "death" in cue or "impact" in cue or "hit" in cue or "attack" in cue or "slash" in cue:
        return -6.0
    if "fireball" in cue or "projectile" in cue or "hazard" in cue:
        return -7.0
    if cue.startswith("player."):
        return -8.0
    return -8.0


def default_tone_safety_effects(spec: CueSpec) -> list[dict[str, Any]]:
    """Return automatic anti-harshness processing for a cue.

    Disable per cue with ``render.safety_profile: none``. Tune with
    ``render.final_peak_db``, ``render.safety_lowpass_hz``,
    ``render.deharsh_hz``, or ``render.deharsh_amount``.
    """
    render_spec = dict(spec.raw.get("render", {}) or {})
    profile = str(render_spec.get("safety_profile", DEFAULT_SAFETY_PROFILE)).lower()
    if profile in {"none", "off", "false", "0"}:
        return []
    cue = spec.cue_id.lower()
    dur = spec.duration_seconds
    if dur > 0.75 or ".loop" in cue or cue.endswith("_loop"):
        lowpass = 6200.0
        deharsh_amount = 0.24
    elif cue.startswith("ui."):
        lowpass = 7800.0
        deharsh_amount = 0.16
    elif "footstep" in cue or "stone" in cue or "metal" in cue:
        lowpass = 5200.0
        deharsh_amount = 0.22
    elif "blink" in cue or "energy" in cue:
        lowpass = 7600.0
        deharsh_amount = 0.25
    else:
        lowpass = 7000.0
        deharsh_amount = 0.20
    return [{
        "effect": "tone_safety",
        "profile": profile,
        "highpass_hz": float(render_spec.get("safety_highpass_hz", 28.0)),
        "lowpass_hz": float(render_spec.get("safety_lowpass_hz", lowpass)),
        "deharsh_hz": float(render_spec.get("deharsh_hz", 3200.0)),
        "deharsh_amount": float(render_spec.get("deharsh_amount", deharsh_amount)),
        "deharsh_q": float(render_spec.get("deharsh_q", 0.9)),
        "drive": float(render_spec.get("safety_drive", 1.06)),
        "clip_mix": float(render_spec.get("safety_clip_mix", 0.55)),
        "target_peak_db": float(render_spec.get("final_peak_db", default_final_peak_db(spec.cue_id, dur))),
        "only_if_louder": bool(render_spec.get("only_if_louder", True)),
        "fade_out_ms": float(render_spec.get("safety_fade_out_ms", 2.0)),
    }]


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
        "final_peak_db": float(raw.get("render", {}).get("final_peak_db", default_final_peak_db(spec.cue_id, duration_seconds))),
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
    final_safety = default_tone_safety_effects(spec)
    if final_safety:
        mix = apply_effects(mix, sample_rate, final_safety, context)
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
        "final_safety": default_tone_safety_effects(spec),
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
            out_dir = Path(out_root) / spec.cue_id
            _ensure_source_symlink(out_dir, path)
            _write_regen(
                out_dir / "regen.sh",
                path,
                out_root,
                cue_id=spec.cue_id,
                format_policy=format_policy,
                no_wav=no_wav,
                no_ogg=no_ogg,
                wav_max_seconds=wav_max_seconds,
            )
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
    _ensure_source_symlink(out_dir, path)
    _write_regen(
        out_dir / "regen.sh",
        path,
        out_root,
        cue_id=spec.cue_id,
        format_policy=format_policy,
        no_wav=no_wav,
        no_ogg=no_ogg,
        wav_max_seconds=wav_max_seconds,
    )
    return report



def _ensure_source_symlink(out_dir: Path, cue_path: Path) -> None:
    """Create a convenient symlink from output/<cue>/ back to the source YAML.

    This makes the render directory useful during tuning: open
    output/<cue>/source.sfx.yaml, edit the real source file, then run
    output/<cue>/regen.sh to re-render and audition.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    link = out_dir / "source.sfx.yaml"
    cue_path = Path(cue_path).resolve()
    try:
        target = os.path.relpath(cue_path, start=out_dir.resolve())
        if link.is_symlink():
            if os.readlink(link) == target:
                return
            link.unlink()
        elif link.exists():
            link.unlink()
        link.symlink_to(target)
    except OSError:
        # Filesystems without symlink support still get a useful pointer.
        link.write_text(str(cue_path) + "\n", encoding="utf8")


def _preferred_playback_output(outputs: dict[str, str]) -> str | None:
    """Choose the most useful file to audition after a render."""
    for key in ("wav", "ogg"):
        value = outputs.get(key)
        if value:
            return value
    return None


def _write_regen(
    path: Path,
    cue_path: Path,
    out_root: Path,
    *,
    cue_id: str,
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

    out_dir = path.parent
    preferred_wav = out_dir / f"{cue_id}.wav"
    preferred_ogg = out_dir / f"{cue_id}.ogg"
    waveform_svg = out_dir / f"{cue_id}.waveform.svg"
    render_cmd = " ".join(shlex.quote(x) for x in cmd)
    # Keep $PLAY_FILE as a shell variable.  Do not pass it through
    # shlex.quote here, because that would generate a literal
    # '${PLAY_FILE}' argument in regen.sh.
    draw_cmd = " ".join(
        [
            shlex.quote(sys.executable),
            "-m",
            "ambition_sfx_renderer",
            "draw",
            '"$PLAY_FILE"',
            "--out",
            shlex.quote(str(waveform_svg)),
        ]
    )
    script = f'''#!/usr/bin/env bash
set -euo pipefail

# Re-render this cue from its source YAML, then optionally audition/draw it.
# Usage:
#   ./regen.sh              # render + ffplay
#   ./regen.sh --no-play    # render only
#   ./regen.sh --play-only  # play existing output only
#   ./regen.sh --draw       # render + write waveform SVG + ffplay
#   ./regen.sh --draw --no-play
#   ./regen.sh --draw-only  # write waveform SVG for current output only

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PLAY=true
RENDER=true
DRAW=false

for arg in "$@"; do
    case "$arg" in
        --no-play)
            PLAY=false
            ;;
        --play-only)
            RENDER=false
            ;;
        --draw)
            DRAW=true
            ;;
        --draw-only)
            RENDER=false
            PLAY=false
            DRAW=true
            ;;
        --help|-h)
            sed -n '1,18p' "$0"
            exit 0
            ;;
        *)
            echo "regen.sh: unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

if [[ "$RENDER" == "true" ]]; then
    {render_cmd}
fi

PLAY_FILE=""
if [[ -f {shlex.quote(str(preferred_wav))} ]]; then
    PLAY_FILE={shlex.quote(str(preferred_wav))}
elif [[ -f {shlex.quote(str(preferred_ogg))} ]]; then
    PLAY_FILE={shlex.quote(str(preferred_ogg))}
elif compgen -G "$SCRIPT_DIR/*.wav" > /dev/null; then
    PLAY_FILE="$(ls -t "$SCRIPT_DIR"/*.wav | head -n 1)"
elif compgen -G "$SCRIPT_DIR/*.ogg" > /dev/null; then
    PLAY_FILE="$(ls -t "$SCRIPT_DIR"/*.ogg | head -n 1)"
fi

if [[ -z "$PLAY_FILE" ]]; then
    echo "regen.sh: no rendered wav/ogg found in $SCRIPT_DIR" >&2
    exit 1
fi

if [[ "$DRAW" == "true" ]]; then
    {draw_cmd}
    echo "waveform: {shlex.quote(str(waveform_svg))}"
fi

if [[ "$PLAY" != "true" ]]; then
    exit 0
fi

if ! command -v ffplay >/dev/null 2>&1; then
    echo "regen.sh: ffplay not found; install ffmpeg to audition: $PLAY_FILE" >&2
    exit 0
fi

echo "ffplay $PLAY_FILE"
ffplay -hide_banner -loglevel error -nodisp -autoexit "$PLAY_FILE"
'''
    path.write_text(script, encoding="utf8")
    path.chmod(0o755)
