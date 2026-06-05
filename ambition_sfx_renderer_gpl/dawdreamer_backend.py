"""DawDreamer backend for Faust and plugin-based offline rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ambition_sfx_renderer.audio import curve_values, fit_length, stereoize
from ambition_sfx_renderer.errors import BackendUnavailableError
from ambition_sfx_renderer.paths import resolve_path


def _import_dawdreamer():
    try:
        import dawdreamer as daw
    except Exception as ex:  # pragma: no cover - depends on local environment
        raise BackendUnavailableError(
            "dawdreamer is required for kind: dawdreamer_faust layers. "
            "Install with `uv pip install dawdreamer`."
        ) from ex
    return daw


def _load_dsp_string(layer: dict[str, Any], context: dict[str, Any]) -> tuple[str, Path | None]:
    if "dsp_string" in layer:
        return str(layer["dsp_string"]), None
    dsp_value = layer.get("dsp") or layer.get("dsp_path")
    if not dsp_value:
        raise ValueError(f"dawdreamer_faust layer {layer.get('name')} requires dsp or dsp_string")
    path = resolve_path(dsp_value, base_dir=context["base_dir"])
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf8"), path


def _set_parameters(
    proc: Any, params: dict[str, Any], duration_seconds: float, sample_rate: int
) -> None:
    n = max(1, int(round(duration_seconds * sample_rate)))
    for address, value in params.items():
        address = str(address)
        if isinstance(value, dict) and "start" in value and "end" in value:
            arr = curve_values(value["start"], value["end"], n, value.get("curve", "linear"))
            if hasattr(proc, "set_automation"):
                proc.set_automation(address, arr)
            else:  # pragma: no cover - old DawDreamer versions
                proc.set_parameter(address, float(value["start"]))
        else:
            proc.set_parameter(address, float(value))


def render_faust_layer(layer: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    """Render a Faust DSP patch through DawDreamer.

    The Faust processor is expected to synthesize its own output. For effects
    that process existing audio, add a dedicated backend later.
    """
    daw = _import_dawdreamer()
    sample_rate = int(context["sample_rate"])
    duration = float(layer.get("duration_seconds", context["duration_seconds"]))
    buffer_size = int(layer.get("buffer_size", context.get("buffer_size", 128)))
    engine = daw.RenderEngine(sample_rate, buffer_size)
    proc = engine.make_faust_processor(str(layer.get("processor_name", "faust")))
    dsp_string, dsp_path = _load_dsp_string(layer, context)
    if dsp_path is not None and hasattr(proc, "set_dsp"):
        # Docs require an absolute path for set_dsp.
        proc.set_dsp(str(dsp_path.resolve()))
    else:
        proc.set_dsp_string(dsp_string)
    if bool(layer.get("compile", True)) and hasattr(proc, "compile"):
        proc.compile()
    _set_parameters(proc, dict(layer.get("parameters") or {}), duration, sample_rate)
    if "midi" in layer:
        for note in layer["midi"] or []:
            proc.add_midi_note(
                int(note["note"]),
                int(note.get("velocity", 100)),
                float(note.get("start", 0.0)),
                float(note.get("duration", duration)),
            )
    engine.load_graph([(proc, [])])
    engine.render(duration, beats=False)
    audio = engine.get_audio()
    audio = stereoize(audio, channels=context["channels"])
    return fit_length(audio, int(round(duration * sample_rate)))


def render_plugin_layer(layer: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    """Render a VST/AU instrument through DawDreamer.

    This is intentionally minimal: specify a plugin path and MIDI notes. Complex
    plugin state, automation, and preset management can be added as real needs
    appear.
    """
    daw = _import_dawdreamer()
    sample_rate = int(context["sample_rate"])
    duration = float(layer.get("duration_seconds", context["duration_seconds"]))
    buffer_size = int(layer.get("buffer_size", context.get("buffer_size", 128)))
    plugin_path = resolve_path(layer["plugin"], base_dir=context["base_dir"])
    engine = daw.RenderEngine(sample_rate, buffer_size)
    proc = engine.make_plugin_processor(
        str(layer.get("processor_name", "plugin")), str(plugin_path)
    )
    if layer.get("preset"):
        proc.load_preset(str(resolve_path(layer["preset"], base_dir=context["base_dir"])))
    if layer.get("state"):
        proc.load_state(str(resolve_path(layer["state"], base_dir=context["base_dir"])))
    for key, value in dict(layer.get("parameters") or {}).items():
        # DawDreamer accepts parameter indices or parameter names depending on plugin.
        try:
            key2: str | int = int(key)
        except (TypeError, ValueError):
            key2 = str(key)
        proc.set_parameter(key2, float(value))
    for note in layer.get("midi") or []:
        proc.add_midi_note(
            int(note["note"]),
            int(note.get("velocity", 100)),
            float(note.get("start", 0.0)),
            float(note.get("duration", duration)),
        )
    engine.load_graph([(proc, [])])
    engine.render(duration, beats=False)
    return stereoize(engine.get_audio(), channels=context["channels"])
