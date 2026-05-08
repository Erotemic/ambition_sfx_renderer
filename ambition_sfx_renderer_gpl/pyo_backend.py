"""pyo backend placeholder.

pyo is a good candidate for richer procedural and granular patches, but it has
more environment-specific behavior than the DawDreamer/Pedalboard/PyFXR path.
This module is kept as an explicit TODO rather than silently emulating it with
lower-quality code.
"""
from __future__ import annotations

from typing import Any


def render_pyo_patch_layer(layer: dict[str, Any], context: dict[str, Any]):
    raise NotImplementedError(
        "kind: pyo_patch is reserved for a future pyo offline patch backend. "
        "TODO: implement a small patch interface using pyo.Server(audio='offline') "
        "and recordOptions()/start() once concrete patches are authored."
    )
