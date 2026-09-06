"""Audit rendered audio loudness / balance."""

from __future__ import annotations

import json
from pathlib import Path

from ambition_sfx_renderer.audio import audit_stats
from ambition_sfx_renderer.io import read_audio


def audit_file(path: Path) -> dict[str, float | str]:
    audio, sr = read_audio(path)
    stats = audit_stats(audio)
    duration = audio.shape[1] / sr
    return {
        "path": str(path),
        "duration_seconds": duration,
        **stats,
    }


def audit_output_tree(root: Path) -> list[dict[str, float | str]]:
    root = Path(root)
    files = sorted([*root.glob("**/*.wav"), *root.glob("**/*.ogg")])
    return [audit_file(path) for path in files]


def print_audit(rows: list[dict[str, float | str]]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="SFX render audit")
        table.add_column("file")
        table.add_column("dur", justify="right")
        table.add_column("peak", justify="right")
        table.add_column("rms", justify="right")
        for row in rows:
            table.add_row(
                str(row["path"]),
                f"{float(row['duration_seconds']):.3f}s",
                f"{float(row['peak_db']):.1f} dB",
                f"{float(row['rms_db']):.1f} dB",
            )
        Console().print(table)
    except Exception:
        print(json.dumps(rows, indent=2))
