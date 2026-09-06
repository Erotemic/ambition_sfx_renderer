"""Simple waveform drawing helpers.

The renderer keeps this dependency-light: it reads audio with soundfile via the
project I/O layer and writes an SVG. No matplotlib/Pillow dependency is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ambition_sfx_renderer.audio import ensure_chans_first
from ambition_sfx_renderer.io import read_audio


def _resolve_audio_path(path: Path) -> Path:
    path = Path(path)
    if path.is_dir():
        candidates = sorted(
            [*path.glob("*.wav"), *path.glob("*.ogg")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"no wav/ogg files found in {path}")
        return candidates[0]
    return path


def _bin_minmax(channel: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    channel = np.asarray(channel, dtype=np.float32)
    if channel.size == 0:
        return np.zeros(bins, dtype=np.float32), np.zeros(bins, dtype=np.float32)
    edges = np.linspace(0, channel.size, bins + 1, dtype=np.int64)
    mins = np.empty(bins, dtype=np.float32)
    maxs = np.empty(bins, dtype=np.float32)
    for idx in range(bins):
        lo = int(edges[idx])
        hi = int(edges[idx + 1])
        if hi <= lo:
            hi = min(channel.size, lo + 1)
        segment = channel[lo:hi]
        mins[idx] = float(np.min(segment)) if segment.size else 0.0
        maxs[idx] = float(np.max(segment)) if segment.size else 0.0
    return mins, maxs


def draw_waveform(
    audio_path: Path,
    *,
    out: Path | None = None,
    width: int = 1400,
    height: int = 420,
    title: str | None = None,
) -> Path:
    """Draw an audio waveform to SVG and return the output path."""
    audio_path = _resolve_audio_path(Path(audio_path))
    audio, sample_rate = read_audio(audio_path)
    audio = ensure_chans_first(audio)
    channels = min(audio.shape[0], 2)
    width = max(240, int(width))
    height = max(120, int(height))
    margin_l = 58
    margin_r = 18
    margin_t = 42
    margin_b = 34
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    row_h = plot_h / channels
    bins = max(32, plot_w)
    peak = float(np.max(np.abs(audio))) if audio.size else 1.0
    peak = max(peak, 1e-9)
    duration = audio.shape[1] / float(sample_rate)

    if out is None:
        out = audio_path.with_suffix(".waveform.svg")
    out = Path(out)
    if title is None:
        title = audio_path.name

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    lines.append('<rect width="100%" height="100%" fill="#111318"/>')
    lines.append(
        f'<text x="{margin_l}" y="24" fill="#d7dce5" font-family="monospace" font-size="15">{esc(title)}  {duration:.3f}s  {sample_rate} Hz  peak={peak:.3f}</text>'
    )
    # Draw second markers first so waveform stays visible.
    if duration > 0:
        sec = 0
        while sec <= int(np.ceil(duration)):
            x = margin_l + min(1.0, sec / duration) * plot_w
            lines.append(
                f'<line x1="{x:.2f}" y1="{margin_t}" x2="{x:.2f}" y2="{margin_t + plot_h}" stroke="#242a33" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{x + 3:.2f}" y="{height - 12}" fill="#697386" font-family="monospace" font-size="11">{sec}s</text>'
            )
            sec += 1
    for ch in range(channels):
        top = margin_t + ch * row_h
        mid = top + row_h * 0.5
        amp = row_h * 0.43
        label = "L" if ch == 0 else "R"
        lines.append(
            f'<text x="18" y="{mid + 5:.1f}" fill="#9aa4b2" font-family="monospace" font-size="14">{label}</text>'
        )
        lines.append(
            f'<line x1="{margin_l}" y1="{mid:.2f}" x2="{margin_l + plot_w}" y2="{mid:.2f}" stroke="#2b313b" stroke-width="1"/>'
        )
        mins, maxs = _bin_minmax(audio[ch] / peak, bins)
        step = plot_w / bins
        for idx, (lo, hi) in enumerate(zip(mins, maxs)):
            x = margin_l + idx * step
            y1 = mid - hi * amp
            y2 = mid - lo * amp
            lines.append(
                f'<line x1="{x:.2f}" y1="{y1:.2f}" x2="{x:.2f}" y2="{y2:.2f}" stroke="#72d0ff" stroke-width="1"/>'
            )
    lines.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf8")
    return out
