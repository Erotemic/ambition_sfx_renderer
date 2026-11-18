"""The gate wormhole (2026-10-31): open / enter / exit SFX candidates, measurements, reels.

The interdimensional gate is a bronze ring of glyph chevrons; its membrane is a
triangle door opening into a tunnel of nested triangles (blue/violet void, cyan
light). The same membrane is used by the smaller in-dimension travel portals.
Jon: "Opening the wormhole also needs a memorable and unique SFX, as well as
stepping into and out of it." The sheet rows the cues sit on:

- charge   30 x 34 ms (~1 s): a light pulse runs round the chevrons, 3 lamps lock
- opening  8 x 80 ms (640 ms) -> stable (loop) -> closing (640 ms)
- swallow  12 x 55 ms (660 ms): a body goes in
- emit     12 x 55 ms (660 ms): a body comes out

Three slots, from descriptions (synthesis only, no samples):

- `open_*`   ~1.1-1.6 s  the signature: a charge, a bloom, a hum settling
- `enter_*`  ~0.55-0.65 s  a body pushed through the membrane: suck inward, falling
- `exit_*`   ~0.55-0.65 s  the mirror: emitted outward, rising. `enter_x` and
             `exit_x` are a matched PAIR (same voices, inverted motion)

THE TRIANGLE MOTIF. Most candidates use the augmented triad D-F#-A# (each note
a major third, 4 semitones, above the last). It divides the octave into three
equal parts: on the circle of pitch classes its notes are the corners of an
equilateral TRIANGLE. It is unresolved and otherworldly, and it is in no other
cue of the bank.

Must not sound like `world.gate.rise` / `world.gate.fall` (portcullis chain
grind), `portal.*` (the portal gun: laser zap, bright attach whoosh),
`world.portal.enter` (the room door) or `boss.gnu_ton.portal_open` (a rising
whomp + tremolo shimmer). The README lists, per candidate, the nearest of those
by a crude spectral-profile correlation (MEASURED, not a listening test).

Every candidate is a sfxir recipe (`safety_profile: none`, own band limits),
written next to its WAV, so a chosen one ships into `sounds/active/` with the
same layers and seed and renders sample for sample the same.

Run with the renderer's interpreter, from anywhere:

    tools/ambition_sfx_renderer/.venv/bin/python tools/ambition_sfx_renderer/scripts/wormhole_candidates.py

Writes `<superproject>/untracked/sfx-candidates/wormhole/` (git-ignored;
`--outdir` elsewhere): `<slot>_<x>_<name>.{sfx.yaml,wav,png}`, `reel_*.wav`
with a `.txt` index of offsets next to each, `README.md`; `_render/` holds the
renderer's own output folders.
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import struct
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

RENDERER = Path(__file__).resolve().parents[1]
SHIPPED = RENDERER / "output"
SUPER = RENDERER.parents[1]
DEFAULT_OUT = SUPER / "untracked" / "sfx-candidates" / "wormhole"
PY = sys.executable
SR = 48000
JOBS = 6  # a shared box: never every CPU

# The bank's cues the new ones must not sound like.
COMPARE = ["world.gate.rise", "world.gate.fall", "world.gate.ripple", "portal.fire", "portal.attach",
           "portal.powerup", "world.portal.enter", "boss.gnu_ton.portal_open", "story.gate_ripple"]

PEAK = {"open": -9.0, "enter": -10.0, "exit": -10.0}

# The augmented triad on D: a triangle on the pitch-class circle.
D3, FS3, AS3 = 146.83, 185.00, 233.08
TRIAD3 = (D3, FS3, AS3)
TRIAD4 = tuple(2 * f for f in TRIAD3)
TRIAD5 = tuple(4 * f for f in TRIAD3)


# ── Recipe pieces ────────────────────────────────────────────────────────────
def env(a, h, r):
    return {"attack_ms": a, "hold_ms": h, "release_ms": r}


def bp(hz, q):
    return {"effect": "bandpass", "center_hz": hz, "q": q}


def lp(hz, order=2):
    return {"effect": "lowpass", "cutoff_hz": hz, "order": order}


def hp(hz, order=2):
    return {"effect": "highpass", "cutoff_hz": hz, "order": order}


def r4(x):
    return round(float(x), 4)


def decay(tau, points=14, floor=0.0, attack=0.0):
    """An exponential fall exp(-u / tau) across the layer (u 0..1) as points;
    with `attack`, a linear rise over the first `attack` of the layer first."""
    us = np.concatenate([[0.0], np.geomspace(0.004, 1.0, points - 1)])
    pts = []
    for u in us:
        if attack and u < attack:
            continue
        v = max(floor, np.exp(-(u - attack) / tau))
        pts.append([r4(u), r4(v)])
    if attack:
        pts = [[0.0, 0.0], [r4(attack), 1.0]] + [p for p in pts if p[0] > attack]
    return pts


def swell(peak_at, tau_after, points=12):
    """A rise (u^1.6) to 1 at `peak_at`, then exp fall with `tau_after`."""
    rise = [[r4(u), r4((u / peak_at) ** 1.6)] for u in np.linspace(0, peak_at, 6)]
    fall = [[r4(u), r4(np.exp(-(u - peak_at) / tau_after))] for u in np.linspace(peak_at, 1, points)[1:]]
    return rise + fall + [[1.0, 0.0]]


def wobble(rate0, rate1, dur_ms, depth, shape, points_per_cycle=8):
    """An amplitude wub whose rate glides (exp) from rate0 to rate1 Hz, under the
    contour `shape(u)`: a membrane flexing faster or slower."""
    n = 4000
    u = np.linspace(0, 1, n)
    rate = rate0 * (rate1 / rate0) ** u
    phase = np.cumsum(rate) * (dur_ms / 1000.0) / n
    cycles = phase[-1]
    k = max(16, int(cycles * points_per_cycle))
    idx = np.searchsorted(phase, np.linspace(0, cycles, k))
    idx = np.clip(idx, 0, n - 1)
    out = []
    for i in idx:
        w = (1 - depth) + depth * (0.5 + 0.5 * np.cos(2 * np.pi * phase[i]))
        out.append([r4(u[i]), r4(w * shape(u[i]))])
    out[-1][0] = 1.0
    return out


def noise(name, seed, color, gain, start, dur, e, effects, pan=0.0):
    return {"name": name, "kind": "noise_burst", "mode": "burst", "seed": seed, "color": color,
            "gain_db": gain, "start_ms": start, "duration_ms": dur, "pan": pan, "envelope": e,
            "effects": effects}


def sweep(name, seed, a, b, q, gain, start, dur, e, effects=None, pan=0.0, color="white"):
    """A moving band of noise from `a` Hz to `b` Hz."""
    return {"name": name, "kind": "noise_burst", "mode": "air_sweep", "seed": seed, "color": color,
            "start_hz": a, "end_hz": b, "q": q, "gain_db": gain, "start_ms": start, "duration_ms": dur,
            "pan": pan, "envelope": e, "effects": effects or []}


def tone(name, wave, freq, gain, start, dur, amp, e=None, harmonics=None, effects=None, seed=None,
         spread_cents=0.0, pan=0.0, vibrato=None, tremolo=None):
    """`freq` is [[u, hz], ...] (exp between points); `amp` is [[u, gain], ...]."""
    layer = {"name": name, "kind": "synth_tone", "waveform": wave,
             "frequency_hz": {"points": [[r4(u), round(float(f), 2)] for u, f in freq], "curve": "exp"},
             "amplitude_points": amp, "gain_db": gain, "start_ms": start, "duration_ms": dur, "pan": pan,
             "envelope": e or env(0.6, 0, 8)}
    if harmonics:
        layer["harmonics"] = harmonics
    if effects:
        layer["effects"] = effects
    if seed is not None:
        layer["seed"] = seed
    if spread_cents:
        layer["stereo_spread_cents"] = spread_cents
    if vibrato:
        layer["vibrato"] = vibrato
    if tremolo:
        layer["tremolo"] = tremolo
    return layer


def bronze(name, hz, gain, start, dur, tau, pan=0.0, spread=3):
    """A struck bronze bell (hum 0.5, prime, minor-third tierce, quint, nominal)."""
    partials = [(0.5, -6), (1.0, 0), (1.19, -5), (1.5, -9), (2.0, -4), (2.52, -14), (3.01, -17)]
    return tone(name, "sine", [[0, hz], [1, hz * 0.997]], gain, start, dur, decay(tau),
                harmonics=[{"ratio": r, "gain_db": g} for r, g in partials], spread_cents=spread, pan=pan)


def clack(name, seed, gain, start, band=(1600, 7000), ms=5.0, pan=0.0):
    """A chevron latching: a few ms of band-limited white noise."""
    centre = round(float(np.sqrt(band[0] * band[1])))
    return noise(name, seed, "white", gain, start, ms + 10, env(0, ms * 0.3, ms),
                 [hp(band[0], 3), lp(band[1], 2), bp(centre, 0.4)], pan=pan)


def sub(name, f0, f1, gain, start, dur, tau, drive_db=6.0, attack=0.0):
    """A low sine bloom; a little drive so a phone speaker hears harmonics."""
    freq = [[0, f0], [0.3, (f0 * f1) ** 0.5], [1, f1]]
    return tone(name, "sine", freq, gain, start, dur, decay(tau, attack=attack),
                harmonics=[{"ratio": 1.0, "gain_db": 0}, {"ratio": 2.0, "gain_db": -8}],
                effects=[{"effect": "distortion", "drive_db": drive_db}, lp(420)])


def post(peak, low=12000, high=30, comp=None, room=None, chorus=None, phaser=None, delay=None, out_ms=40):
    chain = [hp(high, 3), lp(low)]
    if comp:
        thr, ratio = comp
        chain.append({"effect": "compressor", "threshold_db": thr, "ratio": ratio, "attack_ms": 4.0,
                      "release_ms": 120})
    if phaser:
        chain.append({"effect": "phaser", **phaser})
    if chorus:
        chain.append({"effect": "chorus", **chorus})
    if delay:
        chain.append({"effect": "delay", **delay})
    if room:
        size, wet = room
        chain.append({"effect": "reverb", "room_size": size, "damping": 0.55, "wet_level": wet,
                      "dry_level": 1.0, "width": 1.0})
    chain += [{"effect": "highpass", "cutoff_hz": 20, "order": 2},  # DC after the plugins
              {"effect": "fade_edges", "in_ms": 0.2, "out_ms": out_ms},
              {"effect": "normalize_peak", "target_db": peak}]
    return chain


# ── The candidates: (slot, id) -> spec ───────────────────────────────────────
C: dict[tuple[str, str], dict] = {}


def cand(slot, cid, dur, seed, layers, postprocess, note):
    C[(slot, cid)] = dict(duration_ms=dur, seed=seed, layers=layers, postprocess=postprocess, note=note)


# Opening ────────────────────────────────────────────────────────────────────
# a: three chevron locks ring the triad, then the kawoosh, then the hum.
LOCK = 150  # ms between locks
BLOOM = 430
cand("open", "a_triad_kawoosh", 1500, 9101, [
    *[clack(f"latch{i}", 9110 + i, -14, i * LOCK, pan=(-0.3, 0.0, 0.3)[i]) for i in range(3)],
    *[bronze(f"lock{i}", TRIAD4[i], -13, i * LOCK, 1500 - i * LOCK, 0.32, pan=(-0.3, 0.0, 0.3)[i])
      for i in range(3)],
    sweep("charge", 9120, 260, 2400, 0.55, -16, 0, BLOOM + 20, env(380, 0, 30), color="pink"),
    sweep("kawoosh_out", 9121, 220, 3400, 0.5, 0, BLOOM, 320, env(6, 60, 220), color="pink"),
    sweep("kawoosh_back", 9122, 2600, 240, 0.6, -8, BLOOM + 180, 760, env(80, 120, 520), color="pink"),
    noise("body", 9123, "brown", -3, BLOOM, 700, env(4, 40, 600), [lp(700)]),
    sub("bloom", 42, 64, -4, BLOOM, 1000, 0.3, attack=0.03),
    tone("hum", "soft_saw", [[0, D3], [1, D3]], -15, BLOOM + 90, 1500 - BLOOM - 90, swell(0.2, 0.33),
         harmonics=[{"ratio": 1.0, "gain_db": 0}, {"ratio": FS3 / D3, "gain_db": -2},
                    {"ratio": AS3 / D3, "gain_db": -3}],
         effects=[lp(1100)], spread_cents=12, tremolo={"rate_hz": 6.0, "depth": 0.25}),
], post(PEAK["open"], comp=(-20, 2.5), room=(0.55, 0.16), out_ms=160),
    "three bronze chevron locks ring the triangle triad D-F#-A# (150 ms apart), a pink charge riser, then a "
    "kawoosh at 430 ms (noise bursts out 220->3400 Hz and draws back, brown body, 42->64 Hz sub bloom) that "
    "settles into a tremolo hum chord")

# b: locks as low bronze clunks, a reverse swell sucked shut, a glass bloom.
SWELL = 520
cand("open", "b_reverse_swell", 1350, 9201, [
    *[clack(f"latch{i}", 9210 + i, -8, i * 140, band=(900, 5000), ms=7, pan=(-0.35, 0.0, 0.35)[i])
      for i in range(3)],
    *[bronze(f"clunk{i}", TRIAD3[i], -7, i * 140, 420, 0.12, pan=(-0.35, 0.0, 0.35)[i]) for i in range(3)],
    noise("swell", 9220, "pink", -6, 40, SWELL - 40, env(SWELL - 60, 0, 6), [bp(1600, 0.45)]),
    tone("swell_tone", "triangle", [[0, AS3], [1, AS3 * 4]], -16, 60, SWELL - 60, [[0, 0], [0.9, 1], [1, 1]],
         e=env(0.6, 0, 4)),
    sub("thump", 110, 44, -2, SWELL, 520, 0.25, drive_db=8),
    *[tone(f"glass{i}", "bell", [[0, TRIAD5[i]], [1, TRIAD5[i] * 1.003]], -11, SWELL + 10 + 25 * i,
           1350 - SWELL - 10 - 25 * i, decay(0.4, attack=0.01), spread_cents=16,
           tremolo={"rate_hz": 11.0, "depth": 0.35, "phase": 2.1 * i}) for i in range(3)],
    sweep("wash", 9230, 3200, 380, 0.6, -9, SWELL, 700, env(4, 80, 560), color="pink"),
], post(PEAK["open"], comp=(-20, 2.0), chorus={"rate_hz": 0.9, "depth": 0.3, "mix": 0.35},
        room=(0.6, 0.18), out_ms=120),
    "three low bronze clunks (the triad an octave down) as the lamps lock, a pink reverse swell with a "
    "rising triangle tone that is sucked shut at 520 ms, then a pitched thump (110->44 Hz) and a glassy "
    "bell chord (D5-F#5-A#5, tremolo, chorus) washing outward")

# c: three voices spiral up two octaves, then everything drops into the void.
DROP = 560
cand("open", "c_spiral_drop", 1450, 9301, [
    *[tone(f"spiral{i}", "triangle", [[0, f], [1, f * 4]], -10, 0, DROP + 20, [[0, 0.15], [0.85, 1], [1, 0.8]],
           e=env(30, 0, 18), harmonics=[{"ratio": 1, "gain_db": 0}, {"ratio": 2, "gain_db": -12}],
           vibrato={"rate_hz": 7.0, "depth_cents": 18}, pan=(-0.4, 0.0, 0.4)[i]) for i, f in enumerate(TRIAD3)],
    sweep("riser", 9310, 400, 5000, 0.5, -14, 0, DROP, env(DROP - 40, 0, 20)),
    sub("drop", 190, 38, -2, DROP, 600, 0.4, drive_db=9),
    noise("kawoosh", 9320, "brown", -4, DROP, 760, env(3, 60, 640), [lp(1300)]),
    sweep("kawoosh_air", 9321, 1800, 220, 0.55, -10, DROP, 600, env(4, 40, 500), color="pink"),
    tone("hum", "triangle", [[0, D3], [1, D3]], -14, DROP + 120, 1450 - DROP - 120, swell(0.25, 0.3),
         harmonics=[{"ratio": 1.0, "gain_db": 0}, {"ratio": FS3 / D3, "gain_db": -2},
                    {"ratio": AS3 / D3, "gain_db": -2}, {"ratio": 2.0, "gain_db": -9}],
         spread_cents=10, tremolo={"rate_hz": 4.5, "depth": 0.2}),
], post(PEAK["open"], comp=(-18, 2.5), phaser={"rate_hz": 0.6, "depth": 0.6, "centre_frequency_hz": 700,
                                               "feedback": 0.3, "mix": 0.35},
        room=(0.6, 0.15), out_ms=120),
    "three triangle-wave voices in the triad spiral up two octaves (a charge, 0-560 ms), then everything "
    "drops: a driven sub dive 190->38 Hz, a brown kawoosh, and a phased triangle-chord hum settles")

# d: a bright glass arpeggio up the triangle, a falling whomp, echoes.
cand("open", "d_glass_arpeggio", 1150, 9401, [
    *[tone(f"arp{i}", "bell", [[0, f], [1, f]], -9 - (i % 3), 50 * i, 1150 - 50 * i, decay(0.22),
           spread_cents=8, pan=(-0.35, 0.0, 0.35)[i % 3])
      for i, f in enumerate((*TRIAD5, *(2 * f for f in TRIAD5)))],
    sub("whomp", 140, 56, -5, 250, 520, 0.3, drive_db=5),
    sweep("air", 9410, 4200, 900, 0.6, -12, 240, 520, env(10, 60, 380), color="pink"),
    tone("shimmer", "sine", [[0, 2 * TRIAD5[2]], [1, 2 * TRIAD5[2] * 1.01]], -18, 300, 850, swell(0.15, 0.4),
         harmonics=[{"ratio": 1.0, "gain_db": 0}, {"ratio": 1.26, "gain_db": -4}, {"ratio": 1.587, "gain_db": -6}],
         tremolo={"rate_hz": 21.0, "depth": 0.6}, spread_cents=20),
], post(PEAK["open"], low=13000, high=60, delay={"delay_seconds": 0.11, "feedback": 0.3, "mix": 0.22},
        room=(0.5, 0.14), out_ms=100),
    "a quick bell arpeggio up the triangle triad (two octaves, 50 ms a note), a soft falling whomp "
    "140->56 Hz, a 21 Hz tremolo shimmer and 110 ms echoes; light and magical (a travel-portal size)")

# e: the void inhales, a beat of silence, then exhales a low bronze chord.
EXH = 560
cand("open", "e_inhale_exhale", 1600, 9501, [
    sweep("inhale", 9510, 3600, 320, 0.55, -6, 0, 500, env(420, 0, 30), color="pink"),
    tone("inhale_tone", "sine", [[0, 70], [1, 150]], -10, 0, 500, [[0, 0], [0.9, 1], [1, 1]], e=env(0.6, 0, 6),
         harmonics=[{"ratio": 1, "gain_db": 0}, {"ratio": 3, "gain_db": -10}]),
    noise("exhale", 9520, "brown", -3, EXH, 650, env(2, 20, 620), [lp(1600)]),
    noise("exhale_air", 9521, "pink", -10, EXH, 600, env(2, 20, 570), [bp(1100, 0.5)]),
    sub("boom", 50, 34, -2, EXH, 1000, 0.22, drive_db=8),
    tone("chord", "soft_saw", [[0, D3 / 2], [0.08, D3], [1, D3]], -11, EXH, 1600 - EXH, swell(0.08, 0.28),
         e=env(20, 0, 60),
         harmonics=[{"ratio": 1.0, "gain_db": 0}, {"ratio": FS3 / D3, "gain_db": -1},
                    {"ratio": AS3 / D3, "gain_db": -2}, {"ratio": 2.0, "gain_db": -6}],
         effects=[lp(1200)], spread_cents=14),
    tone("cyan_whistle", "sine", [[0, TRIAD5[2]], [1, TRIAD5[2]]], -20, EXH + 150, 1600 - EXH - 150,
         swell(0.3, 0.5), vibrato={"rate_hz": 5.5, "depth_cents": 25, "delay_ms": 80}),
], post(PEAK["open"], room=(0.7, 0.17), out_ms=140),
    "the void inhales (pink sweep 3600->320 Hz over a rising 70->150 Hz tone), 60 ms of silence, then "
    "exhales: brown/pink burst, a 50->34 Hz sub boom and a low soft-saw triad chord that bends up an octave "
    "into place, a faint cyan whistle on top; the heaviest")


# Enter / exit pairs ─────────────────────────────────────────────────────────
DUR = 650

# a: a rubbery membrane.
cand("enter", "a_membrane", DUR, 9601, [
    noise("contact", 9602, "pink", -8, 0, 40, env(0, 4, 30), [bp(900, 0.7)]),
    tone("membrane", "sine", [[0, 230], [1, 68]], -3, 0, DUR, decay(0.3, attack=0.03),
         harmonics=[{"ratio": 1, "gain_db": 0}, {"ratio": 2, "gain_db": -7}, {"ratio": 3, "gain_db": -14}],
         vibrato={"rate_hz": 17.0, "depth_cents": 70}),
    sweep("suck", 9603, 2800, 260, 0.6, -8, 10, 520, env(30, 60, 380), color="pink"),
], post(PEAK["enter"], low=10000, comp=(-18, 2.0), room=(0.35, 0.12), out_ms=110),
    "a membrane contact tap, a rubbery wobbling tone (17 Hz vibrato) falling 230->68 Hz, a pink suck "
    "2800->260 Hz")
cand("exit", "a_membrane", DUR, 9611, [
    tone("membrane", "sine", [[0, 68], [1, 230]], -3, 0, DUR, swell(0.62, 0.15),
         harmonics=[{"ratio": 1, "gain_db": 0}, {"ratio": 2, "gain_db": -7}, {"ratio": 3, "gain_db": -14}],
         vibrato={"rate_hz": 17.0, "depth_cents": 70}),
    sweep("push", 9613, 260, 2800, 0.6, -8, 40, 520, env(200, 60, 200), color="pink"),
    noise("release", 9612, "pink", -6, 370, 50, env(0, 4, 40), [bp(1100, 0.7)]),
], post(PEAK["exit"], low=10000, comp=(-18, 2.0), room=(0.35, 0.12), out_ms=110),
    "the mirror: the wobbling membrane tone swells up 68->230 Hz, a pink push 260->2800 Hz, a release pop "
    "at 370 ms as the body breaks free")

# b: the open's triad glides down (in) or up (out).
cand("enter", "b_triad_glide", DUR, 9701, [
    *[tone(f"voice{i}", "triangle", [[0, TRIAD5[i]], [1, TRIAD4[i] * 0.84]], -6, 15 * i, DUR - 15 * i,
           decay(0.38, attack=0.02), harmonics=[{"ratio": 1, "gain_db": 0}, {"ratio": 2, "gain_db": -14}],
           pan=(-0.3, 0.0, 0.3)[i], spread_cents=8) for i in range(3)],
    sub("whump", 120, 60, -10, 0, 320, 0.25, drive_db=4),
    sweep("suck", 9702, 3000, 400, 0.6, -12, 0, 480, env(20, 40, 380), color="pink"),
], post(PEAK["enter"], low=11000, room=(0.45, 0.14), out_ms=110),
    "the open cue's triangle triad (D5-F#5-A#5, triangle waves) glides down an octave and a bit, fading; a "
    "small whump 120->60 Hz and a pink suck under it")
cand("exit", "b_triad_glide", DUR, 9711, [
    *[tone(f"voice{i}", "triangle", [[0, TRIAD4[i] * 0.84], [0.6, TRIAD5[i]], [1, TRIAD5[i]]], -6, 15 * i,
           DUR - 15 * i, swell(0.55, 0.2), harmonics=[{"ratio": 1, "gain_db": 0}, {"ratio": 2, "gain_db": -14}],
           pan=(-0.3, 0.0, 0.3)[i], spread_cents=8) for i in range(3)],
    sub("whump", 60, 120, -10, 260, 300, 0.3, drive_db=4),
    sweep("push", 9712, 400, 3000, 0.6, -12, 120, 480, env(260, 40, 160), color="pink"),
], post(PEAK["exit"], low=11000, room=(0.45, 0.14), out_ms=110),
    "the mirror: the triad glides UP into D5-F#5-A#5 and lands on it at 60%, a 60->120 Hz whump and a "
    "pink push as it arrives")

# c: a gulp in, a spit out (a time mirror).
GULP = 420
cand("enter", "c_gulp", DUR, 9801, [
    sweep("inhale", 9802, 3200, 360, 0.55, -6, 0, GULP + 10, env(GULP - 40, 0, 10), color="pink"),
    tone("pull", "sine", [[0, 160], [1, 80]], -12, 0, GULP + 10, [[0, 0], [0.9, 1], [1, 1]], e=env(0.6, 0, 6)),
    sub("gulp", 130, 40, -2, GULP, DUR - GULP, 0.2, drive_db=7),
    noise("gulp_body", 9803, "brown", -8, GULP, 150, env(1, 10, 130), [lp(500)]),
], post(PEAK["enter"], low=9000, room=(0.3, 0.1), out_ms=110),
    "an inhale (pink 3200->360 Hz, growing, over a falling 160->80 Hz pull) cut off by a muffled gulp at "
    "420 ms: a 130->40 Hz thump and a brown puff")
cand("exit", "c_spit", DUR, 9811, [
    sub("spit", 40, 130, -2, 0, 220, 0.25, drive_db=7, attack=0.04),
    noise("spit_body", 9812, "brown", -8, 0, 150, env(1, 10, 130), [lp(500)]),
    sweep("exhale", 9813, 360, 3200, 0.55, -6, 20, 560, env(4, 40, 500), color="pink"),
    tone("push", "sine", [[0, 80], [1, 160]], -12, 20, 560, decay(0.45), e=env(0.6, 0, 8)),
], post(PEAK["exit"], low=9000, room=(0.3, 0.1), out_ms=110),
    "the time mirror: a puff at once (a 40->130 Hz thump and brown burst), then an exhale fading outward "
    "(pink 360->3200 Hz over a rising 80->160 Hz push)")

# d: glass ripple, the light travel-portal pair.
cand("enter", "d_glass_ripple", 620, 9901, [
    *[tone(f"glass{i}", "bell", [[0, f * 1.5], [1, f * 0.6]], -9, 20 * i, 600 - 20 * i, decay(0.35),
           tremolo={"rate_hz": 14.0, "depth": 0.5, "phase": 2.1 * i}, pan=(-0.3, 0.0, 0.3)[i], spread_cents=10)
      for i, f in enumerate(TRIAD5)],
    sweep("air", 9902, 5000, 700, 0.6, -14, 0, 460, env(10, 40, 360)),
    sub("dip", 160, 70, -9, 0, 260, 0.25, drive_db=3),
], post(PEAK["enter"], low=13000, high=60, phaser={"rate_hz": 2.0, "depth": 0.7, "centre_frequency_hz": 1500,
                                                   "feedback": 0.4, "mix": 0.4}, room=(0.45, 0.14), out_ms=110),
    "a glassy bell cluster on the triad sliding down (x1.5 to x0.6) with a 14 Hz ripple tremolo through a "
    "phaser, a bright air sweep and a small 160->70 Hz dip; light, for the small travel portals")
cand("exit", "d_glass_ripple", 620, 9911, [
    *[tone(f"glass{i}", "bell", [[0, f * 0.6], [1, f * 1.5]], -9, 20 * i, 600 - 20 * i, swell(0.45, 0.25),
           tremolo={"rate_hz": 14.0, "depth": 0.5, "phase": 2.1 * i}, pan=(-0.3, 0.0, 0.3)[i], spread_cents=10)
      for i, f in enumerate(TRIAD5)],
    sweep("air", 9912, 700, 5000, 0.6, -14, 20, 460, env(200, 40, 160)),
    sub("lift", 70, 160, -9, 200, 260, 0.3, drive_db=3),
], post(PEAK["exit"], low=13000, high=60, phaser={"rate_hz": 2.0, "depth": 0.7, "centre_frequency_hz": 1500,
                                                  "feedback": 0.4, "mix": 0.4}, room=(0.45, 0.14), out_ms=110),
    "the mirror: the rippling bell cluster slides up (x0.6 to x1.5), swelling to 45%, the air rising "
    "700->5000 Hz and a 70->160 Hz lift")

# e: a warp wub that slows (in) or speeds up (out).
cand("enter", "e_warp_wub", 620, 10001, [
    tone("wub", "soft_saw", [[0, 620], [1, 125]], -4, 0, 620,
         wobble(16.0, 4.5, 620, 0.85, lambda u: min(1.0, u / 0.03) * np.exp(-u / 0.55)),
         harmonics=[{"ratio": 1, "gain_db": 0}, {"ratio": 1.26, "gain_db": -6}, {"ratio": 1.587, "gain_db": -8}],
         effects=[lp(2600)], spread_cents=12),
    noise("hiss", 10002, "white", -20, 0, 520, env(10, 40, 420), [bp(4200, 0.6)]),
    sub("sink", 110, 45, -8, 0, 620, 0.35, drive_db=5),
], post(PEAK["enter"], low=10000, comp=(-18, 2.0), room=(0.35, 0.12), out_ms=110),
    "a soft-saw triad tone diving 620->125 Hz while its wub slows from 16 to 4.5 Hz (a warp sinking away), "
    "a faint hiss and a 110->45 Hz sink")
cand("exit", "e_warp_wub", 620, 10011, [
    tone("wub", "soft_saw", [[0, 125], [1, 620]], -4, 0, 620,
         wobble(4.5, 16.0, 620, 0.85, lambda u: min(1.0, u / 0.6) if u < 0.6
                else float(np.exp(-(u - 0.6) / 0.16))),
         harmonics=[{"ratio": 1, "gain_db": 0}, {"ratio": 1.26, "gain_db": -6}, {"ratio": 1.587, "gain_db": -8}],
         effects=[lp(2600)], spread_cents=12),
    noise("hiss", 10012, "white", -20, 100, 520, env(320, 40, 120), [bp(4200, 0.6)]),
    sub("rise", 45, 110, -8, 200, 420, 0.3, drive_db=5),
], post(PEAK["exit"], low=10000, comp=(-18, 2.0), room=(0.35, 0.12), out_ms=110),
    "the mirror: the soft-saw triad tone climbs 125->620 Hz while its wub speeds up from 4.5 to 16 Hz, peaking "
    "at 60%, a hiss and a 45->110 Hz rise")


# The pick per slot, with why (filled in after the measurements; README only).
RECOMMEND: dict[str, tuple[str, str]] = {
    "open": ("a_triad_kawoosh",
             "it follows the sheet: three chevron locks ring the triangle triad (the charge row's three lamps), "
             "the kawoosh lands at 430 ms (fire the cue ~430 ms before the opening row so the bloom meets its "
             "first frame), and the hum hands over to the stable loop. A three-note bell motif is the most "
             "recognisable thing here, and enter_b/exit_b reuse it. Runners-up: open_e (heaviest, for the big "
             "gate) and open_d (lightest, least like the bank, for the small travel portals)"),
    "enter": ("b_triad_glide",
              "the triad falls away as the body is drawn in, so every trip echoes the opening; it is quiet "
              "(RMS -27 dB), so it will not tire on repeat. enter_a (rubber membrane) is the more physical "
              "alternative"),
    "exit": ("b_triad_glide",
             "the pair of enter_b: the triad rises back into place as the body is pushed out"),
}

SLOTS = ("open", "enter", "exit")
PAIRS = [cid[0] for (slot, cid) in C if slot == "enter"]


def name(slot, cid):
    return f"{slot}_{cid}"


def cue_id(slot, cid):
    return f"audition.wormhole.{slot}.{cid}"


def recipe(slot, cid, spec):
    return {"schema": "ambition.sfxir.v1", "id": cue_id(slot, cid), "duration_ms": spec["duration_ms"],
            "render": {"sample_rate": SR, "channels": 2, "seed": spec["seed"], "buffer_size": 128,
                       # Our own band limits: the default safety lowpass dulls the transients.
                       "safety_profile": "none"},
            "layers": copy.deepcopy(spec["layers"]), "postprocess": copy.deepcopy(spec["postprocess"]),
            "notes": f"Wormhole {slot} candidate {cid}: {spec['note']}. From "
                     "tools/ambition_sfx_renderer/scripts/wormhole_candidates.py."}


def render(path: Path, outdir: Path):
    done = subprocess.run([PY, "-m", "ambition_sfx_renderer", "render", str(path), "--outdir", str(outdir),
                           "--format-policy", "wav", "--force"], cwd=RENDERER, capture_output=True, text=True,
                          env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    if done.returncode:
        raise SystemExit(f"{path.name}: {done.stderr[-3000:]}")


def load(cue_dir: Path) -> np.ndarray:
    wavs = sorted(cue_dir.glob("*.wav")) or sorted(cue_dir.glob("*.ogg"))
    if not wavs:
        raise SystemExit(f"no audio in {cue_dir}")
    a, sr = sf.read(wavs[0], always_2d=True)
    assert sr == SR, (cue_dir, sr)
    if a.shape[1] == 1:
        a = np.repeat(a, 2, axis=1)
    return a.astype(np.float32)


# ── Measuring ────────────────────────────────────────────────────────────────
def env_db(a, ms=5):
    mono = np.abs(a).max(axis=1)
    w = int(SR * ms / 1000)
    k = len(mono) // w
    return 20 * np.log10(mono[: k * w].reshape(k, w).max(axis=1) + 1e-9)


def band_gram(a, bands=60, floor=-50.0):
    """A log-band spectrogram (10 ms frames, 60 bands 60 Hz-12 kHz) in dB re
    its own peak, floored: the spectral shape AND its motion in time."""
    mono = a.mean(axis=1)
    hop = SR // 100
    win = 2048
    pad = np.concatenate([mono, np.zeros(win)])
    k = max(1, len(mono) // hop)
    w = np.hanning(win)
    spec = np.stack([np.abs(np.fft.rfft(pad[i * hop: i * hop + win] * w)) ** 2 for i in range(k)])
    freq = np.fft.rfftfreq(win, 1 / SR)
    edges = np.geomspace(60, 12000, bands + 1)
    g = np.stack([spec[:, (freq >= lo) & (freq < max(hi, lo + 24))].sum(axis=1)
                  for lo, hi in zip(edges[:-1], edges[1:])], 1)
    db = 10 * np.log10(g / (g.max() + 1e-20) + 1e-12)
    return np.maximum(db, floor)


def similarity(a, b) -> float:
    """Pearson r of two band-grams: the shorter slid along the longer (20 ms
    steps), the best window. Crude: siblings in the bank (gate rise vs fall)
    score ~0.84, unrelated bank cues a median ~0.4."""
    ga, gb = band_gram(a), band_gram(b)
    if len(ga) < len(gb):
        ga, gb = gb, ga
    best = -1.0
    for off in range(0, len(ga) - len(gb) + 1, 2):
        r = np.corrcoef(ga[off: off + len(gb)].ravel(), gb.ravel())[0, 1]
        if np.isfinite(r):
            best = max(best, float(r))
    return best


def measure(a, ref=None) -> dict:
    mono = a.mean(axis=1)
    peak = float(np.abs(a).max())
    e = env_db(a)
    top = e.max()
    first = int(np.argmax(e > -60)) * 5  # the first 5 ms with anything in it
    onset = int(np.argmax(e > top - 30)) * 5  # within 30 dB of the peak
    loud_at = int(np.argmax(e)) * 5
    end_db = float(e[-4:].max() - top)
    spec = np.abs(np.fft.rfft(mono)) ** 2
    freq = np.fft.rfftfreq(len(mono), 1 / SR)
    out = dict(
        ms=round(1000 * len(mono) / SR),
        peak_db=round(float(20 * np.log10(peak + 1e-12)), 1),
        rms_db=round(float(10 * np.log10((mono ** 2).mean() + 1e-12)), 1),
        clipped=int((np.abs(a) >= 0.999).sum()),
        dc=round(float(np.abs(a.mean(axis=0)).max()), 5),
        first_ms=first,
        onset_ms=onset,
        loudest_ms=loud_at,
        end_db=round(end_db, 1),
        centroid_hz=round(float((spec * freq).sum() / (spec.sum() or 1.0))),
    )
    if ref:
        best = max((similarity(a, r), k) for k, r in ref.items())
        out["nearest"] = f"{best[1]} ({best[0]:.2f})"
    return out


def problems(m, slot) -> list[str]:
    bad = []
    if m["clipped"]:
        bad.append(f"{m['clipped']} clipped samples")
    if m["dc"] > 0.002:
        bad.append(f"DC {m['dc']}")
    if m["first_ms"] > 10:
        bad.append(f"silent for the first {m['first_ms']} ms")
    if m["end_db"] > -30:
        bad.append(f"ends only {m['end_db']} dB below the peak (cut tail?)")
    if abs(m["peak_db"] - PEAK[slot]) > 0.3:
        bad.append(f"peak {m['peak_db']} != {PEAK[slot]}")
    return bad


# ── Pictures (a log-frequency spectrogram, no plotting library) ─────────────
def write_png(path: Path, rgb: np.ndarray):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(kind, data):
        c = struct.pack(">I", len(data)) + kind + data
        return c + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def spectrogram_png(path: Path, a: np.ndarray, px_per_s=400, rows=160):
    mono = a.mean(axis=1)
    hop = SR // px_per_s
    win = 2048
    pad = np.concatenate([np.zeros(win // 2), mono, np.zeros(win)])
    frames = (len(mono)) // hop
    w = np.hanning(win)
    spec = np.stack([np.abs(np.fft.rfft(pad[i * hop: i * hop + win] * w)) for i in range(frames)], axis=1)
    freq = np.fft.rfftfreq(win, 1 / SR)
    edges = np.geomspace(40, 16000, rows + 1)
    img = np.stack([spec[(freq >= lo) & (freq < max(hi, lo + 24))].max(axis=0) if ((freq >= lo) & (freq < max(hi, lo + 24))).any()
                    else np.zeros(frames) for lo, hi in zip(edges[:-1], edges[1:])])
    db = 20 * np.log10(img / (img.max() + 1e-12) + 1e-9)
    v = np.clip((db + 70) / 70, 0, 1)[::-1]
    # Dark violet -> cyan -> white.
    rgb = np.stack([np.clip(1.5 * v - 0.5, 0, 1) * 255 + (1 - v) * 20, v ** 0.8 * 230 + 10, (0.3 + 0.7 * v) * 255],
                   axis=-1)
    # A tick every 100 ms along the bottom, octave lines (D) faintly.
    for x in range(0, frames, px_per_s // 10):
        rgb[-4:, x] = 255
    write_png(path, rgb)


# ── Reels ────────────────────────────────────────────────────────────────────
def concat(items):
    """items: (label, audio or seconds of silence). Returns audio and offset lines."""
    parts, marks, t = [], [], 0.0
    for label, a in items:
        if isinstance(a, (int, float)):
            parts.append(np.zeros((int(a * SR), 2), np.float32))
            t += a
            continue
        marks.append(f"{t:7.3f}s  {label}")
        parts.append(a)
        t += len(a) / SR
    return np.concatenate(parts), marks


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    here = args.outdir
    here.mkdir(parents=True, exist_ok=True)

    paths = []
    for (slot, cid), spec in C.items():
        path = here / f"{name(slot, cid)}.sfx.yaml"
        path.write_text(yaml.safe_dump(recipe(slot, cid, spec), sort_keys=False))
        paths.append(path)
    with ThreadPoolExecutor(JOBS) as pool:
        list(pool.map(lambda q: render(q, here / "_render"), paths))
    got = {key: load(here / "_render" / cue_id(*key)) for key in C}
    ref = {k: load(SHIPPED / k) for k in COMPARE if (SHIPPED / k).is_dir()}

    for key, a in got.items():
        sf.write(here / f"{name(*key)}.wav", a, SR)
        spectrogram_png(here / f"{name(*key)}.png", a)

    # Reels, one gain for all of them so they keep their levels.
    opens = [k for k in C if k[0] == "open"]
    reel_open = []
    for k in opens:
        reel_open += [(name(*k), got[k]), ("", 1.0)]
    reel_trip = []
    if RECOMMEND:  # the recommended trip first
        rec = [(slot, RECOMMEND[slot][0]) for slot in SLOTS]
        reel_trip += [(name(*rec[0]) + " (recommended)", got[rec[0]]), ("", 1.0), (name(*rec[1]), got[rec[1]]),
                      ("", 0.6), (name(*rec[2]), got[rec[2]]), ("", 1.5)]
    for x in PAIRS:
        enter = next(k for k in C if k[0] == "enter" and k[1][0] == x)
        exit_ = next(k for k in C if k[0] == "exit" and k[1][0] == x)
        o = next((k for k in opens if k[1][0] == x), opens[0])
        reel_trip += [(name(*o), got[o]), ("", 1.0), (name(*enter), got[enter]), ("", 0.6),
                      (name(*exit_), got[exit_]), ("", 1.5)]
    reel_all = []
    for slot in SLOTS:
        for k in C:
            if k[0] == slot:
                reel_all += [(name(*k), got[k]), ("", 0.8)]
    reel_all += [("", 1.0)]
    for k, a in ref.items():
        reel_all += [(f"{k} (in the bank, for contrast)", a), ("", 0.8)]
    reels = {"reel_open": reel_open, "reel_trip": reel_trip, "reel_all": reel_all}
    audio = {n: concat(items) for n, items in reels.items()}
    gain = min(1.0, 0.89 / max(float(np.abs(a).max()) for a, _ in audio.values()))
    for n, (a, marks) in audio.items():
        sf.write(here / f"{n}.wav", a * gain, SR)
        (here / f"{n}.txt").write_text("\n".join(marks) + "\n")

    # README.
    keys = list(ref)
    pairs = [(similarity(ref[x], ref[y]), f"{x} vs {y}") for i, x in enumerate(keys) for y in keys[i + 1:]]
    base = [r for r, _ in pairs]
    base_pair = max(pairs)[1]
    lines = ["# Wormhole SFX candidates (open / enter / exit)", "",
             "Generated by `tools/ambition_sfx_renderer/scripts/wormhole_candidates.py` (2026-10-31); rerun it to",
             "regenerate everything here. Each candidate is `<slot>_<x>_<name>.sfx.yaml` (a sfxir recipe that",
             "ports into `sounds/active/` as is: change `id`), its `.wav` and a log-frequency spectrogram `.png`",
             "(40 Hz bottom to 16 kHz top, a tick every 100 ms).", "",
             "The motif: most use the augmented triad D-F#-A#, three notes a major third apart, which split the",
             "octave in three equal parts: a TRIANGLE on the circle of pitch classes.", "",
             "Columns are MEASURED from the WAV: duration, peak dBFS, RMS dBFS (whole cue), clipped samples",
             "(|x| >= 0.999), max |DC| per channel, first sound (first 5 ms window above -60 dBFS), rise (first",
             "5 ms window within 30 dB of the peak; late on purpose for a swell), the loudest 5 ms, the last 20 ms",
             "relative to the peak, spectral centroid, and the most similar bank cue: Pearson r of two 10 ms x",
             "60-band log spectrograms, the shorter slid along the longer (crude, MEASURED, not a listening test).",
             f"For scale, bank cue against bank cue: median r {np.median(base):.2f}, max {max(base):.2f}"
             f" ({base_pair}).", "",
             "Reels (one shared gain; offsets in the `.txt` beside each):", "",
             "- `reel_open.wav`: each open candidate, 1 s of silence between.",
             "- `reel_trip.wav`: the recommended open, 1 s, enter, 0.6 s, exit first; then per pair x: open_x, 1 s,",
             "  enter_x, 0.6 s, exit_x (open_x is not tied to pair x; the letters just line up).",
             "- `reel_all.wav`: every candidate, then the bank's gate and portal cues for contrast.", ""]
    if RECOMMEND:
        lines += ["## Recommended (by design and measurement; NOT yet heard by a person)", ""]
        lines += [f"- {slot}: `{name(slot, cid)}` -- {why}" for slot, (cid, why) in RECOMMEND.items()] + [""]
    hdr = ("| file | what it is | ms | peak dB | rms dB | clip | DC | first ms | rise ms | loudest ms | end dB "
           "| centroid Hz | most similar bank cue (r) |")
    any_bad = []
    for slot in SLOTS:
        lines += [f"## {slot}", "", hdr, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for k in C:
            if k[0] != slot:
                continue
            m = measure(got[k], ref)
            bad = problems(m, slot)
            if bad:
                any_bad.append(f"- `{name(*k)}`: " + "; ".join(bad))
            lines.append(f"| `{name(*k)}.wav` | {C[k]['note']} | {m['ms']} | {m['peak_db']} | {m['rms_db']} | "
                         f"{m['clipped']} | {m['dc']} | {m['first_ms']} | {m['onset_ms']} | {m['loudest_ms']} | "
                         f"{m['end_db']} | {m['centroid_hz']} | {m['nearest']} |")
        lines.append("")
    lines += ["## Pairs", "", "r of exit_x against enter_x played BACKWARDS (a mirror scores high), and against",
              "enter_x forwards.", "", "| pair | r(exit, reversed enter) | r(exit, enter) |", "|---|---:|---:|"]
    for x in PAIRS:
        en = got[next(k for k in C if k[0] == "enter" and k[1][0] == x)]
        ex = got[next(k for k in C if k[0] == "exit" and k[1][0] == x)]
        lines.append(f"| {x} | {similarity(ex, en[::-1]):.2f} | {similarity(ex, en):.2f} |")
    lines += ["", "## The bank's cues, for contrast", "", "| cue | ms | peak dB | rms dB | centroid Hz |",
              "|---|---:|---:|---:|---:|"]
    for k, a in ref.items():
        m = measure(a)
        lines.append(f"| `{k}` | {m['ms']} | {m['peak_db']} | {m['rms_db']} | {m['centroid_hz']} |")
    lines += ["", "## Checks", ""] + (any_bad or ["- every candidate: no clipping, |DC| <= 0.002, sound from the "
                                                  "first 10 ms, tail >= 30 dB down, peak on target"])
    (here / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(any_bad or ["checks ok"]))
    print(f"{len(C)} candidates -> {here}")


if __name__ == "__main__":
    main()
