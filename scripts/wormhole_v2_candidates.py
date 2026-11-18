"""The gate wormhole, round 2 (2026-10-31): NOISE-ONLY open / enter / exit candidates.

Jon on round 1 (`wormhole_candidates.py`): "The SFX shouldn't be musical. It
needs to have more layers and texture to it though. A wormhole woosh. Hard for
me to tell which candidate was which."

So: no tones, notes, chords or tonal glides. Every candidate is layered noise:
filtered-noise bands sweeping at different rates (a low roar, a mid rush, a
high hiss), sub pressure (noise under ~80 Hz, unpitched thumps), crackle
(sparse clicks), swirl (short overlapping noise segments whose pan rotates and
whose band moves; the gaps between them flutter), and a suction / reverse swell
into a burst and a tail. Band filters are wide (an octave or more), so no band
rings at a pitch; the README's "tonal peak" column MEASURES that.

Layer names are `<group>.<detail>`; the script renders each group alone through
the same chain and reports how loud it is against the whole (`groups` column),
so "at least 4 audible layers" is measured, not claimed.

- `open_N_<word>`   0.9-1.6 s: build, whoosh bloom, a swirling tail
- `pair_N_enter_<word>` / `pair_N_exit_<word>`  0.35-0.6 s: enter sucks in and
  darkens (bands sweep down, close to the centre); exit bursts out and
  brightens (bands sweep up, spread wide)

Writes `<superproject>/untracked/sfx-candidates/wormhole_v2/` (git-ignored):
`<name>.sfx.yaml` + `<name>.wav` per candidate, `listen_open_N.wav`,
`listen_pair_N.wav` (enter, 0.5 s, exit), reels with N dry clicks before
candidate N, `README.md`; `png/` spectrograms, `_render/` and `_groups/`.

    tools/ambition_sfx_renderer/.venv/bin/python tools/ambition_sfx_renderer/scripts/wormhole_v2_candidates.py
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy import signal

import multiprocessing as mp
import sys

from wormhole_candidates import RENDERER, SUPER, bp, env, hp, load, lp, spectrogram_png, sweep

if str(RENDERER) not in sys.path:
    sys.path.insert(0, str(RENDERER))
# Imported once here and inherited by the forked workers: on a busy shared
# disk one `python -m ambition_sfx_renderer` start paid ~100 s of imports.
from ambition_sfx_renderer.render import render_file  # noqa: E402


def _render(job):
    path, outdir = job
    render_file(Path(path), out_root=Path(outdir), format_policy="wav", force=True)
    return path

DEFAULT_OUT = SUPER / "untracked" / "sfx-candidates" / "wormhole_v2"
SR = 48000
JOBS = 6  # a shared box: never every CPU
PEAK = {"open": -9.0, "enter": -10.0, "exit": -10.0}


# ── Layer pieces (all noise) ─────────────────────────────────────────────────
def noise(name, seed, color, gain, start, dur, e, effects, pan=0.0, mode="burst", **kw):
    return {"name": name, "kind": "noise_burst", "mode": mode, "seed": seed, "color": color, "gain_db": gain,
            "start_ms": round(start, 1), "duration_ms": round(dur, 1), "pan": round(pan, 3), "envelope": e,
            "effects": effects, **kw}


def band(name, seed, color, gain, start, dur, e, lo, hi, pan=0.0, extra=None):
    """Noise between lo and hi Hz (gentle slopes: no ringing band)."""
    fx = ([hp(lo, 2)] if lo else []) + ([lp(hi, 2)] if hi else []) + (extra or [])
    return noise(name, seed, color, gain, start, dur, e, fx, pan=pan)


def rumble(name, seed, gain, start, dur, e, cutoff=80):
    """Sub pressure: brown noise below `cutoff`."""
    return noise(name, seed, "brown", gain, start, dur, e, [hp(24, 2), lp(cutoff, 4)])


def thump(name, seed, gain, start, dur, cutoff=140):
    """An unpitched low blow: brown noise under a fast exponential decay."""
    return noise(name, seed, "brown", gain, start, dur, env(0, 0, min(40, dur * 0.2)), [hp(26, 2), lp(cutoff, 4)],
                 mode="thud")


def crackle(name, seed, gain, start, dur, count, decay_ms, e, lo=1800, hi=11000, pan=0.0):
    """Sparse clicks: static, crumbly debris."""
    return noise(name, seed, "white", gain, start, dur, e, [hp(lo, 2), lp(hi, 2)], pan=pan, mode="grains",
                 grain_count=count, grain_decay_ms=decay_ms, grain_start_ms=0)


def flanger(rate=0.8, depth=0.7, delay_ms=3.0, feedback=0.25, mix=0.5):
    """A swept comb: the jet whoosh. The sweep keeps its notches moving."""
    return {"effect": "chorus", "rate_hz": rate, "depth": depth, "centre_delay_ms": delay_ms,
            "feedback": feedback, "mix": mix}


def wide(group, seed, color, gain, start, dur, e, lo, hi, spread=0.85, extra=None):
    """Two decorrelated copies panned apart: a wide band."""
    return [band(f"{group}.L", seed, color, gain, start, dur, e, lo, hi, pan=-spread, extra=extra),
            band(f"{group}.R", seed + 1, color, gain, start, dur, e, lo, hi, pan=spread, extra=extra)]


def swirl(group, seed, start, dur, segs, f0, f1, g0, g1, turns, w0, w1, accel=1.0, gap=0.3, color="pink",
          phase=0.0):
    """Short overlapping noise segments: the band moves f0 -> f1 Hz (two
    octaves wide), the pan circles `turns` times (`accel` > 1 speeds up,
    < 1 slows down) with a width w0 -> w1, the level g0 -> g1 dB. `gap`
    opens dips between segments: an amplitude flutter at the segment rate."""
    step = dur / segs
    seg = step * (1.7 - 1.2 * gap)
    out = []
    for k in range(segs):
        u = (k + 0.5) / segs
        c = f0 * (f1 / f0) ** u
        pan = (w0 + (w1 - w0) * u) * np.sin(2 * np.pi * turns * u ** accel + phase)
        t0 = max(start, start + k * step - (seg - step) / 2)
        length = min(seg, start + dur - t0)
        out.append(band(f"{group}.{k:02d}", seed + k, color, g0 + (g1 - g0) * u, t0, length,
                        env(length * 0.45, 0, length * 0.45), round(c / 2), round(c * 2), pan=float(pan)))
    return out


def post(peak, low=14000, high=26, comp=None, room=None, phaser=None, out_ms=40):
    chain = [hp(high, 3), lp(low)]
    if comp:
        chain.append({"effect": "compressor", "threshold_db": comp[0], "ratio": comp[1], "attack_ms": 4.0,
                      "release_ms": 120})
    if phaser:
        chain.append({"effect": "phaser", **phaser})
    if room:
        chain.append({"effect": "reverb", "room_size": room[0], "damping": 0.6, "wet_level": room[1],
                      "dry_level": 1.0, "width": 1.0})
    chain += [{"effect": "highpass", "cutoff_hz": 20, "order": 2},
              {"effect": "fade_edges", "in_ms": 0.2, "out_ms": out_ms},
              {"effect": "normalize_peak", "target_db": peak}]
    return chain


# ── The candidates ───────────────────────────────────────────────────────────
OPENS: dict[int, dict] = {}
PAIRS: dict[int, dict] = {}


def spec(dur, seed, layers, postprocess, hear):
    return dict(duration_ms=dur, seed=seed, layers=layers, postprocess=postprocess, hear=hear)


# Open 1: roar. Three bands rise together, a flanged roar blooms at 450 ms.
B = 450
OPENS[1] = dict(word="roar", **spec(1400, 2101, [
    sweep("roar.build", 2102, 120, 700, 0.6, -2, 0, B + 20, env(B - 40, 0, 20)),
    sweep("rush.build", 2103, 400, 3200, 0.7, -6, 0, B + 20, env(B - 30, 0, 20)),
    sweep("hiss.build", 2104, 3000, 10000, 0.8, -12, 120, B - 100, env(B - 150, 0, 10)),
    rumble("sub.rumble", 2105, 0, 0, 1400, env(380, 100, 900)),
    thump("sub.thump", 2106, 0, B, 420),
    band("roar.bloom", 2107, "brown", 0, B, 900, env(3, 60, 830), 60, 1000, extra=[flanger(0.8, 0.7, 4.0, 0.25, 0.5)]),
    band("rush.bloom", 2108, "pink", -5, B, 950, env(3, 40, 900), 300, 3500, extra=[flanger(1.3, 0.6, 2.5, 0.2, 0.5)]),
    *wide("hiss.bloom", 2109, "pink", -12, B, 700, env(2, 20, 670), 3000, 12000),
    *swirl("swirl.tail", 2120, B + 80, 870, 16, 1800, 500, -9, -16, 2.5, 0.85, 0.85, accel=0.8, gap=0.35),
    crackle("crackle.debris", 2140, -10, B - 60, 900, 45, 2.0, env(10, 200, 650)),
], post(PEAK["open"], comp=(-20, 2.5), room=(0.5, 0.15), out_ms=160),
    "a rising roar, then a big jet-like WHOOSH with a low boom; crackling debris and a fluttering swirl "
    "circling left-right as it dies away"))

# Open 2: suction. A reverse swell sucked shut at 520 ms, a beat, then the burst.
S, BO = 520, 565
OPENS[2] = dict(word="suction", **spec(1300, 2201, [
    sweep("suck.pull", 2202, 7000, 350, 0.7, -2, 0, S, env(S - 40, 0, 6)),
    band("suck.low", 2203, "brown", -6, 0, S, env(S - 20, 0, 6), 60, 450),
    crackle("crackle.in", 2204, -10, 100, S - 100, 30, 1.5, env(S - 140, 0, 8), lo=2500),
    rumble("sub.pressure", 2205, -2, 0, S + 10, env(S - 20, 0, 20)),
    thump("sub.boom", 2206, 0, BO, 420, cutoff=160),
    sweep("burst.L", 2207, 300, 8000, 0.7, -4, BO, 520, env(2, 30, 480), pan=-0.8),
    sweep("burst.R", 2208, 300, 8000, 0.7, -4, BO + 12, 520, env(2, 30, 480), pan=0.8),
    band("roar.body", 2209, "brown", -2, BO, 700, env(2, 40, 650), 50, 800),
    *swirl("swirl.tail", 2220, BO + 80, 1300 - BO - 80, 14, 2400, 700, -9, -16, 2.0, 0.9, 0.9, gap=0.4),
], post(PEAK["open"], comp=(-20, 2.0), room=(0.55, 0.15), out_ms=150,
        phaser={"rate_hz": 0.9, "depth": 0.6, "centre_frequency_hz": 1200, "feedback": 0.1, "mix": 0.35}),
    "air and grit SUCKED inward, cut off dead, a beat of silence, then a burst that blows out to both "
    "sides with a deep boom; a phasey swirl trails off"))

# Open 3: vortex. The swirl is the star: it spins up, blooms, spins down.
B3 = 500
OPENS[3] = dict(word="vortex", **spec(1500, 2301, [
    *swirl("swirl.build", 2310, 0, B3 + 20, 14, 300, 2600, -16, -4, 3.0, 0.9, 0.9, accel=1.8, gap=0.3),
    sweep("hiss.build", 2302, 2500, 9000, 0.8, -14, 150, B3 - 130, env(B3 - 170, 0, 10)),
    rumble("sub.rumble", 2303, -2, 0, 1500, env(420, 200, 860)),
    thump("sub.thump", 2304, -1, B3, 380),
    band("burst.whoosh", 2305, "pink", -1, B3, 600, env(4, 40, 550), 200, 6000, extra=[flanger(1.0, 0.8, 3.0, 0.3, 0.55)]),
    *swirl("swirl.tail", 2330, B3 + 60, 1500 - B3 - 60, 18, 2000, 400, -5, -16, 3.0, 0.9, 0.6, accel=0.55, gap=0.4,
           phase=1.2),
    crackle("crackle.static", 2350, -12, B3 + 100, 900, 25, 3.0, env(10, 100, 780), lo=1200, hi=6000),
], post(PEAK["open"], comp=(-20, 2.0), room=(0.6, 0.17), out_ms=160),
    "a whirling noise that spins faster and faster round your head, bursts in a flanged whoosh, then spins "
    "down slower and slower over a low rumble and static"))

# Open 4: tear. Static builds, space rips, a crunchy roar; embers crackle.
B4 = 440
OPENS[4] = dict(word="tear", **spec(1200, 2401, [
    crackle("static.a", 2402, -14, 0, 220, 6, 2.0, env(10, 100, 100)),
    crackle("static.b", 2403, -12, 150, 220, 14, 2.0, env(10, 100, 100), pan=-0.4),
    crackle("static.c", 2404, -10, 300, B4 - 290, 30, 2.0, env(10, 60, 50), pan=0.4),
    sweep("rush.build", 2405, 500, 4000, 0.7, -9, 0, B4 + 10, env(B4 - 40, 0, 10)),
    noise("rip.tear", 2406, "white", -3, B4 - 40, 170, env(2, 40, 120), [hp(900, 2), lp(9000, 2)], mode="scrape"),
    band("roar.burst", 2407, "brown", -1, B4, 760, env(2, 40, 700), 50, 1100,
         extra=[{"effect": "distortion", "drive_db": 8}, lp(2500, 2)]),
    thump("sub.thump", 2408, 0, B4, 400),
    *wide("hiss.tail", 2409, "pink", -14, B4 + 10, 700, env(5, 50, 640), 4000, 12000),
    crackle("embers.L", 2420, -11, B4 + 60, 1200 - B4 - 60, 40, 2.5, env(10, 50, 630), lo=1500, pan=-0.7),
    crackle("embers.R", 2421, -11, B4 + 80, 1200 - B4 - 80, 40, 2.5, env(10, 50, 610), lo=1500, pan=0.7),
    *swirl("swirl.tail", 2430, B4 + 60, 1200 - B4 - 60, 10, 1200, 600, -12, -18, 1.5, 0.7, 0.7, gap=0.3),
], post(PEAK["open"], comp=(-20, 2.0), room=(0.45, 0.14), out_ms=140),
    "electric static thickening, then space RIPS open with a crunchy roar; sparks crackle left and right "
    "as it fades"))

# Open 5: jet. A flanged jet sweep passes through, led in by pressure pulses.
B5 = 450
OPENS[5] = dict(word="jet", **spec(1100, 2501, [
    band("jet.body", 2502, "pink", -1, 0, 1100, env(B5, 40, 600), 80, 12000, extra=[flanger(0.9, 0.9, 3.0, 0.35, 0.6)]),
    sweep("jet.rise", 2503, 400, 6000, 0.7, -8, 0, B5, env(B5 - 50, 0, 20)),
    sweep("jet.fall", 2504, 6000, 600, 0.7, -8, B5, 650, env(5, 40, 600)),
    thump("pulse.1", 2505, -8, 0, 160, cutoff=120),
    thump("pulse.2", 2506, -6, 200, 160, cutoff=120),
    thump("pulse.3", 2507, -4, 330, 160, cutoff=120),
    thump("pulse.4", 2508, 0, B5, 420, cutoff=150),
    rumble("sub.rumble", 2509, -4, 0, 1100, env(300, 100, 700)),
    crackle("crackle.spit", 2510, -11, 300, 700, 35, 2.0, env(20, 120, 500)),
    *swirl("swirl.tail", 2520, B5 + 50, 1100 - B5 - 50, 12, 1800, 500, -10, -17, 2.0, 0.85, 0.85, gap=0.35),
], post(PEAK["open"], comp=(-20, 2.0), room=(0.5, 0.14), out_ms=140),
    "three quickening low pulses, then a jet-engine whoosh flies through (that hollow 'phwoosh' sweep), "
    "spitting crackle, a swirl trailing behind"))


# Pair 1: membrane. A slap and a dark suck in; a pop and a bright spray out.
PAIRS[1] = dict(word="membrane", enter=spec(500, 3101, [
    noise("slap.contact", 3102, "pink", -3, 0, 70, env(0, 4, 60), [hp(300, 2), lp(2500, 2)], mode="thud"),
    sweep("suck.down", 3103, 5000, 250, 0.7, -3, 0, 480, env(20, 40, 400)),
    *wide("air.narrow", 3104, "pink", -9, 0, 300, env(5, 20, 270), 1000, 7000, spread=0.9),
    band("dark.close", 3106, "brown", -5, 40, 440, env(140, 0, 300), 60, 500),
    rumble("sub.drop", 3107, -3, 0, 480, env(30, 60, 380)),
], post(PEAK["enter"], low=12000, comp=(-18, 2.0), room=(0.3, 0.1), out_ms=90),
    "a soft slap then a dark sucking whoosh closing to the middle, a low drop under it"),
    exit=spec(500, 3111, [
        noise("pop.center", 3112, "brown", -2, 0, 90, env(0, 4, 80), [hp(60, 2), lp(600, 2)], mode="thud"),
        thump("sub.kick", 3113, -2, 0, 250, cutoff=130),
        sweep("push.L", 3114, 300, 7000, 0.7, -4, 20, 460, env(10, 60, 360), pan=-0.85),
        sweep("push.R", 3115, 300, 7000, 0.7, -4, 32, 460, env(10, 60, 360), pan=0.85),
        *wide("hiss.open", 3116, "white", -12, 60, 420, env(170, 30, 220), 4000, 12000),
        crackle("crackle.spray", 3118, -12, 30, 420, 18, 1.8, env(5, 50, 360), lo=2500),
    ], post(PEAK["exit"], low=14000, comp=(-18, 2.0), room=(0.3, 0.1), out_ms=90),
        "a pop and a bright whoosh spraying out to both sides, sparkly hiss opening up"))

# Pair 2: suction. Sound converges from the sides into a gulp; a gulp bursts out to the sides.
def converge(group, seed, start, dur, n, f0, f1, g0, g1, w0, w1):
    out = []
    step = dur / n
    for k in range(n):
        u = (k + 0.5) / n
        c = f0 * (f1 / f0) ** u
        side = -1 if k % 2 else 1
        length = step * 1.6
        t0 = max(start, start + k * step - step * 0.3)
        out.append(band(f"{group}.{k:02d}", seed + k, "pink", g0 + (g1 - g0) * u, t0, length,
                        env(length * 0.45, 0, length * 0.45), round(c / 2), round(c * 2),
                        pan=side * (w0 + (w1 - w0) * u)))
    return out


PAIRS[2] = dict(word="suction", enter=spec(480, 3201, [
    *converge("pull", 3210, 0, 380, 8, 4000, 500, -10, -3, 0.95, 0.05),
    crackle("crackle.in", 3202, -11, 40, 320, 14, 1.5, env(260, 0, 20), lo=2500),
    rumble("sub.pressure", 3203, -3, 0, 400, env(340, 0, 40)),
    thump("gulp.thump", 3204, -1, 370, 110, cutoff=200),
    band("gulp.body", 3205, "brown", -6, 370, 110, env(1, 10, 95), 60, 500),
], post(PEAK["enter"], low=12000, room=(0.3, 0.1), out_ms=60),
    "a hiss pulled in from both sides, darkening as it narrows to the middle, swallowed by a muffled gulp"),
    exit=spec(480, 3211, [
        thump("gulp.thump", 3212, -1, 0, 120, cutoff=200),
        band("gulp.body", 3213, "brown", -6, 0, 110, env(1, 10, 95), 60, 500),
        *converge("push", 3220, 50, 400, 8, 500, 4000, -3, -12, 0.05, 0.95),
        crackle("crackle.out", 3214, -11, 40, 400, 14, 1.5, env(5, 40, 340), lo=2500),
        rumble("sub.release", 3215, -4, 0, 300, env(5, 40, 250)),
    ], post(PEAK["exit"], low=14000, room=(0.3, 0.1), out_ms=90),
        "a muffled gulp bursts and a hiss rushes outward from the middle to both sides, getting brighter"))

# Pair 3: swirl. A spiral drain inward; a spiral out.
PAIRS[3] = dict(word="swirl", enter=spec(560, 3301, [
    *swirl("swirl.in", 3310, 0, 520, 14, 3000, 350, -4, -6, 2.5, 0.95, 0.1, accel=1.6, gap=0.3),
    sweep("hiss.down", 3302, 8000, 1500, 0.8, -13, 0, 360, env(20, 40, 280)),
    rumble("sub.drain", 3303, -4, 60, 480, env(250, 0, 220)),
    thump("whump.end", 3304, -5, 400, 150, cutoff=150),
], post(PEAK["enter"], low=12000, room=(0.35, 0.12), out_ms=80),
    "a whirl circling your head, spinning faster as it drains down and into the middle, ending in a soft "
    "whump"),
    exit=spec(560, 3311, [
        thump("whump.start", 3312, -5, 0, 150, cutoff=150),
        *swirl("swirl.out", 3320, 30, 520, 14, 350, 3000, -4, -9, 2.5, 0.1, 0.95, accel=0.6, gap=0.3),
        sweep("hiss.up", 3313, 1500, 9000, 0.8, -13, 150, 400, env(220, 40, 140)),
        rumble("sub.release", 3314, -4, 0, 320, env(5, 40, 270)),
    ], post(PEAK["exit"], low=14000, room=(0.35, 0.12), out_ms=140),
        "a soft whump, then a whirl spiralling out from the middle, widening, slowing and brightening"))

# Pair 4: static. Crackle swallowed into the dark; sparks thrown out.
PAIRS[4] = dict(word="static", enter=spec(450, 3401, [
    crackle("static.L", 3402, -6, 0, 330, 30, 1.5, env(5, 30, 280), pan=-0.85),
    crackle("static.R", 3407, -6, 0, 330, 30, 1.5, env(5, 30, 280), pan=0.85),
    sweep("dark.down", 3403, 4000, 200, 0.7, -4, 0, 420, env(20, 40, 340)),
    rumble("sub.rumble", 3404, -4, 0, 440, env(150, 0, 280)),
    thump("close.thump", 3405, -3, 330, 120, cutoff=160),
    band("hum.noise", 3406, "pink", -12, 0, 300, env(10, 20, 260), 1500, 5000),
], post(PEAK["enter"], low=12000, room=(0.3, 0.1), out_ms=110),
    "a fizz of crackling static that darkens and thins, shut off by a dull thump"),
    exit=spec(450, 3411, [
        thump("open.thump", 3412, -3, 0, 120, cutoff=160),
        crackle("sparks.L", 3413, -7, 10, 420, 30, 1.5, env(3, 60, 340), pan=-0.8),
        crackle("sparks.R", 3414, -7, 20, 420, 30, 1.5, env(3, 60, 340), pan=0.8),
        sweep("bright.up", 3415, 300, 7000, 0.7, -5, 0, 420, env(30, 40, 330)),
        *wide("hiss.open", 3416, "white", -14, 80, 360, env(150, 30, 180), 4000, 12000),
    ], post(PEAK["exit"], low=14000, room=(0.3, 0.1), out_ms=80),
        "a dull thump and a bright burst of crackling sparks thrown to left and right"))


# ── Open-wormhole ambient loops ──────────────────────────────────────────────
# Jon: "a bit of a liquidy hum or persistent background noise the wormhole
# makes when it is open". A recipe renders T = X + D + X + PAD ms of steady
# texture; the loop is [X, X + D) with the X ms that follow it crossfaded
# (equal power) into its head, so its last sample runs on into its first.
# Every slow motion (band drift, pan turns, throb, plugin LFO) makes a whole
# number of cycles in D, so what is joined is in phase. NOTE: the crossfade is
# this script's step (`make_loop`); the renderer has no loop mode, so shipping
# a loop needs that step too (or a renderer `loop_crossfade_ms`).
LOOP_D, LOOP_X, LOOP_PAD = 5000, 500, 100
LOOP_T = LOOP_D + 2 * LOOP_X + LOOP_PAD
LOOP_PEAK = -18.0


def steady():
    """A layer envelope that is up before the loop region and down after it."""
    return env(LOOP_X * 0.8, 0, LOOP_PAD * 0.6)


def cyc(t_ms, cycles, phase=0.0):
    """Phase (radians) of a motion with `cycles` whole cycles per loop."""
    return 2 * np.pi * cycles * (t_ms - LOOP_X) / LOOP_D + phase


def loop_swirl(group, seed, segs, fc, depth_oct, drift_cycles, turns, width, gain, gap=0.2, gain_wobble=0.0,
               color="pink", octaves=1.0):
    """Overlapping noise segments over the whole render, `segs` per loop: the
    band centre drifts +-depth_oct octaves round fc (`drift_cycles` per loop),
    the pan circles `turns` times per loop, the level wobbles +-gain_wobble dB
    once per loop; the gaps flutter at the segment rate."""
    step = LOOP_D / segs
    seg = step * (1.7 - 1.2 * gap)
    out = []
    t = LOOP_X - step * np.ceil(LOOP_X / step)
    k = 0
    while t < LOOP_T - LOOP_PAD:
        mid = t + step / 2
        c = fc * 2 ** (depth_oct * np.sin(cyc(mid, drift_cycles)))
        pan = width * np.sin(cyc(mid, turns, 0.4))
        g = gain + gain_wobble * np.sin(cyc(mid, 1, 1.9))
        t0 = max(0.0, t - (seg - step) / 2)
        length = min(seg, LOOP_T - t0)
        lo, hi = c / 2 ** octaves, c * 2 ** octaves
        out.append(band(f"{group}.{k:02d}", seed + k, color, round(float(g), 2), t0, length,
                        env(length * 0.45, 0, length * 0.45), round(lo), round(hi), pan=float(pan)))
        t += step
        k += 1
    return out


def blobs(group, rng_seed, seed0, count, f_lo, f_hi, d_lo, d_hi, g_lo, g_hi, rise=(1.3, 2.1), width=0.6,
          kind="sweep", clusters=None):
    """Burbles: short filtered-noise blobs, each a band that slides upward (a
    bubble's chirp, as noise: no fixed pitch) or a still band (`kind: band`).
    With `clusters` (n, spread_ms) they come in gurgling bunches."""
    rng = np.random.default_rng(rng_seed)
    span = LOOP_T - LOOP_PAD - d_hi
    if clusters:
        n, spread = clusters
        centres = np.sort(rng.uniform(0, span, n))
        times = np.clip(np.concatenate([c + rng.uniform(0, spread, count // n) for c in centres]), 0, span)
    else:
        times = rng.uniform(0, span, count)
    out = []
    for i, t0 in enumerate(np.sort(times)):
        c = float(np.exp(rng.uniform(np.log(f_lo), np.log(f_hi))))
        d = float(rng.uniform(d_lo, d_hi))
        g = round(float(rng.uniform(g_lo, g_hi)), 1)
        pan = float(rng.uniform(-width, width))
        e = env(d * 0.3, 0, d * 0.6)
        if kind == "sweep":
            r = float(rng.uniform(*rise))
            out.append(sweep(f"{group}.{i:03d}", seed0 + i, round(c), round(c * r), 0.9, g, round(t0, 1), round(d, 1),
                             e, effects=[lp(round(c * r * 3), 2)], pan=round(pan, 3)))
        else:
            out.append(band(f"{group}.{i:03d}", seed0 + i, "pink", g, round(t0, 1), round(d, 1), e,
                            round(c / 1.6), round(c * 1.6), pan=pan))
    return out


def loop_post(phaser_cycles=None, room=(0.6, 0.18)):
    chain = [hp(26, 3), lp(12000)]
    if phaser_cycles:
        chain.append({"effect": "phaser", "rate_hz": phaser_cycles * 1000.0 / LOOP_D, "depth": 0.5,
                      "centre_frequency_hz": 800, "feedback": 0.1, "mix": 0.3})
    chain += [{"effect": "reverb", "room_size": room[0], "damping": 0.6, "wet_level": room[1], "dry_level": 1.0,
               "width": 1.0},
              {"effect": "highpass", "cutoff_hz": 20, "order": 2},
              {"effect": "normalize_peak", "target_db": LOOP_PEAK}]
    return chain


def watery(cycles=1):
    """A slow chorus whose LFO makes whole cycles per loop: a wet wobble."""
    return {"effect": "chorus", "rate_hz": cycles * 1000.0 / LOOP_D, "depth": 0.5, "centre_delay_ms": 9.0,
            "feedback": 0.1, "mix": 0.45}


LOOPS: dict[int, dict] = {}
LOOPS[1] = dict(word="burble", **spec(LOOP_T, 4101, [
    rumble("bed.pressure", 4102, -6, 0, LOOP_T, steady(), cutoff=90),
    *loop_swirl("bed.hum", 4110, 10, 160, 0.25, 1, 0, 0.15, -6, gap=0.15, octaves=1.0),
    *blobs("bubbles.burble", 41, 4200, 70, 250, 900, 40, 110, -19, -10),
    *loop_swirl("swirl.drift", 4300, 20, 1500, 0.6, 1, 2, 0.8, -11, gap=0.2),
    crackle("crackle.faint", 4400, -18, 0, LOOP_T, 30, 1.2, steady(), lo=3000),
], loop_post(),
    "a steady wet bubbling, like a pot on a low simmer, over a soft low throb; air drifting round left-right"))

LOOPS[2] = dict(word="gurgle", **spec(LOOP_T, 4501, [
    rumble("bed.pressure", 4502, 4, 0, LOOP_T, steady(), cutoff=110),
    *loop_swirl("bed.throb", 4510, 7, 90, 0.2, 1, 0, 0.1, 5, gap=0.4, octaves=1.0, color="brown"),
    *blobs("bubbles.gurgle", 45, 4600, 36, 120, 420, 100, 260, -16, -8, rise=(1.4, 2.4), width=0.5,
           clusters=(8, 320)),
    *loop_swirl("swirl.drift", 4700, 16, 900, 0.5, 1, 1, 0.7, -6, gap=0.25),
    crackle("crackle.faint", 4800, -17, 0, LOOP_T, 22, 1.5, steady(), lo=2500),
], loop_post(phaser_cycles=1),
    "deep, slow gurgles in bunches, like a drain or a swamp, over a low pulsing pressure"))

LOOPS[3] = dict(word="drift", **spec(LOOP_T, 4901, [
    rumble("bed.pressure", 4902, -5, 0, LOOP_T, steady(), cutoff=100),
    band("wind.flow", 4903, "pink", -8, 0, LOOP_T, steady(), 200, 2500, extra=[watery(2)]),
    *loop_swirl("swirl.tunnel", 4910, 24, 1200, 0.8, 2, 3, 0.85, -10, gap=0.25),
    *blobs("bubbles.sparse", 49, 5000, 25, 200, 700, 60, 160, -14, -7, kind="band"),
    crackle("crackle.faint", 5100, -24, 0, LOOP_T, 30, 1.2, steady(), lo=3000),
], loop_post(phaser_cycles=1),
    "a watery wind circling slowly through a tunnel, an occasional blub, a faint low pressure"))

LOOPS[4] = dict(word="simmer", **spec(LOOP_T, 5201, [
    band("bed.hum", 5202, "brown", 14, 0, LOOP_T, steady(), 40, 250, extra=[watery(1)]),
    *blobs("fizz.small", 52, 5300, 160, 600, 2200, 15, 40, -20, -12, rise=(1.2, 1.8), width=0.8),
    *blobs("bubbles.big", 53, 5600, 14, 200, 500, 80, 180, -15, -9, width=0.4),
    *loop_swirl("swirl.drift", 5700, 16, 2000, 0.5, 1, 2, 0.8, -12, gap=0.2),
    crackle("crackle.faint", 5800, -20, 0, LOOP_T, 40, 1.0, steady(), lo=3500),
], loop_post(),
    "a fine fizzing simmer with the odd big bubble rising through it, a warm low hum underneath"))


def make_loop(a):
    """The rendered [X, X + D) with the X ms after it crossfaded into its head
    (equal power: two uncorrelated textures keep their level)."""
    x, d = int(LOOP_X * SR / 1000), int(LOOP_D * SR / 1000)
    out = a[x: x + d].copy()
    u = np.linspace(0, 1, x, endpoint=False)[:, None]
    out[:x] = a[x: 2 * x] * np.sin(0.5 * np.pi * u) + a[x + d: 2 * x + d] * np.cos(0.5 * np.pi * u)
    return (out * (10 ** (LOOP_PEAK / 20) / np.abs(out).max())).astype(np.float32)


def seam(loop):
    """The level step across the wrap (last 200 ms -> first 200 ms) against
    the 90th percentile of the steps between neighbouring 200 ms windows
    inside the loop, and the sample jump at the wrap over the loop's 99th
    percentile sample step (<= 1: no click)."""
    w = SR // 5
    k = len(loop) // w
    rms = 10 * np.log10((loop[: k * w].mean(axis=1).reshape(k, w) ** 2).mean(axis=1) + 1e-15)
    inner = np.percentile(np.abs(np.diff(rms)), 90)
    step = abs(rms[0] - rms[-1])
    jump = np.abs(loop[0] - loop[-1]).max()
    typical = np.percentile(np.abs(np.diff(loop, axis=0)), 99)
    return round(float(step), 1), round(float(inner), 1), round(float(jump / typical), 2)


# ── Names ────────────────────────────────────────────────────────────────────
def cands():
    """(file stem, slot, spec) for every candidate."""
    out = [(f"open_{n}_{o['word']}", "open", o) for n, o in OPENS.items()]
    for n, p in PAIRS.items():
        out += [(f"pair_{n}_enter_{p['word']}", "enter", p["enter"]), (f"pair_{n}_exit_{p['word']}", "exit", p["exit"])]
    out += [(f"loop_{n}_{lo['word']}", "loop", lo) for n, lo in LOOPS.items()]
    return out


def plain(x):
    """Numpy scalars to Python ones (for YAML), rounded."""
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4)
    return x


def recipe(stem, s, layers=None, postprocess=None):
    return plain({"schema": "ambition.sfxir.v1", "id": f"audition.wormhole_v2.{stem}", "duration_ms": s["duration_ms"],
            "render": {"sample_rate": SR, "channels": 2, "seed": s["seed"], "buffer_size": 128,
                       "safety_profile": "none"},
            "layers": copy.deepcopy(layers if layers is not None else s["layers"]),
            "postprocess": copy.deepcopy(postprocess if postprocess is not None else s["postprocess"]),
            "notes": f"Wormhole round 2 candidate {stem} (noise only): {s['hear']}. From "
                     "tools/ambition_sfx_renderer/scripts/wormhole_v2_candidates.py."
                     + (f" A LOOP: the wav is [{LOOP_X}, {LOOP_X + LOOP_D}) ms of this render with the "
                        f"{LOOP_X} ms after it crossfaded into its head (make_loop in the script)."
                        if stem.startswith("loop_") else "")})


# ── Measuring ────────────────────────────────────────────────────────────────
def tonal_peak_db(a):
    """How far the strongest narrow spectral peak (80 Hz-12 kHz) stands above
    the spectrum around it (a running median over ~1/3 octave), from a Welch
    average. Noise through wide filters: a few dB. A tone: 20 dB and more."""
    mono = a.mean(axis=1)
    f, p = signal.welch(mono, SR, nperseg=8192, noverlap=6144)
    db = 10 * np.log10(p + 1e-20)
    out = 0.0
    keep = np.where((f >= 80) & (f <= 12000))[0]
    for i in keep:
        half = max(3, int(f[i] * 0.12 / (f[1] - f[0])))
        local = np.median(db[max(0, i - half): i + half + 1])
        out = max(out, db[i] - local)
    return float(out)


def loud100(a):
    mono = a.mean(axis=1)
    w = SR // 10
    if len(mono) <= w:
        return float(10 * np.log10((mono ** 2).mean() + 1e-15))
    c = np.concatenate([[0.0], np.cumsum(mono.astype(np.float64) ** 2)])
    return float(10 * np.log10(((c[w:] - c[:-w]) / w).max() + 1e-15))


def measure(a):
    e = np.abs(a).max(axis=1)
    w = SR // 200
    k = len(e) // w
    db = 20 * np.log10(e[: k * w].reshape(k, w).max(axis=1) + 1e-9)
    return dict(ms=round(1000 * len(a) / SR), peak_db=round(float(20 * np.log10(np.abs(a).max())), 1),
                clipped=int((np.abs(a) >= 0.999).sum()), dc=round(float(np.abs(a.mean(axis=0)).max()), 5),
                first_ms=int(np.argmax(db > -60)) * 5, end_db=round(float(db[-4:].max() - db.max()), 1),
                tonal_db=round(tonal_peak_db(a), 1),
                width=round(float(np.sqrt(((a[:, 0] - a[:, 1]) ** 2).mean() / ((a[:, 0] + a[:, 1]) ** 2).mean())), 2))


def thirds(a):
    """Spectral centroid (Hz) and width (side/mid RMS) of the first and the
    last third of the cue, over the part within 30 dB of its peak."""
    def cen(z):
        m = z.mean(axis=1)
        sp = np.abs(np.fft.rfft(m)) ** 2
        f = np.fft.rfftfreq(len(m), 1 / SR)
        return float((sp * f).sum() / (sp.sum() or 1))

    def wid(z):
        return float(np.sqrt(((z[:, 0] - z[:, 1]) ** 2).mean() / (((z[:, 0] + z[:, 1]) ** 2).mean() + 1e-15)))
    e = np.abs(a).max(axis=1)
    on = np.where(e > e.max() * 10 ** (-30 / 20))[0]
    z = a[on[0]: on[-1] + 1]
    n = len(z) // 3
    return round(cen(z[:n])), round(cen(z[-n:])), round(wid(z[:n]), 2), round(wid(z[-n:]), 2)


def rms_db(a):
    return float(10 * np.log10((a.mean(axis=1) ** 2).mean() + 1e-15))


def group_jobs(stem, s, gdir):
    """One recipe per layer group with the final normalize swapped for 0 dB,
    plus the whole the same way: the groups' levels against the whole."""
    chain = [fx if fx["effect"] != "normalize_peak" else {"effect": "gain", "gain_db": 0.0} for fx in s["postprocess"]]
    groups = sorted({l["name"].split(".")[0] for l in s["layers"]})
    jobs = {"__all__": [l for l in s["layers"]]}
    for g in groups:
        jobs[g] = [l for l in s["layers"] if l["name"].split(".")[0] == g]
    paths = {}
    for g, layers in jobs.items():
        r = recipe(f"{stem}__{g}", s, layers, chain)
        r["id"] = f"audition.wormhole_v2.groups.{stem}.{g}"
        path = gdir / f"{stem}__{g}.sfx.yaml"
        path.write_text(yaml.safe_dump(r, sort_keys=False))
        paths[g] = (path, r["id"])
    return paths


# ── Reels ────────────────────────────────────────────────────────────────────
def clicks(n):
    """N dry clicks 30 ms apart, then 0.6 s of silence: candidate N follows."""
    rng = np.random.default_rng(7)
    click = rng.standard_normal(int(0.0015 * SR)) * np.exp(-np.arange(int(0.0015 * SR)) / (0.0003 * SR))
    click = signal.sosfilt(signal.butter(2, 1500 / (SR / 2), "highpass", output="sos"), click)
    click = 0.3 * click / np.abs(click).max()
    gap = int(0.030 * SR)
    out = np.zeros((gap * (n - 1) + len(click) + int(0.6 * SR), 2), np.float32)
    for i in range(n):
        out[i * gap: i * gap + len(click)] += click[:, None]
    return out


def silence(s):
    return np.zeros((int(s * SR), 2), np.float32)


def build_reel(entries):
    """entries: (n, label, [audio or seconds of silence...]). Returns audio and lines."""
    parts, lines, t = [], [], 0.0
    for n, label, items in entries:
        c = clicks(n)
        lines.append(f"| {t:6.2f} s | {n} | {t + len(c) / SR:6.2f} s | {label} |")
        parts.append(c)
        t += len(c) / SR
        for x in items:
            a = silence(x) if isinstance(x, (int, float)) else x
            parts.append(a)
            t += len(a) / SR
        parts.append(silence(1.2))
        t += 1.2
    return np.concatenate(parts), lines


# The picks (README only).
PICK = {"open": 1, "pair": 3, "loop": 1}
PICK_WHY = {
    "open": "It is the plainest 'wormhole whoosh': all three bands (roar, rush, hiss) rise together, bloom into a "
            "flanged whoosh over a boom, and leave a swirl and debris, which hands over to a loop. Six audible "
            "layer groups and the least tonal of the opens. Runners-up: 4 (tear: the richest, eight groups, "
            "more violent) and 2 (suction: the most dramatic shape, inhale - silence - burst).",
    "pair": "The swirl is the same motion as the open's tail and the loop, so the trip sounds of one piece: "
            "enter spirals in and darkens, exit spirals out, widens and brightens. Pair 1 (membrane) is the "
            "more physical alternative.",
    "loop": "The most literal 'liquidy hum': a low throbbing bed and a steady simmer of burbles at equal "
            "weight, a slow left-right drift and faint crackle; the open's swirling tail settles into it. "
            "Runner-up: 4 (simmer: finer fizz, lighter, maybe less tiring); 2 (gurgle) if the gate should "
            "sound heavier.",
}


def trip(open_a, loop, enter, exit_, settle=0.55, fade=0.4, before=3.0, after=1.5):
    """The open; the loop fades in under the open's tail (from `settle` of its
    length) and runs on `before` s past its end; enter, 0.6 s, exit over it;
    the loop runs `after` s more and fades out."""
    t_in = int(settle * len(open_a))
    t_enter = len(open_a) + int(before * SR)
    t_exit = t_enter + len(enter) + int(0.6 * SR)
    total = t_exit + len(exit_) + int(after * SR)
    bed = np.tile(loop, (int(np.ceil((total - t_in) / len(loop))) + 1, 1))[: total - t_in].copy()
    f = int(fade * SR)
    bed[:f] *= np.linspace(0, 1, f)[:, None]
    bed[-f:] *= np.linspace(1, 0, f)[:, None]
    out = np.zeros((total, 2), np.float32)
    out[: len(open_a)] += open_a
    out[t_in:] += bed
    out[t_enter: t_enter + len(enter)] += enter
    out[t_exit: t_exit + len(exit_)] += exit_
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--only", default="", help="render and measure only stems containing this (no reels/README)")
    args = p.parse_args(argv)
    if args.only:
        global cands
        every = cands
        keys = args.only.split(",")
        cands = lambda: [c for c in every() if any(k in c[0] for k in keys)]  # noqa: E731
    here = args.outdir
    for d in ("", "png", "_render", "_groups"):
        (here / d).mkdir(parents=True, exist_ok=True)

    todo, ids = [], {}
    for stem, slot, s in cands():
        path = here / f"{stem}.sfx.yaml"
        path.write_text(yaml.safe_dump(recipe(stem, s), sort_keys=False))
        todo.append((path, here / "_render"))
        ids[stem] = f"audition.wormhole_v2.{stem}"
    gpaths = {stem: group_jobs(stem, s, here / "_groups") for stem, slot, s in cands()}
    for stem, gs in gpaths.items():
        todo += [(path, here / "_groups" / "out") for path, _ in gs.values()]
    with mp.get_context("fork").Pool(JOBS) as pool:
        list(pool.imap_unordered(_render, [(str(a), str(b)) for a, b in todo]))

    got = {stem: load(here / "_render" / ids[stem]) for stem, _, _ in cands()}
    rows, bad = {}, []
    for stem, slot, s in cands():
        a = make_loop(got[stem]) if slot == "loop" else got[stem]
        got[stem] = a
        sf.write(here / f"{stem}.wav", a, SR)
        spectrogram_png(here / "png" / f"{stem}.png", a)
        m = measure(a)
        level = rms_db if slot == "loop" else loud100
        whole = level(load(here / "_groups" / "out" / gpaths[stem]["__all__"][1]))
        levels = {g: round(level(load(here / "_groups" / "out" / cid)) - whole, 1)
                  for g, (_, cid) in gpaths[stem].items() if g != "__all__"}
        if slot in ("enter", "exit"):
            m["c0"], m["c1"], m["w0"], m["w1"] = thirds(a)
        m["groups"] = levels
        m["audible"] = sum(v > -20 for v in levels.values())
        rows[stem] = m
        if slot == "loop":
            m["seam_db"], m["seam_inner"], m["seam_jump"] = seam(a)
            if m["clipped"] or m["dc"] > 0.002 or abs(m["peak_db"] - LOOP_PEAK) > 0.3 or m["tonal_db"] > 10 \
                    or m["audible"] < 4 or m["seam_db"] > m["seam_inner"] or m["seam_jump"] > 1.0:
                bad.append(f"- `{stem}`: {m}")
        elif m["clipped"] or m["dc"] > 0.002 or m["first_ms"] > (40 if slot == "open" else 10) \
                or m["end_db"] > -30 or abs(m["peak_db"] - PEAK[slot]) > 0.3 or m["tonal_db"] > 10 \
                or m["audible"] < 4 \
                or (slot == "enter" and not (m["c1"] < m["c0"] and m["w1"] < m["w0"])) \
                or (slot == "exit" and not (m["c1"] > m["c0"] and m["w1"] > m["w0"])):
            bad.append(f"- `{stem}`: {m}")

    if args.only:
        for stem, m in rows.items():
            print(stem, m)
        print("\n".join(bad or ["checks ok"]))
        return

    # Listen files.
    for n, o in OPENS.items():
        sf.write(here / f"listen_open_{n}.wav", np.concatenate([got[f"open_{n}_{o['word']}"], silence(0.3)]), SR)
    for n, pr in PAIRS.items():
        w = pr["word"]
        sf.write(here / f"listen_pair_{n}.wav", np.concatenate(
            [got[f"pair_{n}_enter_{w}"], silence(0.5), got[f"pair_{n}_exit_{w}"], silence(0.3)]), SR)

    for n, lo in LOOPS.items():
        loop = got[f"loop_{n}_{lo['word']}"]
        sf.write(here / f"listen_loop_{n}.wav", np.concatenate([loop, loop, loop]), SR)

    # Reels.
    po = OPENS[PICK["open"]]
    open_pick = got[f"open_{PICK['open']}_{po['word']}"]
    pp = PAIRS[PICK["pair"]]
    pick_enter, pick_exit = got[f"pair_{PICK['pair']}_enter_{pp['word']}"], got[f"pair_{PICK['pair']}_exit_{pp['word']}"]
    reels = {
        "reel_open": [(n, f"open_{n}_{o['word']}", [got[f"open_{n}_{o['word']}"]]) for n, o in OPENS.items()],
        "reel_pairs": [(n, f"pair_{n} {pr['word']}: enter, 0.5 s, exit",
                        [got[f"pair_{n}_enter_{pr['word']}"], 0.5, got[f"pair_{n}_exit_{pr['word']}"]])
                       for n, pr in PAIRS.items()],
        "reel_loop": [(n, f"loop_{n}_{lo['word']}, twice round", [got[f"loop_{n}_{lo['word']}"]] * 2)
                      for n, lo in LOOPS.items()],
        "reel_trip": [(n, f"open_{PICK['open']}, loop_{n} {lo['word']} rising under its tail, 3 s of loop, "
                          f"pair_{PICK['pair']} enter, 0.6 s, exit, over the loop",
                       [trip(open_pick, got[f"loop_{n}_{lo['word']}"], pick_enter, pick_exit)])
                      for n, lo in LOOPS.items()],
    }
    built = {k: build_reel(v) for k, v in reels.items()}
    gain = min(1.0, 0.89 / max(float(np.abs(a).max()) for a, _ in built.values()))
    for k, (a, _) in built.items():
        sf.write(here / f"{k}.wav", a * gain, SR)

    # README.
    L = ["# Wormhole SFX, round 2: noise only", "",
         "Click a `listen_*` file to hear one candidate (a pair plays enter, 0.5 s, exit).",
         "No tones anywhere: layered noise bands, sub rumble, crackle and swirl.", "",
         "## Picks (from measurements and spectrograms; I could not listen)", "",
         f"- open: `listen_open_{PICK['open']}.wav` ({OPENS[PICK['open']]['word']}). {PICK_WHY['open']}",
         f"- enter/exit: `listen_pair_{PICK['pair']}.wav` ({PAIRS[PICK['pair']]['word']}). {PICK_WHY['pair']}",
         f"- open loop: `listen_loop_{PICK['loop']}.wav` ({LOOPS[PICK['loop']]['word']}). {PICK_WHY['loop']}", "",
         "## Opens", "", "| file | what you'd hear | ms | peak dB |", "|---|---|---:|---:|"]
    for n, o in OPENS.items():
        stem = f"open_{n}_{o['word']}"
        L.append(f"| `listen_open_{n}.wav` | {o['hear']} | {rows[stem]['ms']} | {rows[stem]['peak_db']} |")
    L += ["", "## Enter / exit pairs", "", "| file | what you'd hear | ms | peak dB |", "|---|---|---:|---:|"]
    for n, pr in PAIRS.items():
        for side in ("enter", "exit"):
            stem = f"pair_{n}_{side}_{pr['word']}"
            L.append(f"| `{stem}.wav` | {pr[side]['hear']} | {rows[stem]['ms']} | {rows[stem]['peak_db']} |")
    L += ["", "## Open-wormhole loops (each loop 5 s; the listen file plays it 3 times, to hear the seam)", "",
          "| file | what you'd hear | ms | peak dB |", "|---|---|---:|---:|"]
    for n, lo in LOOPS.items():
        stem = f"loop_{n}_{lo['word']}"
        L.append(f"| `listen_loop_{n}.wav` | {lo['hear']} | {rows[stem]['ms']} | {rows[stem]['peak_db']} |")
    L += ["", "## Reels", "",
          "Before candidate N: N short clicks 30 ms apart, then 0.6 s of silence. The times below are where "
          "the clicks start and where the sound starts.", ""]
    for k, (_, lines) in built.items():
        L += [f"`{k}.wav`", "", "| clicks at | N | sound at | what |", "|---:|---:|---:|---|"] + lines + [""]
    L += ["## Checks (measured)", "",
          "Tonal peak = how far the strongest narrow spectral peak stands above its surroundings (a tone is 20 dB+; "
          "round 1's open_a measured TONAL_V1 dB). Audible groups = layer groups within 20 dB of the whole "
          "(loudest 100 ms). Width = side/mid RMS.", "",
          "Loops: group levels are whole-loop RMS; seam = the level step across the wrap (200 ms windows) vs the "
          "90th percentile of the loop's own window steps, and the sample jump at the wrap over the loop's "
          "99th-percentile sample step (<= 1: no click). Pairs: centroid Hz and width, first third -> last third "
          "(enter should darken and narrow, exit brighten and widen).", "",
          "| file | tonal peak dB | audible groups | groups (dB re whole) | motion / seam | clipped | DC | end dB |",
          "|---|---:|---:|---|---|---:|---:|---:|"]
    for stem, m in rows.items():
        gs = ", ".join(f"{g} {v}" for g, v in sorted(m["groups"].items(), key=lambda kv: -kv[1]))
        if "seam_db" in m:
            mo = f"seam step {m['seam_db']} dB (inner p90 {m['seam_inner']}), jump {m['seam_jump']}"
        elif "c0" in m:
            mo = f"{m['c0']} -> {m['c1']} Hz, width {m['w0']} -> {m['w1']}"
        else:
            mo = f"width {m['width']}"
        end = "loop" if "seam_db" in m else m["end_db"]
        L.append(f"| `{stem}` | {m['tonal_db']} | {m['audible']} | {gs} | {mo} | {m['clipped']} | {m['dc']} | {end} |")
    v1 = SUPER / "untracked" / "sfx-candidates" / "wormhole" / "open_a_triad_kawoosh.wav"
    tv1 = round(tonal_peak_db(sf.read(v1, always_2d=True)[0]), 1) if v1.exists() else "n/a"
    L += [""] + (bad or ["All pass: no clipping, |DC| <= 0.002, sound within 10 ms (opens: 40 ms, they build from silence), tail >= 30 dB down (not loops), "
                         "peak on target, tonal peak <= 10 dB, >= 4 audible groups."])
    (here / "README.md").write_text("\n".join(L).replace("TONAL_V1", str(tv1)) + "\n")
    print("\n".join(bad or ["checks ok"]))
    for stem, m in rows.items():
        print(f"{stem:30s} tonal {m['tonal_db']:5} aud {m['audible']} width {m['width']} end {m['end_db']} "
              f"seam {m.get('seam_db')}/{m.get('seam_inner')} {m.get('seam_jump')} "
              f"motion {m.get('c0')}->{m.get('c1')} {m.get('w0')}->{m.get('w1')} "
              f"{sorted(m['groups'].items(), key=lambda kv: -kv[1])}")


if __name__ == "__main__":
    main()
