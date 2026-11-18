"""Smash feel P2 (2026-10-29): hit tiers and swings, candidates, measurements, reels.

155 of ~200 authored hit volumes play one generic `player.hit` (a 120 ms
brown-noise thud). Nothing scales with the strength of the hit, and no move
has a swing sound. Ultimate has weak/mid/strong/kill impact families and a
whoosh per move class; the kill-hit sound is a feel pillar. This writes the
twelve cues of that family, from descriptions (synthesis only):

Impacts, blunt (the cast's generic hit) and slash (bladed fighters), four
tiers each. The tiers are ORDERED: light < mid < heavy < kill in peak, in
loudness and in length, so a player hears the strength of a hit.

- `hit.impact.light`  ~70 ms   a short dry snap or slap, little low end (a jab)
- `hit.impact.mid`    ~110 ms  a punchy thud: low body under a click (a tilt)
- `hit.impact.heavy`  ~180 ms  a deep thump plus a crack, a touch of sub (a smash)
- `hit.impact.kill`   ~290 ms  heavy plus a sub boom, a bright crack, a noise tail
- `hit.slash.*`       the same four, with a cutting metallic edge instead of a thud

Swings (the whoosh when a move starts, before contact):

- `swing.light`           ~90 ms   an airy whoosh (jab, tilt)
- `swing.heavy`           ~180 ms  a deeper, slower whoosh with weight (a smash)
- `swing.aerial`          ~120 ms  a rising whoosh (an aerial)
- `swing.charge_release`  ~120 ms  a charged smash letting go: a snap, then whoosh

Every cue has two candidates (`a_*`, `b_*`). A chosen one ships as
`sounds/active/<id>.sfx.yaml` with the SAME layers and seed (`ship` below),
so the shipped render equals the candidate sample for sample.

Run with the renderer's interpreter, from anywhere:

    python scripts/smash_feel_candidates.py            # render, measure, reels, INDEX.md
    python scripts/smash_feel_candidates.py measure    # only the table of the cues in use
    python scripts/smash_feel_candidates.py ship       # write the chosen recipes into sounds/active/

Writes `<superproject>/untracked/sfx-candidates/smash-feel/` (git-ignored;
`--outdir` elsewhere): `recipes/`, `out/`, `solo/<cue>__<candidate>.wav`,
`reel/*.wav` with a `.txt` index of offsets next to each, and `INDEX.md`.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

RENDERER = Path(__file__).resolve().parents[1]
SHIPPED = RENDERER / "output"
ACTIVE = RENDERER / "sounds" / "active"
SUPER = RENDERER.parents[1]
DEFAULT_OUT = SUPER / "untracked" / "sfx-candidates" / "smash-feel"
PY = sys.executable
SR = 48000
JOBS = 6  # a shared box: never every CPU
GAP_S = 0.4  # silence between the cues of a reel

# The cues in use that the new ones replace or sit beside.
IN_USE = {
    "player_hit": "player.hit",
    "sword_smash": "fighter.sword.smash_forward",
    "brawler_jab": "fighter.brawler.jab",
}

# Each family in tier order. The ORDER is the contract the measurements check.
TIERS = ("light", "mid", "heavy", "kill")
CUES = {
    "hit.impact": [f"hit.impact.{t}" for t in TIERS],
    "hit.slash": [f"hit.slash.{t}" for t in TIERS],
    "swing": ["swing.light", "swing.heavy", "swing.aerial", "swing.charge_release"],
}
ALL_CUES = [c for family in CUES.values() for c in family]

# Peak ceiling per tier (dBFS). The bank's impacts sit at -8 to -11 dB peak;
# the robot's blade impacts reach -5 to -6. A kill blow may be the loudest
# thing in the bank, a jab is quiet.
PEAK = {"light": -11.0, "mid": -9.0, "heavy": -7.5, "kill": -6.0}

# The candidate each cue ships (the `ship` command), chosen by the
# measurements in INDEX.md and by ear against the neighbours.
CHOSEN = {
    "hit.impact.light": "a_snap",
    "hit.impact.mid": "a_thud",
    "hit.impact.heavy": "a_thump",
    "hit.impact.kill": "a_boom",
    "hit.slash.light": "a_nick",
    "hit.slash.mid": "a_bite",
    "hit.slash.heavy": "a_cleave",
    "hit.slash.kill": "a_ring",
    "swing.light": "a_airy",
    "swing.heavy": "a_deep",
    "swing.aerial": "a_rise",
    "swing.charge_release": "a_snap",
}


# ── Recipe pieces ────────────────────────────────────────────────────────────
def env(a, h, r):
    return {"attack_ms": a, "hold_ms": h, "release_ms": r}


def bp(hz, q):
    return {"effect": "bandpass", "center_hz": hz, "q": q}


def lp(hz, order=2):
    return {"effect": "lowpass", "cutoff_hz": hz, "order": order}


def hp(hz, order=2):
    return {"effect": "highpass", "cutoff_hz": hz, "order": order}


def decay(tau, points=14, floor=0.0):
    """An exponential fall exp(-u / tau) across the layer (u from 0 to 1), as
    straight-line points: the amplitude of a struck body."""
    us = np.concatenate([[0.0], np.geomspace(0.004, 1.0, points - 1)])
    return [[round(float(u), 4), round(float(max(floor, np.exp(-u / tau))), 4)] for u in us]


def noise(name, seed, color, gain, start, dur, e, effects, pan=0.0):
    return {"name": name, "kind": "noise_burst", "mode": "burst", "seed": seed, "color": color,
            "gain_db": gain, "start_ms": start, "duration_ms": dur, "pan": pan, "envelope": e,
            "effects": effects}


def thud(name, seed, color, gain, start, dur, effects, pan=0.0):
    """Noise under exp(-t / (0.42 dur)) with two clicks at its front."""
    return {"name": name, "kind": "noise_burst", "mode": "thud", "seed": seed, "color": color,
            "gain_db": gain, "start_ms": start, "duration_ms": dur, "pan": pan,
            "envelope": env(0, 0, min(30, dur * 0.2)), "effects": effects}


def grains(name, seed, hz, q, gain, start, dur, count, grain_ms, spread, color="white", pan=0.0, e=None):
    return {"name": name, "kind": "noise_burst", "mode": "grains", "seed": seed, "color": color,
            "grain_count": count, "grain_decay_ms": grain_ms, "spread_ms": spread, "grain_start_ms": 0,
            "gain_db": gain, "start_ms": start, "duration_ms": dur, "pan": pan,
            "envelope": e or env(0, dur * 0.3, dur * 0.7), "effects": [bp(hz, q)]}


def sweep(name, seed, a, b, q, gain, start, dur, e, effects=None, pan=0.0, color="white"):
    """A moving band of noise from `a` Hz to `b` Hz: air past a limb or a blade."""
    return {"name": name, "kind": "noise_burst", "mode": "air_sweep", "seed": seed, "color": color,
            "start_hz": a, "end_hz": b, "q": q, "gain_db": gain, "start_ms": start, "duration_ms": dur,
            "pan": pan, "envelope": e, "effects": effects or []}


def mode(name, seed, m, gain, start, dur, e, effects=None, pan=0.0):
    return {"name": name, "kind": "noise_burst", "mode": m, "seed": seed, "gain_db": gain,
            "start_ms": start, "duration_ms": dur, "pan": pan, "envelope": e, "effects": effects or []}


def tone(name, wave, freq, gain, start, dur, amp, e=None, harmonics=None, effects=None, seed=None,
         noise_amount=0.0, spread_cents=0.0, pan=0.0):
    """`freq` is [[u, hz], ...]; `amp` is [[u, gain], ...]."""
    layer = {"name": name, "kind": "synth_tone", "waveform": wave,
             "frequency_hz": {"points": freq, "curve": "exp"}, "amplitude_points": amp,
             "gain_db": gain, "start_ms": start, "duration_ms": dur, "pan": pan,
             "envelope": e or env(0.6, 0, 6)}
    if harmonics:
        layer["harmonics"] = harmonics
    if effects:
        layer["effects"] = effects
    if seed is not None:
        layer["seed"] = seed
    if noise_amount:
        layer["noise_amount"] = noise_amount
    if spread_cents:
        layer["stereo_spread_cents"] = spread_cents
    return layer


def crack(seed, gain, start=0.0, band=(1800, 9000), ms=6.0, pan=0.0, name="crack"):
    """A broadband snap: white noise, near no attack, gone in a few ms."""
    centre = round(float(np.sqrt(band[0] * band[1])))
    return noise(name, seed, "white", gain, start, ms + 8, env(0, ms * 0.25, ms),
                 [hp(band[0], 3), lp(band[1], 2), bp(centre, 0.35)], pan=pan)


def thump(name, f0, f1, sweep_frac, gain, start, dur, tau, drive_db=0.0, harm=-9.0):
    """A pitched body blow: a sine falling from f0 to f1, struck."""
    freq = [[0, f0], [round(sweep_frac * 0.35, 3), round((f0 * f1) ** 0.5, 1)], [sweep_frac, f1],
            [1, round(f1 * 0.92, 1)]]
    effects = [{"effect": "distortion", "drive_db": drive_db}, lp(900)] if drive_db else None
    return tone(name, "sine", freq, gain, start, dur, decay(tau), harmonics=[
        {"ratio": 1.0, "gain_db": 0}, {"ratio": 2.0, "gain_db": harm}, {"ratio": 3.0, "gain_db": harm - 8}],
        effects=effects)


def sub(name, f0, f1, gain, start, dur, tau, drive_db=0.0):
    """The sub boom under a kill blow: a sine below 60 Hz, a little drive so a
    phone speaker still hears its harmonics."""
    freq = [[0, f0], [0.3, round((f0 * f1) ** 0.5, 1)], [1, f1]]
    effects = [{"effect": "distortion", "drive_db": drive_db}, lp(260)] if drive_db else [lp(160)]
    return tone(name, "sine", freq, gain, start, dur, decay(tau), effects=effects)


def ring(name, partials, gain, start, dur, tau, seed=None, spread=2):
    """An inharmonic ring: struck steel, decaying (partials = [(hz, dB), ...])."""
    base = partials[0][0]
    return tone(name, "sine", [[0, base], [1, round(base * 0.995, 2)]], gain, start, dur, decay(tau),
                harmonics=[{"ratio": round(hz / base, 4), "gain_db": db} for hz, db in partials],
                spread_cents=spread, seed=seed)


def post(peak, low=14000, high=28, comp=None, room=None, clip=None, limit=None, out_ms=12):
    chain = [hp(high, 3), lp(low)]
    if comp:
        thr, ratio = comp
        chain.append({"effect": "compressor", "threshold_db": thr, "ratio": ratio, "attack_ms": 1.0,
                      "release_ms": 70})
    if room:
        size, wet = room
        chain.append({"effect": "reverb", "room_size": size, "damping": 0.6, "wet_level": wet,
                      "dry_level": 1.0, "width": 1.0})
    if clip:
        chain.append({"effect": "soft_clip", "drive": clip, "mix": 0.5})
    if limit is not None:
        # Takes `limit` dB off the top (the crack's first ms), so the body is
        # louder under one peak: at 0 dB peak first, so the amount is exact.
        chain += [{"effect": "normalize_peak", "target_db": 0.0},
                  {"effect": "limiter", "threshold_db": limit, "release_ms": 40}]
    chain += [{"effect": "fade_edges", "in_ms": 0.1, "out_ms": out_ms},
              {"effect": "normalize_peak", "target_db": peak}]
    return chain


# ── The candidates: (cue, id) -> spec ────────────────────────────────────────
C: dict[tuple[str, str], dict] = {}


def cand(cue, cid, dur, seed, layers, postprocess, note):
    C[(cue, cid)] = dict(duration_ms=dur, seed=seed, layers=layers, postprocess=postprocess, note=note)


# Blunt impacts ─────────────────────────────────────────────────────────────
cand("hit.impact.light", "a_snap", 70, 5101, [
    crack(5102, -4, band=(1400, 7000), ms=5),
    noise("slap", 5103, "pink", -12, 1, 34, env(0, 3, 26), [bp(950, 0.9), lp(3400)]),
], post(PEAK["light"], low=8000, high=120, clip=0.25, out_ms=8),
    "a dry snap: a 5 ms crack and a short pink slap around 950 Hz, no low end")

cand("hit.impact.light", "b_slap", 75, 5111, [
    crack(5112, -8, band=(900, 5000), ms=6),
    noise("slap", 5113, "pink", -9, 1, 44, env(0, 4, 34), [bp(620, 0.8), lp(2600)]),
    grains("knuckle", 5114, 2200, 1.1, -14, 0, 14, 2, 2.0, 4),
], post(PEAK["light"], low=7000, high=110, clip=0.3, out_ms=8),
    "a flat-hand slap: the pink body sits lower (620 Hz) and longer, the crack is duller")

cand("hit.impact.mid", "a_thud", 110, 5201, [
    grains("click", 5202, 2600, 1.0, 0, 0, 12, 3, 2.2, 3),
    crack(5203, -1, band=(1200, 6000), ms=5),
    thud("smack", 5205, "pink", -5, 1, 60, [bp(1100, 0.6), lp(3200)]),
    thud("body", 5204, "brown", -3, 1, 88, [lp(640)]),
    thump("thump", 150, 70, 0.55, -14, 1, 80, 0.33),
], post(PEAK["mid"], low=8000, high=45, clip=0.35, out_ms=10),
    "a punchy thud: a click, then brown-noise body under 640 Hz with a small 150 to 70 Hz thump")

cand("hit.impact.mid", "b_punch", 110, 5211, [
    grains("click", 5212, 3000, 1.0, 0, 0, 10, 2, 2.0, 2.5),
    crack(5213, 0, band=(1400, 7000), ms=5),
    thud("smack", 5215, "pink", -4, 1, 56, [bp(1300, 0.6), lp(3400)]),
    thud("body", 5214, "brown", -7, 1, 72, [lp(560)]),
    thump("thump", 175, 62, 0.5, -9, 0.5, 92, 0.3, drive_db=6),
], post(PEAK["mid"], low=8000, high=45, clip=0.4, out_ms=10),
    "a punch: the pitched 175 to 62 Hz thump leads, driven a little; the noise body is under it")

cand("hit.impact.heavy", "a_thump", 180, 5301, [
    crack(5302, 0, band=(1500, 9000), ms=7),
    grains("click", 5303, 2400, 0.9, -4, 0, 12, 3, 2.4, 3),
    thud("smack", 5305, "pink", 0, 1, 80, [bp(900, 0.6), lp(3000)]),
    thud("body", 5304, "brown", -8, 1, 130, [lp(420)]),
    thump("thump", 125, 48, 0.5, -9, 1, 160, 0.3, drive_db=4),
    sub("sub", 52, 40, -18, 2, 120, 0.35),
], post(PEAK["heavy"], low=9000, high=32, comp=(-14, 2.2), clip=0.4, limit=-2.5, out_ms=14),
    "a deep thump plus a crack: 125 to 48 Hz body, brown thud under 420 Hz, a touch of 52 Hz sub")

cand("hit.impact.heavy", "b_crack", 185, 5311, [
    crack(5312, 0, band=(1800, 11000), ms=9),
    grains("splinter", 5313, 1900, 0.8, -1, 2, 26, 6, 3.0, 14),
    thud("smack", 5315, "pink", -1, 1, 90, [bp(800, 0.6), lp(2800)]),
    thud("body", 5314, "brown", -8, 1, 140, [lp(500)]),
    thump("thump", 110, 50, 0.5, -10, 1, 150, 0.3, drive_db=8),
    sub("sub", 48, 38, -18, 2, 110, 0.35),
], post(PEAK["heavy"], low=10000, high=32, comp=(-14, 2.2), clip=0.5, limit=-2.5, out_ms=14),
    "the crack leads: a 9 ms broadband crack with splinter grains over a driven thump")

cand("hit.impact.kill", "a_boom", 290, 5401, [
    crack(5402, 0, band=(2200, 13000), ms=9),
    grains("click", 5403, 2600, 0.9, -9, 0, 14, 3, 2.4, 3),
    thud("body", 5404, "brown", -7, 1, 150, [lp(460)]),
    thud("slam", 5406, "pink", 2, 1, 110, [bp(900, 0.5), lp(2600)]),
    thump("thump", 130, 46, 0.45, -7, 1, 190, 0.3, drive_db=8),
    sub("boom", 58, 30, -14, 3, 270, 0.42, drive_db=12),
    sweep("tail", 5405, 3400, 320, 0.55, -2, 28, 230, env(6, 40, 180), color="pink"),
], post(PEAK["kill"], low=12000, high=26, comp=(-13, 2.4), room=(0.22, 0.09), clip=0.45, limit=-3.5, out_ms=24),
    "the kill blow: heavy plus a 58 to 30 Hz sub boom with drive, a bright 9 ms crack, a falling noise tail, a small room")

cand("hit.impact.kill", "b_crunch", 300, 5411, [
    crack(5412, -1, band=(2000, 14000), ms=10),
    grains("crunch", 5413, 1500, 0.7, 1, 1, 60, 12, 3.5, 40),
    thud("body", 5414, "brown", -6, 1, 180, [lp(520)]),
    thud("slam", 5416, "pink", 2, 1, 120, [bp(800, 0.5), lp(2400)]),
    thump("thump", 140, 44, 0.45, -8, 1, 200, 0.3, drive_db=12),
    sub("boom", 55, 28, -15, 3, 280, 0.45, drive_db=14),
    sweep("tail", 5415, 2600, 240, 0.5, -1, 30, 260, env(6, 60, 190), color="pink"),
], post(PEAK["kill"], low=12000, high=26, comp=(-12, 2.6), clip=0.6, limit=-3.5, out_ms=28),
    "the kill blow as a crunch: a grain crunch across 60 ms, more drive, a longer dry tail, no room")

# Slash impacts ─────────────────────────────────────────────────────────────
cand("hit.slash.light", "a_nick", 75, 5501, [
    crack(5502, -6, band=(2600, 11000), ms=4),
    mode("chink", 5503, "metal_chink", -9, 0, 52, env(0, 4, 42), [hp(900, 2)]),
    sweep("edge", 5504, 5200, 1600, 0.8, -12, 0, 30, env(1, 8, 20)),
], post(PEAK["light"], low=11000, high=260, clip=0.25, out_ms=8),
    "a nick: a 4 ms crack, a short metal chink, a 30 ms edge of air; no body")

cand("hit.slash.light", "b_tick", 70, 5511, [
    crack(5512, -5, band=(3000, 12000), ms=4),
    grains("tick", 5513, 4600, 1.2, -6, 0, 22, 3, 2.0, 5),
    sweep("edge", 5514, 6000, 1800, 0.9, -10, 0, 34, env(1, 8, 24)),
    ring("steel", [(3300, 0), (5100, -6)], -16, 1, 48, 0.25, seed=5515),
], post(PEAK["light"], low=12000, high=300, clip=0.2, out_ms=8),
    "a tick: bandpassed grains at 4.6 kHz and a short two-partial steel ring, no chink mode")

cand("hit.slash.mid", "a_bite", 115, 5601, [
    crack(5602, -9, band=(2000, 10000), ms=5),
    mode("bite", 5603, "blade_impact", 2, 0, 95, env(0, 6, 80)),
    ring("steel", [(2300, 0), (3500, -5), (5200, -10)], -7, 1, 80, 0.28, seed=5604),
    thud("body", 5606, "brown", -14, 1, 70, [lp(600)]),
    sweep("edge", 5605, 4400, 1200, 0.7, -14, 0, 40, env(1, 10, 28)),
], post(PEAK["mid"], low=11000, high=110, clip=0.3, out_ms=10),
    "a bite: the blade_impact mode (body, crack, bite grains, drag) with a short steel ring")

cand("hit.slash.mid", "b_edge", 115, 5611, [
    crack(5612, -8, band=(2400, 11000), ms=5),
    grains("teeth", 5613, 2800, 0.8, -4, 1, 36, 8, 2.6, 22),
    ring("steel", [(2100, 0), (3300, -4), (4900, -8), (6600, -14)], -5, 1, 96, 0.3, seed=5614),
    thud("body", 5615, "brown", -10, 1, 60, [lp(700)]),
    sweep("edge", 5616, 5000, 1400, 0.7, -12, 0, 44, env(1, 10, 32)),
], post(PEAK["mid"], low=12000, high=110, clip=0.3, out_ms=10),
    "an edge: the steel ring leads with grain teeth, a small brown body, more air")

cand("hit.slash.heavy", "a_cleave", 185, 5701, [
    crack(5702, -6, band=(1800, 11000), ms=7),
    mode("bite", 5703, "blade_impact", 2, 0, 150, env(0, 8, 130)),
    ring("steel", [(1600, 0), (2500, -4), (3900, -9), (5600, -15)], -4, 2, 160, 0.3, seed=5704),
    thump("thump", 115, 55, 0.5, -14, 1, 130, 0.3, drive_db=6),
    sweep("edge", 5705, 4000, 900, 0.65, -6, 0, 60, env(1, 14, 44)),
], post(PEAK["heavy"], low=12000, high=50, comp=(-14, 2.2), clip=0.4, limit=-2.5, out_ms=14),
    "a cleave: blade bite over a 115 to 55 Hz thump, a longer and lower steel ring")

cand("hit.slash.heavy", "b_chop", 180, 5711, [
    crack(5712, -7, band=(1500, 12000), ms=8),
    mode("bite", 5713, "blade_impact", 1, 0, 120, env(0, 8, 100)),
    grains("teeth", 5714, 2200, 0.8, -4, 1, 40, 9, 2.8, 26),
    ring("steel", [(1900, 0), (3000, -5), (4700, -11)], -9, 2, 120, 0.28, seed=5715),
    thud("body", 5716, "brown", -11, 1, 130, [lp(480)]),
    thump("thump", 120, 50, 0.5, -14, 1, 140, 0.3, drive_db=6),
], post(PEAK["heavy"], low=12000, high=45, comp=(-14, 2.2), clip=0.45, limit=-2.5, out_ms=14),
    "a chop: more body and thump under the bite, a shorter ring; nearer the blunt heavy")

cand("hit.slash.kill", "a_ring", 300, 5801, [
    crack(5802, -2, band=(2400, 14000), ms=9),
    mode("bite", 5803, "blade_impact", 2, 0, 160, env(0, 8, 140)),
    ring("steel", [(1300, 0), (2100, -3), (3400, -8), (5200, -13), (7400, -19)], 2, 2, 280, 0.38, seed=5804),
    thump("thump", 125, 48, 0.45, -13, 1, 180, 0.3, drive_db=8),
    sub("boom", 56, 30, -22, 3, 250, 0.4, drive_db=12),
    sweep("tail", 5805, 3600, 400, 0.55, -3, 30, 220, env(6, 40, 170), color="pink"),
], post(PEAK["kill"], low=13000, high=28, comp=(-13, 2.4), room=(0.22, 0.08), clip=0.45, limit=-3.5, out_ms=26),
    "the kill cut: heavy plus a sub boom, a bright crack, a long five-partial steel ring, a noise tail")

cand("hit.slash.kill", "b_shear", 300, 5811, [
    crack(5812, -2, band=(2200, 14000), ms=10),
    mode("bite", 5813, "blade_impact", 2, 0, 170, env(0, 8, 150)),
    ring("steel", [(1500, 0), (2400, -4), (3800, -9), (5600, -15)], -3, 2, 200, 0.32, seed=5814),
    mode("shear", 5815, "scrape", 2, 24, 190, env(4, 40, 140), [bp(2400, 0.6)]),
    thump("thump", 130, 46, 0.45, -13, 1, 190, 0.3, drive_db=10),
    sub("boom", 54, 28, -22, 3, 260, 0.42, drive_db=14),
], post(PEAK["kill"], low=13000, high=28, comp=(-12, 2.6), clip=0.55, limit=-3.5, out_ms=26),
    "the kill cut as a shear: a scrape drags through the tail instead of a long ring; dry")

# Swings ────────────────────────────────────────────────────────────────────
cand("swing.light", "a_airy", 90, 5901, [
    sweep("air", 5902, 5600, 900, 0.8, -3, 0, 88, env(8, 30, 46)),
    sweep("close", 5903, 2800, 500, 0.65, -11, 6, 70, env(6, 26, 36), color="pink", pan=0.1),
], post(-10.5, low=9000, high=160, clip=0.2, out_ms=10),
    "an airy whoosh: white air from 5.6 kHz to 900 Hz, a quieter pink wake under it")

cand("swing.light", "b_soft", 95, 5911, [
    sweep("air", 5912, 3600, 600, 0.7, -3, 0, 92, env(10, 34, 46), color="pink"),
    sweep("close", 5913, 1800, 380, 0.6, -12, 8, 70, env(6, 24, 38), color="pink", pan=-0.1),
], post(-10.5, low=8000, high=140, clip=0.2, out_ms=10),
    "a softer whoosh: pink air lower, from 3.6 kHz to 600 Hz; less hiss")

cand("swing.heavy", "a_deep", 180, 6001, [
    sweep("air", 6002, 2400, 240, 0.6, -3, 0, 176, env(26, 60, 88), color="pink"),
    sweep("wake", 6003, 1200, 160, 0.55, -8, 14, 150, env(20, 50, 78), color="pink", pan=0.12),
    noise("weight", 6004, "brown", -10, 10, 150, env(24, 50, 74), [lp(300)]),
], post(-9.0, low=7000, high=40, comp=(-16, 2.0), clip=0.3, out_ms=16),
    "a deep slow whoosh: pink air from 2.4 kHz to 240 Hz over a brown weight under 300 Hz")

cand("swing.heavy", "b_wide", 185, 6011, [
    sweep("air_l", 6012, 3000, 300, 0.6, -5, 0, 180, env(28, 62, 88), color="pink", pan=-0.35),
    sweep("air_r", 6013, 2600, 260, 0.6, -5, 8, 172, env(26, 60, 84), color="pink", pan=0.35),
    noise("weight", 6014, "brown", -8, 12, 160, env(26, 54, 78), [lp(360)]),
], post(-9.0, low=7500, high=40, comp=(-16, 2.0), clip=0.3, out_ms=16),
    "a wide whoosh: two pink sweeps panned apart, a little more weight and mids")

cand("swing.aerial", "a_rise", 120, 6101, [
    sweep("rise", 6102, 650, 4200, 0.75, -3, 0, 118, env(10, 40, 66)),
    sweep("close", 6103, 420, 2200, 0.6, -11, 8, 100, env(8, 34, 56), color="pink", pan=-0.1),
], post(-10.0, low=9000, high=130, clip=0.2, out_ms=12),
    "a rising whoosh: white air from 650 Hz up to 4.2 kHz; a pink wake under it")

cand("swing.aerial", "b_lift", 125, 6111, [
    sweep("rise", 6112, 900, 3200, 0.7, -3, 0, 122, env(6, 50, 64), color="pink"),
    sweep("tip", 6113, 2600, 6000, 0.9, -12, 40, 80, env(8, 30, 40)),
], post(-10.0, low=9500, high=130, clip=0.2, out_ms=12),
    "a lift: a quicker pink rise to 3.2 kHz, a white tip that climbs to 6 kHz late")

cand("swing.charge_release", "a_snap", 120, 6201, [
    grains("snap", 6202, 3000, 1.0, -5, 0, 12, 3, 2.0, 3),
    crack(6203, -7, band=(1600, 9000), ms=4),
    sweep("air", 6204, 4200, 520, 0.7, -4, 6, 112, env(4, 36, 70)),
    sweep("wake", 6205, 1800, 300, 0.6, -11, 12, 100, env(6, 30, 62), color="pink", pan=0.1),
], post(-9.5, low=9000, high=90, clip=0.3, out_ms=12),
    "a charged smash letting go: a grain snap and crack, then a white whoosh with a pink wake")

cand("swing.charge_release", "b_pop", 120, 6211, [
    tone("pop", "sine", [[0, 420], [0.5, 140], [1, 90]], -6, 0, 18, decay(0.4), e=env(0.3, 0, 8)),
    crack(6212, -6, band=(1800, 10000), ms=4),
    sweep("air", 6213, 3600, 480, 0.7, -4, 8, 110, env(4, 34, 70), color="pink"),
    noise("weight", 6214, "brown", -14, 10, 90, env(10, 30, 48), [lp(320)]),
], post(-9.5, low=9000, high=70, clip=0.3, out_ms=12),
    "a pitched pop (420 to 90 Hz in 18 ms) and crack, then a pink whoosh with a little weight")


def cand_id(cue, cid):
    return f"audition.smash_feel.{cue}.{cid}"


def recipe(cue_id, spec, note):
    return {"schema": "ambition.sfxir.v1", "id": cue_id, "duration_ms": spec["duration_ms"],
            "render": {"sample_rate": SR, "channels": 2, "seed": spec["seed"], "buffer_size": 128,
                       # Our own band limits: the default safety lowpass takes a hit's crack.
                       "safety_profile": "none"},
            "layers": copy.deepcopy(spec["layers"]), "postprocess": copy.deepcopy(spec["postprocess"]),
            "notes": note}


def render(path: Path, outdir: Path, policy="wav"):
    done = subprocess.run([PY, "-m", "ambition_sfx_renderer", "render", str(path), "--outdir", str(outdir),
                           "--format-policy", policy, "--force"], cwd=RENDERER, capture_output=True, text=True,
                          env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    if done.returncode:
        raise SystemExit(f"{path.name}: {done.stderr[-2000:]}")


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
BANDS = ((0, 90), (90, 200), (200, 800), (800, 3000), (3000, 8000), (8000, 24000))


def moving_mean(x, win):
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    return (c[win:] - c[:-win]) / win


def loudness(a) -> float:
    """The loudest 400 ms RMS in dB (or the whole, when shorter)."""
    mono = a.mean(axis=1)
    win = int(0.4 * SR)
    if len(mono) <= win:
        return float(10 * np.log10((mono ** 2).mean() + 1e-12))
    return float(10 * np.log10(moving_mean(mono ** 2, win).max() + 1e-12))


def env10(a, n=None):
    """RMS dB of each 10 ms window."""
    mono = a.mean(axis=1)
    w = SR // 100
    k = len(mono) // w
    r = np.sqrt((mono[: k * w].reshape(k, w) ** 2).mean(axis=1))
    db = 20 * np.log10(r + 1e-9)
    return db[:n] if n else db


def measure(a) -> dict:
    mono = a.mean(axis=1)
    e = env10(a)
    top = e.max()
    heard = np.where(e > top - 20)[0]
    spec = np.abs(np.fft.rfft(mono)) ** 2
    freq = np.fft.rfftfreq(len(mono), 1 / SR)
    total = spec.sum() or 1.0
    head = mono[: int(0.015 * SR)]
    hs = np.abs(np.fft.rfft(head, n=4096)) ** 2
    hf = np.fft.rfftfreq(4096, 1 / SR)
    first30 = mono[: int(0.03 * SR)]
    return dict(
        ms=round(1000 * len(mono) / SR),
        heard_ms=(int(heard[-1]) - int(heard[0]) + 1) * 10 if len(heard) else 0,
        loud_db=round(loudness(a), 1),
        peak_db=round(float(20 * np.log10(np.abs(a).max() + 1e-12)), 1),
        bands=[round(100 * spec[(freq >= lo) & (freq < hi)].sum() / total) for lo, hi in BANDS],
        centroid_hz=round(float((spec * freq).sum() / total)),
        # The phone-speaker view: the RMS of what lies above 300 Hz. A tier
        # step that lives only in the sub band is not a step on a phone.
        loud300_db=round(float(10 * np.log10((mono ** 2).mean() * spec[freq >= 300].sum() / total + 1e-12)), 1),
        crack_hf=round(100 * hs[hf >= 3000].sum() / (hs.sum() or 1.0)),
        crest30_db=round(float(20 * np.log10(np.abs(first30).max() / (np.sqrt((first30 ** 2).mean()) + 1e-12))), 1),
    )


HEAD = ("| cue | ms | heard (-20 dB) | loud dB | loud >300 Hz dB | peak dB | bands % <90/<200/<800/<3k/<8k/8k+ "
        "| centroid | HF% first 15 ms | crest 30 ms dB |")


def table(rows) -> list[str]:
    out = [HEAD, "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for name, a in rows:
        m = measure(a)
        out.append(f"| {name} | {m['ms']} | {m['heard_ms']} | {m['loud_db']} | {m['loud300_db']} | {m['peak_db']} | "
                   f"{'/'.join(map(str, m['bands']))} | {m['centroid_hz']} | {m['crack_hf']} | {m['crest30_db']} |")
    return out


def tier_order(got) -> list[str]:
    """Check light < mid < heavy < kill in peak, loudness and heard length for
    the chosen impacts. Returns the lines of the report; a line starts with
    FAIL when an order breaks."""
    lines = []
    for family in ("hit.impact", "hit.slash"):
        ms = [measure(got[(f"{family}.{t}", CHOSEN[f"{family}.{t}"])]) for t in TIERS]
        for key in ("peak_db", "loud_db", "loud300_db", "heard_ms"):
            values = [m[key] for m in ms]
            ok = all(values[i] < values[i + 1] for i in range(3))
            lines.append(f"- {'ok  ' if ok else 'FAIL'} {family} {key}: " + " < ".join(map(str, values)))
    return lines


# ── Reels ────────────────────────────────────────────────────────────────────
def concat(items):
    """items: (label, audio). One wav, GAP_S of silence between, and the
    offsets as text lines."""
    gap = np.zeros((int(GAP_S * SR), 2), np.float32)
    parts, marks, t = [], [], 0.0
    for label, a in items:
        marks.append(f"{t:7.3f}s  {label}")
        parts += [a, gap]
        t += len(a) / SR + GAP_S
    return np.concatenate(parts), marks


def write_reels(here, cur, got):
    (here / "reel").mkdir(exist_ok=True)
    reels = {}
    every = []
    for cue in ALL_CUES:
        for (c, cid) in C:
            if c == cue:
                star = " *" if CHOSEN[cue] == cid else ""
                every.append((f"{cue} {cid}{star}", got[(c, cid)]))
    every.append(("player.hit (in use)", cur["player_hit"]))
    every.append(("fighter.brawler.jab (in use)", cur["brawler_jab"]))
    every.append(("fighter.sword.smash_forward (in use)", cur["sword_smash"]))
    reels["all_cues"] = every
    chosen = [(cue + " *", got[(cue, CHOSEN[cue])]) for cue in ALL_CUES]
    reels["chosen"] = chosen + [("player.hit (in use)", cur["player_hit"])]
    for family in ("hit.impact", "hit.slash"):
        tiers = [(f"{family}.{t} " + CHOSEN[f"{family}.{t}"], got[(f"{family}.{t}", CHOSEN[f"{family}.{t}"])])
                 for t in TIERS]
        reels[f"tiers_{family.split('.')[1]}"] = tiers + [("player.hit (in use)", cur["player_hit"])]
        # Each tier twice in a row, then the next: the step between neighbours.
        reels[f"tiers_{family.split('.')[1]}_pairs"] = [x for x in tiers for _ in (0, 1)]
    # A swing, then the hit it leads to, 70 ms after the swing's peak.
    combos = []
    for swing, hit in (("swing.light", "hit.impact.light"), ("swing.light", "hit.impact.mid"),
                       ("swing.heavy", "hit.impact.heavy"), ("swing.charge_release", "hit.impact.kill"),
                       ("swing.aerial", "hit.impact.mid"), ("swing.light", "hit.slash.light"),
                       ("swing.light", "hit.slash.mid"), ("swing.heavy", "hit.slash.heavy"),
                       ("swing.charge_release", "hit.slash.kill"), ("swing.aerial", "hit.slash.mid")):
        s, h = got[(swing, CHOSEN[swing])], got[(hit, CHOSEN[hit])]
        lead = int(0.5 * len(s)) + int(0.07 * SR)
        out = np.zeros((max(len(s), lead + len(h)), 2), np.float32)
        out[: len(s)] += s
        out[lead: lead + len(h)] += h
        combos.append((f"{swing} + {hit}", out))
    reels["swing_then_hit"] = combos

    index = []
    # One gain for every reel, so they keep their levels against each other.
    audio = {name: concat(items) for name, items in reels.items()}
    gain = min(1.0, 0.89 / max(float(np.abs(a).max()) for a, _ in audio.values()))
    for name, (a, marks) in audio.items():
        sf.write(here / "reel" / f"{name}.wav", a * gain, SR)
        (here / "reel" / f"{name}.txt").write_text("\n".join(marks) + "\n")
        index.append(f"- `reel/{name}.wav` ({len(marks)} cues; offsets in `reel/{name}.txt`)")
    return index


# ── Main ─────────────────────────────────────────────────────────────────────
def in_use():
    return {k: load(SHIPPED / v) for k, v in IN_USE.items()}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("command", nargs="?", default="all", choices=["all", "measure", "ship"])
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    here = args.outdir

    if args.command == "ship":
        for cue, cid in CHOSEN.items():
            spec = C[(cue, cid)]
            note = (f"Smash feel hit tier or swing, {cue}: {spec['note']}. Candidate {cid} of "
                    f"scripts/smash_feel_candidates.py (same layers and seed).")
            path = ACTIVE / f"{cue}.sfx.yaml"
            path.write_text(yaml.safe_dump(recipe(cue, spec, note), sort_keys=False))
            print(path)
        return

    cur = in_use()
    index = ["# Smash feel: hit tiers and swings, SFX auditions", "",
             "Written by `tools/ambition_sfx_renderer/scripts/smash_feel_candidates.py` (2026-10-29).",
             "Solos are as rendered (no level matching: each cue's level is its own choice); the reels",
             "share one gain. `loud dB` is the RMS of the loudest 400 ms (the whole cue when shorter).",
             "", "## The cues in use", ""]
    index += table([(f"`{v}`", cur[k]) for k, v in IN_USE.items()])
    if args.command == "measure":
        print("\n".join(index))
        return

    (here / "recipes").mkdir(parents=True, exist_ok=True)
    paths = []
    for (cue, cid), spec in C.items():
        path = here / "recipes" / f"{cand_id(cue, cid)}.sfx.yaml"
        path.write_text(yaml.safe_dump(recipe(cand_id(cue, cid), spec, spec["note"]), sort_keys=False))
        paths.append(path)
    with ThreadPoolExecutor(JOBS) as pool:
        list(pool.map(lambda q: render(q, here / "out"), paths))
    got = {key: load(here / "out" / cand_id(*key)) for key in C}

    (here / "solo").mkdir(exist_ok=True)
    for key, a in got.items():
        sf.write(here / "solo" / f"{key[0]}__{key[1]}.wav", a, SR)
    for k, v in IN_USE.items():
        sf.write(here / "solo" / f"{v}__in_use.wav", cur[k], SR)

    index += ["", "## The candidates", "", "`*` marks the one shipped (`CHOSEN`).", ""]
    for cue in ALL_CUES:
        index += [f"### `{cue}`", ""]
        for (c, cid), spec in C.items():
            if c == cue:
                star = "*" if CHOSEN[cue] == cid else ""
                index.append(f"- {star}`solo/{cue}__{cid}.wav`: {spec['note']}")
        index.append("")
        rows = [(("*" if CHOSEN[cue] == cid else "") + cid, got[(c, cid)]) for c, cid in C if c == cue]
        index += table(rows) + [""]

    index += ["## Tier order (chosen candidates)", ""] + tier_order(got) + [""]
    index += ["## The reels", "", f"Cues in a row with {GAP_S:.1f} s of silence between; `*` = chosen.", ""]
    index += write_reels(here, cur, got)
    (here / "INDEX.md").write_text("\n".join(index) + "\n")
    print("\n".join(tier_order(got)))
    print(f"{len(C)} candidates -> {here}")


if __name__ == "__main__":
    main()
