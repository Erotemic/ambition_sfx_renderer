"""GNU-ton's giant blows (2026-10-23): candidates, measurements and reels.

GNU-ton's giant punches and kicks THROUGH a pair of portals: the limb goes
into a line of light beside the giant and comes out of a far line beside the
player. Jon wants the blows to have "oomph", to "feel dangerous", "a sense of
power... impactful, especially if they land". Every land plays the old
`boss.gnu_ton.stomp` today, which is weak. This writes the cues of a blow, from
descriptions (synthesis only; no recording is in them):

- `windup`  a giant limb drawn back: sinew creak, a low rising air pull
- `whoosh`  the fist or hoof swung fast: a deep, wide whoosh with a low body
- `land`    the blow lands on the player: crack, big low thump, ringing tail
- `floor`   the fist hits the station floor and misses: boom, debris, rumble
- `kick`    a hoof kick: a woody, hard crack of hoof plus thump
- `portal`  a portal line easing open: a soft rising shimmer and whomp

`land` and `floor` have four candidates each, the others one or two. Each
candidate is a recipe; a chosen one is shipped as `sounds/active/<id>.sfx.yaml`
with the SAME layers and seed (`ship` below writes them), so the shipped
render equals the candidate sample for sample.

Run with the renderer's interpreter, from anywhere:

    python scripts/gnu_ton_blow_candidates.py            # render, measure, reels, INDEX.md
    python scripts/gnu_ton_blow_candidates.py measure    # only the table of the cues in use
    python scripts/gnu_ton_blow_candidates.py ship       # write the chosen recipes into sounds/active/

Writes `<superproject>/untracked/sfx-candidates/gnu_ton_blows/` (git-ignored;
`--outdir` elsewhere): `recipes/`, `out/`, `solo/<sound>__<candidate>.wav`,
`reel/<reel>.wav` and `INDEX.md`.
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
DEFAULT_OUT = SUPER / "untracked" / "sfx-candidates" / "gnu_ton_blows"
PY = sys.executable
SR = 48000
JOBS = 6  # a shared box: never every CPU

IN_USE = {
    "stomp": "boss.gnu_ton.stomp",
    "snort": "boss.gnu_ton.snort",
    "charge": "boss.gnu_ton.charge",
    "portal_attach": "portal.attach",
    "world_portal_enter": "world.portal.enter",
    "flesh_deep": "player.robot.slash.impact.flesh.deep",
}

# The candidate each new cue ships (the `ship` command), chosen by the
# measurements in INDEX.md.
CHOSEN = {
    "windup": "a_sinew",
    "whoosh": "a_wide",
    "land": "b_crush_robot",
    "floor": "b_deck_plate",
    "kick": "a_hoof",
    "portal": "a_whomp",
}
CUE_ID = {
    "windup": "boss.gnu_ton.blow_windup",
    "whoosh": "boss.gnu_ton.blow_whoosh",
    "land": "boss.gnu_ton.blow_land",
    "floor": "boss.gnu_ton.blow_floor",
    "kick": "boss.gnu_ton.hoof_kick",
    "portal": "boss.gnu_ton.portal_open",
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


def sweep(name, seed, a, b, q, gain, start, dur, e, effects=None, pan=0.0):
    return {"name": name, "kind": "noise_burst", "mode": "air_sweep", "seed": seed, "start_hz": a,
            "end_hz": b, "q": q, "gain_db": gain, "start_ms": start, "duration_ms": dur, "pan": pan,
            "envelope": e, "effects": effects or []}


def mode(name, seed, m, gain, start, dur, e, effects=None, pan=0.0):
    return {"name": name, "kind": "noise_burst", "mode": m, "seed": seed, "gain_db": gain,
            "start_ms": start, "duration_ms": dur, "pan": pan, "envelope": e, "effects": effects or []}


def tone(name, wave, freq, gain, start, dur, amp, e=None, harmonics=None, effects=None, seed=None,
         noise_amount=0.0, spread_cents=0.0, tremolo=None, pan=0.0):
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
    if tremolo:
        layer["tremolo"] = tremolo
    return layer


def crack(seed, gain, start=0.0, band=(1800, 9000), ms=6.0, pan=0.0, name="crack"):
    """A broadband snap: white noise, near no attack, gone in a few ms."""
    centre = round(float(np.sqrt(band[0] * band[1])))
    return noise(name, seed, "white", gain, start, ms + 8, env(0, ms * 0.25, ms),
                 [hp(band[0], 3), lp(band[1], 2), bp(centre, 0.35)], pan=pan)


def thump(name, f0, f1, sweep_frac, gain, start, dur, tau, drive_db=0.0, harm=-9.0):
    """A pitched body blow: a sine falling from f0 to f1, struck."""
    freq = [[0, f0], [round(sweep_frac * 0.35, 3), round((f0 * f1) ** 0.5, 1)], [sweep_frac, f1], [1, round(f1 * 0.92, 1)]]
    effects = [{"effect": "distortion", "drive_db": drive_db}, lp(900)] if drive_db else None
    return tone(name, "sine", freq, gain, start, dur, decay(tau), harmonics=[
        {"ratio": 1.0, "gain_db": 0}, {"ratio": 2.0, "gain_db": harm}, {"ratio": 3.0, "gain_db": harm - 8}],
        effects=effects)


def ring(name, partials, gain, start, dur, tau, seed=None):
    """An inharmonic ring: struck metal, decaying (partials = [(hz, dB), ...])."""
    base = partials[0][0]
    return tone(name, "sine", [[0, base], [1, round(base * 0.995, 2)]], gain, start, dur, decay(tau),
                harmonics=[{"ratio": round(hz / base, 4), "gain_db": db} for hz, db in partials],
                spread_cents=2, seed=seed)


def post(peak, low=14000, comp=None, room=None, clip=None, limit=None, out_ms=25):
    chain = [hp(28, 3), lp(low)]
    if comp:
        thr, ratio = comp
        chain.append({"effect": "compressor", "threshold_db": thr, "ratio": ratio, "attack_ms": 1.5,
                      "release_ms": 90})
    if room:
        size, wet = room
        chain.append({"effect": "reverb", "room_size": size, "damping": 0.55, "wet_level": wet,
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


# ── The candidates: (sound, id) -> spec ──────────────────────────────────────
C: dict[tuple[str, str], dict] = {}


def cand(sound, cid, dur, seed, layers, postprocess, note):
    C[(sound, cid)] = dict(duration_ms=dur, seed=seed, layers=layers, postprocess=postprocess, note=note)


# WINDUP: the limb drawn back. A creak is stick-slip: a pulse train through a
# resonance, here a saw at 26-60 Hz through a sinew's 400-900 Hz band; the air
# pull rises under it. It swells to the end, where the whoosh takes over.
cand("windup", "a_sinew", 450, 7101, [
    tone("creak_low", "saw", [[0, 26], [0.6, 34], [1, 44]], -2, 0, 440,
         [[0, 0], [0.3, 0.55], [0.75, 1], [0.92, 0.8], [1, 0]], seed=7102, noise_amount=0.18,
         effects=[bp(430, 2.2), hp(180)], pan=-0.2),
    tone("creak_high", "saw", [[0, 41], [0.5, 52], [1, 63]], -6, 70, 370,
         [[0, 0], [0.4, 0.6], [0.8, 1], [1, 0]], seed=7103, noise_amount=0.25,
         effects=[bp(880, 2.8), hp(400)], pan=0.25),
    sweep("air_pull", 7104, 150, 650, 0.55, -7, 40, 410, env(260, 60, 90), [lp(1800)]),
    tone("mass", "sine", [[0, 42], [1, 64]], -20, 0, 450, [[0, 0], [0.7, 1], [0.9, 0.9], [1, 0]]),
    grains("tendon_ticks", 7105, 1500, 1.4, -24, 150, 280, 9, 2.5, 260, pan=0.1),
], post(-12, low=9000), "creak of sinew (two stick-slip voices) over a rising air pull, swelling to the end")
cand("windup", "b_leather", 450, 7111, [
    tone("creak", "saw", [[0, 22], [0.5, 30], [1, 38]], -1, 0, 440,
         [[0, 0], [0.25, 0.7], [0.5, 0.45], [0.8, 1], [1, 0]], seed=7112, noise_amount=0.35,
         effects=[bp(620, 3.2), hp(250)]),
    sweep("air_pull", 7114, 120, 520, 0.5, -6, 0, 450, env(320, 40, 90), [lp(1400)]),
    tone("mass", "sine", [[0, 38], [1, 58]], -19, 0, 450, [[0, 0], [0.75, 1], [1, 0]]),
], post(-12, low=8000), "one rougher leather-stretch creak, a slower air pull")

# WHOOSH: the fist or hoof swung fast. Deep and wide: two air sweeps with
# their own noise, panned apart, a brown-noise push and a low thrum of the mass.
cand("whoosh", "a_wide", 350, 7201, [
    sweep("air_l", 7202, 1300, 170, 0.5, -4, 0, 340, env(20, 40, 120), pan=-0.45),
    sweep("air_r", 7203, 1100, 150, 0.5, -4, 10, 330, env(20, 40, 120), pan=0.45),
    noise("push", 7204, "brown", -6, 30, 300, env(110, 30, 150), [lp(260)]),
    tone("thrum", "sine", [[0, 72], [0.5, 58], [1, 44]], -8, 20, 320,
         [[0, 0], [0.4, 1], [0.6, 0.85], [1, 0]], harmonics=[{"ratio": 1, "gain_db": 0}, {"ratio": 2, "gain_db": -12}]),
], post(-7, low=9000, comp=(-20, 2.0)), "two wide air sweeps 1.3k-150 Hz, a brown push and a 72-44 Hz thrum")
cand("whoosh", "b_heavy", 350, 7211, [
    sweep("air", 7212, 900, 110, 0.42, -3, 0, 350, env(40, 60, 140)),
    noise("push", 7214, "brown", -4, 20, 320, env(140, 20, 150), [lp(200)]),
    tone("thrum", "sine", [[0, 60], [1, 38]], -6, 30, 310, [[0, 0], [0.45, 1], [1, 0]]),
    noise("flutter", 7215, "pink", -18, 60, 220, env(60, 20, 120), [bp(420, 0.8)]),
], post(-7, low=7000, comp=(-20, 2.0)), "darker and heavier: one 900-110 Hz sweep, a big push, little air")

# LAND: the blow lands on the player (a robot). The scariest of the set: a
# crack in the first ms, a 40-90 Hz thump, and a short ringing tail.
cand("land", "a_crush_bone", 600, 7301, [
    crack(7302, 0, band=(1700, 10000), ms=8),
    crack(7303, -6, start=9, band=(2500, 9000), ms=6, pan=0.3, name="crack_2"),
    thump("sub", 95, 48, 0.14, -8, 0, 600, 0.12, drive_db=8),
    thump("body", 220, 90, 0.08, 1, 0, 300, 0.10, drive_db=4),
    noise("smack", 7304, "pink", 4, 0.5, 90, env(0, 8, 70), [bp(420, 0.7)]),
    grains("bone_rattle", 7305, 2100, 0.9, -1, 6, 160, 34, 2.0, 140),
    thud("mass", 7306, "brown", -5, 0, 420, [lp(320)]),
    ring("ring", [(520, 0), (837, -4), (1291, -8), (1960, -14)], -6, 4, 560, 0.30),
], post(-3, comp=(-16, 3.0), clip=2.2, limit=-3), "crack + crack, 95-48 Hz driven sub, 220-90 Hz body, smack, bone rattle, short ring")
cand("land", "b_crush_robot", 600, 7311, [
    crack(7312, 0, band=(1600, 11000), ms=9),
    thump("sub", 88, 46, 0.16, -8, 0, 600, 0.13, drive_db=10),
    thump("body", 190, 82, 0.10, 0, 0, 320, 0.10, drive_db=6),
    noise("smack", 7316, "pink", 3, 0.5, 90, env(0, 8, 70), [bp(480, 0.7)]),
    mode("shell_crunch", 7313, "robot_crunch", 4, 1, 260, env(0, 40, 200)),
    grains("rattle", 7314, 2600, 1.0, -2, 8, 200, 40, 1.8, 180, pan=-0.2),
    thud("mass", 7315, "brown", -6, 0, 400, [lp(300)]),
    ring("ring", [(611, 0), (947, -3), (1433, -7), (2187, -12)], -5, 3, 590, 0.26),
], post(-3, comp=(-16, 3.0), clip=2.2, limit=-3), "crack, 88-46 Hz driven sub, 190-82 Hz body, a smack, a robot shell crunching, rattle, metal ring")
cand("land", "c_flesh_strike", 600, 7321, [
    mode("strike", 7322, "flesh_strike", 4, 0, 420, env(0, 0, 40)),
    crack(7323, -3, band=(2000, 10000), ms=8),
    thump("sub", 80, 40, 0.15, -3, 0, 600, 0.14, drive_db=8),
    ring("ring", [(470, 0), (760, -5), (1180, -10)], -8, 4, 560, 0.28),
], post(-3, comp=(-16, 3.0), clip=2.2, limit=-3), "the heavy flesh_strike (crack, thump, smack, wet tail) over an 80-40 Hz sub and a faint ring")
cand("land", "d_thunder_fist", 600, 7331, [
    crack(7332, 0, band=(1500, 12000), ms=10),
    crack(7333, -5, start=14, band=(1500, 7000), ms=8, pan=-0.35, name="crack_2"),
    thump("sub", 70, 38, 0.20, -6, 0, 600, 0.15, drive_db=12),
    thump("body", 140, 62, 0.12, 2, 0, 380, 0.12, drive_db=6),
    noise("blast", 7334, "pink", 3, 0, 160, env(0, 10, 140), [lp(1600)]),
    grains("debris", 7335, 1700, 0.8, -3, 20, 320, 30, 3.0, 280),
    ring("ring", [(392, 0), (588, -4), (931, -9)], -6, 6, 580, 0.32),
], post(-3, comp=(-14, 3.5), room=(0.42, 0.10), clip=2.4, limit=-3), "two cracks, a 70-38 Hz sub, a blast, debris, a ring in a room: the biggest")

# FLOOR: the fist hits the station floor and misses. A deep boom, debris and
# grit, a short rumble; heavier and longer than the stomp, not as sharp as land.
cand("floor", "a_stone_boom", 700, 7401, [
    noise("contact", 7402, "pink", 0, 0, 22, env(0, 4, 16), [bp(1300, 0.5)]),
    thump("boom", 72, 34, 0.22, -6, 0, 700, 0.16, drive_db=8),
    thump("knock", 170, 80, 0.1, -4, 0, 400, 0.12, drive_db=4),
    thud("body", 7403, "brown", -6, 0, 620, [lp(400)]),
    grains("debris", 7404, 1800, 0.8, -4, 15, 480, 46, 3.5, 420),
    noise("grit", 7405, "pink", -8, 10, 380, env(5, 40, 330), [bp(900, 0.6)]),
    noise("rumble", 7406, "brown", -9, 40, 660, env(30, 120, 500), [lp(110)]),
], post(-5, low=9000, comp=(-18, 2.5), clip=1.3), "soft contact, a 72-34 Hz boom, brown body, debris, grit, a 110 Hz rumble")
cand("floor", "b_deck_plate", 700, 7411, [
    noise("contact", 7412, "pink", 0, 0, 24, env(0, 4, 18), [bp(1500, 0.5)]),
    thump("boom", 68, 33, 0.22, -6, 0, 700, 0.17, drive_db=9),
    thump("knock", 160, 76, 0.1, -4, 0, 400, 0.12, drive_db=4),
    thud("body", 7413, "brown", -7, 0, 600, [lp(380)]),
    mode("deck_ring", 7414, "metal_gong", -2, 2, 640, env(0, 60, 500)),
    grains("debris", 7415, 2000, 0.8, 0, 18, 520, 52, 3.0, 480, pan=0.15),
    grains("bolts", 7416, 3400, 1.2, -6, 40, 400, 14, 1.5, 360, pan=-0.3),
    noise("rumble", 7417, "brown", -9, 50, 650, env(30, 120, 480), [lp(100)]),
], post(-5, low=10000, comp=(-18, 2.5), clip=1.3), "the station floor: a 68-33 Hz boom, a deck plate ringing, debris, rattling bolts, rumble")
cand("floor", "c_quake", 700, 7421, [
    noise("contact", 7422, "pink", -1, 0, 24, env(0, 4, 18), [bp(1100, 0.5)]),
    thump("boom", 60, 30, 0.26, -6, 0, 700, 0.2, drive_db=10),
    thump("knock", 150, 70, 0.1, -5, 0, 400, 0.12, drive_db=4),
    sweep("shock_air", 7423, 700, 120, 0.45, -3, 5, 360, env(5, 40, 300)),
    grains("gravel", 7424, 1500, 0.7, -3, 20, 600, 70, 4.0, 560),
    noise("rumble", 7425, "brown", -6, 20, 680, env(20, 160, 480), [lp(90)]),
], post(-5, low=8000, comp=(-18, 2.5), clip=1.3), "a quake: a 60-30 Hz boom, a shock of air, much gravel, a long low rumble")
cand("floor", "d_hollow_hull", 700, 7431, [
    noise("contact", 7432, "pink", 0, 0, 22, env(0, 4, 16), [bp(1400, 0.5)]),
    thump("boom", 76, 36, 0.2, -6, 0, 700, 0.16, drive_db=7),
    ring("hull", [(94, 0), (143, -2), (226, -5), (332, -9)], -5, 4, 690, 0.30),
    thud("body", 7433, "brown", -7, 0, 500, [lp(400)]),
    grains("debris", 7434, 1900, 0.8, -4, 15, 440, 40, 3.0, 400),
], post(-5, low=9000, comp=(-18, 2.5), clip=1.3), "a hollow hull: the boom and the hull's own modes (94-332 Hz) ringing, debris")

# KICK: a hoof kick, at the floor or at the end of its reach. Woody and hard:
# a crack lower than the fist's, the damped modes of a hoof, a thump.
cand("kick", "a_hoof", 450, 7501, [
    crack(7502, 0, band=(1200, 6500), ms=5),
    ring("hoof_knock", [(690, 0), (1130, -3), (1710, -6), (2480, -11)], 3, 0, 160, 0.10),
    thump("thump", 150, 58, 0.12, -4, 0, 450, 0.12, drive_db=6),
    thud("mass", 7503, "brown", -9, 0, 300, [lp(300)]),
    grains("grit", 7504, 1700, 0.8, -8, 10, 260, 22, 2.5, 230),
], post(-5, comp=(-16, 2.8), clip=1.3, limit=-3), "a 1.2-6.5k crack, a hoof's knock (690 Hz modes, 16 ms), a 150-58 Hz thump, grit")
cand("kick", "b_clack", 450, 7511, [
    crack(7512, -2, band=(1500, 8000), ms=3),
    ring("hoof_knock", [(820, 0), (1260, -2), (2050, -7)], 0, 0, 110, 0.12),
    crack(7513, -6, start=22, band=(1500, 6000), ms=3, pan=0.3, name="clack_2"),
    thump("thump", 130, 52, 0.12, -4, 0, 450, 0.13, drive_db=6),
], post(-5, comp=(-16, 2.8), clip=1.3, limit=-3), "a brighter clack with a second smaller clack 22 ms on, a 130-52 Hz thump")

# PORTAL: a portal line easing open. A soft rising whomp and a shimmer of
# partials over it; quiet, so it is heard as a telegraph.
cand("portal", "a_whomp", 400, 7601, [
    tone("whomp", "sine", [[0, 48], [0.55, 96], [1, 112]], -2, 0, 400,
         [[0, 0], [0.45, 1], [0.7, 0.7], [1, 0]], harmonics=[{"ratio": 1, "gain_db": 0}, {"ratio": 2, "gain_db": -9}]),
    tone("shimmer", "sine", [[0, 760], [1, 980]], -4, 40, 360, [[0, 0], [0.5, 1], [0.8, 0.6], [1, 0]],
         harmonics=[{"ratio": 1, "gain_db": 0}, {"ratio": 1.5, "gain_db": -4}, {"ratio": 2.01, "gain_db": -7},
                    {"ratio": 3.02, "gain_db": -12}], spread_cents=14, tremolo={"rate_hz": 17, "depth": 0.45}),
    sweep("air", 7602, 260, 2200, 0.6, -4, 0, 400, env(200, 60, 140), [lp(3500)]),
], post(-14, low=9000), "a 48-112 Hz whomp rising, a tremolo shimmer 760-980 Hz, a soft rising air")
cand("portal", "b_bloom", 400, 7611, [
    tone("whomp", "sine", [[0, 60], [0.6, 120], [1, 128]], -3, 0, 400, [[0, 0], [0.55, 1], [1, 0]]),
    tone("shimmer", "triangle", [[0, 1180], [1, 1460]], -6, 60, 340, [[0, 0], [0.6, 1], [1, 0]],
         harmonics=[{"ratio": 1, "gain_db": 0}, {"ratio": 1.26, "gain_db": -3}, {"ratio": 1.5, "gain_db": -5}],
         spread_cents=20, tremolo={"rate_hz": 23, "depth": 0.5}),
    noise("hiss", 7612, "pink", -8, 0, 400, env(240, 40, 120), [bp(3000, 0.6)]),
], post(-14, low=9000), "brighter: a 60-128 Hz bloom, a chord shimmer at 1.2-1.5 kHz, a soft hiss")


# ── Recipes and rendering ────────────────────────────────────────────────────
def cand_id(sound, cid):
    return f"audition.gnu_ton_blows.{sound}.{cid}"


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
    # The crack: the share of energy above 3 kHz in the first 15 ms, and the
    # crest (peak over RMS) of the first 30 ms.
    head = mono[: int(0.015 * SR)]
    hs = np.abs(np.fft.rfft(head, n=4096)) ** 2
    hf = np.fft.rfftfreq(4096, 1 / SR)
    first30 = mono[: int(0.03 * SR)]
    return dict(
        ms=round(1000 * len(mono) / SR),
        onset_ms=int(np.argmax(e > top - 20)) * 10,
        peak_at_ms=int(e.argmax()) * 10,
        heard_ms=(int(heard[-1]) - int(heard[0]) + 1) * 10 if len(heard) else 0,
        fall20_ms=(int(heard[-1]) - int(e.argmax()) + 1) * 10 if len(heard) else 0,
        loud_db=round(loudness(a), 1),
        peak_db=round(float(20 * np.log10(np.abs(a).max() + 1e-12)), 1),
        bands=[round(100 * spec[(freq >= lo) & (freq < hi)].sum() / total) for lo, hi in BANDS],
        centroid_hz=round(float((spec * freq).sum() / total)),
        width_db=round(float(10 * np.log10((((a[:, 0] - a[:, 1]) / 2) ** 2).sum() / ((((a[:, 0] + a[:, 1]) / 2) ** 2).sum() + 1e-12) + 1e-12)), 1),
        punch_db=round(float(10 * np.log10((mono[: int(0.1 * SR)] ** 2).mean() + 1e-12)), 1),
        crack_hf=round(100 * hs[hf >= 3000].sum() / (hs.sum() or 1.0)),
        crest30_db=round(float(20 * np.log10(np.abs(first30).max() / (np.sqrt((first30 ** 2).mean()) + 1e-12))), 1),
    )


HEAD = ("| cue | ms | onset | peak@ | heard (-20 dB) | fall | loud dB | peak dB | bands % <90/<200/<800/<3k/<8k/8k+ "
        "| centroid | HF% first 15 ms | crest 30 ms dB | first 100 ms dB | side/mid dB |")


def table(rows) -> list[str]:
    out = [HEAD, "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for name, a in rows:
        m = measure(a)
        out.append(f"| {name} | {m['ms']} | {m['onset_ms']} | {m['peak_at_ms']} | {m['heard_ms']} | {m['fall20_ms']} | "
                   f"{m['loud_db']} | {m['peak_db']} | {'/'.join(map(str, m['bands']))} | {m['centroid_hz']} | "
                   f"{m['crack_hf']} | {m['crest30_db']} | {m['punch_db']} | {m['width_db']} |")
    return out


def env_line(name, a, n=70):
    return f"- `{name}` (dB per 10 ms): " + " ".join(f"{v:.0f}" for v in env10(a, n))


# ── Reels ────────────────────────────────────────────────────────────────────
def mix(events, length_s):
    """events: (time s, audio, gain)."""
    out = np.zeros((int(length_s * SR) + 1, 2), np.float32)
    for t, a, g in events:
        i = int(t * SR)
        j = min(len(out), i + len(a))
        out[i:j] += a[: j - i] * g
    return out


def write_reels(here, cur, got):
    reels, audio = [], {}

    def reel(name, what, events, length):
        audio[name] = mix([e[:3] for e in events], length)
        reels.append((name, what, "; ".join(f"{e[0]:.2f}s {e[3]}" for e in events)))

    w, s = got[("windup", CHOSEN["windup"])], got[("whoosh", CHOSEN["whoosh"])]
    for sound, old in (("land", "stomp"), ("floor", "stomp")):
        for cid in ["current"] + [c for k, c in C if k == sound]:
            hit = cur[old] if cid == "current" else got[(sound, cid)]
            label = f"{IN_USE[old]} (in use)" if cid == "current" else f"{sound} {cid}"
            reel(f"{sound}_whoosh_hit__{cid}", "the whoosh, then the hit 60 ms in",
                 [(0.0, s, 1.0, "whoosh"), (0.06, hit, 1.0, label)], 1.2)
            reel(f"{sound}_strike__{cid}", "windup, whoosh as it ends, the hit as the swing peaks",
                 [(0.0, w, 1.0, "windup"), (0.42, s, 1.0, "whoosh"), (0.60, hit, 1.0, label)], 1.9)
    for cid in [c for k, c in C if k == "kick"]:
        reel(f"kick__{cid}", "windup, whoosh, the hoof kick; then a kick at the floor alone",
             [(0.0, w, 1.0, "windup"), (0.42, s, 1.0, "whoosh"), (0.60, got[("kick", cid)], 1.0, f"kick {cid}"),
              (1.6, got[("kick", cid)], 1.0, "kick")], 2.4)
    for cid in [c for k, c in C if k == "portal"]:
        reel(f"portal__{cid}", "both portals ease open (0.0 s by the giant, 0.25 s by the player), windup, whoosh, land",
             [(0.0, got[("portal", cid)], 1.0, f"portal {cid}"), (0.25, got[("portal", cid)], 0.8, "far portal"),
              (0.55, w, 1.0, "windup"), (0.97, s, 1.0, "whoosh"),
              (1.15, got[("land", CHOSEN["land"])], 1.0, "land (chosen)")], 2.2)
    for cid in [c for k, c in C if k == "whoosh"]:
        reel(f"whoosh_miss__{cid}", "a miss in the air: windup, whoosh, nothing; then a strike that floors",
             [(0.0, w, 1.0, "windup"), (0.42, got[("whoosh", cid)], 1.0, f"whoosh {cid}"),
              (1.3, got[("whoosh", cid)], 1.0, "whoosh"), (1.48, got[("floor", CHOSEN["floor"])], 1.0, "floor (chosen)")], 2.4)
    reel("fight_sequence", "the chosen set: open, windup, land; open, windup, floor; kick; whoosh miss",
         [(0.0, got[("portal", CHOSEN["portal"])], 1.0, "portal"), (0.45, w, 1.0, "windup"),
          (0.87, s, 1.0, "whoosh"), (1.05, got[("land", CHOSEN["land"])], 1.0, "land"),
          (2.2, got[("portal", CHOSEN["portal"])], 1.0, "portal"), (2.65, w, 1.0, "windup"),
          (3.07, s, 1.0, "whoosh"), (3.25, got[("floor", CHOSEN["floor"])], 1.0, "floor"),
          (4.4, s, 1.0, "whoosh"), (4.58, got[("kick", CHOSEN["kick"])], 1.0, "kick"),
          (5.5, w, 1.0, "windup"), (5.92, s, 1.0, "whoosh (miss)")], 6.8)

    (here / "reel").mkdir(exist_ok=True)
    # One gain for every reel, so they keep their levels against each other.
    gain = 0.89 / max(float(np.abs(a).max()) for a in audio.values())
    for name, a in audio.items():
        sf.write(here / "reel" / f"{name}.wav", a * gain, SR)
    return reels


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
        for sound, cid in CHOSEN.items():
            spec = C[(sound, cid)]
            note = (f"GNU-ton's giant blow, {sound}: {spec['note']}. Candidate {cid} of "
                    f"scripts/gnu_ton_blow_candidates.py (same layers and seed).")
            path = ACTIVE / f"{CUE_ID[sound]}.sfx.yaml"
            path.write_text(yaml.safe_dump(recipe(CUE_ID[sound], spec, note), sort_keys=False))
            print(path)
        return

    cur = in_use()
    index = ["# GNU-ton's giant blows: SFX auditions", "",
             "Written by `tools/ambition_sfx_renderer/scripts/gnu_ton_blow_candidates.py` (2026-10-23).",
             "Solos are as rendered (no level matching: each cue's level is its own choice); the reels",
             "share one gain.", "", "## The cues in use", ""]
    index += table([(f"`{v}`", cur[k]) for k, v in IN_USE.items()])
    index += ["", "### Envelopes", ""] + [env_line(IN_USE[k], cur[k]) for k in ("stomp", "snort", "charge")]
    if args.command == "measure":
        print("\n".join(index))
        return

    (here / "recipes").mkdir(parents=True, exist_ok=True)
    paths = []
    for (sound, cid), spec in C.items():
        path = here / "recipes" / f"{cand_id(sound, cid)}.sfx.yaml"
        path.write_text(yaml.safe_dump(recipe(cand_id(sound, cid), spec, spec["note"]), sort_keys=False))
        paths.append(path)
    with ThreadPoolExecutor(JOBS) as pool:
        list(pool.map(lambda q: render(q, here / "out"), paths))
    got = {key: load(here / "out" / cand_id(*key)) for key in C}

    (here / "solo").mkdir(exist_ok=True)
    for key, a in got.items():
        sf.write(here / "solo" / f"{key[0]}__{key[1]}.wav", a, SR)
    for k in ("stomp",):
        sf.write(here / "solo" / f"{k}__current.wav", cur[k], SR)

    index += ["", "## The candidates", "", "`*` marks the one shipped (`CHOSEN`).", ""]
    for sound in dict.fromkeys(s for s, _ in C):
        index += [f"### {sound} -> `{CUE_ID[sound]}`", ""]
        for (s, cid), spec in C.items():
            if s == sound:
                star = "*" if CHOSEN[sound] == cid else ""
                index.append(f"- {star}`solo/{sound}__{cid}.wav`: {spec['note']}")
        index.append("")
        rows = [("stomp (in use)", cur["stomp"])] if sound in ("land", "floor", "kick") else []
        if sound == "portal":
            rows = [("portal.attach", cur["portal_attach"]), ("world.portal.enter", cur["world_portal_enter"])]
        rows += [(("*" if CHOSEN[sound] == cid else "") + cid, got[(s, cid)]) for s, cid in C if s == sound]
        index += table(rows) + [""]
        index += [env_line(cid, got[(s, cid)]) for s, cid in C if s == sound] + [""]

    reels = write_reels(here, cur, got)
    index += ["## The reels", "", "`__current` is the stomp in use, in the same place.", ""]
    for name, what, marks in reels:
        index.append(f"- `reel/{name}.wav` ({what}): {marks}")
    (here / "INDEX.md").write_text("\n".join(index) + "\n")
    print(f"{len(C)} candidates, {len(reels)} reels -> {here}")


if __name__ == "__main__":
    main()
