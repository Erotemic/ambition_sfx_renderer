# Ambition SFX Renderer

Standalone, data-driven offline sound-effect renderer for short game cues such as
player.jump, player.dash, projectile.fireball.impact, ui.menu.accept, and other namespaced game cues.

This repository is meant to live as its own repo or as a git submodule at:

```text
tools/ambition_sfx_renderer/
```

It intentionally does **not** hook into the Rust game runtime. It renders YAML
recipes into local `output/<cue>/` folders.

## Design goals

- No `src/` layout; package directories live at the repo root.
- YAML recipes define sound cues.
- Heavyweight/high-quality offline Python backends are allowed.
- Runtime-game integration is out of scope for this repo.
- No custom low-quality fallback synth unless it is genuinely useful later.
- Missing/nonessential backends should fail loudly with helpful TODO errors.

## Package split

```text
ambition_sfx_renderer/       # core orchestration, schema, mixing, CLI
ambition_sfx_renderer_gpl/   # GPL/LGPL-heavy backend integrations
sounds/                      # YAML cue recipes
patches/                     # Faust / plugin patch source files
output/                      # generated files, ignored by git
```

The split is deliberately simple: the command-line interface feels like one
tool, but the heavyweight backend integrations are isolated in a clearly named
module.

## Install

Using `uv`:

```bash
cd tools/ambition_sfx_renderer
uv python install 3.11
uv venv --python 3.11
uv pip install -e .
```

The default development interpreter is Python 3.11 (`.python-version` and `setup.sh`).
DawDreamer currently supports Python 3.8+ with published classifiers through 3.12,
so avoiding whatever newest system Python you have installed is intentional.
Override with `PYTHON_VERSION=3.12 ./setup.sh` if you want to try 3.12.


Then render one cue:

```bash
uv run python -m ambition_sfx_renderer render player.dash
```

Render everything under `sounds/active/` in parallel. This is the main batch command:

```bash
uv run python -m ambition_sfx_renderer render-all --jobs auto
```

`render-all` skips cues whose manifest hash is current. Use `--force` after editing DSP patches, samples, Python backend code, or when you simply want to regenerate every file:

```bash
uv run python -m ambition_sfx_renderer render-all --jobs auto --force
```

Use `--jobs 1` to debug one cue at a time or `--fail-fast` to stop after the first failure.

Or use the console script after installing:

```bash
ambition-sfx-renderer render player.dash
ambition-sfx-renderer render-all --jobs auto --force
ambition-sfx-renderer audit output
```

## Outputs

Each cue renders into:

```text
output/<cue>/
  <cue>.wav
  <cue>.ogg
  <cue>.render.json
  regen.sh
```

The WAV is useful for waveform inspection. The OGG is the likely future game
asset. The JSON manifest records the source YAML, renderer version, duration,
peak/RMS, and generated file paths.

## YAML quick example

```yaml
schema: ambition.sfxir.v1
id: player.dash
duration_ms: 170

render:
  sample_rate: 48000
  channels: 2
  seed: 1301

layers:
  - name: tonal_body
    kind: dawdreamer_faust
    dsp: patches/faust/tone.dsp
    gain_db: -13
    parameters:
      /AmbitionTone/freq:
        start: 360
        end: 95
        curve: exp
      /AmbitionTone/gain: -10
    envelope:
      attack_ms: 1
      hold_ms: 22
      release_ms: 120

  - name: game_seed
    kind: pyfxr
    preset: laser
    seed: 1302
    gain_db: -22

postprocess:
  - effect: highpass
    cutoff_hz: 55
  - effect: compressor
    threshold_db: -18
    ratio: 2.5
    attack_ms: 3
    release_ms: 70
  - effect: reverb
    room_size: 0.07
    damping: 0.65
    wet_level: 0.035
    dry_level: 1.0
  - effect: limiter
    threshold_db: -2.0
  - effect: normalize_peak
    target_db: -3.0
```

## Implemented layer kinds

- `sample` — load a WAV/OGG/FLAC layer from disk with `soundfile`.
- `pyfxr` — render a game-SFX seed with `pyfxr` presets or explicit SFX params.
- `dawdreamer_faust` — render a Faust DSP patch through DawDreamer.
- `pyo_patch` — intentionally TODO / `NotImplementedError` for now.

## Effects

Postprocess and per-layer effects use Pedalboard when available:

- `highpass`
- `lowpass`
- `compressor`
- `gain`
- `reverb`
- `chorus`
- `phaser`
- `distortion`
- `delay`
- `limiter`
- `bitcrush`
- `pitch_shift`
- `vst3` / `audio_unit` via `pedalboard.load_plugin`
- `normalize_peak` is handled in core code after plugin/effect processing.

## Path resolution

Paths in YAML are resolved in this order:

1. absolute path as written;
2. relative to the YAML file's directory;
3. relative to the tool repository root.

That means `dsp: patches/faust/tone.dsp` works from any cue file in this repo.

## Licensing note

This repository contains mixed-license FOSS tooling.

- The renderer implementation is separate from files processed by the renderer.
- `ambition_sfx_renderer_gpl/` contains integrations with GPL/LGPL audio libraries.
- `sounds/*.sfx.yaml` are sound recipe data files and are Apache-2.0 unless a nearer license file says otherwise.
- Files written to `output/` are generated artifacts and are not automatically relicensed by the renderer.

Generated audio may also be subject to the licenses of any source samples,
soundfonts, plugins, presets, or other source materials used during rendering.

## Namespacing convention

Cue ids are namespaced by the actor or system that emits them:

```text
player.jump
player.wall_jump
player.footstep.stone.01
projectile.fireball.shoot
projectile.fireball.impact
enemy.goblin.hit
hazard.spike.hit
world.pickup.generic
ui.menu.accept
```

The file name does not have to match the cue id exactly, but the cue id is the
stable name that future runtime integration should use.


## Current overlay notes

Apply this overlay from the main Ambition checkout with:

```bash
cd /home/joncrall/code/ambition
unzip -o ~/Downloads/ambition_sfx_renderer_overlay_duration_policy.zip
```

Render every active cue in parallel:

```bash
cd /home/joncrall/code/ambition/tools/ambition_sfx_renderer
uv run python -m ambition_sfx_renderer render-all --jobs auto
```

The default output policy is now duration-aware:

- cues with duration `<= 0.300s` write `.wav`;
- cues with duration `> 0.300s` write `.ogg`;
- use `--format-policy both` when you explicitly want both debug WAV and OGG;
- use `--format-policy wav` or `--format-policy ogg` to force one format.

Useful commands:

```bash
uv run python -m ambition_sfx_renderer list
uv run python -m ambition_sfx_renderer render player.dash
uv run python -m ambition_sfx_renderer render player.death --force
uv run python -m ambition_sfx_renderer render-all --jobs auto --force
uv run python -m ambition_sfx_renderer render-all --jobs auto --format-policy both
```

Several cues now have longer tails when that makes sense: `player.death`,
`player.respawn`, `player.precision_blink`, `projectile.fireball.impact`,
`world.portal.enter`, `world.checkpoint.activate`, plus loop-ish beds like
`player.fly.loop`, `player.glide.loop`, `projectile.fireball.travel_loop`,
`hazard.wind.gust_loop`, and `world.platform.loop`.
