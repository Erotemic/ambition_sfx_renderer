# Patch: parallel render-all and namespaced SFX cues

Apply from the parent Ambition checkout:

```bash
cd /home/joncrall/code/ambition
unzip -o /path/to/ambition_sfx_renderer_overlay_parallel_namespaced.zip
```

Render all active sounds efficiently:

```bash
cd /home/joncrall/code/ambition/tools/ambition_sfx_renderer
uv run python -m ambition_sfx_renderer render-all --jobs auto
```

Useful variants:

```bash
# OGG-only batch render; faster and smaller if you do not need debug WAVs.
uv run python -m ambition_sfx_renderer render-all --jobs auto --no-wav

# Force every cue to regenerate even if the manifest hash is current.
uv run python -m ambition_sfx_renderer render-all --jobs auto --force

# Debug serially.
uv run python -m ambition_sfx_renderer render-all --jobs 1 --fail-fast --force
```

Notes:

- `render-all` now uses `ProcessPoolExecutor` when `--jobs` is greater than 1.
- `--jobs auto` uses `os.cpu_count()` worker processes.
- By default, `render-all` skips cues whose output manifest hash is current.
- The cache key includes the cue YAML and renderer version. Use `--force` after
  editing Faust DSP patches, samples, backend Python code, or dependency versions.
- Existing legacy cue YAML filenames are overwritten in place, but their cue ids
  are now namespaced, e.g. `player.jump`, `player.dash`, `player.blink`.

Added namespaces include:

- `player.*`
- `projectile.fireball.*`
- `enemy.goblin.*`
- `hazard.*`
- `world.*`
- `ui.*`
