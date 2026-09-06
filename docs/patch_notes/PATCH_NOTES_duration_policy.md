# Duration-aware output + expanded cue pass

Apply from the main Ambition checkout:

```bash
cd /home/joncrall/code/ambition
unzip -o ~/Downloads/ambition_sfx_renderer_overlay_duration_policy.zip
```

Main render command:

```bash
cd /home/joncrall/code/ambition/tools/ambition_sfx_renderer
uv run python -m ambition_sfx_renderer render-all --jobs auto
```

Changes:

- Default output format policy is duration-aware:
  - `<= 0.300s` writes WAV.
  - `> 0.300s` writes OGG.
- Added `--format-policy auto|both|wav|ogg`.
- Added `--wav-max-seconds` for the auto policy threshold.
- Lengthened cues where tails/loops are useful.
- Revised existing cue recipes by extending envelopes / adding tail processing.
- Added additional namespaced cues for movement, water, enemies, projectiles,
  hazards, world objects, and UI.
- Bumped renderer version to 0.2.0 so cached manifests invalidate.
