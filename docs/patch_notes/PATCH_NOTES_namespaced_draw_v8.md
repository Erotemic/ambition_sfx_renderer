# Namespaced cue IDs + waveform draw regen v8

- Canonical player cue YAML filenames are now namespaced, e.g.
  `sounds/active/player.blink.sfx.yaml`.
- `find_cue()` now resolves canonical namespaced cue IDs like
  `player.blink` directly.
- Added `python -m ambition_sfx_renderer draw <audio-or-output-dir>` to write a
  dependency-light SVG waveform.
- Newly generated `output/<cue>/regen.sh` scripts support optional waveform
  drawing:
  - `./regen.sh --draw`
  - `./regen.sh --draw --no-play`
  - `./regen.sh --draw-only`
- `player.blink` was retuned toward a short comical mouth-pop / cheek-pop sound.

After applying this overlay, remove old non-namespaced cue files from the
submodule if they still exist:

```bash
rm -f sounds/active/{blink,dash,death,double_jump,hit,jump,pogo,precision_blink,reset,respawn,slash}.sfx.yaml
```
