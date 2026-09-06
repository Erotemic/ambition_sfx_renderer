#!/usr/bin/env bash
set -euo pipefail

# Render and draw preview waveforms for the generic_explosions sprite variants.
# This intentionally stays inside the standalone SFX renderer and does not hook
# the cues into the Rust game runtime.

cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH="${PYTHONPATH:-.}"

cues=(
  vfx.explosion.classic_burst
  vfx.explosion.burst_round
  vfx.explosion.shockwave
  vfx.explosion.smoke_burst
  vfx.explosion.starburst
)

for cue in "${cues[@]}"; do
  "$PYTHON_BIN" -m ambition_sfx_renderer render "$cue" --format-policy both --force
  "$PYTHON_BIN" -m ambition_sfx_renderer draw "output/$cue" --title "$cue"
done

if [[ "${1:-}" == "--audit" ]]; then
  "$PYTHON_BIN" -m ambition_sfx_renderer audit output
fi
