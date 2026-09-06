# Patch: regen.sh --draw variable expansion fix

Fixes generated `regen.sh --draw` scripts passing the literal string
`${PLAY_FILE}` into `ambition_sfx_renderer draw`.

The render script template now preserves `$PLAY_FILE` as a shell variable.  This
patch also overwrites the existing `output/player.blink/regen.sh` so the command
that failed can be retried immediately without first regenerating the cue.
