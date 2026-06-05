Story/character SFX design pack r20260605_07

This is a fixed reissue of r20260605_06. It keeps the same 112 authored
story/character/world cue ids, but fixes invalid dialogue blip YAML that had
noise colors accidentally written into top-level duration_ms and numeric
frequency hints written into noise color fields.

No generated WAV/OGG files and no runtime wiring are included.

Render this pack from tools/ambition_sfx_renderer with:

while read -r cue; do
  [ -z "$cue" ] && continue
  uv run python -m ambition_sfx_renderer render "$cue" --format-policy both --force
done < sounds/active/story_character_sfx_r20260605_07_cues.txt
