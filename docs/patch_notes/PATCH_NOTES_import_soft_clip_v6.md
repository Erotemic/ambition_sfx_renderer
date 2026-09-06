# Import hotfix v6

Fixes startup errors caused by applying a newer effect router over an older
`audio.py`:

```text
ImportError: cannot import name 'soft_clip' from 'ambition_sfx_renderer.audio'
```

The patch makes `effects.py` self-contained for `soft_clip` and the newer
`peak_normalize(..., only_if_louder=...)` behavior, so it works whether or not
those helpers exist in `audio.py`.

Apply from the Ambition repo root:

```bash
unzip -o ~/Downloads/ambition_sfx_renderer_overlay_import_soft_clip_v6.zip
```

Then retry:

```bash
cd /home/joncrall/code/ambition/tools/ambition_sfx_renderer
uv run python -m ambition_sfx_renderer render-all --jobs auto --force
```
