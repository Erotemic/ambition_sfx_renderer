# Effect router v5

Fixes crashes like:

```text
ValueError: unknown pedalboard effect: 'soft_clip'
```

`soft_clip`, `tone_safety`, `deharsh`, `lowpass`, `highpass`, and related
corrective processors are core NumPy/SciPy effects, not Pedalboard plugins.
The effect router now walks YAML effect lists in order and only sends
contiguous non-core chunks to Pedalboard. This both preserves the intended
postprocess order and prevents core-only effect names from reaching the
Pedalboard adapter.

The Pedalboard adapter also defensively ignores core effect names if an older
caller path sends them directly.
