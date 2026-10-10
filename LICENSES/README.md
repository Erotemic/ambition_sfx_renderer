# License notes

This repository intentionally separates sound recipe data from renderer implementation code.

- `ambition_sfx_renderer/` is the core orchestration package.
- `ambition_sfx_renderer_gpl/` contains integrations with GPL/LGPL audio backends.
- `sounds/**/*.sfx.yaml` are data files. They are Apache-2.0 unless a nearer license says otherwise.
- `output/` contains generated artifacts and is not committed by default.
- `sounds/public_samples.yaml` lists the third-party recordings that some cues use. Each entry
  states its own license and source. The files are downloaded into `public_samples/` and are
  not committed.

The full legal texts can be added here by the downstream repo maintainer if desired.
