# Authored VFX/SFX companions

The detached VFX sprite catalogs have matching procedural SFX recipes under
`sounds/active/vfx/`. The cross-domain authoring map is
`sounds/metadata/vfx_sfx_companions.yaml`.

The map is **content metadata**, not a runtime API. A presentation integration
should resolve cue IDs through generic content data rather than branching on
sprite animation names or character identities.

## Author-owned playback intent

Each companion records:

- the sprite target and animation it accompanies;
- the authored visual duration;
- one-shot versus sustained-loop intent;
- a synchronization hint such as visual start, release, or contact;
- an audibility tier (`primary`, `supporting`, or `subtle`);
- a reusable sound-design family.

`primary` means the cue normally owns the audible punctuation. `supporting`
means it should yield when a stronger gameplay cue already owns the same
moment. `subtle` is intended for sustained visual texture and should yield to
music, ambience, accessibility settings, and density limits.

Loop cue IDs use a `.loop` suffix. Their recipe duration matches one authored
visual cycle so an integration layer has an explicit synchronization period.
The current renderer does not promise sample-perfect seamless loop boundaries;
a runtime may crossfade/retrigger these beds, and a future renderer loop
contract can promote that requirement without changing the content IDs.

The five pre-existing generic explosion sounds remain the companions for the
five `generic_explosions` rows and are referenced, rather than duplicated, by
the cross-domain manifest.
