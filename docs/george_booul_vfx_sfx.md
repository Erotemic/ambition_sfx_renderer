# George Booul detached VFX/SFX authoring

George Boo'l / George Booul is a classic bedsheet ghost whose gameplay language
is Boolean logic. The detached sprite target is `george_booul_vfx`; procedural
sound companions are namespaced under `vfx.george_booul.*`. The authoritative
cross-domain content map is `sounds/metadata/george_booul_vfx_sfx.yaml`.

These files are **content authoring**, not runtime integration. Presentation
should consume the mapping generically rather than switching on George's name.

## Up-B: `excluded_middle`

The move table gives this effect unusually useful event timing:

- press at 0 ms;
- commanded vertical `Set` impulse at 180 ms;
- approximately 450 ms from impulse to apex under baseline gravity;
- committed recovery through 1150 ms.

The 180 ms `excluded_middle_windup` / `vfx.george_booul.up_b.windup` pair is
therefore an exact gameplay tell, not an approximate animation. The launch pair
belongs on the actual impulse event at 0.18 s. `excluded_middle_ascent` is a
short motion companion after launch; `excluded_middle_gate` is lower-priority
spatial support; `excluded_middle_tail` is intentionally weak so the committed
recovery never sounds or looks like a second boost.

The gravity-up orientation in authoring metadata is semantic. Rotate the whole
effect through the resolved presentation/reference frame rather than drawing
separate rows for alternate gravity directions.

## Mix priority

Every sound recipe declares `primary`, `supporting`, or `subtle` audibility.
Primary cues own gameplay punctuation. Supporting cues should yield to stronger
contact/ability sounds at the same instant. Subtle cues are optional texture and
should be the first suppressed by density, accessibility, or quality policy.

`ghost_afterimage.loop` is the only loop in this pack. The current renderer does
not promise a sample-perfect loop seam, so presentation may crossfade or
retrigger it until a generic seamless-loop contract exists.
