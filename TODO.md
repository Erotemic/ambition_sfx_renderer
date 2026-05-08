# SFX renderer TODO

Living list of sounds the game wants but the renderer doesn't yet
produce, plus consumer-side wiring gaps that are blocked on either
new content or on engine-side hooks that haven't been added. Maintained
from the parent repo (`/home/joncrall/code/ambition`); the consumer of
these SFX is `crates/ambition_sandbox` via the `.sfxbank` packed by
`tools/ambition_sfx_pack/pack.py`.

When a sound here gets generated, also add a `pub const` entry under
`ids::*` in `crates/ambition_sfx/src/lib.rs` and (if it has a clear
gameplay event) wire it in `crates/ambition_sandbox/src/app.rs::handle_feature_events`
or wherever the event originates.

## Wanted but not yet generated

### Footstep variants
- `player.footstep.grass.{01,02,03}` — outdoor / hub zones
- `player.footstep.wood.{01,02,03}` — interior / dock surfaces
- `player.footstep.water.{01,02,03}` — wading
- `player.footstep.ice.{01,02,03}` — frozen biomes
- `player.footstep.sand.{01,02,03}` — desert / beach biomes
- `player.footstep.snow.{01,02,03}` — soft, muted, with squeak
- `player.footstep.glass.{01,02}` — cracked-glass walking moments

### Pickup / reward variants
- `world.coin.large` / `world.coin.huge` — bigger denominations
  (current `world.coin.pickup` is the standard hit)
- `world.key.pickup` — distinct from generic pickup
- `world.lore.pickup` — note / journal / scroll
- `world.ability.unlock` — multi-layer "you got it!" sting
- `world.heart_container.collect` — max-HP boost (vs the standard
  `world.health.collect` heal)
- `world.upgrade.permanent` — major progression milestone

### Doors / interactables (variants beyond what's installed)
- `world.door.heavy_open` / `world.door.heavy_close` — boss / vault
  doors, stone scrape
- `world.door.locked.rattle` — interaction with locked door before
  the key is acquired
- `world.gate.rise` / `world.gate.fall` — portcullis / lift gate
- `world.lever.engage` / `world.lever.disengage` — chunkier than
  the existing `world.switch.toggle`
- `world.pressure_plate.click_on` / `world.pressure_plate.click_off`
- `world.save_point.activate` — reuse `world.checkpoint.activate` or
  produce a richer one-shot

### Enemies (currently only goblin)
- `enemy.slime.alert` / `.attack` / `.death` / `.hit` / `.split`
- `enemy.skeleton.alert` / `.attack` / `.death` / `.hit` / `.bones_fall`
- `enemy.archer.draw` / `.shoot` / `.hit` / `.death`
- `enemy.mage.cast.charge` / `.cast.release` / `.hit` / `.death`
- `enemy.knight.swing` / `.shield_block` / `.hit` / `.death`
- `enemy.flyer.flap_loop` / `.attack_dive` / `.death`
- Per-enemy footstep variants (heavy / light / shuffling / no-footstep flyers)

### Boss
- `boss.<id>.intro_roar` — used at encounter start
- `boss.<id>.weak_point.hit` — distinct ping when player hits the
  weak spot
- `boss.<id>.phase_transition` — sting on phase change
- `boss.<id>.defeat` — multi-layer death sting + crumble

### Environmental ambient loops
- `ambient.wind.indoor_loop` — hub interior tone
- `ambient.water.river_loop`
- `ambient.fire.torch_loop`
- `ambient.machinery.hum_loop` — basement / lab biome
- `ambient.cave.drip_loop`
- `ambient.crowd.market_loop` — populated hub area

### Spells / abilities
- `ability.ice.charge` / `.release` / `.shatter`
- `ability.lightning.charge` / `.release` / `.arc_loop`
- `ability.heal.cast` / `.bloom`
- `ability.shield.up` / `.down` / `.absorb_hit`
- Ability-impact variants (ice/fire/lightning hitting enemies)

### UI polish
- `ui.confirm.warning` — destructive actions (delete save, etc.)
- `ui.slider.tick` — settings menu volume scrub feedback
- `ui.toggle.on` / `ui.toggle.off`
- `ui.tooltip.appear` — subtle "hint became available" sting
- `ui.notification.quest_complete` — bigger than `ui.save.complete`
- `ui.notification.discovery` — first-time-seen-X chime

### Cutscene stingers
- `cutscene.dramatic_chord.{tense,reveal,defeat}`
- `cutscene.impact.heavy` — for screen-shake punctuation
- `cutscene.transition.dissolve_in` / `.dissolve_out`

### Damage-type variants (player taking specific damage)
- `player.hit.fire`
- `player.hit.ice`
- `player.hit.lightning`
- `player.hit.poison`

## Consumer-side wiring gaps (sounds exist but aren't played in-game yet)

Each of these has a bank entry but no `SfxMessage::Play` emit site in
`crates/ambition_sandbox/`. The blocker is engine-side hook work, not
a missing sound.

### Player movement (need engine state-diff hooks)
- `player.land` — fires when ground contact is freshly acquired this
  frame. Needs `Player::just_landed` (or equivalent diff against the
  previous frame's `on_ground`) and an emit in `app.rs` movement path.
- `player.fast_fall` — when the fast-fall ability activates (currently
  has no SFX emit; check `Player::fast_falling` edge).
- `player.wall_jump` — engine emits `Player::wall_jumped_this_frame`?
  if not, wire one. Distinct from the generic `Jump` cue.
- `player.wall_slide` / `player.wall_cling` — looped contact tones,
  need a "currently sliding" / "currently clinging" flag and a
  start/stop pair on the audio side.
- `player.ledge_grab` — needs `Player::just_grabbed_ledge` edge.
- `player.rebound` — pogo bounce off an enemy (separate from
  `Pogo` typed cue which is the input action).
- `player.run.start` / `player.run.stop` — needs a `Player::running`
  edge from the movement state machine. Footstep timing comes from
  the same place once stride detection lands.
- `player.respawn` — fires alongside `Reset`/`Death`; pick one to
  retire or scope semantically (death vs respawn vs reset are three
  states).
- `player.low_health.pulse` — periodic emit while HP ≤ threshold;
  needs a sandbox-side timer + threshold check.

### Footsteps (need stride timer + surface lookup)
- `player.footstep.{stone,metal,soft}.{01,02}` are in the bank but
  not emitted. Two pieces missing:
  1. Stride timing — emit one footstep every N seconds while the
     player is grounded and moving horizontally faster than a small
     threshold. N derived from horizontal speed (faster = shorter
     interval).
  2. Surface lookup — query the LDtk IntGrid (or a per-tile
     `SurfaceKind` resource) under the player's feet to pick which
     SFX family to play. Random between `_01` and `_02` for variety.
- Note: there are two naming schemes in the bank
  (`player.footstep.metal.01` and `player.footstep_metal_01`). Pick
  one before wiring; the dotted form already has consts in
  `ambition_sfx::ids`.

### Doors / loading zones (need to find the LoadingZone activation path)
- `world.door.open` / `world.door.close` — emit when player triggers
  a `LoadingZone` Door interaction. Currently `LoadingZoneActivation`
  is consumed by room transitions in `app.rs`; the SFX hook should be
  next to that consumer.

### Ladder / climbable
- `player.ladder.grab` — fires when `Player::climbable_contact`
  transitions from `None` to `Some` AND player presses up.
- `player.ladder.climb_loop` — looped while climbing. Same start/stop
  pattern as `wall_slide`.
- Decision: do these as one-shots tied to climb-state edges, or as a
  loop layer that fades in/out with vertical speed?

### Enemies (goblin sounds in bank, no emit sites)
- `enemy.goblin.alert` — when an `EnemyRuntime` first targets the
  player (state edge: idle → alerted).
- `enemy.goblin.attack` — windup edge of attack state.
- `enemy.goblin.hit` — when the enemy takes damage.
- `enemy.goblin.death` — when an enemy dies (currently only fires
  the existing physics-burst VFX).
- `enemy.goblin.jump` / `enemy.goblin.land` — same hooks as the
  player's land, applied to enemy state.
- `enemy.goblin.footstep` — same stride-timer story as the player.
- `enemy.goblin.taunt` — periodic / cooldown-gated emit while
  the enemy is alerted but out of attack range.

### Hazards
- `hazard.acid.splash` / `hazard.lava.splash` — fires when the player
  enters a hazard volume. Hook: `FeatureEvents::player_damage` with
  `mode == Lava` / similar; map mode → SFX in `handle_feature_events`.
- `hazard.spike.hit` — same idea, distinct hazard mode.
- `hazard.wind.gust_loop` — looped per-zone ambient (needs a per-room
  registry of active loops + position-based volume / falloff).

### Projectiles
- `projectile.energy_orb.charge` / `.release` — bind to the existing
  charge ability state machine.
- `projectile.fireball.shoot` / `.travel_loop` / `.bounce` / `.impact`
  — bind to projectile spawn / lifetime events. `travel_loop` needs
  to attach to the projectile entity and despawn with it.
- `projectile.generic.spawn` / `.impact` — fallback for projectiles
  without a typed family.

### UI
- `ui.menu.move` / `.accept` / `.back` / `.error` — bind to
  `MenuInputState` / `MenuSelect` / focus-change events.
- `ui.tab.change` — settings menu tab switch.
- `ui.pause.open` / `.close` — pause menu show/hide.
- `ui.save.complete` — bind to save-write completion (already has a
  TODO entry around save points elsewhere).

### Movement / abilities (existing typed cues live alongside these)
- `player.swim.stroke` — emit per stroke while in water; needs a
  stroke timer in the swim state.
- `player.fly.start` / `player.fly.loop` — bind to fly ability edges.
- `player.glide.start` / `player.glide.loop` — same for glide.
- `player.water.enter` / `player.water.exit` — water-volume edges.
- `player.directional_primary` / `player.directional_special` — bind
  to whatever the directional special move dispatch is.
- `player.attack.charge` — bind to attack-charge state.

### World / progression
- `world.checkpoint.activate` — bind to checkpoint trigger.
- `world.portal.enter` — bind to portal interaction.
- `world.secret.reveal` — bind to hidden-room discovery.
- `world.platform.start` / `.loop` / `.stop` — for moving platforms;
  loop layer attached per active platform with falloff by distance.

## Notes

- Naming convention: dot-separated lowercase, hierarchy = `<family>.<thing>.<variant>`.
  The bank currently has duplicates with underscored variants
  (`player.footstep.metal.01` and `player.footstep_metal_01`). Pick one
  and remove the other from the renderer spec to avoid drift.
- Variants for the same conceptual sound stay as sibling ids;
  gameplay code does the random pick. The bank doesn't model "choose
  between these N" as a first-class concept.
- Keep loop sounds (anything ending `.loop`) under a couple of
  seconds and seamless at the boundaries — `bevy_kira_audio` plays
  them with `looped()` and a non-seamless splice will click each
  cycle.
