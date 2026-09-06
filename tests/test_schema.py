from pathlib import Path

from ambition_sfx_renderer.schema import find_cue, load_cue


def test_jump_cue_loads():
    """One cue loads end to end: found by id, parsed, with real layers.

    The cue is NAMESPACED. This test asked for a bare ``jump`` until 2026-08-26
    and had been red ever since the cues were namespaced -- all 490 of them
    carry a domain prefix now, so ``find_cue("jump")`` returns None and the
    assertion below fires. Its sibling
    ``test_namespaced_active_cues_are_discoverable`` was asserting the new world
    the whole time; this one was left behind, in a suite the Rust gate never
    runs.
    """
    root = Path(__file__).resolve().parents[1] / "sounds"
    path = find_cue("player.jump", root=root)
    assert path is not None
    spec = load_cue(path)
    assert spec.cue_id == "player.jump"
    assert spec.sample_rate == 48000
    assert spec.layers


def test_namespaced_active_cues_are_discoverable():
    from ambition_sfx_renderer.schema import find_cue, iter_cue_files, load_cue

    paths = iter_cue_files(group="active")
    ids = {load_cue(p).cue_id for p in paths}
    assert "player.jump" in ids
    assert "projectile.fireball.impact" in ids
    assert find_cue("player.jump") is not None
    assert find_cue("projectile.fireball.impact") is not None


def test_auto_format_policy_boundary():
    from pathlib import Path
    from ambition_sfx_renderer.render import select_output_formats
    from ambition_sfx_renderer.schema import load_cue

    # Namespaced, and chosen to STRADDLE the boundary rather than to be short
    # and long in the abstract: the policy splits at 300ms, `player.jump` is
    # 95ms and `player.death` is 430ms. Same pair the bare names meant.
    jump = load_cue(Path("sounds/active/player.jump.sfx.yaml"))
    death = load_cue(Path("sounds/active/player.death.sfx.yaml"))
    assert select_output_formats(jump, format_policy="auto")[:2] == (True, False)
    assert select_output_formats(death, format_policy="auto")[:2] == (False, True)


def test_expanded_namespaced_cues_are_discoverable():
    from ambition_sfx_renderer.schema import find_cue, iter_cue_files, load_cue

    paths = iter_cue_files(group="active")
    ids = {load_cue(p).cue_id for p in paths}
    assert "player.jump" in ids
    assert "projectile.fireball.impact" in ids
    assert "world.platform.loop" in ids
    assert "hazard.wind.gust_loop" in ids
    assert find_cue("player.jump") is not None
    assert find_cue("world.platform.loop") is not None
