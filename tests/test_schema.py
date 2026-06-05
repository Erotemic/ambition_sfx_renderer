from pathlib import Path

from ambition_sfx_renderer.schema import find_cue, load_cue


def test_jump_cue_loads():
    root = Path(__file__).resolve().parents[1] / "sounds"
    path = find_cue("jump", root=root)
    assert path is not None
    spec = load_cue(path)
    assert spec.cue_id == "jump"
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

    jump = load_cue(Path("sounds/active/jump.sfx.yaml"))
    death = load_cue(Path("sounds/active/death.sfx.yaml"))
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
