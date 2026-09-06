from pathlib import Path

import yaml

from ambition_sfx_renderer.schema import load_cue


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "sounds" / "metadata" / "george_booul_vfx_sfx.yaml"


def test_george_booul_vfx_companion_manifest_is_complete_and_unique():
    data = yaml.safe_load(META.read_text(encoding="utf8"))
    assert data["schema"] == "ambition.vfx_sfx_companions.v1"
    assert data["sprite_target"] == "george_booul_vfx"
    assert data["counts"] == {"mapped": 21, "up_b": 5}
    entries = data["companions"]
    assert len(entries) == 21
    assert len({e["animation"] for e in entries}) == 21
    assert len({e["sfx_cue_id"] for e in entries}) == 21


def test_up_b_companions_preserve_authored_recovery_timing():
    data = yaml.safe_load(META.read_text(encoding="utf8"))
    timing = data["up_b_timing"]
    assert timing["move_id"] == "excluded_middle"
    assert timing["windup_ms"] == 180
    assert timing["set_impulse_at_ms"] == 180
    assert timing["nominal_time_to_apex_ms"] == 450
    assert timing["committed_until_ms"] == 1150

    by_anim = {e["animation"]: e for e in data["companions"]}
    assert by_anim["excluded_middle_windup"]["visual_duration_ms"] == 180
    assert by_anim["excluded_middle_launch"]["sync_hint"] == "set_impulse_at_0.18s"
    assert by_anim["excluded_middle_tail"]["audibility"] == "subtle"
    assert by_anim["excluded_middle_tail"]["visual_duration_ms"] == 670


def test_all_companion_recipes_load_and_repeat_manifest_authoring():
    data = yaml.safe_load(META.read_text(encoding="utf8"))
    for entry in data["companions"]:
        path = ROOT / entry["sfx_recipe"]
        assert path.is_file(), path
        cue = load_cue(path)
        assert cue.cue_id == entry["sfx_cue_id"]
        assert round(cue.duration_seconds * 1000) == entry["visual_duration_ms"]
        authored = cue.raw["authoring"]
        assert authored["playback"]["sync_hint"] == entry["sync_hint"]
        assert authored["playback"]["audibility"] == entry["audibility"]
        assert authored["sound_design_family"] == entry["sound_design_family"]
        assert authored["companion_for"]["animation"] == entry["animation"]
