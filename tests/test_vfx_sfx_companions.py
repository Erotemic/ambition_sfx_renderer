from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from ambition_sfx_renderer.schema import load_cue


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "sounds" / "metadata" / "vfx_sfx_companions.yaml"


def _manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf8"))


def test_vfx_companion_manifest_is_complete_and_unique() -> None:
    manifest = _manifest()
    companions = manifest["new_companions"]
    assert manifest["schema"] == "ambition.vfx_sfx_companions.v1"
    assert manifest["counts"] == {
        "new": 140,
        "existing_explosions": 5,
        "total_mapped": 145,
    }
    assert len(companions) == 140
    assert len({entry["sfx_cue_id"] for entry in companions}) == 140
    assert len({(entry["sprite_target"], entry["animation"]) for entry in companions}) == 140

    by_catalog = Counter(entry["catalog"] for entry in companions)
    assert by_catalog == {
        "generic_action": 18,
        "generic_world": 18,
        "generic_exotic": 24,
        "carl_stargan": 12,
        "noether": 12,
        "patent_clerk": 14,
        "pca": 14,
        "pirate_admiral": 14,
        "ninja_shadow_oni_leader": 14,
    }


def test_every_vfx_companion_recipe_matches_manifest_authoring() -> None:
    for entry in _manifest()["new_companions"]:
        recipe_path = ROOT / entry["sfx_recipe"]
        assert recipe_path.is_file(), recipe_path
        spec = load_cue(recipe_path)
        assert spec.cue_id == entry["sfx_cue_id"]
        assert round(spec.duration_seconds * 1000) == entry["visual_duration_ms"]

        authoring = spec.raw["authoring"]
        assert authoring["companion_for"] == {
            "sprite_target": entry["sprite_target"],
            "animation": entry["animation"],
            "visual_duration_ms": entry["visual_duration_ms"],
        }
        assert authoring["playback"]["mode"] == entry["playback_mode"]
        assert authoring["playback"]["sync_hint"] == entry["sync_hint"]
        assert authoring["playback"]["audibility"] == entry["audibility"]
        assert authoring["sound_design_family"] == entry["sound_design_family"]

        if entry["visual_loop"]:
            assert entry["playback_mode"] == "loop"
            assert spec.cue_id.endswith(".loop")
        else:
            assert entry["playback_mode"] == "one_shot"
            assert not spec.cue_id.endswith(".loop")


def test_existing_explosion_companions_remain_external_to_new_pack() -> None:
    manifest = _manifest()
    existing = manifest["existing_explosion_companions"]
    assert [entry["animation"] for entry in existing] == [
        "classic_burst",
        "burst_round",
        "shockwave",
        "smoke_burst",
        "starburst",
    ]
    new_ids = {entry["sfx_cue_id"] for entry in manifest["new_companions"]}
    assert not new_ids.intersection(entry["sfx_cue_id"] for entry in existing)
