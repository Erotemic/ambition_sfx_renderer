from pathlib import Path

from ambition_sfx_renderer.schema import iter_cue_files, load_cue


ROOT = Path(__file__).resolve().parents[1]
OILER_DIR = ROOT / "sounds" / "active" / "vfx" / "oiler"

EXPECTED = {
    "chalk_spiral": 504,
    "curve_trace": 520,
    "invariant_loop": 840,
    "convergence_ticks": 414,
    "tolerance_brackets": 400,
    "error_term_collapse": 432,
    "bearing_ping": 308,
    "friction_tick": 252,
    "gauge_sweep": 450,
    "stabilizer_spinup": 720,
    "stabilizer_lock": 336,
    "gate_calibration": 768,
    "brass_spark": 294,
    "wrench_strike": 264,
    "oil_drip": 522,
    "oil_splash": 368,
    "oil_slick": 640,
    "pressure_vent": 540,
    "portal_leak": 840,
    "unit_circle_rotation": 560,
    "oil_geyser_emerge": 432,
    "oil_geyser_stream": 672,
    "oil_geyser_impact": 368,
}
LOOPS = {"invariant_loop", "gate_calibration", "portal_leak", "oil_geyser_stream"}


def test_oiler_sfx_are_auto_discovered_recursively_without_registry_edits():
    discovered = set(iter_cue_files(ROOT / "sounds", group="active"))
    local = set(OILER_DIR.glob("*.sfx.yaml"))
    assert len(local) == 23
    assert {path.resolve() for path in local}.issubset(discovered)


def test_every_oiler_vfx_has_one_matching_valid_cue():
    specs = [load_cue(path) for path in sorted(OILER_DIR.glob("*.sfx.yaml"))]
    assert len(specs) == len(EXPECTED)
    by_animation = {spec.raw["authoring"]["companion_for"]["animation"]: spec for spec in specs}
    assert set(by_animation) == set(EXPECTED)
    for animation, duration_ms in EXPECTED.items():
        spec = by_animation[animation]
        assert round(spec.duration_seconds * 1000) == duration_ms
        assert spec.raw["authoring"]["auto_registration"]["central_registry_edit_required"] is False
        expected_mode = "loop" if animation in LOOPS else "one_shot"
        assert spec.raw["authoring"]["playback"]["mode"] == expected_mode
        assert spec.cue_id.endswith(".loop") == (animation in LOOPS)
