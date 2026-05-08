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
