from ambition_sfx_renderer import render


def test_sample_and_renderer_changes_invalidate_cached_audio(tmp_path, monkeypatch):
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"first sample")
    cue = tmp_path / "test.sfx.yaml"
    cue.write_text("""schema: ambition.sfxir.v1
id: fixture.sample
duration_ms: 20
layers:
  - kind: sample
    path: sample.wav
""")
    initial = render.cue_hash(cue)
    sample.write_bytes(b"second sample")
    changed_sample = render.cue_hash(cue)
    assert initial != changed_sample
    monkeypatch.setattr(render, "renderer_fingerprint", lambda: "new renderer")
    assert render.cue_hash(cue) != changed_sample
