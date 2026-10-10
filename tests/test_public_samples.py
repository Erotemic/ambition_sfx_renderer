import hashlib

import pytest

from ambition_sfx_renderer.errors import SfxRenderError
from ambition_sfx_renderer.public_samples import ensure_public_sample, load_manifest


def _manifest(tmp_path, payload: bytes, sha256: str | None = None):
    """A manifest whose one sample is a local file, so no network is used."""
    source = tmp_path / "upstream" / "scream.ogg"
    source.parent.mkdir()
    source.write_bytes(payload)
    manifest = tmp_path / "public_samples.yaml"
    manifest.write_text(
        "schema: ambition.public_samples.v1\n"
        "samples:\n"
        "  scream:\n"
        f"    url: {source.as_uri()}\n"
        "    filename: scream.ogg\n"
        f"    sha256: {sha256 or hashlib.sha256(payload).hexdigest()}\n"
        "    license: CC0-1.0\n"
        "    source_page: https://example.invalid/scream\n",
        encoding="utf8",
    )
    return manifest, source


def test_an_absent_sample_is_downloaded_and_then_reused(tmp_path):
    manifest, source = _manifest(tmp_path, b"scream bytes")
    root = tmp_path / "downloads"
    path = ensure_public_sample("scream", manifest=manifest, root=root)
    assert path.read_bytes() == b"scream bytes"
    # The second call has no upstream to read, so it can only use the local file.
    source.unlink()
    assert ensure_public_sample("scream", manifest=manifest, root=root) == path


def test_a_local_file_with_the_wrong_hash_is_replaced(tmp_path):
    manifest, _ = _manifest(tmp_path, b"scream bytes")
    root = tmp_path / "downloads"
    root.mkdir()
    (root / "scream.ogg").write_bytes(b"a truncated download")
    path = ensure_public_sample("scream", manifest=manifest, root=root)
    assert path.read_bytes() == b"scream bytes"


def test_a_download_with_the_wrong_hash_is_refused(tmp_path):
    manifest, _ = _manifest(tmp_path, b"not the pinned file", sha256="0" * 64)
    root = tmp_path / "downloads"
    with pytest.raises(SfxRenderError, match="sha256"):
        ensure_public_sample("scream", manifest=manifest, root=root)
    assert not (root / "scream.ogg").exists()


def test_every_shipped_sample_states_its_license_and_source():
    for sample in load_manifest().values():
        assert sample.license and sample.source_page.startswith("https://")
        assert len(sample.sha256) == 64
