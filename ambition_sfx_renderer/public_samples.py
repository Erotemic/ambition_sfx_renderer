"""Public samples: free recordings that the renderer downloads.

``sounds/public_samples.yaml`` is the one list. Each entry pins a URL and the
sha256 of the file. The download directory is not in git, so a new checkout
gets each file from its URL.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ambition_sfx_renderer import __version__
from ambition_sfx_renderer.errors import SfxRenderError
from ambition_sfx_renderer.paths import package_root, sounds_root
from ambition_sfx_renderer.schema import load_yaml

SUPPORTED_SCHEMA = "ambition.public_samples.v1"
# Wikimedia refuses a request that has no descriptive User-Agent.
USER_AGENT = (
    f"ambition-sfx-renderer/{__version__} (https://github.com/Erotemic/ambition_sfx_renderer)"
)
REQUIRED_FIELDS = ("url", "filename", "sha256", "license", "source_page")


@dataclass(frozen=True)
class PublicSample:
    sample_id: str
    url: str
    filename: str
    sha256: str
    license: str
    source_page: str


def manifest_path() -> Path:
    return sounds_root() / "public_samples.yaml"


def download_root() -> Path:
    return package_root() / "public_samples"


def load_manifest(path: Path | None = None) -> dict[str, PublicSample]:
    path = path or manifest_path()
    data = load_yaml(path)
    if data.get("schema") != SUPPORTED_SCHEMA:
        raise SfxRenderError(f"{path}: expected schema {SUPPORTED_SCHEMA!r}")
    samples: dict[str, PublicSample] = {}
    for sample_id, entry in (data.get("samples") or {}).items():
        missing = [field for field in REQUIRED_FIELDS if not entry.get(field)]
        if missing:
            raise SfxRenderError(f"{path}: sample {sample_id!r} is missing {', '.join(missing)}")
        if Path(entry["filename"]).name != entry["filename"]:
            raise SfxRenderError(f"{path}: sample {sample_id!r} filename must not have a directory")
        samples[sample_id] = PublicSample(
            sample_id=sample_id,
            **{field: str(entry[field]) for field in REQUIRED_FIELDS},
        )
    return samples


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_public_sample(
    sample_id: str, *, manifest: Path | None = None, root: Path | None = None
) -> Path:
    """Return the local file for ``sample_id``. Download it when it is absent.

    A file that does not have the pinned sha256 is not used. A local file with
    a different hash is downloaded again; a download with a different hash is
    an error.
    """
    samples = load_manifest(manifest)
    if sample_id not in samples:
        known = ", ".join(sorted(samples)) or "none"
        raise SfxRenderError(f"unknown public sample {sample_id!r}; known samples: {known}")
    sample = samples[sample_id]
    dest = (root or download_root()) / sample.filename
    if dest.exists() and file_sha256(dest) == sample.sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(sample.url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except OSError as ex:
        raise SfxRenderError(
            f"could not download public sample {sample_id!r} from {sample.url}: {ex}. "
            f"Put the file at {dest} to render without a network."
        ) from ex
    got = hashlib.sha256(payload).hexdigest()
    if got != sample.sha256:
        raise SfxRenderError(
            f"public sample {sample_id!r} from {sample.url} has sha256 {got}, "
            f"but {manifest or manifest_path()} pins {sample.sha256}"
        )
    # Parallel renders can download the same sample. Each one writes its own
    # temporary file and then replaces the destination in one step.
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{sample.filename}.")
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(payload)
        # `mkstemp` makes an owner-only file. Other users of a shared checkout
        # must be able to read the sample.
        os.chmod(tmp, 0o644)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return dest
