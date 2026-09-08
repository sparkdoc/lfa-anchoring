"""Getting a published p(h) artifact onto disk, by id, with the checksum that pins it.

An artifact is the only large thing Layerwise Function Anchoring (LFA) ships, and it is also the
one piece a user cannot check by reading it: it is a few hundred megabytes of second-moment
statistics whose only visible failure mode is a *worse anchor*. A silently truncated download or a
mirror serving the wrong file would not raise anywhere -- the shapes would still be right, the
run would still train, and the preservation number would just be off. So every entry in
:data:`ARTIFACTS` carries a SHA-256, the download is verified against it before the file is put
in place, and a mismatch is an error rather than a warning.

Two artifacts are published for Qwen3-0.6B:

* ``qwen3-0.6b-gmm1543k-int8`` -- the recipe artifact: correlated covariance on the linear sites
  and a K=32 GMM on the MLP sites, blockwise-int8 (~108 MB). This is what the paper's lambda was
  calibrated against; anything else moves the operating point (:meth:`lfa.recipe.Recipe.warnings`).
* ``qwen3-0.6b-diagonal`` -- a ~1 MB diagonal artifact, kept as a budget floor. It is a
  *known-inferior* option, not a cheaper equivalent: a diagonal p(h) misprices the linear sites by
  up to 3-4x, so a run against it needs its own lambda.

``n_samples_total`` is a **per-site** count -- every site sees the same token stream -- and it is
what :func:`lfa.artifact.extend.extend_artifact` needs to weight a new domain against the base
pool when the artifact itself carries no per-block counts (the shipped one predates that field).

Until the release assets exist, every ``sha256`` here is :data:`PLACEHOLDER_SHA256` and every
fetch raises :class:`ArtifactNotPublished` rather than downloading something unverifiable.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable

import requests

logger = logging.getLogger("lfa.artifact.fetch")

__all__ = [
    "ARTIFACTS",
    "PLACEHOLDER_SHA256",
    "ArtifactNotPublished",
    "ChecksumMismatch",
    "list_artifacts",
    "sha256_file",
    "fetch_artifact",
]

#: The value a ``sha256`` carries until the release asset it names has been uploaded.
PLACEHOLDER_SHA256 = "<filled at release>"

_RELEASE_BASE = "https://github.com/sparkdoc/lfa-anchoring/releases/download/artifacts-v1"

#: The published artifacts, by id. ``n_samples_total`` is per site; ``size_mb`` is the download.
ARTIFACTS: dict[str, dict] = {
    "qwen3-0.6b-gmm1543k-int8": {
        "model_id": "Qwen/Qwen3-0.6B",
        "url": f"{_RELEASE_BASE}/qwen3-0.6b-gmm1543k-int8.pt",
        "sha256": "acfbc0ecbae60e5806fe356004508fa26880feef6f948903c9c9959bb8cf4485",
        "n_samples_total": 1_543_040,
        "kind": "correlated-linear + GMM K=32 MLP (int8)",
        "size_mb": 108,
    },
    "qwen3-0.6b-diagonal": {
        "model_id": "Qwen/Qwen3-0.6B",
        "url": f"{_RELEASE_BASE}/qwen3-0.6b-diagonal.pt",
        "sha256": "ae66b9513037abecbfa6ca71bf0db2f9f5e5cf7f36d62e8d0a3e923f4c943522",
        "n_samples_total": 1_200_000,
        "kind": "diagonal (budget floor)",
        "size_mb": 1,
    },
}

#: Read size for hashing and for streaming a download.
_CHUNK = 1 << 20


class ArtifactNotPublished(RuntimeError):
    """Raised when an artifact's release asset does not exist yet (placeholder checksum)."""


class ChecksumMismatch(RuntimeError):
    """Raised when a file's SHA-256 is not the one the registry records for that artifact."""


def list_artifacts() -> list[dict]:
    """Every registry entry as a flat dict -- its ``id``, its fields, and whether it is published.

    Sorted by id, so the listing a user sees is stable.
    """
    return [dict(entry, id=artifact_id, published=entry["sha256"] != PLACEHOLDER_SHA256)
            for artifact_id, entry in sorted(ARTIFACTS.items())]


def sha256_file(path: str | Path) -> str:
    """The SHA-256 of a file, read in chunks so an artifact never has to fit in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_with_requests(url: str, path: Path) -> None:
    """Stream ``url`` into ``path``. The default downloader; injectable for tests and mirrors."""
    with requests.get(url, stream=True, timeout=(10, 300)) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=_CHUNK):
                handle.write(chunk)


def fetch_artifact(
    artifact_id: str,
    dest_dir: str | Path,
    *,
    force: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> Path:
    """Put the artifact ``artifact_id`` in ``dest_dir`` as ``<id>.pt``, checksum-verified.

    A file already at the destination is *verified*, not re-downloaded: an artifact is large and
    a repeated fetch is a common thing to do. It is verified rather than trusted, because the
    usual reason for a stale file is an interrupted download, which leaves a plausible-looking
    prefix behind.

    Args:
        artifact_id: a key of :data:`ARTIFACTS`.
        dest_dir: directory to download into (created if needed).
        force: download again even when a verified copy is already there.
        downloader: ``downloader(url, path)``; defaults to the streaming ``requests`` one.

    Returns:
        The path to the verified file.

    Raises:
        ValueError: no such artifact id (the message lists the ones there are).
        ArtifactNotPublished: the release asset does not exist yet.
        ChecksumMismatch: the downloaded (or already-present) file is not what the registry
            records. A failed download is removed; a pre-existing file is left alone, since
            deleting a user's file on a mismatch is not this function's call.
    """
    entry = ARTIFACTS.get(artifact_id)
    if entry is None:
        raise ValueError(
            f"Unknown artifact id {artifact_id!r}. Available: {', '.join(sorted(ARTIFACTS))}."
        )
    if entry["sha256"] == PLACEHOLDER_SHA256:
        raise ArtifactNotPublished(
            f"Artifact {artifact_id!r} has no published release asset yet: its checksum in the "
            "registry is still the placeholder, so a download could not be verified. See "
            "RELEASING.md for how the assets are uploaded and the checksums filled in. Until "
            "then, build an artifact locally (`lfa build-artifact`), or pass a copy of it as a "
            f"path together with the id it is a copy of: `--artifact <file> --artifact-id "
            f"{artifact_id}`. Pass both -- with the path alone the workspace knows the file only "
            "by its name, warns at every stage that lambda was calibrated elsewhere, and cannot "
            "extend, because the id is where the base sample count comes from."
        )

    # After the two refusals above, never before them: a destination directory created for a
    # download that was then refused is debris that looks like a half-made workspace.
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{artifact_id}.pt"

    if dest.exists() and not force:
        if sha256_file(dest) == entry["sha256"]:
            logger.info("Artifact %s already present at %s (checksum verified)", artifact_id, dest)
            return dest
        raise ChecksumMismatch(
            f"{dest} is not artifact {artifact_id!r}: its SHA-256 does not match the registry's "
            f"{entry['sha256']}. It is most likely an interrupted download. Delete it, or pass "
            "force=True to fetch it again."
        )

    download = downloader or _download_with_requests
    # Downloaded beside the destination and moved into place only once verified, so an
    # interrupted or corrupt fetch never leaves something that looks like an artifact.
    part = dest.with_name(dest.name + ".part")
    logger.info("Fetching artifact %s (~%s MB) from %s", artifact_id, entry["size_mb"],
                entry["url"])
    try:
        download(entry["url"], part)
        digest = sha256_file(part)
        if digest != entry["sha256"]:
            raise ChecksumMismatch(
                f"Downloaded artifact {artifact_id!r} has SHA-256 {digest}, but the registry "
                f"records {entry['sha256']}. The download was discarded; retry, and if it "
                "persists the release asset or the mirror is wrong."
            )
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)

    logger.info("Artifact %s verified and written to %s", artifact_id, dest)
    return dest
