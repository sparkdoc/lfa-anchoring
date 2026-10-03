"""Published self-generated artifacts: the pinned list, and downloading one into the store.

A self-generated artifact is made once per model checkpoint and frame. When one has been made and
published, :func:`lfa.artifact.store.obtain_self_generated` downloads it into the store on a miss
instead of spending hours of GPU time building the same thing again.

What may be downloaded is pinned in this package, in ``published.json`` beside this module: one
pin per published store entry, keyed exactly as the store keys an entry -- ``model_id`` (the Hub
id), ``writer_sha256`` (:func:`lfa.selfgen.generate.checkpoint_sha256`) and ``frame_sha256``
(:func:`lfa.selfgen.artifact_corpus.frame_sha256`) -- and naming the entry's three files, each by
URL, file sha256 and size:

==================  ==================  =========================  ==========================
file                URL                 sha256 of the file         size in bytes
==================  ==================  =========================  ==========================
``artifact.pt``     ``artifact_url``    ``artifact_file_sha256``   ``artifact_size_bytes``
``corpus.jsonl``    ``corpus_url``      ``corpus_file_sha256``     ``corpus_size_bytes``
its manifest        ``manifest_url``    ``manifest_file_sha256``   ``manifest_size_bytes``
==================  ==================  =========================  ==========================

A ``*_file_sha256`` is the sha256 of the file's bytes. It is not the corpus hash the artifact's
meta and the manifest record as ``corpus_sha256``, which :func:`lfa.selfgen.generate.sha256_text`
computes over the corpus rows' text. A model, a snapshot of its weights or a frame with no pin is
built locally.

A download is accepted only when every file has its pinned size and sha256, the artifact is one
this package reads (:func:`lfa.artifact.schema.load_artifact`) whose meta names the model, the
frame asked for and a corpus, the corpus rows hash to that corpus, and the manifest agrees with
the artifact and the pin. Anything else is refused with :class:`PublishedArtifactUnavailable`,
naming the URL and offering ``--rebuild``.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import re
import urllib.error
import urllib.request
from importlib import resources
from pathlib import Path

from .. import __version__
from ..selfgen.artifact_corpus import SelfGenOptions
from ..selfgen.generate import sha256_text
from .schema import SELF_GENERATED, load_artifact

logger = logging.getLogger(__name__)

__all__ = ["PUBLISHED_LIST", "FILES", "PublishedArtifactUnavailable", "published_artifacts",
           "check_pin", "find_published", "fetch_published"]

#: The pinned list, shipped as package data beside this module.
PUBLISHED_LIST = "published.json"

#: The files a pin names, by the prefix of their pin fields, in the order they are downloaded.
FILES = ("manifest", "corpus", "artifact")

#: Seconds to wait for the connection and for each read before giving up.
TIMEOUT_S = 60

_IDENTITY = ("model_id", "writer_sha256", "frame_sha256")
_REQUIRED = _IDENTITY + tuple(f"{name}_{field}" for name in FILES
                              for field in ("url", "file_sha256", "size_bytes"))
_HEX64 = re.compile(r"[0-9a-f]{64}")
_CHUNK = 1 << 20
_REPORT_ABOVE = 16 * _CHUNK          # smaller files are not worth progress lines
_REBUILD = ("Nothing was stored. Run the same command again to retry the download, or pass "
            "--rebuild to `lfa init` to build the artifact here instead (hours on one GPU).")


class PublishedArtifactUnavailable(RuntimeError):
    """Raised when a pinned published artifact cannot be downloaded, or fails verification."""


def _hex64(value) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def check_pin(pin: dict) -> dict:
    """Return ``pin`` once it has exactly the pin fields, each in the expected form.

    Raises:
        ValueError: naming the first field that is missing, unknown or malformed.
    """
    missing = [field for field in _REQUIRED if field not in pin]
    if missing:
        raise ValueError(f"The published-artifact pin {pin!r} lacks {', '.join(missing)}.")
    unknown = sorted(set(pin) - set(_REQUIRED))
    if unknown:
        raise ValueError(f"The published-artifact pin {pin!r} has unknown fields {unknown}.")
    for field in _REQUIRED:
        value = pin[field]
        if field.endswith("sha256") and not _hex64(value):
            problem = "not 64 lower-case hex digits"
        elif field.endswith("size_bytes") and (type(value) is not int or value <= 0):
            problem = "not a positive integer"
        elif field.endswith("_url") and not str(value).startswith(("https://", "http://")):
            problem = "not an http(s) URL"
        else:
            continue
        raise ValueError(f"The published-artifact pin for {pin['model_id']!r} has {field} "
                         f"{value!r}, {problem}.")
    return pin


def published_artifacts() -> list[dict]:
    """The pinned list this package ships (``lfa/artifact/published.json``), each pin checked."""
    text = resources.files(__package__).joinpath(PUBLISHED_LIST).read_text(encoding="utf-8")
    return [check_pin(pin) for pin in json.loads(text)["artifacts"]]


def find_published(model_id: str, writer_sha256: str, frame_sha256: str,
                   pins: list[dict]) -> dict | None:
    """The pin for exactly this model id, checkpoint and frame, or ``None``."""
    for pin in pins:
        if (pin["model_id"] == model_id and pin["writer_sha256"] == writer_sha256
                and pin["frame_sha256"] == frame_sha256):
            return pin
    return None


def _refuse(pin: dict, url: str, what: str) -> PublishedArtifactUnavailable:
    return PublishedArtifactUnavailable(
        f"The published self-generated artifact for {pin['model_id']} could not be used: "
        f"{url} {what}. {_REBUILD}")


def _download(pin: dict, name: str, dest: Path) -> None:
    """Stream the pin's ``name`` file to ``dest``, checking its size and sha256 against the pin."""
    url, expected = pin[f"{name}_url"], pin[f"{name}_size_bytes"]
    pinned = pin[f"{name}_file_sha256"]
    logger.info("Downloading %s (%.1f MB) from %s", name, expected / 2**20, url)
    request = urllib.request.Request(url, headers={"User-Agent": f"lfa-anchoring/{__version__}"})
    digest = hashlib.sha256()
    received = 0
    next_report = 10
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response, \
                open(dest, "wb") as out:
            while True:
                block = response.read(_CHUNK)
                if not block:
                    break
                received += len(block)
                if received > expected:
                    raise _refuse(pin, url, f"sent more than the pinned {expected} bytes")
                digest.update(block)
                out.write(block)
                percent = received * 100 // expected
                if expected > _REPORT_ABOVE and percent >= next_report:
                    logger.info("Downloaded %d%% (%.0f of %.0f MB)", percent, received / 2**20,
                                expected / 2**20)
                    next_report = (percent // 10 + 1) * 10
    except urllib.error.HTTPError as error:
        error.close()
        raise _refuse(pin, url, f"answered HTTP {error.code} {error.reason}") from None
    except urllib.error.URLError as error:
        raise _refuse(pin, url, f"could not be reached ({error.reason})") from None
    except (OSError, http.client.HTTPException) as error:
        # A reset, a timeout or a body cut short part-way through the download.
        raise _refuse(pin, url, f"failed part-way through the download "
                                f"({type(error).__name__}: {error})") from None
    if received != expected:
        raise _refuse(pin, url, f"sent {received} bytes, and the pin says {expected} bytes")
    found = digest.hexdigest()
    if found != pinned:
        raise _refuse(pin, url, f"sent a file whose sha256 is {found}, and the pin says {pinned}")


def _verify_artifact(pin: dict, path: Path, model_id: str, options: SelfGenOptions) -> dict:
    """Load the downloaded artifact and check its meta against the pin; return the meta."""
    url = pin["artifact_url"]
    try:
        meta = load_artifact(path)["__meta__"]
    except Exception as error:  # noqa: BLE001 - any unreadable file is the same refusal
        raise _refuse(pin, url, f"sent a file this package does not accept as an artifact: "
                                f"{' '.join(str(error).split())}") from None
    frame = options.artifact_frame()
    if meta.get("provenance") != SELF_GENERATED:
        raise _refuse(pin, url, f"sent an artifact whose meta says provenance "
                                f"{meta.get('provenance')!r}, not {SELF_GENERATED!r}")
    if meta.get("model_id") != model_id:
        raise _refuse(pin, url, f"sent an artifact built from {meta.get('model_id')!r}, not "
                                f"{model_id!r}")
    if meta.get("selfgen_frame") != frame:
        raise _refuse(pin, url, f"sent an artifact fitted at the frame "
                                f"{meta.get('selfgen_frame')!r}, not the frame asked for, "
                                f"{frame!r}")
    if not _hex64(meta.get("corpus_sha256")):
        raise _refuse(pin, url, f"sent an artifact whose meta names no corpus (corpus_sha256 "
                                f"{meta.get('corpus_sha256')!r})")
    return meta


def _verify_corpus(pin: dict, path: Path, meta: dict) -> None:
    """The corpus rows hash (as the build hashes them) to the artifact's ``corpus_sha256``."""
    url = pin["corpus_url"]
    try:
        with open(path, encoding="utf-8") as handle:
            found = sha256_text(json.loads(line)["text"] for line in handle)
    except (ValueError, KeyError, TypeError) as error:
        raise _refuse(pin, url, f"sent a corpus that is not one JSON row with a text per line "
                                f"({type(error).__name__}: {error})") from None
    if found != meta["corpus_sha256"]:
        raise _refuse(pin, url, f"sent a corpus whose text hashes to {found}, and the artifact "
                                f"was fitted on the corpus {meta['corpus_sha256']}")


def _verify_manifest(pin: dict, path: Path, meta: dict, model_id: str,
                     options: SelfGenOptions) -> None:
    """The manifest names the artifact's corpus, the pinned writer and model, and the frame."""
    url = pin["manifest_url"]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        recorded = {name: manifest.get("frame", {}).get(name) for name in options.corpus_frame()}
    except (ValueError, AttributeError) as error:
        raise _refuse(pin, url, f"sent a manifest that is not a JSON object with a frame "
                                f"({type(error).__name__}: {error})") from None
    disagreements = []
    if manifest.get("corpus_sha256") != meta["corpus_sha256"]:
        disagreements.append(f"corpus_sha256 {manifest.get('corpus_sha256')!r} (the artifact's "
                             f"is {meta['corpus_sha256']})")
    if manifest.get("writer_sha256") != pin["writer_sha256"]:
        disagreements.append(f"writer_sha256 {manifest.get('writer_sha256')!r} (the pin's is "
                             f"{pin['writer_sha256']})")
    if manifest.get("model_id") != model_id:
        disagreements.append(f"model_id {manifest.get('model_id')!r} (asked {model_id!r})")
    if recorded != options.corpus_frame():
        disagreements.append(f"frame {recorded!r} (asked {options.corpus_frame()!r})")
    if disagreements:
        raise _refuse(pin, url, "sent a manifest that disagrees: " + "; ".join(disagreements))


def fetch_published(pin: dict, staged: dict[str, Path], model_id: str,
                    options: SelfGenOptions) -> dict:
    """Download the pin's three files to ``staged`` (keyed by :data:`FILES`) and verify them.

    Each file's size and sha256 are checked against the pin while it streams; then the artifact
    is loaded and its meta checked (provenance, model, frame, a corpus hash), the corpus rows are
    hashed as the build hashes them and compared with the meta's ``corpus_sha256``, and the
    manifest is checked against the artifact, the pin and the frame. Every staged file is
    removed on any failure, an interrupt included, so a refused download leaves nothing behind.

    Returns:
        The artifact's meta block.

    Raises:
        PublishedArtifactUnavailable: a download failed (network, HTTP status, timeout, a body
            cut short), a size or sha256 is not the pinned one, or the files fail a check above.
            The message names the URL and ``--rebuild``.
    """
    logger.info("Downloading the published self-generated artifact for %s (%.0f MB with its "
                "corpus)", model_id,
                sum(pin[f"{name}_size_bytes"] for name in FILES) / 2**20)
    try:
        for name in FILES:
            _download(pin, name, staged[name])
        meta = _verify_artifact(pin, staged["artifact"], model_id, options)
        _verify_corpus(pin, staged["corpus"], meta)
        _verify_manifest(pin, staged["manifest"], meta, model_id, options)
    except BaseException:
        for path in staged.values():
            path.unlink(missing_ok=True)
        raise
    logger.info("Verified the published artifact %s (sha256 %s) and its corpus %s for %s at "
                "the frame asked for", pin["artifact_url"], pin["artifact_file_sha256"],
                meta["corpus_sha256"], model_id)
    return meta
