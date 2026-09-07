"""Fetching a published p(h) artifact by id: the registry, the checksum, the release gate.

Nothing here touches the network. The registry entry under test is monkeypatched to carry the
real digest of the `tiny_artifact` fixture file, and the downloader is injected (or, for the two
tests that exercise the shipped one, `requests.get` is replaced by a fake streaming response).
"""

import hashlib

import pytest

import lfa.artifact.fetch as fetch_module
from lfa.artifact.fetch import (
    ARTIFACTS,
    PLACEHOLDER_SHA256,
    ArtifactNotPublished,
    ChecksumMismatch,
    fetch_artifact,
    list_artifacts,
    sha256_file,
)


@pytest.fixture
def published(monkeypatch, tiny_artifact):
    """A registry entry named "tiny", whose checksum is the fixture file's real digest."""
    _, path = tiny_artifact
    monkeypatch.setitem(ARTIFACTS, "tiny", {
        "model_id": "tiny",
        "url": "https://example.invalid/artifacts-v1/tiny.pt",
        "sha256": sha256_file(path),
        "n_samples_total": 1000,
        "kind": "test fixture",
        "size_mb": 1,
    })
    return path


def copier(source, *, corrupt=False):
    """A downloader that copies ``source`` (or writes junk) and counts its calls."""
    calls = []

    def download(url, path):
        calls.append(url)
        path.write_bytes(b"not an artifact" if corrupt else source.read_bytes())

    download.calls = calls
    return download


# ------------------------------------------------------------------------------- the registry

def test_the_registry_carries_both_published_artifacts():
    assert set(ARTIFACTS) == {"qwen3-0.6b-gmm1543k-int8", "qwen3-0.6b-diagonal"}

    recipe_artifact = ARTIFACTS["qwen3-0.6b-gmm1543k-int8"]
    assert recipe_artifact["model_id"] == "Qwen/Qwen3-0.6B"
    assert recipe_artifact["url"].endswith("/artifacts-v1/qwen3-0.6b-gmm1543k-int8.pt")
    assert recipe_artifact["n_samples_total"] == 1_543_000        # per site, not a cross-site sum
    assert recipe_artifact["size_mb"] == 108
    assert "GMM" in recipe_artifact["kind"]

    floor = ARTIFACTS["qwen3-0.6b-diagonal"]
    assert floor["model_id"] == "Qwen/Qwen3-0.6B"
    assert floor["url"].endswith("/artifacts-v1/qwen3-0.6b-diagonal.pt")
    assert floor["n_samples_total"] == 1_200_000
    assert (floor["kind"], floor["size_mb"]) == ("diagonal (budget floor)", 1)


def test_list_artifacts_names_every_id_and_whether_it_is_published():
    listed = list_artifacts()
    assert [entry["id"] for entry in listed] == sorted(ARTIFACTS)
    for entry in listed:
        assert entry["published"] is (entry["sha256"] != PLACEHOLDER_SHA256)
        assert set(entry) >= {"id", "model_id", "url", "sha256", "n_samples_total", "kind",
                              "size_mb", "published"}


def test_an_unknown_id_lists_the_ones_there_are(tmp_path):
    with pytest.raises(ValueError, match="qwen3-0.6b-gmm1543k-int8"):
        fetch_artifact("qwen3-0.6b-gmm42", tmp_path)


# --------------------------------------------------------------------------- the release gate

def test_fetching_before_release_names_the_artifact_and_the_release_doc(tmp_path, tiny_artifact):
    """Every shipped entry still carries the placeholder digest, so every fetch must refuse."""
    _, path = tiny_artifact
    downloader = copier(path)

    with pytest.raises(ArtifactNotPublished) as excinfo:
        fetch_artifact("qwen3-0.6b-gmm1543k-int8", tmp_path, downloader=downloader)

    assert issubclass(ArtifactNotPublished, RuntimeError)
    message = str(excinfo.value)
    assert "qwen3-0.6b-gmm1543k-int8" in message and "RELEASING.md" in message
    assert downloader.calls == []                       # refused before anything was downloaded
    assert not list(tmp_path.iterdir())


# ------------------------------------------------------------------------------- the download

def test_a_fetch_verifies_the_checksum_and_returns_the_path(tmp_path, published):
    downloader = copier(published)

    out = fetch_artifact("tiny", tmp_path, downloader=downloader)

    assert out == tmp_path / "tiny.pt"
    assert sha256_file(out) == ARTIFACTS["tiny"]["sha256"]
    assert downloader.calls == [ARTIFACTS["tiny"]["url"]]
    assert list(tmp_path.iterdir()) == [out]            # no part file left behind


def test_a_corrupted_download_is_rejected_and_leaves_no_file(tmp_path, published):
    with pytest.raises(ChecksumMismatch, match="tiny"):
        fetch_artifact("tiny", tmp_path, downloader=copier(published, corrupt=True))

    assert issubclass(ChecksumMismatch, RuntimeError)
    assert not list(tmp_path.iterdir())


def test_an_artifact_already_on_disk_is_not_downloaded_again(tmp_path, published):
    first = copier(published)
    fetch_artifact("tiny", tmp_path, downloader=first)

    second = copier(published)
    out = fetch_artifact("tiny", tmp_path, downloader=second)

    assert out == tmp_path / "tiny.pt"
    assert second.calls == []


def test_force_downloads_over_a_file_that_is_already_there(tmp_path, published):
    fetch_artifact("tiny", tmp_path, downloader=copier(published))

    again = copier(published)
    fetch_artifact("tiny", tmp_path, downloader=again, force=True)

    assert again.calls == [ARTIFACTS["tiny"]["url"]]


def test_a_local_file_with_the_wrong_digest_is_reported_not_used(tmp_path, published):
    (tmp_path / "tiny.pt").write_bytes(b"half a download")
    downloader = copier(published)

    with pytest.raises(ChecksumMismatch, match="force"):
        fetch_artifact("tiny", tmp_path, downloader=downloader)

    assert downloader.calls == []                       # the caller decides, not the fetcher
    assert (tmp_path / "tiny.pt").read_bytes() == b"half a download"


# -------------------------------------------------------------------- the shipped downloader

class _FakeResponse:
    def __init__(self, chunks, status_error=None):
        self._chunks, self._status_error = chunks, status_error
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error

    def iter_content(self, chunk_size):
        yield from self._chunks


def test_the_shipped_downloader_streams_the_response_to_the_file(tmp_path, monkeypatch):
    seen = {}

    def fake_get(url, **kwargs):
        seen.update(url=url, **kwargs)
        return _FakeResponse([b"chunk one ", b"chunk two"])

    monkeypatch.setattr(fetch_module.requests, "get", fake_get)
    out = tmp_path / "streamed.pt"

    fetch_module._download_with_requests("https://example.invalid/a.pt", out)

    assert out.read_bytes() == b"chunk one chunk two"
    assert seen["url"] == "https://example.invalid/a.pt"
    assert seen["stream"] is True and seen["timeout"]


def test_the_shipped_downloader_raises_on_an_http_error(tmp_path, monkeypatch):
    boom = RuntimeError("404 Not Found")
    monkeypatch.setattr(fetch_module.requests, "get",
                        lambda url, **kwargs: _FakeResponse([], status_error=boom))

    with pytest.raises(RuntimeError, match="404"):
        fetch_module._download_with_requests("https://example.invalid/a.pt", tmp_path / "x.pt")


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"anchoring on sampled hidden states" * 1000)
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()
