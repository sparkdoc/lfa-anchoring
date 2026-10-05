"""The local store self-generated artifacts are built into and reused from."""
import json
import os

import pytest
import torch

import lfa.artifact.store as store
from lfa.artifact.schema import ARTIFACT_FORMAT, ForeignArtifact, make_meta
from lfa.selfgen.artifact_corpus import SelfGenOptions


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
    monkeypatch.setattr(store, "checkpoint_sha256", lambda model_id: "a" * 64)
    return tmp_path / "store"


def _artifact(format_version=ARTIFACT_FORMAT):
    """The smallest file a store hit is checked against: a meta block this package wrote."""
    return {"__meta__": dict(make_meta("m", 8, 1, ["pre_mlp"], 10, provenance="self-generated"),
                             format_version=format_version)}


def _fake_build(calls, *, fail=False):
    def build(model_id, out_path, options, *, corpus_path, generate=None, writer=None):
        calls.append(out_path)
        corpus_path.write_text('{"text": "x", "source": "selfgen_raw"}\n')
        manifest = {"corpus_sha256": "e" * 64, "writer_sha256": "a" * 64,
                    "frame": options.frame()}
        corpus_path.with_name(corpus_path.name + ".manifest.json").write_text(json.dumps(manifest))
        if fail:
            raise MemoryError("fit ran out of host memory")
        torch.save(_artifact(), out_path)
        return out_path
    return build


def test_the_store_lives_where_the_environment_says(isolated_store):
    assert store.store_root() == isolated_store


def test_a_first_build_lands_in_the_entry_and_a_second_reuses_it(isolated_store, monkeypatch,
                                                                 caplog):
    calls = []
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls))
    options = SelfGenOptions(n_raw=60)
    first, manifest = store.obtain_self_generated("Qwen/Qwen3-0.6B", options)
    assert first == store.entry_dir("Qwen/Qwen3-0.6B", "a" * 64, options) / "artifact.pt"
    assert first.is_file() and manifest["corpus_sha256"] == "e" * 64
    assert json.loads((first.parent / "entry.json").read_text())["model_id"] == "Qwen/Qwen3-0.6B"
    with caplog.at_level("INFO"):
        second, _ = store.obtain_self_generated("Qwen/Qwen3-0.6B", SelfGenOptions(n_raw=60,
                                                                               batch_size=4))
    assert second == first and len(calls) == 1
    assert any("Reused the self-generated artifact" in r.message for r in caplog.records)


def test_a_different_frame_or_checkpoint_is_a_different_entry(isolated_store):
    a = store.entry_dir("m", "a" * 64, SelfGenOptions())
    assert a != store.entry_dir("m", "a" * 64, SelfGenOptions(n_raw=60))
    assert a != store.entry_dir("m", "b" * 64, SelfGenOptions())
    assert a.name.startswith("m-aaaaaaaaaaaa-")


def test_a_failed_fit_keeps_the_corpus_and_the_next_call_fits_without_regenerating(
        isolated_store, monkeypatch):
    calls = []
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls, fail=True))
    with pytest.raises(MemoryError):
        store.obtain_self_generated("m", SelfGenOptions(n_raw=60))
    entry = store.entry_dir("m", "a" * 64, SelfGenOptions(n_raw=60))
    assert (entry / "corpus.jsonl").is_file() and not (entry / "artifact.pt").exists()
    assert not (entry / ".lock").exists()
    assert store.list_store()[0]["state"] == "corpus complete, not fitted"
    # The real builder reuses a complete corpus (Task 1); here the retry just has to be routed
    # to the same entry and corpus path.
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls))
    path, _ = store.obtain_self_generated("m", SelfGenOptions(n_raw=60))
    assert path.is_file() and calls[0] == calls[1] == entry / "artifact.partial.pt"


def test_ctrl_c_during_a_build_names_the_entry_and_releases_the_lock(isolated_store, monkeypatch,
                                                                      caplog):
    """Spec §2.3: an interrupted build prints one line naming the store entry and saying the same
    command resumes it. The interrupt itself still propagates, so the CLI exits 130."""
    def interrupted(model_id, out_path, options, *, corpus_path, generate=None, writer=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(store, "build_artifact_self_generated", interrupted)
    options = SelfGenOptions(n_raw=60)
    entry = store.entry_dir("m", "a" * 64, options)
    with caplog.at_level("INFO"), pytest.raises(KeyboardInterrupt):
        store.obtain_self_generated("m", options)
    lines = [r.getMessage() for r in caplog.records if str(entry) in r.getMessage()]
    assert len(lines) == 1 and "lfa init" in lines[0] and "resume" in lines[0]
    assert not (entry / ".lock").exists()
    assert (entry / "entry.json").is_file()      # the entry is kept for the resume


def test_rebuild_moves_the_old_entry_aside(isolated_store, monkeypatch):
    calls = []
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls))
    store.obtain_self_generated("m", SelfGenOptions(n_raw=60))
    store.obtain_self_generated("m", SelfGenOptions(n_raw=60), rebuild=True)
    assert len(calls) == 2
    assert len(list(isolated_store.glob("*.replaced-*"))) == 1
    assert len(store.list_store()) == 1


def test_a_held_lock_refuses(isolated_store, monkeypatch):
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    entry = store.entry_dir("m", "a" * 64, SelfGenOptions(n_raw=60))
    entry.mkdir(parents=True)
    (entry / ".lock").write_text(str(os.getpid()))           # this process: alive
    with pytest.raises(store.StoreLocked, match=str(os.getpid())):
        store.obtain_self_generated("m", SelfGenOptions(n_raw=60))


def test_a_stale_lock_is_taken_over(isolated_store, monkeypatch, caplog):
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    entry = store.entry_dir("m", "a" * 64, SelfGenOptions(n_raw=60))
    entry.mkdir(parents=True)
    (entry / ".lock").write_text("999999999")                # no such pid
    with caplog.at_level("WARNING"):
        path, _ = store.obtain_self_generated("m", SelfGenOptions(n_raw=60))
    assert path.is_file() and any("stale" in r.message for r in caplog.records)


def test_list_store_shows_a_build_in_progress(isolated_store):
    entry = store.entry_dir("m", "a" * 64, SelfGenOptions(n_raw=60))
    entry.mkdir(parents=True)
    (entry / "corpus.jsonl.progress.json").write_text(json.dumps(
        {"shares": {"selfgen_raw": {"kept": 20}, "selfgen_chatfmt": {"kept": 0}}}))
    (entry / "entry.json").write_text(json.dumps(
        {"model_id": "m", "frame": SelfGenOptions(n_raw=60).artifact_frame(), "asked": 60}))
    [row] = store.list_store()
    assert row["state"] == "in progress: 20/60 documents"


@pytest.mark.parametrize("staged", ["corpus.jsonl.manifest.json.download",
                                    "corpus.jsonl.download", "artifact.partial.pt"])
def test_list_store_shows_a_download_in_progress(isolated_store, staged):
    # A download writes no corpus progress file, so "0/? documents" would say nothing true.
    entry = store.entry_dir("m", "a" * 64, SelfGenOptions(n_raw=60))
    entry.mkdir(parents=True)
    (entry / ".lock").write_text(str(os.getpid()))
    (entry / staged).write_bytes(b"x")
    [row] = store.list_store()
    assert row["state"] == "downloading"


def test_list_store_shows_a_fit_under_way_as_a_build_not_a_download(isolated_store):
    # A local build also writes artifact.partial.pt, but only after its corpus is complete.
    entry = store.entry_dir("m", "a" * 64, SelfGenOptions(n_raw=60))
    entry.mkdir(parents=True)
    (entry / "corpus.jsonl").write_text('{"text": "x"}\n')
    (entry / "artifact.partial.pt").write_bytes(b"x")
    [row] = store.list_store()
    assert row["state"] == "corpus complete, not fitted"


_HOLD_LOCK = """
import sys
from pathlib import Path
import lfa.artifact.store as store
with store._lock(Path(sys.argv[1])):
    print("held", flush=True)
    sys.stdin.read()
"""


def test_a_build_running_in_another_process_refuses_naming_its_lock_and_pid(isolated_store,
                                                                            monkeypatch):
    import subprocess
    import sys
    calls = []
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls))
    entry = store.entry_dir("m", "a" * 64, SelfGenOptions(n_raw=60))
    other = subprocess.Popen([sys.executable, "-c", _HOLD_LOCK, str(entry)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert other.stdout.readline().strip() == "held"
        with pytest.raises(store.StoreLocked) as refused:
            store.obtain_self_generated("m", SelfGenOptions(n_raw=60))
        assert str(entry / ".lock") in str(refused.value)
        assert f"process {other.pid}" in str(refused.value)
        with pytest.raises(store.StoreLocked):
            store.obtain_self_generated("m", SelfGenOptions(n_raw=60), rebuild=True)
        assert calls == [] and not list(isolated_store.glob("*.replaced-*"))
    finally:
        other.stdin.close()
        other.wait(timeout=30)
        other.stdout.close()
    assert not (entry / ".lock").exists()
    path, _ = store.obtain_self_generated("m", SelfGenOptions(n_raw=60))
    assert path.is_file() and len(calls) == 1


def test_an_artifact_finished_while_waiting_for_the_lock_is_reused(isolated_store, monkeypatch):
    calls = []
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls))
    entry = store.entry_dir("m", "a" * 64, SelfGenOptions(n_raw=60))
    real_lock = store._lock

    def lock_after_another_build(e):
        e.mkdir(parents=True, exist_ok=True)
        _fake_build([])("m", e / "artifact.pt", SelfGenOptions(n_raw=60),
                        corpus_path=e / "corpus.jsonl")
        return real_lock(e)

    monkeypatch.setattr(store, "_lock", lock_after_another_build)
    path, manifest = store.obtain_self_generated("m", SelfGenOptions(n_raw=60))
    assert path == entry / "artifact.pt" and calls == [] and manifest["corpus_sha256"] == "e" * 64


def test_the_slug_is_one_path_component_of_at_most_60_characters():
    assert store._slug("Qwen/Qwen3-0.6B") == "Qwen--Qwen3-0.6B"
    assert store._slug("/home/u/my model") == "--home--u--my--model"
    assert len(store._slug("org/" + "x" * 100)) == 60


def test_a_store_whose_filesystem_refuses_flock_still_builds_on_the_exclusive_lock(
        isolated_store, monkeypatch, caplog):
    import errno
    import fcntl

    def no_flock(fd, operation):
        raise OSError(errno.ENOSYS, "Function not implemented")

    calls = []
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls))
    monkeypatch.setattr(fcntl, "flock", no_flock)
    monkeypatch.setattr(store, "_flock_unsupported_logged", False)
    with caplog.at_level("WARNING"):
        path, _ = store.obtain_self_generated("m", SelfGenOptions(n_raw=60))
    assert path.is_file() and len(calls) == 1
    assert any("flock" in r.message for r in caplog.records)
    assert [row["path"] for row in store.list_store()] == [path.parent]


def test_the_guard_file_is_not_listed_as_an_entry(isolated_store, monkeypatch):
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    path, _ = store.obtain_self_generated("m", SelfGenOptions(n_raw=60))
    assert (isolated_store / ".store.lock").is_file()
    assert [row["path"] for row in store.list_store()] == [path.parent]


def test_a_stored_artifact_in_a_format_this_release_does_not_read_is_refused_on_reuse(
        isolated_store, monkeypatch):
    """A store hit is checked before it is handed out: an entry written in another format (by a
    later release, or before a downgrade) is refused naming the entry and `--rebuild`, and a
    rebuild moves it aside and builds afresh."""
    calls = []
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls))
    path, _ = store.obtain_self_generated("m", SelfGenOptions(n_raw=60))
    torch.save(_artifact(ARTIFACT_FORMAT + 1), path)

    with pytest.raises(ForeignArtifact) as refusal:
        store.obtain_self_generated("m", SelfGenOptions(n_raw=60))
    assert str(path.parent) in str(refusal.value) and "--rebuild" in str(refusal.value)
    assert f"uses artifact format {ARTIFACT_FORMAT + 1}," in str(refusal.value)

    rebuilt, _ = store.obtain_self_generated("m", SelfGenOptions(n_raw=60), rebuild=True)
    assert rebuilt == path and len(calls) == 2
    assert torch.load(rebuilt, weights_only=False)["__meta__"]["format_version"] == ARTIFACT_FORMAT


# ------------------------------------------------------------------ published artifacts (fetch)

import functools                                                    # noqa: E402
import hashlib                                                      # noqa: E402
import http.server                                                  # noqa: E402
import socket                                                       # noqa: E402
import threading                                                    # noqa: E402
from pathlib import Path                                            # noqa: E402

from lfa.artifact.published import PublishedArtifactUnavailable     # noqa: E402
from lfa.selfgen.generate import sha256_text                        # noqa: E402

_MODEL = "Qwen/Qwen3-0.6B"
_FILES = {"artifact": "artifact.pt", "corpus": "corpus.jsonl",
          "manifest": "corpus.jsonl.manifest.json"}
_TEXTS = ["the first document", "a second one", "and a third"]


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves the fixture directory and records each path asked for; prints nothing.

    A path under ``/truncated/`` serves that file with its full Content-Length but only half its
    body, then closes the connection: a download cut short.
    """

    requests: list = []

    def do_GET(self):
        self.requests.append(self.path)
        if self.path.startswith("/truncated/"):
            body = (Path(self.directory) / self.path.removeprefix("/truncated/")).read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[:len(body) // 2])
            self.close_connection = True
            return
        super().do_GET()

    def log_message(self, *args):
        pass


@pytest.fixture
def served(tmp_path, monkeypatch):
    """``(directory, base_url, requests)``: an HTTP server on 127.0.0.1 over ``directory``."""
    root = tmp_path / "served"
    root.mkdir()
    for name in ("no_proxy", "NO_PROXY"):                    # never route 127.0.0.1 via a proxy
        monkeypatch.setenv(name, "127.0.0.1,localhost")
    requests = []
    handler = type("Handler", (_Handler,), {"requests": requests})
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(handler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02},
                              daemon=True)
    thread.start()
    try:
        yield root, f"http://127.0.0.1:{server.server_address[1]}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def _published_entry(root, options, *, meta_overrides=None, manifest_overrides=None,
                     texts=None, foreign=False):
    """A published entry's three files in ``root``, consistent unless told otherwise.

    ``texts`` replaces the corpus rows actually written (the meta and manifest keep the hash of
    the real ones).
    """
    corpus_sha256 = sha256_text(_TEXTS)
    rows = [{"text": text, "source": "selfgen_raw"} for text in (texts or _TEXTS)]
    (root / "corpus.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"kind": "artifact-corpus", "model_id": _MODEL, "writer_sha256": "a" * 64,
                "frame": options.frame(), "corpus_sha256": corpus_sha256}
    manifest.update(manifest_overrides or {})
    (root / "corpus.jsonl.manifest.json").write_text(json.dumps(manifest))
    if foreign:
        torch.save({"0_pre_mlp": {"mean": torch.zeros(8)}}, root / "artifact.pt")
    else:
        meta = make_meta(_MODEL, 8, 1, ["pre_mlp"], 10, provenance="self-generated",
                         corpus_sha256=corpus_sha256, selfgen_frame=options.artifact_frame())
        meta.update(meta_overrides or {})
        torch.save({"__meta__": meta}, root / "artifact.pt")
    return root


def _pin(base_url, root, options, **overrides):
    pin = {"model_id": _MODEL, "writer_sha256": "a" * 64,
           "frame_sha256": store.frame_sha256(options)}
    for name, filename in _FILES.items():
        data = (root / filename).read_bytes()
        pin.update({f"{name}_url": f"{base_url}/{filename}",
                    f"{name}_file_sha256": hashlib.sha256(data).hexdigest(),
                    f"{name}_size_bytes": len(data)})
    pin.update(overrides)
    return pin


def _leftovers(entry):
    return sorted(p.name for p in entry.iterdir()) if entry.exists() else []


def test_a_miss_with_a_pinned_entry_downloads_verifies_and_then_reuses_it(
        isolated_store, served, monkeypatch, caplog):
    root, base, requests = served
    calls = []
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls))
    options = SelfGenOptions(n_raw=60)
    pin = _pin(base, _published_entry(root, options), options)

    with caplog.at_level("INFO"):
        path, manifest = store.obtain_self_generated(_MODEL, options, published=[pin])
    entry = store.entry_dir(_MODEL, "a" * 64, options)
    assert path == entry / "artifact.pt"
    assert calls == []
    assert sorted(requests) == sorted(f"/{name}" for name in _FILES.values())
    # Exactly a built entry's layout, byte for byte what was published.
    assert _leftovers(entry) == ["artifact.pt", "corpus.jsonl", "corpus.jsonl.manifest.json",
                                 "entry.json"]
    for filename in _FILES.values():
        assert (entry / filename).read_bytes() == (root / filename).read_bytes()
    assert manifest == json.loads((root / "corpus.jsonl.manifest.json").read_text())
    record = json.loads((entry / "entry.json").read_text())
    assert record["provenance"] == "published"
    for name in _FILES:
        assert record[f"{name}_url"] == pin[f"{name}_url"]
        assert record[f"{name}_file_sha256"] == pin[f"{name}_file_sha256"]
    assert record["model_id"] == _MODEL and record["frame"] == options.artifact_frame()
    assert any(pin["artifact_url"] in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level("INFO"):
        again, again_manifest = store.obtain_self_generated(_MODEL, options, published=[pin])
    assert again == path and len(requests) == 3 and calls == []
    assert again_manifest == manifest
    assert any("Reused the self-generated artifact downloaded" in r.message
               for r in caplog.records)
    [row] = store.list_store()
    assert row["state"] == "published"


def test_a_local_build_records_its_provenance(isolated_store, monkeypatch):
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    path, _ = store.obtain_self_generated("m", SelfGenOptions(n_raw=60), published=[])
    assert json.loads((path.parent / "entry.json").read_text())["provenance"] == "built"
    assert store.list_store()[0]["state"] == "built"


@pytest.mark.parametrize("field, value", [("model_id", "Qwen/Qwen3-1.7B"),
                                          ("writer_sha256", "b" * 64),
                                          ("frame_sha256", "f" * 64)])
def test_a_pin_for_another_model_snapshot_or_frame_is_not_fetched(isolated_store, served,
                                                                  monkeypatch, field, value):
    """Another snapshot of the weights (a different checkpoint sha) is a different artifact."""
    root, base, requests = served
    calls = []
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls))
    options = SelfGenOptions(n_raw=60)
    pin = _pin(base, _published_entry(root, options), options, **{field: value})
    path, _ = store.obtain_self_generated(_MODEL, options, published=[pin])
    assert requests == [] and len(calls) == 1 and path.is_file()


def test_rebuild_never_fetches(isolated_store, served, monkeypatch):
    root, base, requests = served
    calls = []
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls))
    options = SelfGenOptions(n_raw=60)
    pin = _pin(base, _published_entry(root, options), options)
    path, _ = store.obtain_self_generated(_MODEL, options, rebuild=True, published=[pin])
    assert requests == [] and len(calls) == 1
    assert json.loads((path.parent / "entry.json").read_text())["provenance"] == "built"


def _refused(options, pin, *, url, contains):
    entry = store.entry_dir(_MODEL, "a" * 64, options)
    with pytest.raises(PublishedArtifactUnavailable) as refusal:
        store.obtain_self_generated(_MODEL, options, published=[pin])
    message = str(refusal.value)
    assert url in message and "--rebuild" in message and "hours on one GPU" in message
    for needle in contains:
        assert needle in message, message
    assert not entry.exists()               # nothing kept: no file, no empty entry directory
    assert store.list_store() == []
    return message


@pytest.mark.parametrize("name", list(_FILES))
def test_a_file_sha256_mismatch_is_refused_and_nothing_is_kept(isolated_store, served,
                                                               monkeypatch, name):
    root, base, _ = served
    calls = []
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls))
    options = SelfGenOptions(n_raw=60)
    pin = _pin(base, _published_entry(root, options), options,
               **{f"{name}_file_sha256": "0" * 64})
    _refused(options, pin, url=pin[f"{name}_url"], contains=["sha256", "0" * 64])
    assert calls == []                      # never a silent hours-long build instead


def test_a_size_mismatch_is_refused(isolated_store, served, monkeypatch):
    root, base, _ = served
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    options = SelfGenOptions(n_raw=60)
    _published_entry(root, options)
    pin = _pin(base, root, options)
    pin["corpus_size_bytes"] += 7
    _refused(options, pin, url=pin["corpus_url"], contains=["bytes"])


def test_an_http_error_is_refused_naming_the_url(isolated_store, served, monkeypatch):
    root, base, _ = served
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    options = SelfGenOptions(n_raw=60)
    pin = _pin(base, _published_entry(root, options), options,
               artifact_url=f"{base}/no-such-asset.pt")
    _refused(options, pin, url=pin["artifact_url"], contains=["404"])


def test_an_unreachable_host_is_refused_naming_the_url(isolated_store, served, monkeypatch):
    root, base, _ = served
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    with socket.socket() as probe:                 # a port nothing listens on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    options = SelfGenOptions(n_raw=60)
    pin = _pin(base, _published_entry(root, options), options,
               manifest_url=f"http://127.0.0.1:{port}/corpus.jsonl.manifest.json")
    _refused(options, pin, url=pin["manifest_url"], contains=[])


def test_a_download_cut_short_is_refused_and_nothing_is_kept(isolated_store, served,
                                                             monkeypatch):
    root, base, _ = served
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    options = SelfGenOptions(n_raw=60)
    pin = _pin(base, _published_entry(root, options), options,
               artifact_url=f"{base}/truncated/artifact.pt")
    # http.client reports a body cut short either as an error or as a short read; both refuse.
    message = _refused(options, pin, url=pin["artifact_url"], contains=[])
    assert "part-way" in message or f"sent {pin['artifact_size_bytes'] // 2} bytes" in message


def test_an_interrupted_download_keeps_nothing_and_says_it_starts_again(isolated_store, served,
                                                                        monkeypatch, caplog):
    import lfa.artifact.published as published
    root, base, _ = served
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    options = SelfGenOptions(n_raw=60)
    pin = _pin(base, _published_entry(root, options), options)
    real_download = published._download

    def interrupted(pin, name, dest):
        real_download(pin, name, dest)
        if name == "artifact":                   # all three on disk, none verified yet
            raise KeyboardInterrupt
    monkeypatch.setattr(published, "_download", interrupted)
    with caplog.at_level("INFO"), pytest.raises(KeyboardInterrupt):
        store.obtain_self_generated(_MODEL, options, published=[pin])
    assert not store.entry_dir(_MODEL, "a" * 64, options).exists()
    assert store.list_store() == []
    assert any("starts it again" in r.getMessage() for r in caplog.records)


def test_a_download_that_is_not_an_artifact_this_package_built_is_refused(isolated_store, served,
                                                                         monkeypatch):
    root, base, _ = served
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    options = SelfGenOptions(n_raw=60)
    pin = _pin(base, _published_entry(root, options, foreign=True), options)
    _refused(options, pin, url=pin["artifact_url"], contains=["not built by lfa-anchoring"])


def test_a_download_in_another_format_is_refused(isolated_store, served, monkeypatch):
    root, base, _ = served
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    options = SelfGenOptions(n_raw=60)
    _published_entry(root, options, meta_overrides={"format_version": ARTIFACT_FORMAT + 1})
    pin = _pin(base, root, options)
    _refused(options, pin, url=pin["artifact_url"], contains=["artifact format"])


@pytest.mark.parametrize("overrides, needle", [
    ({"selfgen_frame": SelfGenOptions(n_raw=61).artifact_frame()}, "frame"),
    ({"model_id": "Qwen/Qwen3-1.7B"}, "Qwen/Qwen3-1.7B"),
    ({"provenance": None}, "self-generated"),
    ({"corpus_sha256": None}, "names no corpus"),
])
def test_an_artifact_whose_meta_disagrees_is_refused(isolated_store, served, monkeypatch,
                                                     overrides, needle):
    root, base, _ = served
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    options = SelfGenOptions(n_raw=60)
    _published_entry(root, options, meta_overrides=overrides)
    pin = _pin(base, root, options)
    _refused(options, pin, url=pin["artifact_url"], contains=[needle])


def test_a_corpus_that_is_not_the_one_the_artifact_was_fitted_on_is_refused(
        isolated_store, served, monkeypatch):
    """The pin's file hash matches the served corpus, but its text is not the meta's corpus."""
    root, base, _ = served
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    options = SelfGenOptions(n_raw=60)
    _published_entry(root, options, texts=["another text altogether"])
    pin = _pin(base, root, options)
    _refused(options, pin, url=pin["corpus_url"], contains=["hashes to", sha256_text(_TEXTS)])


@pytest.mark.parametrize("overrides, needle", [
    ({"corpus_sha256": "d" * 64}, "corpus_sha256"),
    ({"writer_sha256": "b" * 64}, "writer_sha256"),
    ({"model_id": "Qwen/Qwen3-1.7B"}, "model_id"),
    ({"frame": SelfGenOptions(n_raw=61).frame()}, "frame"),
])
def test_a_manifest_that_disagrees_is_refused(isolated_store, served, monkeypatch, overrides,
                                              needle):
    root, base, _ = served
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    options = SelfGenOptions(n_raw=60)
    _published_entry(root, options, manifest_overrides=overrides)
    pin = _pin(base, root, options)
    _refused(options, pin, url=pin["manifest_url"], contains=[needle])


def test_an_unfinished_local_build_is_moved_aside_for_a_published_entry(isolated_store, served,
                                                                       monkeypatch):
    root, base, requests = served
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    options = SelfGenOptions(n_raw=60)
    entry = store.entry_dir(_MODEL, "a" * 64, options)
    entry.mkdir(parents=True)
    (entry / "corpus.jsonl.partial").write_text('{"text": "x"}\n')
    pin = _pin(base, _published_entry(root, options), options)
    _, manifest = store.obtain_self_generated(_MODEL, options, published=[pin])
    assert len(requests) == 3 and manifest["corpus_sha256"] == sha256_text(_TEXTS)
    assert "corpus.jsonl.partial" not in _leftovers(entry)
    [aside] = isolated_store.glob("*.replaced-*")
    assert (aside / "corpus.jsonl.partial").is_file()


def test_an_artifact_finished_just_before_the_move_aside_is_reused_not_moved(isolated_store,
                                                                            served, monkeypatch):
    # Another process finishes its build after this one saw no artifact.pt and before it moves
    # the entry aside for the published one: the finished artifact is reused, and nothing moves.
    root, base, requests = served
    calls = []
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls))
    options = SelfGenOptions(n_raw=60)
    entry = store.entry_dir(_MODEL, "a" * 64, options)
    entry.mkdir(parents=True)
    (entry / "corpus.jsonl.partial").write_text('{"text": "x"}\n')
    pin = _pin(base, _published_entry(root, options), options)
    real_guard = store._guard
    finished = []

    def guard_after_another_build_finished(directory):
        if not finished:             # the other process finishes once, just before the guard
            finished.append(_fake_build([])(_MODEL, entry / "artifact.pt", options,
                                            corpus_path=entry / "corpus.jsonl"))
        return real_guard(directory)

    monkeypatch.setattr(store, "_guard", guard_after_another_build_finished)
    path, manifest = store.obtain_self_generated(_MODEL, options, published=[pin])
    assert path == entry / "artifact.pt" and path.is_file()
    assert manifest["corpus_sha256"] == "e" * 64
    assert requests == [] and calls == []
    assert not list(isolated_store.glob("*.replaced-*"))


def test_a_download_meeting_a_live_one_waits_rather_than_rebuilding(isolated_store, served,
                                                                    monkeypatch):
    root, base, requests = served
    options = SelfGenOptions(n_raw=60)
    entry = store.entry_dir(_MODEL, "a" * 64, options)
    entry.mkdir(parents=True)
    (entry / ".lock").write_text(str(os.getpid()))           # this process: alive
    (entry / "artifact.partial.pt").write_bytes(b"x")         # a download under way
    pin = _pin(base, _published_entry(root, options), options)
    with pytest.raises(store.StoreLocked) as refused:
        store.obtain_self_generated(_MODEL, options, published=[pin])
    message = str(refused.value)
    assert "built or downloaded" in message and "rebuilding" not in message
    assert "run the same command again" in message
    assert requests == [] and (entry / "artifact.partial.pt").is_file()


def test_the_shipped_pin_list_loads_and_a_malformed_pin_is_refused(monkeypatch):
    import lfa.artifact.published as published
    pins = published.published_artifacts()
    assert isinstance(pins, list)
    for pin in pins:
        published.check_pin(pin)
    good = {"model_id": "m", "writer_sha256": "a" * 64, "frame_sha256": "b" * 64}
    for name in ("artifact", "corpus", "manifest"):
        good.update({f"{name}_url": f"https://example.org/{name}",
                     f"{name}_file_sha256": "c" * 64, f"{name}_size_bytes": 3})
    assert published.check_pin(dict(good)) == good
    with pytest.raises(ValueError, match="corpus_size_bytes"):
        published.check_pin({k: v for k, v in good.items() if k != "corpus_size_bytes"})
    with pytest.raises(ValueError, match="manifest_file_sha256"):
        published.check_pin(dict(good, manifest_file_sha256="not-hex"))
    with pytest.raises(ValueError, match="unknown fields"):
        published.check_pin(dict(good, corpus_sha256="c" * 64))   # the text hash is not a pin field
    # The shipped pins are package data: plain http is refused there. (A list passed to
    # `obtain_self_generated(published=...)` is not checked, which is how the tests here serve
    # from 127.0.0.1 over http.)
    with pytest.raises(ValueError, match="not an https URL"):
        published.check_pin(dict(good, corpus_url="http://example.org/corpus"))
    with pytest.raises(ValueError, match="not an https URL"):
        published.check_pin(dict(good, artifact_url="ftp://example.org/artifact"))


def test_obtain_reads_the_shipped_list_when_none_is_given(isolated_store, monkeypatch):
    seen = []
    monkeypatch.setattr(store, "published_artifacts", lambda: seen.append(1) or [])
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    store.obtain_self_generated("m", SelfGenOptions(n_raw=60))
    assert seen == [1]


def test_list_store_skips_an_empty_directory(isolated_store):
    (isolated_store / "stray-aaaaaaaaaaaa-bbbbbbbbbbbb").mkdir(parents=True)
    assert store.list_store() == []


def test_a_workspace_over_a_published_entry_gets_its_corpus_as_from_a_build(
        isolated_store, served, monkeypatch, tmp_path):
    from lfa.workspace import Workspace
    root, base, _ = served
    options = SelfGenOptions(n_raw=60)
    pin = _pin(base, _published_entry(root, options), options)
    monkeypatch.setattr(store, "published_artifacts", lambda: [pin])
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    workspace = Workspace.init(tmp_path / "ws", _MODEL, "self-generated", selfgen=options)
    artifacts = tmp_path / "ws" / "artifacts"
    assert (artifacts / "v1.pt").read_bytes() == (root / "artifact.pt").read_bytes()
    assert (artifacts / "v1.corpus.jsonl").read_bytes() == (root / "corpus.jsonl").read_bytes()
    assert (artifacts / "v1.corpus.jsonl.manifest.json").is_file()
    assert workspace.state["artifact_id"] == "self-generated:" + sha256_text(_TEXTS)[:12]
    assert workspace.state["artifact_provenance"] == "self-generated"
