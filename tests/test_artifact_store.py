"""The local store self-generated artifacts are built into and reused from."""
import json
import os

import pytest

import lfa.artifact.store as store
from lfa.selfgen.artifact_corpus import SelfGenOptions


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
    monkeypatch.setattr(store, "checkpoint_sha256", lambda model_id: "a" * 64)
    return tmp_path / "store"


def _fake_build(calls, *, fail=False):
    def build(model_id, out_path, options, *, corpus_path, generate=None, writer=None):
        calls.append(out_path)
        corpus_path.write_text('{"text": "x", "source": "selfgen_raw"}\n')
        manifest = {"corpus_sha256": "e" * 64, "writer_sha256": "a" * 64,
                    "frame": options.frame()}
        corpus_path.with_name(corpus_path.name + ".manifest.json").write_text(json.dumps(manifest))
        if fail:
            raise MemoryError("fit ran out of host memory")
        out_path.write_bytes(b"artifact")
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
    assert first.read_bytes() == b"artifact" and manifest["corpus_sha256"] == "e" * 64
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
