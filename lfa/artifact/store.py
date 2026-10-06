"""Where self-generated artifacts are built and found again.

A self-generated artifact costs hours to build and is fixed by three things: the model
checkpoint (by its sha256), the frame (:meth:`SelfGenOptions.artifact_frame`), and this package's
builder. The store keys an entry on the first two, so a second workspace over the same model at
the same frame copies the finished file in rather than building it again, and a build that
stopped part-way resumes in the same entry. ``$LFA_ARTIFACT_STORE`` moves it; the default is
``~/.cache/lfa/artifacts``.

On a miss, an entry published for exactly this model id, checkpoint and frame -- pinned in the
package (:mod:`lfa.artifact.published`) -- is downloaded and verified instead of built;
``rebuild`` always builds here.

An entry is a directory ``<model-slug>-<writer_sha256[:12]>-<frame_sha256[:12]>/`` holding
``artifact.pt``, ``corpus.jsonl`` and its manifest, and ``entry.json`` (model id, frame, documents
asked for, and ``provenance``: ``"built"`` here, or ``"published"`` with the URL and file sha256
of each file it was downloaded from). While a build runs it also holds the durable writer's
``corpus.jsonl.partial`` and ``corpus.jsonl.progress.json``, and ``.lock`` with the building
process's pid: a second build of the same entry refuses rather than interleave its batches into
the same corpus, and a lock whose process is gone is taken over with a warning. A download holds
the same lock and writes ``artifact.partial.pt``, ``corpus.jsonl.download`` and
``corpus.jsonl.manifest.json.download`` until all three are verified; ``artifact.pt`` is renamed
into place last, so in an entry it always means a complete one.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

import torch

try:  # POSIX; elsewhere the lock is the exclusive create alone
    import fcntl
except ImportError:  # pragma: no cover - not reached on the supported platforms
    fcntl = None

from .. import __version__
from ..selfgen.artifact_corpus import SelfGenOptions, frame_sha256, progress_path
from ..selfgen.generate import checkpoint_sha256, generate_texts
from .build import build_artifact_self_generated
from .published import fetch_published, find_published, published_artifacts
from .schema import require_own_artifact

logger = logging.getLogger(__name__)

__all__ = ["STORE_ENV", "StoreLocked", "store_root", "entry_dir", "obtain_self_generated",
           "list_store"]

STORE_ENV = "LFA_ARTIFACT_STORE"

_ARTIFACT = "artifact.pt"
_PARTIAL_ARTIFACT = "artifact.partial.pt"
_CORPUS = "corpus.jsonl"
_MANIFEST = "corpus.jsonl.manifest.json"
_DOWNLOAD = ".download"
_ENTRY_JSON = "entry.json"
_LOCK = ".lock"
_REPLACED = ".replaced-"
_GUARD = ".store.lock"               # a file, never an entry directory: entry names end in hex
# flock errors that mean "this filesystem does not do flock", not "something is wrong"
_FLOCK_UNSUPPORTED = frozenset({errno.ENOSYS, errno.ENOLCK, errno.EBADF, errno.EINVAL,
                                errno.EOPNOTSUPP, errno.ENOTSUP})
_flock_unsupported_logged = False


class StoreLocked(RuntimeError):
    """Raised when another live process is building the same store entry."""


def store_root() -> Path:
    """The store's directory: ``$LFA_ARTIFACT_STORE``, else ``~/.cache/lfa/artifacts``."""
    configured = os.environ.get(STORE_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "lfa" / "artifacts"


def _slug(model_id: str) -> str:
    """The model id as one path component: runs outside ``[A-Za-z0-9._-]`` become ``--``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "--", model_id)[-60:]


def entry_dir(model_id: str, writer_sha256: str, options: SelfGenOptions) -> Path:
    """The entry for this checkpoint at this frame (it need not exist yet)."""
    return store_root() / f"{_slug(model_id)}-{writer_sha256[:12]}-{frame_sha256(options)[:12]}"


def _manifest(corpus: Path) -> dict:
    manifest_file = corpus.with_name(corpus.name + ".manifest.json")
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"The store entry {corpus.parent} has an artifact but no corpus manifest at "
            f"{manifest_file}, so what the artifact was fitted on cannot be recorded. Build the "
            "entry again with rebuild (``lfa init --rebuild``); the old entry is moved aside, "
            "not deleted.")
    return json.loads(manifest_file.read_text())


def _write_entry_json(entry: Path, model_id: str, writer_sha256: str,
                      options: SelfGenOptions, published: dict | None = None) -> None:
    """``published``: the download's record (URLs and file sha256s); ``None`` for a build."""
    record = {"model_id": model_id, "writer_sha256": writer_sha256,
              "frame": options.artifact_frame(), "asked": options.n_raw + options.n_chat,
              "lfa_version": __version__,
              "provenance": "published" if published else "built", **(published or {})}
    tmp = entry / f"{_ENTRY_JSON}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, entry / _ENTRY_JSON)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:          # it exists, under another user
        return True
    except (OverflowError, ValueError):
        return False
    return True


def _holder(lock: Path) -> int | None:
    """The pid recorded in ``lock``, or ``None`` when it is gone or holds no pid."""
    try:
        return int(lock.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _locked(entry: Path, pid: int, then: str) -> StoreLocked:
    return StoreLocked(f"{entry} is being built or downloaded by process {pid} (lock "
                       f"{entry / _LOCK}). {then}")


#: What a refused second build or download of an entry is told to do.
_WAIT = ("Wait for it to finish, then run the same command again: it will reuse the finished "
         "artifact.")


@contextmanager
def _guard(root: Path):
    """Serialise the lock's check-and-take across processes; held for microseconds.

    The exclusive create alone keeps two live builds apart. What it cannot do is take over a
    stale lock safely: two processes that both find it stale would both overwrite it. An
    ``flock`` on ``<store>/.store.lock`` (a regular file opened for writing, which NFS's
    byte-range emulation needs) makes the read, the liveness check and the take one step. A
    filesystem that refuses ``flock`` (Lustre mounted without ``-o flock``, some NFS setups)
    gets the exclusive create alone, with one warning.
    """
    global _flock_unsupported_logged
    if fcntl is None:  # pragma: no cover
        yield
        return
    fd = os.open(root / _GUARD, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as exc:
            if exc.errno not in _FLOCK_UNSUPPORTED:
                raise
            if not _flock_unsupported_logged:
                _flock_unsupported_logged = True
                logger.warning("The filesystem under %s does not support flock (%s); store "
                               "entries are locked by exclusive create alone, so two builds "
                               "that start at the same moment over a stale lock are not kept "
                               "apart.", root, exc.strerror)
        yield
    finally:
        os.close(fd)                 # closing releases the flock


@contextmanager
def _lock(entry: Path):
    """Hold ``<entry>/.lock`` for the duration; refuse while a live process holds it."""
    lock = entry / _LOCK
    entry.parent.mkdir(parents=True, exist_ok=True)
    with _guard(entry.parent):
        entry.mkdir(exist_ok=True)   # under the guard: a rebuild may just have moved it aside
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pid = _holder(lock)
            if pid is not None and _alive(pid):
                raise _locked(entry, pid, _WAIT) from None
            logger.warning("Took over the stale lock %s: process %s, which held it, is no "
                           "longer running. A build it left resumes here; a download starts "
                           "again.", lock, pid)
            fd = os.open(lock, os.O_WRONLY | os.O_TRUNC)
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _move_aside(entry: Path, then: str = "Wait for it to finish before rebuilding; nothing "
                                          "was moved.", *, keep_finished: bool = False) -> bool:
    """Rename ``entry`` aside to ``<entry>.replaced-<timestamp>``; refuse while it is locked.

    With ``keep_finished``, an entry that holds a finished ``artifact.pt`` by the time the guard
    is held is left where it is and ``False`` is returned: another process can finish its build
    between the caller's own check and this one. The holder is read before the artifact is
    looked for, and a build renames its artifact into place before it releases its lock, so a
    build that finishes in between is seen either way. Returns ``True`` once the entry is moved.
    """
    with _guard(entry.parent):
        pid = _holder(entry / _LOCK)
        if keep_finished and (entry / _ARTIFACT).is_file():
            return False
        if pid is not None and _alive(pid):
            raise _locked(entry, pid, then)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        aside = entry.with_name(f"{entry.name}{_REPLACED}{stamp}")
        n = 1
        while aside.exists():
            n += 1
            aside = entry.with_name(f"{entry.name}{_REPLACED}{stamp}-{n}")
        entry.rename(aside)
    logger.info("Moved the previous store entry aside to %s", aside)
    return True


def _record(entry: Path) -> dict:
    try:
        return json.loads((entry / _ENTRY_JSON).read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _reused(artifact: Path, corpus: Path) -> tuple[Path, dict]:
    # Checked before it is handed out, so an entry this release cannot read (one written in
    # another artifact format, by a later release or before a downgrade) is refused here, naming
    # the entry, rather than first at `train`.
    require_own_artifact(torch.load(artifact, map_location="cpu", weights_only=False),
                         f"The stored artifact {artifact}")
    when = time.strftime("%Y-%m-%d", time.localtime(artifact.stat().st_mtime))
    if _record(artifact.parent).get("provenance") == "published":
        logger.info("Reused the self-generated artifact downloaded %s into %s", when,
                    artifact.parent)
    else:
        logger.info("Reused the self-generated artifact built %s from %s", when, artifact.parent)
    return artifact, _manifest(corpus)


def _holds_a_build(entry: Path) -> bool:
    """Whether ``entry`` holds anything of a local build (an empty directory does not)."""
    return entry.is_dir() and any(path.name != _LOCK for path in entry.iterdir())


def _fetch(entry: Path, pin: dict, model_id: str, writer_sha256: str,
           options: SelfGenOptions) -> Path:
    """Download ``pin``'s three files into ``entry`` (held under its lock), verify them, and
    record the entry as published. ``artifact.pt`` is renamed into place last."""
    final = {"manifest": entry / _MANIFEST, "corpus": entry / _CORPUS,
             "artifact": entry / _ARTIFACT}
    staged = {"manifest": entry / (_MANIFEST + _DOWNLOAD), "corpus": entry / (_CORPUS + _DOWNLOAD),
              "artifact": entry / _PARTIAL_ARTIFACT}
    try:
        fetch_published(pin, staged, model_id, options)
        published = {f"{name}_{field}": pin[f"{name}_{field}"] for name in final
                     for field in ("url", "file_sha256")}
        # Provenance first and the artifact last, so `artifact.pt` is never there without its
        # corpus, its manifest and its provenance.
        _write_entry_json(entry, model_id, writer_sha256, options, published)
        for name in ("corpus", "manifest", "artifact"):
            os.replace(staged[name], final[name])
    except BaseException as error:
        for path in (*staged.values(), final["corpus"], final["manifest"], entry / _ENTRY_JSON):
            path.unlink(missing_ok=True)
        if isinstance(error, KeyboardInterrupt):
            logger.info("Interrupted: the download into %s was discarded, and running the same "
                        "`lfa init` command again starts it again.", entry)
        raise
    logger.info("Published self-generated artifact stored at %s", final["artifact"])
    return final["artifact"]


def _remove_if_empty(entry: Path) -> None:
    """Remove ``entry`` when nothing is in it: what a refused download leaves behind."""
    with _guard(entry.parent):       # an empty entry under the guard is held by no one
        try:
            entry.rmdir()
        except OSError:              # not empty (another process took it), or already gone
            pass


def obtain_self_generated(model_id: str, options: SelfGenOptions, *, rebuild: bool = False,
                          generate=generate_texts, writer=None,
                          published: list[dict] | None = None) -> tuple[Path, dict]:
    """The store's artifact for this checkpoint at this frame, fetched or built if needed.

    A finished entry is returned as it is, once it is known to be one this release reads
    (:func:`~lfa.artifact.schema.require_own_artifact`). On a miss, an artifact published for
    exactly this model id, checkpoint sha256 and frame (:mod:`lfa.artifact.published`) is
    downloaded into the entry under its lock -- the artifact, its corpus and the manifest -- and
    verified (each file's size and sha256, the format contract, a meta block naming this model and
    frame, the corpus hashing to the meta's ``corpus_sha256``, a manifest that agrees) before
    ``artifact.pt`` is renamed into place; a download that fails
    any of that is refused, and nothing is left in the store. An unfinished local build in the
    entry is first moved aside, as ``rebuild`` moves an entry.

    With no published artifact for it, the build runs in the entry under its lock: a partial
    corpus resumes at its next batch, a complete corpus whose fit failed is fitted without
    generating again (the durable writer, :func:`write_artifact_corpus`, does both), and an empty
    entry is built from scratch. The artifact is written as ``artifact.partial.pt`` and renamed
    only once complete, so ``artifact.pt`` in an entry always means a finished artifact.

    Args:
        rebuild: move any existing entry aside to ``<entry>.replaced-<timestamp>`` first and
            build afresh, here; never downloads. A finished artifact is never deleted.
        generate, writer: passed to :func:`build_artifact_self_generated`.
        published: the pins to look the entry up in; ``None`` reads the list this package ships
            (:func:`~lfa.artifact.published.published_artifacts`).

    Returns:
        ``(entry / "artifact.pt", the corpus manifest)``, built or published alike.

    Raises:
        StoreLocked: another live process is building or downloading this entry.
        lfa.artifact.schema.ForeignArtifact: the finished entry is in an artifact format this
            release does not read; ``rebuild`` builds it afresh.
        lfa.artifact.published.PublishedArtifactUnavailable: a published artifact for this entry
            could not be downloaded or failed verification; ``rebuild`` builds it here instead.
    """
    writer_sha256 = checkpoint_sha256(model_id)
    entry = entry_dir(model_id, writer_sha256, options)
    if rebuild and entry.exists():
        _move_aside(entry)
    artifact = entry / _ARTIFACT
    corpus = entry / _CORPUS
    if artifact.is_file():
        return _reused(artifact, corpus)
    pin = None
    if not rebuild:
        pins = published_artifacts() if published is None else published
        pin = find_published(model_id, writer_sha256, frame_sha256(options), pins)
    if pin is not None:
        if _holds_a_build(entry):
            # The published artifact replaces an unfinished local build, which is kept aside;
            # a live build or download of the entry refuses here, and one that finished since
            # the check above is reused rather than moved.
            if not _move_aside(entry, then=_WAIT, keep_finished=True):
                return _reused(artifact, corpus)
        try:
            with _lock(entry):
                if artifact.is_file():   # another process finished between the check and lock
                    return _reused(artifact, corpus)
                return _fetch(entry, pin, model_id, writer_sha256, options), _manifest(corpus)
        except BaseException:
            _remove_if_empty(entry)
            raise
    with _lock(entry):
        if artifact.is_file():       # another process finished between the check and the lock
            return _reused(artifact, corpus)
        _write_entry_json(entry, model_id, writer_sha256, options)
        partial = entry / _PARTIAL_ARTIFACT
        try:
            build_artifact_self_generated(model_id, partial, options, corpus_path=corpus,
                                          generate=generate, writer=writer)
        except KeyboardInterrupt:
            # The one place that knows which entry was being built; the lock is released by
            # `_lock` on the way out, and the interrupt goes on up so the command still stops.
            logger.info("Interrupted: the build is kept in the store entry %s, and running the "
                        "same `lfa init` command again resumes it at its next batch.", entry)
            raise
        os.replace(partial, artifact)
    logger.info("Self-generated artifact stored at %s", artifact)
    return artifact, _manifest(corpus)


def _entry_size_mb(entry: Path) -> int:
    total = 0
    for f in entry.rglob("*"):
        try:
            total += f.stat().st_size if f.is_file() else 0
        except FileNotFoundError:    # a running build renamed it away
            pass
    return round(total / 1e6)      # decimal MB, as every size the package prints


def _state(entry: Path, record: dict) -> str:
    if (entry / _ARTIFACT).is_file():
        return "published" if record.get("provenance") == "published" else "built"
    asked = record.get("asked")
    if (entry / _CORPUS).is_file():
        return "corpus complete, not fitted"
    staged = (entry / (_MANIFEST + _DOWNLOAD), entry / (_CORPUS + _DOWNLOAD),
              entry / _PARTIAL_ARTIFACT)
    if not progress_path(entry / _CORPUS).is_file() and any(p.is_file() for p in staged):
        return "downloading"         # a download's files, with no corpus being written
    done = 0
    try:
        progress = json.loads(progress_path(entry / _CORPUS).read_text())
        done = sum(share.get("kept", 0) for share in progress.get("shares", {}).values())
    except (FileNotFoundError, ValueError, AttributeError):
        pass
    return f"in progress: {done}/{asked if asked is not None else '?'} documents"


def list_store() -> list[dict]:
    """One row per store entry (entries moved aside by a rebuild are left out).

    Each row is ``{"path", "model_id", "frame", "built_at", "size_mb", "state"}``: ``built_at``
    is the artifact's modification time (``None`` until there is one; for a published artifact,
    when it was downloaded), ``size_mb`` the whole entry's size on disk (corpus included), and
    ``state`` one of ``"built"`` (here), ``"published"`` (downloaded),
    ``"corpus complete, not fitted"`` (a fit failed; the next build fits without generating),
    ``"downloading"`` (a published artifact's files are being fetched, or a fetching process was
    killed) or ``"in progress: <done>/<asked> documents"``.
    """
    root = store_root()
    if not root.is_dir():
        return []
    rows = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or _REPLACED in entry.name or not any(entry.iterdir()):
            continue                 # an empty directory is no entry
        record = _record(entry)
        artifact = entry / _ARTIFACT
        built_at = (time.strftime("%Y-%m-%d %H:%M", time.localtime(artifact.stat().st_mtime))
                    if artifact.is_file() else None)
        rows.append({"path": entry, "model_id": record.get("model_id"),
                     "frame": record.get("frame"), "built_at": built_at,
                     "size_mb": _entry_size_mb(entry), "state": _state(entry, record)})
    return rows
