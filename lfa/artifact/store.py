"""Where self-generated artifacts are built and found again.

A self-generated artifact costs hours to build and is fixed by three things: the model
checkpoint (by its sha256), the frame (:meth:`SelfGenOptions.artifact_frame`), and this package's
builder. The store keys an entry on the first two, so a second workspace over the same model at
the same frame copies the finished file in rather than building it again, and a build that
stopped part-way resumes in the same entry. ``$LFA_ARTIFACT_STORE`` moves it; the default is
``~/.cache/lfa/artifacts``.

An entry is a directory ``<model-slug>-<writer_sha256[:12]>-<frame_sha256[:12]>/`` holding
``artifact.pt``, ``corpus.jsonl`` and its manifest, and ``entry.json`` (model id, frame, documents
asked for). While a build runs it also holds the durable writer's ``corpus.jsonl.partial`` and
``corpus.jsonl.progress.json``, and ``.lock`` with the building process's pid: a second build of
the same entry refuses rather than interleave its batches into the same corpus, and a lock whose
process is gone is taken over with a warning.
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

try:  # POSIX; elsewhere the lock is the exclusive create alone
    import fcntl
except ImportError:  # pragma: no cover - not reached on the supported platforms
    fcntl = None

from .. import __version__
from ..selfgen.artifact_corpus import SelfGenOptions, frame_sha256, progress_path
from ..selfgen.generate import checkpoint_sha256, generate_texts
from .build import build_artifact_self_generated

logger = logging.getLogger(__name__)

__all__ = ["STORE_ENV", "StoreLocked", "store_root", "entry_dir", "obtain_self_generated",
           "list_store"]

STORE_ENV = "LFA_ARTIFACT_STORE"

_ARTIFACT = "artifact.pt"
_PARTIAL_ARTIFACT = "artifact.partial.pt"
_CORPUS = "corpus.jsonl"
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
                      options: SelfGenOptions) -> None:
    record = {"model_id": model_id, "writer_sha256": writer_sha256,
              "frame": options.artifact_frame(), "asked": options.n_raw + options.n_chat,
              "lfa_version": __version__}
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
    return StoreLocked(f"{entry} is being built by process {pid} (lock {entry / _LOCK}). {then}")


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
                raise _locked(entry, pid, "Wait for it to finish, then run the same command "
                              "again: it will reuse the finished artifact.") from None
            logger.warning("Took over the stale lock %s: process %s, which held it, is no "
                           "longer running. Its build resumes here.", lock, pid)
            fd = os.open(lock, os.O_WRONLY | os.O_TRUNC)
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _move_aside(entry: Path) -> None:
    with _guard(entry.parent):
        pid = _holder(entry / _LOCK)
        if pid is not None and _alive(pid):
            raise _locked(entry, pid, "Wait for it to finish before rebuilding; nothing was "
                          "moved.")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        aside = entry.with_name(f"{entry.name}{_REPLACED}{stamp}")
        n = 1
        while aside.exists():
            n += 1
            aside = entry.with_name(f"{entry.name}{_REPLACED}{stamp}-{n}")
        entry.rename(aside)
    logger.info("Moved the previous store entry aside to %s", aside)


def _reused(artifact: Path, corpus: Path) -> tuple[Path, dict]:
    built = time.strftime("%Y-%m-%d", time.localtime(artifact.stat().st_mtime))
    logger.info("Reused the self-generated artifact built %s from %s", built, artifact.parent)
    return artifact, _manifest(corpus)


def obtain_self_generated(model_id: str, options: SelfGenOptions, *, rebuild: bool = False,
                          generate=generate_texts, writer=None) -> tuple[Path, dict]:
    """The store's artifact for this checkpoint at this frame, built into the store if needed.

    A finished entry is returned as it is. Otherwise the build runs in the entry under its lock:
    a partial corpus resumes at its next batch, a complete corpus whose fit failed is fitted
    without generating again (the durable writer, :func:`write_artifact_corpus`, does both), and
    an empty entry is built from scratch. The artifact is written as ``artifact.partial.pt`` and
    renamed only once complete, so ``artifact.pt`` in an entry always means a finished build.

    Args:
        rebuild: move any existing entry aside to ``<entry>.replaced-<timestamp>`` first and
            build afresh; a finished artifact is never deleted.
        generate, writer: passed to :func:`build_artifact_self_generated`.

    Returns:
        ``(entry / "artifact.pt", the corpus manifest)``.

    Raises:
        StoreLocked: another live process is building this entry.
    """
    writer_sha256 = checkpoint_sha256(model_id)
    entry = entry_dir(model_id, writer_sha256, options)
    if rebuild and entry.exists():
        _move_aside(entry)
    artifact = entry / _ARTIFACT
    corpus = entry / _CORPUS
    if artifact.is_file():
        return _reused(artifact, corpus)
    with _lock(entry):
        if artifact.is_file():       # another process finished between the check and the lock
            return _reused(artifact, corpus)
        _write_entry_json(entry, model_id, writer_sha256, options)
        partial = entry / _PARTIAL_ARTIFACT
        build_artifact_self_generated(model_id, partial, options, corpus_path=corpus,
                                      generate=generate, writer=writer)
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
    return round(total / 2**20)


def _state(entry: Path, asked) -> str:
    if (entry / _ARTIFACT).is_file():
        return "built"
    if (entry / _CORPUS).is_file():
        return "corpus complete, not fitted"
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
    is the artifact's modification time (``None`` until it is built), ``size_mb`` the whole
    entry's size on disk (corpus included), and ``state`` one of ``"built"``,
    ``"corpus complete, not fitted"`` (a fit failed; the next build fits without generating) or
    ``"in progress: <done>/<asked> documents"``.
    """
    root = store_root()
    if not root.is_dir():
        return []
    rows = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or _REPLACED in entry.name:
            continue
        try:
            record = json.loads((entry / _ENTRY_JSON).read_text())
        except (FileNotFoundError, ValueError):
            record = {}
        artifact = entry / _ARTIFACT
        built_at = (time.strftime("%Y-%m-%d %H:%M", time.localtime(artifact.stat().st_mtime))
                    if artifact.is_file() else None)
        rows.append({"path": entry, "model_id": record.get("model_id"),
                     "frame": record.get("frame"), "built_at": built_at,
                     "size_mb": _entry_size_mb(entry), "state": _state(entry, record.get("asked"))})
    return rows
