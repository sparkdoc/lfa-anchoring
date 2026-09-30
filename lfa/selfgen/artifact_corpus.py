"""The corpus p(h) is estimated on, written by the model with no input data.

The recorded frame: 2,500 raw documents of up to 2,048 tokens started from the model's bare
document-start token at temperature 1.0 and top-p 1.0, **unfiltered**, seed 42, and nothing else.
The artifact fitted on it at 600k samples per site, K=32, matched an artifact fitted on real text
at every lambda tried and was at least as good at the recipe's lambda -- one model (Qwen3-0.6B),
one seed, one domain.

The chat-format share (``n_chat`` documents started from the bare user-turn header, header kept,
seed 43) is an option outside that frame and off by default: the launcher intended 250 such
documents, but the research run's own audit found the fitted corpus held none of them. Its effect
on the artifact is unmeasured.

The writer is durable: each finished batch is appended to ``<corpus>.partial`` and
``<corpus>.progress.json`` is rewritten, so an interrupted build resumes at the next batch with
the same per-batch seed; a complete corpus with a matching frame and writer is reused without
generating.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .. import __version__
from .generate import (boundary_markers, chat_user_header, checkpoint_sha256, clean_raw,
                       drop_burn_in, generate_texts, load_writer, passes_filters, pick_seed_prefix,
                       sha256_text)

logger = logging.getLogger(__name__)

__all__ = ["SelfGenOptions", "DegenerateCorpus", "CorpusFrameMismatch", "write_artifact_corpus",
           "frame_sha256", "partial_path", "progress_path"]

_GENERATION_FIELDS = ("n_raw", "n_chat", "max_new_tokens", "batch_size", "seed", "chat_seed",
                      "min_chars", "max_repeat_ratio", "burn_in_tokens")
_CORPUS_FIELDS = ("n_raw", "n_chat", "max_new_tokens", "seed", "chat_seed", "min_chars",
                  "max_repeat_ratio", "burn_in_tokens")
_FIT_FIELDS = ("max_samples", "gmm_k", "pca_variance", "reservoir_size")
_SHARES = ("selfgen_raw", "selfgen_chatfmt")


class DegenerateCorpus(ValueError):
    """Raised when the generated corpus is too small or too empty to fit an artifact on."""


class CorpusFrameMismatch(ValueError):
    """Raised when an existing corpus or partial build was written under another frame or writer."""


def partial_path(corpus_path) -> Path:
    """Where a build in progress appends its finished batches: ``<corpus>.partial``."""
    corpus_path = Path(corpus_path)
    return corpus_path.with_name(corpus_path.name + ".partial")


def progress_path(corpus_path) -> Path:
    """Where a build in progress records how far it got: ``<corpus>.progress.json``."""
    corpus_path = Path(corpus_path)
    return corpus_path.with_name(corpus_path.name + ".progress.json")


def frame_sha256(options: "SelfGenOptions") -> str:
    """What identifies a self-generated artifact's frame: the corpus and fit fields, not speed."""
    blob = json.dumps(options.artifact_frame(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


@dataclass
class SelfGenOptions:
    """The generation frame and the fit frame of a self-generated artifact.

    Defaults are the recorded frame.
    """

    # -- generation
    n_raw: int = 2500
    n_chat: int = 0                    # the recorded frame has no chat-format share
    max_new_tokens: int = 2048
    batch_size: int = 32
    seed: int = 42
    chat_seed: int = 43
    min_chars: int = 1                 # unfiltered, as recorded
    max_repeat_ratio: float = 1.0
    burn_in_tokens: int = 0
    min_docs: int = 50
    max_empty_fraction: float = 0.2
    # -- fit
    max_samples: int = 600_000
    gmm_k: int = 32
    pca_variance: float = 0.95
    layer_group_size: int | None = None   # None: the fit chooses the group from host RAM
    reservoir_size: int = 200_000
    device: str = "cuda:0"

    def frame(self) -> dict:
        return {name: getattr(self, name) for name in _GENERATION_FIELDS}

    def corpus_frame(self) -> dict:
        """The fields that shape the corpus (not ``batch_size``: speed, not content frame)."""
        return {name: getattr(self, name) for name in _CORPUS_FIELDS}

    def artifact_frame(self) -> dict:
        """The corpus frame plus the fit fields: what the artifact was estimated at."""
        return {**self.corpus_frame(), **{name: getattr(self, name) for name in _FIT_FIELDS}}

    def build_kwargs(self) -> dict:
        return dict(max_samples=self.max_samples, gmm_k=self.gmm_k,
                    pca_variance=self.pca_variance, layer_group_size=self.layer_group_size,
                    reservoir_size=self.reservoir_size, device=self.device)


def _stop_ids(tokenizer, markers) -> list[int]:
    ids = []
    for marker in markers:
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(token_id, int) and token_id >= 0 and token_id not in ids:
            ids.append(token_id)
    return ids or [tokenizer.eos_token_id]


def _write_share(model, tokenizer, *, prefix: str, keep_prefix: bool, n_docs: int, seed: int,
                 source: str, options: SelfGenOptions, markers, max_empties: float, batch_size: int,
                 state: dict, sink, generate) -> None:
    """Draw from ``prefix`` in batches until ``state["kept"]`` reaches ``n_docs``.

    ``state`` is the share's entry in the progress record (``batch_index``, ``kept``,
    ``empties``); it is advanced in place after each batch and the batch's surviving rows are
    handed to ``sink``, so a resumed build starts at the recorded batch index -- the same
    per-batch seed -- with the empties already counted. A draw that cleans to nothing or fails
    the filters is an empty, and a further draw replaces it. The loop stops short only once
    ``empties`` exceeds ``max_empties``, the share's part of the corpus-wide limit, at which point
    the caller's refusal is certain.
    """
    stop = _stop_ids(tokenizer, markers)
    while state["kept"] < n_docs and state["empties"] <= max_empties:
        size = min(batch_size, n_docs - state["kept"])
        texts = generate(model, tokenizer, [prefix] * size, max_new_tokens=options.max_new_tokens,
                         temperature=1.0, top_p=1.0, stop_token_ids=stop, seed=seed,
                         batch_index=state["batch_index"])
        rows, empties = [], 0
        for text in texts:
            body = drop_burn_in(clean_raw(text, markers), tokenizer, options.burn_in_tokens)
            if not body or not passes_filters(body, min_chars=options.min_chars,
                                              max_repeat_ratio=options.max_repeat_ratio):
                empties += 1
                continue
            rows.append({"text": (prefix + body) if keep_prefix else body, "source": source})
        state["batch_index"] += 1
        state["kept"] += len(rows)
        state["empties"] += empties
        sink(rows)


def _rebuild_remedy(out_path: Path) -> str:
    return (f"Re-run with the frame it was built with, or discard it: `lfa init ... --rebuild` "
            f"(store) or delete {out_path}, {partial_path(out_path)} and "
            f"{progress_path(out_path)}.")


def _check_reusable(manifest: dict, options: SelfGenOptions, writer_sha256: str,
                    out_path: Path) -> None:
    """Refuse a complete corpus written by another writer or under another corpus frame."""
    if manifest.get("writer_sha256") != writer_sha256:
        raise CorpusFrameMismatch(
            f"{out_path} is a complete corpus written by another writer (checkpoint "
            f"{str(manifest.get('writer_sha256'))[:12]}..., this one {writer_sha256[:12]}...). "
            + _rebuild_remedy(out_path))
    recorded = {name: manifest.get("frame", {}).get(name) for name in _CORPUS_FIELDS}
    wanted = options.corpus_frame()
    if recorded != wanted:
        differ = ", ".join(f"{name} {recorded[name]!r} (asked {wanted[name]!r})"
                           for name in _CORPUS_FIELDS if recorded[name] != wanted[name])
        raise CorpusFrameMismatch(
            f"{out_path} is a complete corpus under a different frame: {differ}. "
            + _rebuild_remedy(out_path))


def _load_progress(out_path: Path, options: SelfGenOptions, writer_sha256: str) -> dict:
    """The progress record of a build in progress, or a fresh one (discarding a stray partial)."""
    record_file = progress_path(out_path)
    if record_file.is_file():
        progress = json.loads(record_file.read_text())
        if (progress.get("frame_sha256") != frame_sha256(options)
                or progress.get("writer_sha256") != writer_sha256):
            raise CorpusFrameMismatch(
                f"{partial_path(out_path)} is a partial build under a different frame or writer. "
                "Re-run with the frame it was started with, or discard it: `lfa init ... "
                f"--rebuild` (store) or delete {partial_path(out_path)} and {record_file}.")
        logger.info("Resuming the self-generated corpus at %s from %d documents on disk",
                    out_path, progress["rows_written"])
        if progress["batch_size"] != options.batch_size:
            logger.info("Resuming with the batch size the build started with, %d, not %d: each "
                        "batch is seeded as a unit, so the batch size shapes the documents",
                        progress["batch_size"], options.batch_size)
        return progress
    partial_path(out_path).unlink(missing_ok=True)
    return {
        "frame_sha256": frame_sha256(options),
        "writer_sha256": writer_sha256,
        "batch_size": options.batch_size,
        "rows_written": 0,
        "shares": {share: {"batch_index": 0, "kept": 0, "empties": 0} for share in _SHARES},
    }


def _save_progress(out_path: Path, progress: dict) -> None:
    """Rewrite the progress record atomically: to a temporary file, then renamed over it."""
    record_file = progress_path(out_path)
    tmp = record_file.with_name(record_file.name + ".tmp")
    tmp.write_text(json.dumps(progress, indent=2))
    os.replace(tmp, record_file)


def _truncate_partial(out_path: Path, n: int) -> int:
    """Keep the first ``n`` lines of the partial file: rows past the last progress write go."""
    partial = partial_path(out_path)
    if not partial.is_file():
        lines = []
    else:
        with open(partial, encoding="utf-8") as handle:
            lines = handle.readlines()
    if len(lines) < n:
        raise ValueError(
            f"{partial} holds {len(lines)} rows but {progress_path(out_path)} records {n}; the "
            f"partial build is damaged. Delete {partial} and {progress_path(out_path)} to start "
            "over.")
    if len(lines) > n:                                  # a torn last line is one of these
        logger.info("Dropping %d rows written after the last progress record from %s",
                    len(lines) - n, partial)
        tmp = partial.with_name(partial.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.writelines(lines[:n])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, partial)
    return n


def write_artifact_corpus(model_id: str, out_path, options: SelfGenOptions, *,
                          generate=generate_texts, writer=None) -> dict:
    """Write the unconditional corpus to ``out_path`` (JSONL) and its manifest beside it.

    The build is durable. Each finished batch is appended to :func:`partial_path` and the
    progress record at :func:`progress_path` is rewritten, so an interrupted build resumes at
    the next batch of each share, with the batch size it started with and the same per-batch
    seed, and the empties counted before the interruption still count against the limit. Rows
    appended after the last progress write are dropped on resume. Only a complete build is
    renamed to ``out_path``. A complete corpus already at ``out_path`` whose manifest records the
    same writer and corpus frame is returned as it is, without loading the model; ``batch_size``
    is speed, not frame, so it may differ.

    Args:
        writer: an already-loaded ``(model, tokenizer)``; loaded with :func:`load_writer` otherwise.
        generate: the sampling function (injectable for tests).

    Raises:
        CorpusFrameMismatch: the complete corpus or the partial build at ``out_path`` was written
            under another frame or by another writer.
        DegenerateCorpus: fewer than ``options.min_docs`` documents, or more than
            ``options.max_empty_fraction`` of the documents asked for came out empty. The corpus
            is not written in either case; the partial build is kept, so a knowing retry with a
            looser limit resumes it.
    """
    out_path = Path(out_path)
    writer_sha256 = checkpoint_sha256(model_id)          # before the hours of sampling, not after
    manifest_file = Path(str(out_path) + ".manifest.json")
    if out_path.is_file() and manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text())
        _check_reusable(manifest, options, writer_sha256, out_path)
        progress_path(out_path).unlink(missing_ok=True)  # left by a crash after the rename
        logger.info("Self-generated corpus already complete at %s; not generating again",
                    out_path)
        return manifest
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress = _load_progress(out_path, options, writer_sha256)
    batch_size = progress["batch_size"]
    _truncate_partial(out_path, progress["rows_written"])

    model, tokenizer = writer if writer is not None else load_writer(model_id, options.device)
    prefix = pick_seed_prefix(tokenizer, model)
    markers = boundary_markers(tokenizer, prefix)
    header = chat_user_header(tokenizer)
    if options.n_chat and not header:
        logger.warning("%s has no chat template: the artifact corpus carries no chat-format "
                       "share", model_id)
    n_chat = options.n_chat if header else 0
    asked = options.n_raw + n_chat
    limit = options.max_empty_fraction * max(asked, 1)
    raw_state = progress["shares"]["selfgen_raw"]
    chat_state = progress["shares"]["selfgen_chatfmt"]

    def sink(rows):
        with open(partial_path(out_path), "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        progress["rows_written"] += len(rows)
        _save_progress(out_path, progress)
        done = sum(share["kept"] for share in progress["shares"].values())
        empty = sum(share["empties"] for share in progress["shares"].values())
        logger.info("self-generated corpus: %d/%d documents (%d empty)", done, asked, empty)

    _write_share(model, tokenizer, prefix=prefix, keep_prefix=False, n_docs=options.n_raw,
                 seed=options.seed, source="selfgen_raw", options=options, markers=markers,
                 max_empties=limit, batch_size=batch_size, state=raw_state, sink=sink,
                 generate=generate)
    if n_chat:
        _write_share(model, tokenizer, prefix=header, keep_prefix=True, n_docs=n_chat,
                     seed=options.chat_seed, source="selfgen_chatfmt", options=options,
                     markers=markers, max_empties=limit - raw_state["empties"],
                     batch_size=batch_size, state=chat_state, sink=sink, generate=generate)

    kept = raw_state["kept"] + chat_state["kept"]
    empties = raw_state["empties"] + chat_state["empties"]
    if empties > limit:
        raise DegenerateCorpus(
            f"{empties} of {kept + empties} draws from {model_id} came out empty or "
            f"degenerate (limit {options.max_empty_fraction:.0%} of the {asked} documents asked "
            "for). An artifact fitted on such a corpus fails nowhere downstream, so this is a "
            f"refusal: check the seed prefix ({prefix!r}) and the model, or raise "
            "max_empty_fraction knowingly.")
    if kept < options.min_docs:
        raise DegenerateCorpus(
            f"only {kept} documents were generated; at least {options.min_docs} are "
            "needed to fit p(h) on (options.min_docs).")

    partial = partial_path(out_path)
    partial.touch()                                     # a zero-document build appends nothing
    with open(partial, encoding="utf-8") as handle:
        texts = [json.loads(line)["text"] for line in handle]
    manifest = {
        "kind": "artifact-corpus",
        "model_id": model_id,
        "writer_sha256": writer_sha256,
        "seed_prefix": prefix,
        "chat_header": header,
        "frame": {**options.frame(), "batch_size": batch_size},   # the one drawn with
        "decoding": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0},
        "counts": {"raw": raw_state["kept"], "chat": chat_state["kept"], "empty": empties},
        "corpus_sha256": sha256_text(texts),
        "lfa_version": __version__,
    }
    # The manifest goes first: a crash before the rename leaves a partial build that resumes
    # straight to this point, and one after it leaves a complete corpus with its manifest.
    tmp = manifest_file.with_name(manifest_file.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, manifest_file)
    os.replace(partial, out_path)
    progress_path(out_path).unlink()
    logger.info("Self-generated corpus: %d raw + %d chat documents -> %s", raw_state["kept"],
                chat_state["kept"], out_path)
    return manifest
