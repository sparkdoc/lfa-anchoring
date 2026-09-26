"""The corpus p(h) is estimated on, written by the model with no input data.

The recorded frame (mr-fusion ``scripts/_w11_artifact_corpus.sh``, the artifact behind C12):
2,500 raw documents of up to 2,048 tokens started from the model's document-boundary token at
temperature 1.0 and top-p 1.0, **unfiltered**, seed 42; plus 250 documents started from the
bare user-turn header with the header kept, seed 43, so the corpus carries the chat-format share
the real seed corpus has. The artifact fitted on it at 600k samples per site, K=32, tied the
shipped artifact at every lambda tried -- one model, one seed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .. import __version__
from .generate import (boundary_markers, chat_user_header, checkpoint_sha256, clean_raw,
                       drop_burn_in, generate_texts, load_writer, passes_filters, pick_seed_prefix,
                       sha256_text)

logger = logging.getLogger(__name__)

__all__ = ["SelfGenOptions", "DegenerateCorpus", "write_artifact_corpus"]

_GENERATION_FIELDS = ("n_raw", "n_chat", "max_new_tokens", "batch_size", "seed", "chat_seed",
                      "min_chars", "max_repeat_ratio", "burn_in_tokens")


class DegenerateCorpus(ValueError):
    """Raised when the generated corpus is too small or too empty to fit an artifact on."""


@dataclass
class SelfGenOptions:
    """The generation frame and the fit frame of a self-generated artifact. Defaults are the record."""

    # -- generation
    n_raw: int = 2500
    n_chat: int = 250
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
                 source: str, options: SelfGenOptions, markers, max_empties: float,
                 generate) -> tuple[list[dict], int]:
    """Draw from ``prefix`` in batches until ``n_docs`` documents are kept; returns (rows, empties).

    A draw that cleans to nothing or fails the filters is an empty, and a further draw replaces
    it. The loop stops short only once ``empties`` exceeds ``max_empties``, the share's part of
    the corpus-wide limit, at which point the caller's refusal is certain.
    """
    rows, empties, batch_index = [], 0, 0
    stop = _stop_ids(tokenizer, markers)
    while len(rows) < n_docs and empties <= max_empties:
        size = min(options.batch_size, n_docs - len(rows))
        texts = generate(model, tokenizer, [prefix] * size, max_new_tokens=options.max_new_tokens,
                         temperature=1.0, top_p=1.0, stop_token_ids=stop, seed=seed,
                         batch_index=batch_index)
        batch_index += 1
        for text in texts:
            body = drop_burn_in(clean_raw(text, markers), tokenizer, options.burn_in_tokens)
            if not body or not passes_filters(body, min_chars=options.min_chars,
                                              max_repeat_ratio=options.max_repeat_ratio):
                empties += 1
                continue
            rows.append({"text": (prefix + body) if keep_prefix else body, "source": source})
    return rows, empties


def write_artifact_corpus(model_id: str, out_path, options: SelfGenOptions, *,
                          generate=generate_texts, writer=None) -> dict:
    """Write the unconditional corpus to ``out_path`` (JSONL) and its manifest beside it.

    Args:
        writer: an already-loaded ``(model, tokenizer)``; loaded with :func:`load_writer` otherwise.
        generate: the sampling function (injectable for tests).

    Raises:
        DegenerateCorpus: fewer than ``options.min_docs`` documents, or more than
            ``options.max_empty_fraction`` of the documents asked for came out empty. Nothing is
            written in either case.
    """
    out_path = Path(out_path)
    writer_sha256 = checkpoint_sha256(model_id)          # before the hours of sampling, not after
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

    raw, raw_empty = _write_share(model, tokenizer, prefix=prefix, keep_prefix=False,
                                  n_docs=options.n_raw, seed=options.seed, source="selfgen_raw",
                                  options=options, markers=markers, max_empties=limit,
                                  generate=generate)
    chat, chat_empty = [], 0
    if n_chat:
        chat, chat_empty = _write_share(model, tokenizer, prefix=header, keep_prefix=True,
                                        n_docs=n_chat, seed=options.chat_seed,
                                        source="selfgen_chatfmt", options=options,
                                        markers=markers, max_empties=limit - raw_empty,
                                        generate=generate)

    rows = raw + chat
    empties = raw_empty + chat_empty
    if empties > limit:
        raise DegenerateCorpus(
            f"{empties} of {len(rows) + empties} draws from {model_id} came out empty or "
            f"degenerate (limit {options.max_empty_fraction:.0%} of the {asked} documents asked "
            "for). An artifact fitted on such a corpus fails nowhere downstream, so this is a "
            f"refusal: check the seed prefix ({prefix!r}) and the model, or raise "
            "max_empty_fraction knowingly.")
    if len(rows) < options.min_docs:
        raise DegenerateCorpus(
            f"only {len(rows)} documents were generated; at least {options.min_docs} are "
            "needed to fit p(h) on (options.min_docs).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "kind": "artifact-corpus",
        "model_id": model_id,
        "writer_sha256": writer_sha256,
        "seed_prefix": prefix,
        "chat_header": header,
        "frame": options.frame(),
        "decoding": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0},
        "counts": {"raw": len(raw), "chat": len(chat), "empty": empties},
        "corpus_sha256": sha256_text(r["text"] for r in rows),
        "lfa_version": __version__,
    }
    Path(str(out_path) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Self-generated corpus: %d raw + %d chat documents -> %s", len(raw), len(chat),
                out_path)
    return manifest
