"""The training corpus: files on disk to a stream of fixed-length causal-LM chunks.

A document is tokenized once and then *cut* into chunks of ``max_length`` tokens every epoch.
:meth:`ChunkedCorpus.rechunk` re-cuts from a per-epoch random offset, so the same text is seen at
different positions in different epochs; epoch 0 is always offset 0. Nothing is re-tokenized, and
the chunking is deterministic in ``(seed, epoch)`` — two runs with the same seed see the same
stream.

The offset **rotates the chunk boundaries; it does not truncate the document**. The leading
segment ``[0, offset)`` is emitted as a chunk of its own alongside the offset-aligned ones, so
every token is trained on in every epoch while the boundaries still move between epochs. That is
simply what this loader does; there is no switch for it.

``keep_short_whole`` is the one choice the chunker leaves open, and it is a choice about short
documents rather than about matching anyone else's stream: see :class:`ChunkedCorpus`.

Tokenization is **eager**. The whole corpus is held as token tensors for the whole run, at a
measured ~35 bytes a token (:data:`BYTES_PER_TOKEN`) — about eight times its size on disk — so a
corpus that cannot fit in host memory is refused here, as :class:`CorpusTooLarge`, rather than by
the OOM killer part-way through tokenizing it.

Three corpus shapes train badly without failing: too few chunks for the batch, one document
contributing most of the gradient, and more epochs than the amount of text can carry.
:meth:`ChunkedCorpus.shape_warnings` names them, with the measured numbers, and
:func:`lfa.train.train` logs them before the first step.

The only per-chunk targets are the inputs themselves (``labels == input_ids``); the model shifts
them internally. A written question-and-answer supplement (:func:`load_corpus`'s ``supplement``) is
mixed in as further whole documents, so there is no prompt masking and no per-token loss
weighting: what the loader yields is exactly what the content loss sees.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = (".txt", ".md")
JSON_EXTENSIONS = (".json", ".jsonl")

# JSON fields that carry a document, in preference order.
_TEXT_FIELDS = ("text", "content", "body", "document", "passage")

# A chunk shorter than this is not worth a training step; it is dropped. Only a document's
# *final* segment can be this short: a leading segment below the minimum is folded away by
# taking the epoch-0 cut for that document instead (:meth:`ChunkedCorpus._chunk_bounds`).
MIN_CHUNK_TOKENS = 10

# ------------------------------------------------------------------------------------------
# What a corpus costs, and what it is refused for
# ------------------------------------------------------------------------------------------

#: Host RAM per token of corpus, measured end to end through :func:`load_corpus` (2026-09-08, the
#: Qwen3 tokenizer, 512-token chunks): **29.3 bytes a token** on a 10.1 M-token corpus and **34.1**
#: on a 40.4 M-token one, i.e. about eight times the corpus's size on disk. Twenty-four of those
#: bytes are the tensors themselves — ``input_ids`` and ``attention_mask`` at eight bytes a token
#: each, plus the eight-byte ``labels`` clone — and the rest is per-chunk object overhead and
#: allocator retention. This is HOST memory, resident for the whole run, and is unrelated to the
#: ~9 GB the run needs on the GPU.
BYTES_PER_TOKEN = 35

#: Characters a token, English prose through a byte-level BPE tokenizer (measured 3.88). Used
#: only to size a corpus *before* it is tokenized. Denser-tokenizing scripts get more tokens per
#: character than this, so the estimate it feeds is an under-estimate: the guard below sooner
#: misses a corpus it should have caught than refuses one that would have fitted.
CHARS_PER_TOKEN = 4

#: Environment variable that overrides the memory guard: a number of GiB to allow, or ``off``.
MEMORY_LIMIT_ENV = "LFA_CORPUS_MEMORY_LIMIT_GB"

# ------------------------------------------------------------------------------------------
# Corpus shapes that train badly without failing (:meth:`ChunkedCorpus.shape_warnings`)
# ------------------------------------------------------------------------------------------

#: Fewer optimizer steps an epoch than this and every step's gradient is computed on more than an
#: eighth of the corpus, so successive steps are near-copies of one another and of the full-batch
#: gradient — the shuffle then buys almost nothing. Eight is also where the recipe's 50-step
#: warmup stops being incidental: below it a 15-epoch run spends its first six epochs or more
#: under the learning rate the operating point was tuned at.
MIN_STEPS_PER_EPOCH = 8

#: The batch size assumed when :meth:`ChunkedCorpus.shape_warnings` is not told the run's own
#: (the shipped recipe's value). The message says when it is assuming.
ASSUMED_BATCH_SIZE = 6

#: A single document contributing more than this share of the chunks contributes more gradient
#: than the whole of the rest of the corpus put together. Strictly more, so that an evenly split
#: two-document corpus — where a document is half of it by arithmetic rather than by dominating
#: — does not trip it.
DOMINANT_DOCUMENT_SHARE = 0.5

#: Training tokens below which a run of more than :data:`SMALL_CORPUS_EPOCHS` epochs is told to let
#: the held-out curve choose its dose. At the bundled recipes' lambda (1,000,000, both models) this
#: package's own two-domain walkthrough's Darwin text (~189 k training tokens an epoch) turned late
#: and shallowly over the recipe's 15 epochs: its held-out perplexity was lowest at epoch 10
#: (Qwen3-0.6B) or 11 (Qwen3-1.7B) and ended 2.6 % or 0.2 % above it (0.6B: 17.62 at epoch 10 ->
#: 18.09; 1.7B: 12.69 at epoch 11 -> 12.71); no run at 10 or 11 epochs was measured. At
#: lambda = 100,000 the same text turned by epoch 4 (0.6B) or 5 (1.7B), and the 0.6B ended worse
#: than the base model on both axes. So a corpus this small carries the dose only
#: at a strong enough anchor; below half a million tokens (~2 MB of English) the curve is the
#: thing to read.
SMALL_CORPUS_TOKENS = 500_000

#: Epochs up to which a small corpus is not remarked on: where the walkthrough's text turned at
#: lambda = 100,000.
SMALL_CORPUS_EPOCHS = 5


class CorpusTooLarge(MemoryError):
    """Raised when tokenizing a corpus would not fit in host memory (see :data:`BYTES_PER_TOKEN`)."""


def _available_memory_bytes() -> int | None:
    """Host memory a new allocation can actually get, or ``None`` where that is not readable.

    ``MemAvailable`` rather than ``MemFree``: it is the kernel's own estimate of what is
    obtainable without swapping, and it counts reclaimable page cache. Linux only — on a platform
    without ``/proc/meminfo`` the guard does not fire at all, which is why the ceiling is
    documented (``docs/faq.md``) as well as enforced.
    """
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _memory_budget_bytes() -> int | None:
    """What the corpus is allowed to cost: :data:`MEMORY_LIMIT_ENV` if set, else what is free."""
    raw = os.environ.get(MEMORY_LIMIT_ENV, "").strip()
    if not raw:
        return _available_memory_bytes()
    if raw.lower() in {"off", "none", "0"}:
        return None
    try:
        return int(float(raw) * 1024 ** 3)
    except ValueError:
        raise ValueError(
            f"{MEMORY_LIMIT_ENV}={raw!r}: expected a number of GiB, or 'off' to disable the "
            "corpus memory guard."
        ) from None


def _refuse_a_corpus_that_cannot_fit(texts: list[str]) -> None:
    """Raise :class:`CorpusTooLarge` if tokenizing ``texts`` would not fit in host memory.

    Sized from characters, before a single document is tokenized, because the failure it replaces
    happens *during* tokenization: the tensors are about eight times the corpus's size on disk and
    the process is killed part-way through, with no message of its own and nothing written. The
    estimate leans towards letting a run start (see :data:`CHARS_PER_TOKEN`), and the guard is
    silent when the budget cannot be read.
    """
    budget = _memory_budget_bytes()
    if budget is None:
        return
    n_chars = sum(len(text) for text in texts)
    n_tokens = n_chars // CHARS_PER_TOKEN
    estimate = n_tokens * BYTES_PER_TOKEN
    if estimate <= budget:
        return
    raise CorpusTooLarge(
        f"This corpus is about {n_chars / 1e6:,.1f} M characters, which tokenizes to roughly "
        f"{n_tokens:,} tokens and needs about {estimate / 1024 ** 3:.2f} GiB of host RAM to hold "
        f"({BYTES_PER_TOKEN} bytes a token, measured); about {budget / 1024 ** 3:.2f} GiB is "
        "available. The corpus is tokenized eagerly and stays resident for the whole run, so this "
        "is an out-of-memory kill part-way through tokenizing rather than a slow run. Split it and "
        "train the parts as successive domains — a chain is what LFA is for — or set "
        f"{MEMORY_LIMIT_ENV}=<GiB>, or {MEMORY_LIMIT_ENV}=off, if this machine can take it."
    )


class ChunkedCorpus(Dataset):
    """Documents tokenized once, cut into ``max_length``-token chunks every epoch.

    Args:
        texts: the documents. Blank ones are dropped.
        tokenizer: any HF tokenizer; documents are tokenized with ``add_special_tokens=True``
            and never truncated (they are chunked instead).
        max_length: chunk length in tokens.
        stride: distance between chunk starts. ``0`` means ``max_length``, i.e. no overlap.
        supplement_flags: one flag per text, ``True`` for a written supplement pair rather than a
            domain document (:func:`load_corpus` passes them). They change nothing about the
            chunking; they let :meth:`shape_warnings` count the DOMAIN's documents, so a corpus
            of one book and four hundred pairs is read as the one document it is. ``None`` means
            every text is a domain document.
        keep_short_whole: what to do with a document that fits in a single chunk
            (``<= max_length`` tokens). Under ``True``, the default, it is trained **whole** in
            every epoch: it ignores the epoch offset entirely. A document that is already one
            chunk gains nothing from having its boundaries moved, and cutting it in two makes the
            second piece a fragment that begins mid-sentence with none of its own text in front of
            it. Under ``False`` it is cut at the offset like any longer document, which is the
            only way a corpus of *exclusively* short documents gets any positional variety between
            epochs at all — worth having when the "documents" are themselves arbitrary slices of
            something longer, and not worth having when each one is a whole unit (an article, a
            page, a recipe). Either way every token is trained on; what changes is whether a short
            document arrives whole or in two pieces, and a piece under :data:`MIN_CHUNK_TOKENS` is
            dropped as any other short piece is. It is recorded per run, because it changes the
            stream.

    Raises:
        CorpusTooLarge: the corpus would not fit in host memory as token tensors.

    Attributes:
        report: counts for the *current* chunking, refreshed on construction and on every
            :meth:`rechunk` — ``n_docs`` (documents kept, supplement pairs included),
            ``n_supplement_docs`` (how many of those are supplement pairs), ``n_short_docs``
            (documents of
            ``<= max_length`` tokens), ``n_dropped_short_chunks`` (documents that produced no
            chunk at all, which can only mean they are under the ten-token minimum), and
            ``n_chunks``. An epoch that chunks to nothing is reported here and
            logged at WARNING; it is never an error.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer,
        max_length: int = 512,
        stride: int = 0,
        keep_short_whole: bool = True,
        supplement_flags: list[bool] | None = None,
    ):
        if supplement_flags is not None and len(supplement_flags) != len(texts):
            raise ValueError(f"supplement_flags has {len(supplement_flags)} flag(s) for "
                             f"{len(texts)} text(s); it needs one per text")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride if stride > 0 else max_length
        self.keep_short_whole = keep_short_whole

        self.examples: list[dict[str, torch.Tensor]] = []
        self._docs: list[tuple[torch.Tensor, torch.Tensor]] = []   # (input_ids, attention_mask)
        #: Chunks contributed by each kept document under the CURRENT chunking, and enough of its
        #: opening to name it in a warning. Both are per-document, so `shape_warnings` can say
        #: which document a lopsided corpus is lopsided towards.
        self._doc_chunks: list[int] = []
        self._excerpts: list[str] = []
        #: Whether each kept document is a supplement pair: the shape checks read the domain's
        #: own documents, not the pairs mixed in beside them.
        self._is_supplement: list[bool] = []
        self.report: dict[str, int] = {}

        # Before anything is tokenized: this is the allocation that gets a process killed.
        _refuse_a_corpus_that_cannot_fit([t for t in texts if t.strip()])

        flags = supplement_flags if supplement_flags is not None else [False] * len(texts)
        for text, is_supplement in zip(texts, flags):
            if not text.strip():
                continue
            encoded = tokenizer(text, add_special_tokens=True, truncation=False,
                                return_tensors="pt")
            self._docs.append((encoded["input_ids"][0], encoded["attention_mask"][0]))
            self._excerpts.append(" ".join(text.split())[:60])
            self._is_supplement.append(bool(is_supplement))

        self._chunk_all(offset=0, epoch=0)

    # -- chunking ----------------------------------------------------------------------

    def _chunk_bounds(self, n_tokens: int, offset: int) -> list[tuple[int, int]]:
        """The half-open ``[start, end)`` spans one document is cut into, in document order.

        The offset moves the *boundaries* rather than the document's starting point: the leading
        segment is cut first and the offset-aligned chunks follow, so the spans tile the whole
        document and no token is left out of an epoch.

        A leading segment below :data:`MIN_CHUNK_TOKENS` is folded away by giving the document the
        epoch-0 cut (offset 0) rather than by dropping it. Dropping it would be the same defect in
        miniature — nine tokens, on 9 of every 512 offsets — and the alternative costs nothing: an
        offset of one to nine tokens moves the boundaries so little that the epoch-0 cut is the
        same variety. The minimum-chunk rule itself is untouched: no chunk below it is ever
        emitted, and the only segment that can still be dropped for being too short is a
        document's final one, exactly as before.

        With a ``stride`` below ``max_length`` the offset-aligned chunks overlap as they always
        have; the leading segment does not overlap the one after it, so the seam is the one place
        a token appears once rather than twice.
        """
        if 0 < offset < MIN_CHUNK_TOKENS:
            offset = 0

        bounds: list[tuple[int, int]] = []
        if offset > 0:
            # `min(..., offset)` matters only for a stride above max_length, where the leading
            # segment is longer than one chunk; normally offset < stride <= max_length and this
            # is the single span [0, offset).
            bounds += [(start, min(start + self.max_length, offset))
                       for start in range(0, offset, self.stride)]
        bounds += [(start, start + self.max_length)
                   for start in range(offset, n_tokens, self.stride)]
        return bounds

    def _chunk_document(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                        offset: int) -> int:
        """Append this document's chunks to ``self.examples``; return how many were appended."""
        n_appended = 0
        for start, end in self._chunk_bounds(len(input_ids), offset):
            chunk_ids = input_ids[start:end]
            if len(chunk_ids) < MIN_CHUNK_TOKENS:
                continue
            self.examples.append({
                "input_ids": chunk_ids,
                "attention_mask": attention_mask[start:end],
                "labels": chunk_ids.clone(),
            })
            n_appended += 1
        return n_appended

    def _chunk_all(self, offset: int, epoch: int) -> None:
        """(Re)cut every document from ``offset`` and refresh :attr:`report`."""
        self.examples = []
        self._doc_chunks = []
        n_short = 0
        n_dropped = 0

        for input_ids, attention_mask in self._docs:
            is_short = len(input_ids) <= self.max_length
            n_short += is_short
            # A document that fits in one chunk gains nothing from a positional offset, and would
            # only be split into a fragment by one, so with keep_short_whole it ignores the offset.
            doc_offset = 0 if (is_short and self.keep_short_whole) else offset
            n_chunks = self._chunk_document(input_ids, attention_mask, doc_offset)
            self._doc_chunks.append(n_chunks)
            if n_chunks == 0 and is_short:
                n_dropped += 1

        self.report = {
            "n_docs": len(self._docs),
            "n_supplement_docs": sum(self._is_supplement),
            "n_short_docs": n_short,
            "n_dropped_short_chunks": n_dropped,
            "n_chunks": len(self.examples),
        }

        if not self.examples:
            logger.warning(
                "Epoch %d chunked to 0 examples from %s at offset %d "
                "(max_length=%d, stride=%d, keep_short_whole=%s). Every document is shorter than "
                "the %d-token minimum, so this epoch trains on nothing.",
                epoch, self._document_count(), offset, self.max_length, self.stride,
                self.keep_short_whole, MIN_CHUNK_TOKENS,
            )

    def rechunk(self, epoch: int, seed: int = 42) -> None:
        """Re-cut every document at a per-epoch offset in ``[0, stride)``.

        The offset is deterministic in ``(seed, epoch)``; epoch 0 uses offset 0, so it reproduces
        the chunking a freshly constructed corpus starts with. It moves where the cuts fall, and
        nothing is left out of an epoch by moving them.
        """
        rng = random.Random(seed + epoch)
        offset = rng.randint(0, self.stride - 1) if epoch > 0 else 0
        self._chunk_all(offset=offset, epoch=epoch)

    # -- Dataset interface -------------------------------------------------------------

    def total_tokens(self) -> int:
        """Tokens in the current chunking (chunks overlap when ``stride < max_length``)."""
        return sum(len(ex["input_ids"]) for ex in self.examples)

    # -- shapes that train badly ---------------------------------------------------------

    def shape_warnings(self, *, batch_size: int | None = None,
                       gradient_accumulation_steps: int = 1,
                       epochs: int | None = None) -> list[str]:
        """Corpus shapes that train badly without failing, as sentences; ``[]`` for a sound one.

        Three of them, measured off the *current* chunking. None is an error and none is a
        refusal: each names the numbers it was read from and what to change, because each is
        invisible in everything else a run prints — the losses fall, the chunk counts look
        ordinary, and the model that comes out is quietly worse than the corpus could have made
        it.

        1. **Too few chunks for the batch.** Below :data:`MIN_STEPS_PER_EPOCH` optimizer steps an
           epoch each step's gradient comes from more than an eighth of the corpus, so successive
           steps are near-copies of one another.
        2. **One document dominating.** A document past :data:`DOMINANT_DOCUMENT_SHARE` of the
           DOMAIN's chunks contributes more gradient than the rest of the domain together, so the
           run is at least as much a fine-tune on that one document as on the corpus. Supplement
           pairs are not counted as documents here: they are written from the domain documents,
           so a single book with four hundred pairs beside it is still one document, and it fires
           (a lone document with nothing beside it is left to the held-out split's own warning,
           :func:`split_documents`).
        3. **More epochs than the text can carry.** Under :data:`SMALL_CORPUS_TOKENS` tokens more
           than :data:`SMALL_CORPUS_EPOCHS` epochs over-train unless lambda holds them (at the
           bundled lambda the walkthrough's 15 ended within 3 % of their minimum; at a tenth of
           it, far above); the trainer's
           held-out curve says so afterwards
           (:func:`lfa.train.held_out_summary`), and this says it before the run.

        Args:
            batch_size: the run's batch size. Without it, :data:`ASSUMED_BATCH_SIZE` is used and
                the message says that it was assumed.
            gradient_accumulation_steps: the run's, since it is the *effective* batch that decides
                how many optimizer steps an epoch holds.
            epochs: the run's epoch count. Without it, check 3 is skipped — an epoch count is not
                a property of a corpus.
        """
        notes: list[str] = []
        n_chunks = len(self.examples)
        if n_chunks == 0:
            return notes                     # an empty chunking is already logged as such

        # 1. too few chunks for the batch
        assumed = batch_size is None
        effective = (ASSUMED_BATCH_SIZE if assumed else batch_size) * max(
            1, gradient_accumulation_steps)
        steps = math.ceil(n_chunks / effective)
        if steps < MIN_STEPS_PER_EPOCH:
            batch_phrase = (f"the shipped batch of {ASSUMED_BATCH_SIZE} (this corpus was not told "
                            f"the run's own)" if assumed else
                            f"a batch of {effective:,} ({batch_size} x "
                            f"{gradient_accumulation_steps} accumulation step(s))")
            run_phrase = (f", {steps * epochs:,} in the whole {epochs}-epoch run"
                          if epochs is not None else "")
            notes.append(
                f"Corpus shape: {n_chunks:,} chunk(s) at {batch_phrase} is {steps} optimizer "
                f"step(s) an epoch{run_phrase}. Every step's gradient then comes from about "
                f"{1 / steps:.0%} of the corpus, so consecutive steps see nearly the same "
                f"examples and the shuffle buys little. Add documents, lower batch_size or "
                f"gradient_accumulation_steps, or lower sequence_length ({self.max_length}) so "
                f"that each document yields more chunks."
            )

        # 2. one document dominating -- among the DOMAIN's documents; supplement pairs are text
        #    written from them, not further documents, and counting them hid the one-book case.
        domain = [i for i, flag in enumerate(self._is_supplement) if not flag]
        domain_chunks = sum(self._doc_chunks[i] for i in domain)
        n_pairs = len(self._docs) - len(domain)
        if domain_chunks and (len(domain) > 1 or n_pairs):
            worst = max(domain, key=self._doc_chunks.__getitem__)
            share = self._doc_chunks[worst] / domain_chunks
            if share > DOMINANT_DOCUMENT_SHARE:
                of_what = (f"the {n_chunks:,} chunk(s)" if not n_pairs else
                           f"the domain's {domain_chunks:,} chunk(s) ({n_chunks:,} with the "
                           f"{n_pairs:,} supplement pair(s))")
                notes.append(
                    f"Corpus shape: one document is {share:.0%} of {of_what} "
                    f"({len(self._docs[worst][0]):,} of "
                    f"{sum(len(self._docs[i][0]) for i in domain):,} tokens), so most of the "
                    f"gradient the domain's own text gives every epoch comes from it -- more "
                    f"than from the rest of the domain's documents together: document "
                    f"{domain.index(worst) + 1} of "
                    f"{len(domain)}, beginning {self._excerpts[worst]!r}. Split it at its own "
                    f"section boundaries -- `lfa prepare-domain <file> --out <new dir> "
                    f"--split-chars 3500` cuts a long file into documents of about 3,500 "
                    f"characters at paragraph boundaries -- or add documents, so that the corpus "
                    f"is not one document with company."
                )

        # 3. more epochs than the text can carry
        n_tokens = self.total_tokens()
        if (epochs is not None and epochs > SMALL_CORPUS_EPOCHS
                and n_tokens < SMALL_CORPUS_TOKENS):
            notes.append(
                f"Corpus shape: this run is {epochs} epochs over {n_tokens:,} training token(s) "
                f"an epoch ({n_chunks:,} chunk(s), {self._document_count()}) -- under "
                f"{SMALL_CORPUS_TOKENS:,} tokens at more than {SMALL_CORPUS_EPOCHS} epochs, where "
                f"a run can train past its held-out minimum and ship a model worse on its own "
                f"domain than one it passed through. Read the held-out line the trainer prints "
                f"when the run ends: it names the epoch the curve bottomed at and the gap to the "
                f"final epoch, and says whether a re-run at that epoch is advised (keep "
                f"val_fraction above 0, so there is a curve). For scale, at the bundled lambda "
                f"(1,000,000) the walkthrough's Darwin text (~189 k tokens an epoch) bottomed at "
                f"epoch 10-11 of 15 and ended 0.2-2.6 % above it; at lambda 100,000 it turned by "
                f"epoch 4-5."
            )

        return notes

    def _document_count(self) -> str:
        """``"60 document(s)"``, or ``"60 document(s) + 293 supplement pair(s)"`` when pairs are
        mixed in: a pair is not a document of the domain, and one count of both hides how many
        documents the domain actually has."""
        n_pairs = sum(self._is_supplement)
        phrase = f"{len(self._docs) - n_pairs:,} document(s)"
        return phrase + (f" + {n_pairs:,} supplement pair(s)" if n_pairs else "")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


# ------------------------------------------------------------------------------------------
# Reading documents off disk
# ------------------------------------------------------------------------------------------

def _extract_text(obj: Any, tokenizer=None) -> str | None:
    """The document carried by a JSON record, or ``None`` if it carries none.

    A ``prompt``/``response`` pair is rendered with ``tokenizer``'s chat template when one is
    given, as the non-thinking turn :func:`render_pair` renders, so that the text carries the
    control tokens the model actually sees in use; without a tokenizer (or if the template fails)
    the two fields are joined by a newline.
    """
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in _TEXT_FIELDS:
            if isinstance(obj.get(key), str):
                return obj[key]
        if obj.get("prompt"):
            prompt, response = str(obj.get("prompt", "")), str(obj.get("response", ""))
            if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": response}]
                try:
                    return tokenizer.apply_chat_template(messages, tokenize=False,
                                                         add_generation_prompt=False,
                                                         enable_thinking=False)
                except Exception:                     # no template, or one that rejects the pair
                    logger.debug("Chat template failed for an instruction pair; joining plainly.")
            return (prompt + "\n" + response).strip()
    return None


def render_pair(tokenizer, prompt: str, response: str) -> str:
    """One question-and-answer pair as a full chat turn, ``enable_thinking=False``.

    Qwen3 then inserts the empty ``<think>\\n\\n</think>`` block that the research code's training
    format carries, so a written pair is trained in the format the model answers in. Without a
    template the two are joined plainly.
    """
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=False, enable_thinking=False)
    except Exception:
        logger.debug("Chat template failed for a pair; joining plainly.")
        return (prompt + "\n" + response).strip()


def load_supplement(path, tokenizer) -> list[str]:
    """The written pairs as training documents, in file order (the prefix rule needs it)."""
    texts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("prompt"):
                texts.append(render_pair(tokenizer, str(record["prompt"]),
                                         str(record.get("response", ""))))
    return texts


@dataclass
class Selection:
    """How many pairs a token-fraction target selects, and what share they actually make."""
    n_used: int
    achieved_fraction: float
    under_target: bool


def select_supplement_prefix(raw_tokens: list[int], pair_tokens: list[int],
                             target: float) -> Selection:
    """The pair prefix whose token share ``used / (raw + used)`` is closest to ``target``.

    Ported verbatim from the research mixer (``prepare_domain_qa.select_qa_for_fraction``):
    pairs are added in order, the largest count under the need is compared with the first count
    over it, and the closer one wins.

    ``under_target`` is True only when the pool was exhausted and still fell short of ``target``.
    This deliberately departs from the research *expression* (``achieved < target``), which also
    fires whenever the closer candidate happens to land just below the target with pairs to
    spare -- about half of ordinary runs -- and implements what the research docstring says the
    flag means: "True when the pool was too small".
    """
    if not (0.0 <= target < 1.0):
        raise ValueError(f"target must be in [0, 1): got {target}")
    total_raw = sum(raw_tokens)
    if target == 0.0:
        return Selection(0, 0.0, under_target=False)
    if not pair_tokens or total_raw == 0:
        return Selection(0, 0.0, under_target=True)

    needed = target / (1.0 - target) * total_raw
    cumsums = list(itertools.accumulate(pair_tokens))
    k = 0
    for i, c in enumerate(cumsums):
        if c <= needed:
            k = i + 1
        else:
            break

    def achieved(n: int) -> float:
        used = cumsums[n - 1] if n > 0 else 0
        return used / (total_raw + used)

    candidates = [k] + ([k + 1] if k < len(pair_tokens) else [])
    best = min(candidates, key=lambda n: abs(achieved(n) - target))
    exhausted = best == len(pair_tokens)
    return Selection(best, achieved(best),
                     under_target=exhausted and achieved(best) < target - 1e-9)


def _load_json_texts(file_path: Path, tokenizer=None) -> list[str]:
    """Documents from a ``.json`` (object or array) or ``.jsonl`` (one object per line) file."""
    texts: list[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.suffix.lower() == ".jsonl":
            records: list[Any] = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            records = data if isinstance(data, list) else [data]
    for record in records:
        text = _extract_text(record, tokenizer)
        if text and text.strip():
            texts.append(text)
    return texts


def load_texts(path: str | Path, tokenizer=None) -> list[str]:
    """Read documents from a file or a directory tree, in sorted path order.

    ``.txt``/``.md`` files are one document each; ``.jsonl`` files are one record per line and
    ``.json`` files a single object or an array, each record contributing its ``text`` (or
    ``content``/``body``/``document``/``passage``) field, else its ``prompt`` and ``response``.
    Other extensions are ignored, as are blank documents.

    Args:
        tokenizer: when given, ``prompt``/``response`` records are rendered with its chat
            template rather than joined by a newline. Pass one when the texts are being used to
            estimate p(h) -- the hidden states of a chat-formatted exchange are not those of the
            same words run together -- and leave it out for training documents.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus path not found: {path}")

    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())

    texts: list[str] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix in JSON_EXTENSIONS:
            texts.extend(_load_json_texts(file_path, tokenizer))
        elif suffix in TEXT_EXTENSIONS:
            content = file_path.read_text(encoding="utf-8").strip()
            if content:
                texts.append(content)
    return texts


def _n_training_documents(n_documents: int, val_fraction: float) -> int:
    """How many of ``n_documents`` the split trains on; the rest are held out."""
    return int(n_documents * (1 - val_fraction))


def min_documents_for_held_out(val_fraction: float) -> int:
    """The fewest documents :func:`split_documents` holds anything out of at ``val_fraction``.

    ``1`` at ``val_fraction <= 0`` (nothing is asked to be held out, so nothing is missing). At the
    bundled recipes' 0.1 it is ``2``: one document cannot be split, and two train on one and
    hold out the other. A large fraction needs more, since a split that would hold out every
    document is refused rather than taken.

    Raises:
        ValueError: ``val_fraction`` outside ``[0, 1)``, the range a recipe accepts -- at 1.0 or
            above no number of documents leaves anything to train on.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(
            f"val_fraction must be in [0, 1), got {val_fraction}: it is the share of DOCUMENTS "
            "held out of training, so 1.0 would leave nothing to train on")
    if val_fraction == 0.0:
        return 1
    n = 2
    while not 1 <= _n_training_documents(n, val_fraction) < n:
        n += 1
    return n


def split_documents(path, val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """The seed-shuffled raw documents, training side then held-out side.

    This is the one split: the trainer, ``evaluate`` and the supplement writer all read it, so
    a pair is never written from a document the stage is scored on. The first
    ``int(n * (1 - val_fraction))`` shuffled documents train and the rest are held out.

    A one-document corpus cannot be split, so it trains on its document and holds nothing out
    (logged at WARNING: its domain number is then a fit). Any other split that would hold out
    every document is refused.
    """
    texts = load_texts(path)
    if not texts:
        raise ValueError(f"No texts found in {path}")
    random.Random(seed).shuffle(texts)
    if val_fraction <= 0.0:
        return texts, []
    if len(texts) == 1:
        logger.warning(
            "val_fraction=%s on the one document found in %s: a single document cannot be "
            "split, so it is trained on and nothing is held out. The domain number is then a "
            "fit rather than a held-out measurement, and there is no per-epoch held-out curve "
            "to choose the number of epochs by. A long file (a book, a report) is one document "
            "until it is split: `lfa prepare-domain <file> --out <new dir> --split-chars 3500` "
            "cuts it into documents of about 3,500 characters at paragraph boundaries.",
            val_fraction, path)
        return texts, []
    split_idx = _n_training_documents(len(texts), val_fraction)
    if split_idx == 0:
        raise ValueError(
            f"val_fraction={val_fraction} holds out all {len(texts)} document(s) found in "
            f"{path}, leaving nothing to train on. Lower it, or pass val_fraction=0.0 to train "
            "on everything and read the domain number as a fit rather than a measurement.")
    return texts[:split_idx], texts[split_idx:]


def load_corpus(
    path: str | Path,
    tokenizer,
    max_length: int = 512,
    stride: int = 0,
    val_fraction: float = 0.0,
    seed: int = 42,
    keep_short_whole: bool = True,
    supplement: str | Path | None = None,
    supplement_fraction: float = 0.0,
) -> tuple[ChunkedCorpus, ChunkedCorpus | None]:
    """Read ``path`` and build the training corpus, optionally holding documents out and
    mixing a written supplement in.

    The held-out split is taken from the RAW documents first (:func:`split_documents`), so the
    held-out perplexity stays a raw-text number comparable across runs; then a prefix of the
    supplement is chosen for ``supplement_fraction`` of training tokens
    (:func:`select_supplement_prefix`) and shuffled into the training side under ``seed``. The
    training corpus's ``supplement_report`` says what was used (``None`` when nothing was
    mixed); a pool short of the target is logged at WARNING and used whole, never refused.

    Returns ``(train, val)``; ``val`` is ``None`` when nothing is held out -- ``val_fraction <= 0``,
    or a one-document corpus, which cannot be split -- and never an empty corpus, which a trainer
    would score as a perfect held-out curve. ``val_fraction <= 0`` is this function's default
    because most callers (the artifact extension, an explicit evaluation of a
    named corpus) want every document. A training run does not: the shipped recipe sets
    ``val_fraction=0.1`` and :meth:`lfa.workspace.Workspace.train` passes it here, so the stage's
    domain number is a held-out measurement rather than a fit.
    """
    train_texts, val_texts = split_documents(path, val_fraction, seed)

    report, train_flags = None, None
    if supplement is not None and supplement_fraction > 0.0:
        pairs = load_supplement(supplement, tokenizer)
        raw_tokens = [len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in train_texts]
        pair_tokens = [len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in pairs]
        chosen = select_supplement_prefix(raw_tokens, pair_tokens, supplement_fraction)
        report = {"n_available": len(pairs), "n_used": chosen.n_used,
                  "target_fraction": supplement_fraction,
                  "achieved_fraction": chosen.achieved_fraction,
                  "under_target": chosen.under_target}
        if chosen.under_target:
            logger.warning("Supplement pool short of the target: %d pairs give a token share of "
                           "%.3f against %.3f asked", chosen.n_used, chosen.achieved_fraction,
                           supplement_fraction)
        # Shuffled as (text, is_pair) so the shape checks can tell the pairs from the domain's
        # documents; `shuffle` permutes by length and seed alone, so the stream is the one a
        # shuffle of the bare texts gives.
        mixed = [(t, False) for t in train_texts] + [(t, True) for t in pairs[:chosen.n_used]]
        random.Random(seed + 1).shuffle(mixed)
        train_texts = [t for t, _ in mixed]
        train_flags = [flag for _, flag in mixed]

    def build(subset: list[str], flags: list[bool] | None = None) -> ChunkedCorpus:
        corpus = ChunkedCorpus(subset, tokenizer, max_length=max_length, stride=stride,
                               keep_short_whole=keep_short_whole, supplement_flags=flags)
        corpus.supplement_report = None
        return corpus

    train = build(train_texts, train_flags)
    train.supplement_report = report
    if not val_texts:
        return train, None
    return train, build(val_texts)


# ------------------------------------------------------------------------------------------
# Batching
# ------------------------------------------------------------------------------------------

def _collate_left_pad(batch: list[dict[str, torch.Tensor]],
                      pad_token_id: int) -> dict[str, torch.Tensor]:
    """Stack ragged chunks, padding on the LEFT.

    Left-padding puts the last real token at the same position in every row, which is what
    generation needs. Padded positions get ``attention_mask=0`` and ``labels=-100`` so they
    contribute to neither attention nor the loss.
    """
    max_len = max(len(ex["input_ids"]) for ex in batch)

    input_ids, attention_mask, labels = [], [], []
    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(torch.cat([
            torch.full((pad_len,), pad_token_id, dtype=ex["input_ids"].dtype),
            ex["input_ids"],
        ]))
        attention_mask.append(torch.cat([
            torch.zeros(pad_len, dtype=ex["attention_mask"].dtype),
            ex["attention_mask"],
        ]))
        labels.append(torch.cat([
            torch.full((pad_len,), -100, dtype=ex["labels"].dtype),
            ex["labels"],
        ]))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
    pad_token_id: int = 0,
) -> DataLoader:
    """A ``DataLoader`` over ``dataset`` with left-padding collation and a seeded shuffle."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=partial(_collate_left_pad, pad_token_id=pad_token_id),
        pin_memory=torch.cuda.is_available(),
    )
