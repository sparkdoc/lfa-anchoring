"""The training corpus: files on disk to a stream of fixed-length causal-LM chunks.

A document is tokenized once and then *cut* into chunks of ``max_length`` tokens every epoch.
:meth:`ChunkedCorpus.rechunk` re-cuts from a per-epoch random offset, so the same text is seen at
different positions in different epochs; epoch 0 is always offset 0. Nothing is re-tokenized, and
the chunking is deterministic in ``(seed, epoch)`` — two runs with the same seed see the same
stream.

The offset **rotates the chunk boundaries; it does not truncate the document**. The leading
segment ``[0, offset)`` is emitted as a chunk of its own alongside the offset-aligned ones, so
every token is trained on in every epoch while the boundaries still move between epochs.

**Two defaults deliberately differ from the research code**, and both are recorded per run as
*frame* fields: ``keep_short_whole`` defaults to ``True`` here, and ``rotate_offset`` defaults to
``True``. Under the research code's ``rotate_offset=False`` the epoch offset discards each
document's first ``offset`` tokens: a 600-token document keeps 57.4 % of its tokens in an average
epoch and 14.8 % in the worst, and a 1,024-token one 75.0 % / 50.0 %. Long documents — the shape
of the corpora the published runs used — sit in the harmless tail (5,000 tokens: 94.9 %), which is
why the defect was invisible there. See :class:`ChunkedCorpus`.

The only per-chunk targets are the inputs themselves (``labels == input_ids``); the model shifts
them internally. There is no QA supplement in the companion, so there is no prompt masking and no
per-token loss weighting: what the loader yields is exactly what the content loss sees.
"""

from __future__ import annotations

import json
import logging
import random
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


class ChunkedCorpus(Dataset):
    """Documents tokenized once, cut into ``max_length``-token chunks every epoch.

    Args:
        texts: the documents. Blank ones are dropped.
        tokenizer: any HF tokenizer; documents are tokenized with ``add_special_tokens=True``
            and never truncated (they are chunked instead).
        max_length: chunk length in tokens.
        stride: distance between chunk starts. ``0`` means ``max_length``, i.e. no overlap.
        keep_short_whole: if ``True`` (the companion's default), a document that fits in one
            chunk (``<= max_length`` tokens) ignores the epoch offset and is therefore cut the
            same way in every epoch. Under ``False`` such a document is cut by the offset like
            any other; combined with ``rotate_offset=False`` that is the research default, under
            which the offset loop ``range(offset, len(doc), stride)`` yields no chunk at all for a
            document shorter than the epoch's offset — short documents drop out of most epochs.
            It is a frame field (:mod:`lfa.workspace`), so perplexities are not comparable across
            the two settings.
        rotate_offset: if ``True`` (the default), the epoch offset ROTATES the chunk boundaries:
            the leading segment ``[0, offset)`` is emitted as a chunk of its own beside the
            offset-aligned chunks, so no token is lost. ``False`` is the research behaviour, which
            starts at ``offset`` and therefore **discards** each document's first ``offset``
            tokens every epoch — 42.6 % of a 600-token document in an average epoch, 25.0 % of a
            1,024-token one. It is offered only so a run can be matched deliberately to the stream
            the published numbers were produced under; it is a frame field like the one above, and
            nothing in this package sets it.

    Attributes:
        report: counts for the *current* chunking, refreshed on construction and on every
            :meth:`rechunk` — ``n_docs`` (documents kept), ``n_short_docs`` (documents of
            ``<= max_length`` tokens), ``n_dropped_short_chunks`` (short documents that produced
            no chunk this epoch — only possible with ``keep_short_whole=False`` *and*
            ``rotate_offset=False``, or for a document under the ten-token minimum), and
            ``n_chunks``. An epoch that chunks to nothing is reported here and logged at WARNING;
            it is never an error.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer,
        max_length: int = 512,
        stride: int = 0,
        keep_short_whole: bool = True,
        rotate_offset: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride if stride > 0 else max_length
        self.keep_short_whole = keep_short_whole
        self.rotate_offset = rotate_offset

        self.examples: list[dict[str, torch.Tensor]] = []
        self._docs: list[tuple[torch.Tensor, torch.Tensor]] = []   # (input_ids, attention_mask)
        self.report: dict[str, int] = {}

        for text in texts:
            if not text.strip():
                continue
            encoded = tokenizer(text, add_special_tokens=True, truncation=False,
                                return_tensors="pt")
            self._docs.append((encoded["input_ids"][0], encoded["attention_mask"][0]))

        self._chunk_all(offset=0, epoch=0)

    # -- chunking ----------------------------------------------------------------------

    def _chunk_bounds(self, n_tokens: int, offset: int) -> list[tuple[int, int]]:
        """The half-open ``[start, end)`` spans one document is cut into, in document order.

        Under :attr:`rotate_offset` the offset moves the *boundaries* rather than the document's
        starting point: the leading segment is cut first and the offset-aligned chunks follow, so
        the spans tile the whole document. Under ``rotate_offset=False`` the leading segment is
        never produced, and the document's first ``offset`` tokens are simply not trained on this
        epoch — the historical behaviour, kept reachable and nothing else.

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
        if self.rotate_offset and 0 < offset < MIN_CHUNK_TOKENS:
            offset = 0

        bounds: list[tuple[int, int]] = []
        if self.rotate_offset and offset > 0:
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
        n_short = 0
        n_dropped = 0

        for input_ids, attention_mask in self._docs:
            is_short = len(input_ids) <= self.max_length
            n_short += is_short
            # A document that fits in one chunk gains nothing from a positional offset, so with
            # keep_short_whole it ignores the offset and survives every epoch.
            doc_offset = 0 if (is_short and self.keep_short_whole) else offset
            if self._chunk_document(input_ids, attention_mask, doc_offset) == 0 and is_short:
                n_dropped += 1

        self.report = {
            "n_docs": len(self._docs),
            "n_short_docs": n_short,
            "n_dropped_short_chunks": n_dropped,
            "n_chunks": len(self.examples),
        }

        if not self.examples:
            logger.warning(
                "Epoch %d chunked to 0 examples from %d document(s) at offset %d "
                "(max_length=%d, stride=%d, keep_short_whole=%s, rotate_offset=%s). Every "
                "document is shorter than the %d-token minimum, or (under rotate_offset=False) "
                "than the epoch offset; this epoch trains on nothing.",
                epoch, len(self._docs), offset, self.max_length, self.stride,
                self.keep_short_whole, self.rotate_offset, MIN_CHUNK_TOKENS,
            )

    def rechunk(self, epoch: int, seed: int = 42) -> None:
        """Re-cut every document at a per-epoch offset in ``[0, stride)``.

        The offset is deterministic in ``(seed, epoch)``; epoch 0 uses offset 0, so it reproduces
        the chunking a freshly constructed corpus starts with. Under the default
        :attr:`rotate_offset` the offset moves where the cuts fall and nothing is left out; under
        ``rotate_offset=False`` it is where each document *starts*, and the tokens before it are
        not seen this epoch.
        """
        rng = random.Random(seed + epoch)
        offset = rng.randint(0, self.stride - 1) if epoch > 0 else 0
        self._chunk_all(offset=offset, epoch=epoch)

    # -- Dataset interface -------------------------------------------------------------

    def total_tokens(self) -> int:
        """Tokens in the current chunking (chunks overlap when ``stride < max_length``)."""
        return sum(len(ex["input_ids"]) for ex in self.examples)

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
    given, so that the text carries the control tokens the model actually sees in use; without a
    tokenizer (or if the template fails) the two fields are joined by a newline.
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
                                                         add_generation_prompt=False)
                except Exception:                     # no template, or one that rejects the pair
                    logger.debug("Chat template failed for an instruction pair; joining plainly.")
            return (prompt + "\n" + response).strip()
    return None


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


def load_corpus(
    path: str | Path,
    tokenizer,
    max_length: int = 512,
    stride: int = 0,
    val_fraction: float = 0.0,
    seed: int = 42,
    keep_short_whole: bool = True,
    rotate_offset: bool = True,
) -> tuple[ChunkedCorpus, ChunkedCorpus | None]:
    """Read ``path`` and build the training corpus, optionally holding documents out.

    Documents are shuffled with ``seed`` before the split, so the same seed gives the same split.
    Returns ``(train, val)``; ``val`` is ``None`` when ``val_fraction <= 0``, which is this
    function's default because most callers (the artifact extension, an explicit evaluation of a
    named corpus) want every document. A training run does not: the shipped recipe sets
    ``val_fraction=0.1`` and :meth:`lfa.workspace.Workspace.train` passes it here, so the stage's
    domain number is a held-out measurement rather than a fit.
    """
    texts = load_texts(path)
    if not texts:
        raise ValueError(f"No texts found in {path}")

    random.Random(seed).shuffle(texts)

    def build(subset: list[str]) -> ChunkedCorpus:
        return ChunkedCorpus(subset, tokenizer, max_length=max_length, stride=stride,
                             keep_short_whole=keep_short_whole, rotate_offset=rotate_offset)

    if val_fraction <= 0.0:
        return build(texts), None

    split_idx = int(len(texts) * (1 - val_fraction))
    if split_idx == 0:
        raise ValueError(
            f"val_fraction={val_fraction} holds out all {len(texts)} document(s) found in "
            f"{path}, leaving nothing to train on. Lower it, or pass val_fraction=0.0 to train "
            "on everything and read the domain number as a fit rather than a measurement."
        )
    return build(texts[:split_idx]), build(texts[split_idx:])


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
