"""The training corpus: files on disk to a stream of fixed-length causal-LM chunks.

A document is tokenized once and then *cut* into chunks of ``max_length`` tokens every epoch.
:meth:`ChunkedCorpus.rechunk` re-cuts from a per-epoch random offset, so the same text is seen at
different positions in different epochs; epoch 0 is always offset 0. Nothing is re-tokenized, and
the chunking is deterministic in ``(seed, epoch)`` — two runs with the same seed see the same
stream.

**One default deliberately differs from the research code**: ``keep_short_whole``
defaults to ``True`` here. See :class:`ChunkedCorpus` for what the flag does and why the research
code keeps the other default.

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

# A chunk shorter than this is not worth a training step; it is dropped.
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
            chunk (``<= max_length`` tokens) ignores the epoch offset and is therefore present in
            EVERY epoch. ``False`` is the research default, under which the offset loop
            ``range(offset, len(doc), stride)`` yields no chunk at all for a document shorter than
            the epoch's offset — short documents drop out of most epochs. ``False`` is offered
            only so a run can be matched deliberately to a corpus chunked that way; nothing in
            this package sets it, and it is a frame field (:mod:`lfa.workspace`), so perplexities
            are not comparable across the two settings.

    Attributes:
        report: counts for the *current* chunking, refreshed on construction and on every
            :meth:`rechunk` — ``n_docs`` (documents kept), ``n_short_docs`` (documents of
            ``<= max_length`` tokens), ``n_dropped_short_chunks`` (short documents that produced
            no chunk this epoch — only possible with ``keep_short_whole=False``), and
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
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride if stride > 0 else max_length
        self.keep_short_whole = keep_short_whole

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

    def _chunk_document(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                        offset: int) -> int:
        """Append this document's chunks to ``self.examples``; return how many were appended."""
        n_appended = 0
        for start in range(offset, len(input_ids), self.stride):
            end = start + self.max_length
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
                "(max_length=%d, stride=%d, keep_short_whole=%s). Every document is shorter than "
                "the epoch offset or than the %d-token minimum; this epoch trains on nothing.",
                epoch, len(self._docs), offset, self.max_length, self.stride,
                self.keep_short_whole, MIN_CHUNK_TOKENS,
            )

    def rechunk(self, epoch: int, seed: int = 42) -> None:
        """Re-cut every document from a per-epoch offset in ``[0, stride)``.

        The offset is deterministic in ``(seed, epoch)``; epoch 0 uses offset 0, so it reproduces
        the chunking a freshly constructed corpus starts with.
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
                             keep_short_whole=keep_short_whole)

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
