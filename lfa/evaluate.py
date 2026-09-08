"""Perplexity: the two numbers the paper reports for every run, and the table that pairs them.

An LFA run is read on two axes at once, and one number alone says nothing:

* **general** -- WikiText-2 test perplexity, the *preservation* axis. It is measured with the
  sliding window the research record was produced under (window 2048, stride 512, labels
  ``-100`` outside the stride so every token is scored exactly once with the most context
  available to it). A different window gives a different number, so the loop here is a
  deliberate port rather than a re-derivation: the paper's seed-drift figures are only
  comparable against a companion number computed the same way.
* **domain** -- held-out perplexity on the new domain, the *adaptation* axis, as a token-weighted
  mean over the held-out chunks: ``exp(sum_i nll_i * n_i / sum_i n_i)``. Weighting by ``n_i``
  matters because chunks are ragged at the tail of every document, and an unweighted mean of
  per-chunk means would quietly over-weight the short ones.

Both go through one primitive, :func:`sequence_perplexity`, which is ``exp()`` of the mean
negative log-likelihood over the labelled tokens of a batch, computed from the model's logits
with the same shift the Hugging Face causal-LM loss uses. It is checked against
``exp(model(..., labels=...).loss)`` in the tests, so the two can never silently disagree.

⚠ **Domain perplexity is not comparable across corpus formats.** It scores whatever text the
held-out corpus contains, so a run trained on chat-formatted Q&A and a run trained on raw prose
produce domain numbers on different scales even for the same domain. Compare domain perplexity
only within one corpus format.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .corpus import ChunkedCorpus, make_dataloader
from .models import DEFAULT_DEVICE

logger = logging.getLogger("lfa.evaluate")

__all__ = [
    "GENERAL_ROW",
    "DOMAIN_ROW",
    "DOMAIN_BATCH_SIZE",
    "DatasetUnavailable",
    "sequence_perplexity",
    "wikitext2_perplexity",
    "domain_perplexity",
    "perplexity_table",
]

#: Row labels of :func:`perplexity_table`, keyed in its inputs by ``"general"`` and ``"domain"``.
GENERAL_ROW = "general (WikiText-2)"
DOMAIN_ROW = "domain"

class DatasetUnavailable(RuntimeError):
    """Raised when the WikiText-2 test split cannot be loaded.

    The general axis is the one measurement here that is not computed from the caller's own files:
    it reads a dataset from the Hugging Face Hub. No network, an offline cache that does not hold
    it, or a moved dataset id are all ordinary situations rather than bugs, so this arrives as one
    line naming the way out (``--n-windows none`` scores the domain axis alone) rather than as a
    ``datasets`` traceback. It is in :data:`lfa.cli.USER_FACING_ERRORS`.

    :meth:`lfa.workspace.Workspace.evaluate` does not let it out at all: it logs the reason and
    reports the general axis as unmeasured, because an offline machine should still get the domain
    number. This class is what a *direct* caller of :func:`wikitext2_perplexity` sees, and what
    that log line names.
    """


#: Batch size for :func:`domain_perplexity`, as in the research code's plain-perplexity path.
#: The result is token-weighted, so this trades memory against speed and not accuracy -- except
#: that a batch mixing chunks of different lengths is left-padded, and a padded row's first real
#: token is then predicted from a masked position. That is inherited from the source and is why
#: the tail chunk of a document is worth exactly one slightly pessimistic token.
DOMAIN_BATCH_SIZE = 4


# ------------------------------------------------------------------------------------------
# The shared primitive
# ------------------------------------------------------------------------------------------

@contextmanager
def _eval_mode(model):
    """Run with dropout off, then put the model back the way it was found."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


def _model_device(model, fallback: str) -> torch.device:
    """Where to put a batch: beside the model's own parameters, else ``fallback``.

    Following the model rather than the caller's ``device`` argument is what makes these
    evaluators work unchanged on a model that transformers has already placed (including a
    sharded one, whose first parameter's device is the entry point). ``fallback`` is only
    reached by a model with no parameters at all.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def _mean_nll(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None,
              labels: torch.Tensor) -> tuple[float, int]:
    """``(mean NLL over labelled tokens, how many there were)`` for one batch.

    The shift is the Hugging Face causal-LM one -- position ``t``'s logits predict token
    ``t + 1`` -- and logits are upcast to float32 before the cross-entropy exactly as
    ``ForCausalLMLoss`` does, so this reproduces ``model(..., labels=labels).loss`` bit for bit
    on the same forward pass. Tokens labelled ``-100`` (padding, and anything the caller masked)
    enter neither the sum nor the count.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        labels = labels.unsqueeze(0)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0)

    targets = labels[:, 1:]
    n_scored = int((targets != -100).sum())
    if n_scored == 0:
        return float("nan"), 0

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float()

    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return float(nll), n_scored


def _exp(mean_nll: float) -> float:
    """``exp`` that reports an unrepresentable perplexity as ``inf`` rather than raising."""
    try:
        return math.exp(mean_nll)
    except OverflowError:
        return float("inf")


def sequence_perplexity(model, input_ids, attention_mask, labels) -> float:
    """Perplexity of one batch: ``exp`` of the mean NLL over its labelled tokens.

    Accepts a 1-D sequence or a 2-D batch (the three tensors must agree). Returns ``inf`` when
    nothing is labelled, which is the honest answer for "no evidence" and keeps a caller's
    token-weighted accumulation from silently absorbing a zero.
    """
    with _eval_mode(model):
        mean_nll, n_scored = _mean_nll(model, input_ids, attention_mask, labels)
    return float("inf") if n_scored == 0 else _exp(mean_nll)


# ------------------------------------------------------------------------------------------
# General: WikiText-2
# ------------------------------------------------------------------------------------------

def _wikitext2_tokens(tokenizer, n_windows: int, stride: int) -> torch.Tensor:
    """The WikiText-2 test split as one token stream, truncated to ``n_windows`` windows.

    The whole split is joined with ``"\\n\\n"`` and tokenized once, then cut to
    ``n_windows * stride`` tokens -- a deterministic prefix, no sampling. ``n_windows <= 0``
    keeps the full split.
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as error:                       # noqa: BLE001 - see DatasetUnavailable
        # Wide on purpose: a missing `datasets`, an offline cache, a Hub outage and a failed
        # download all surface differently, and none of them is a defect in this package.
        raise DatasetUnavailable(
            "Could not load the WikiText-2 test split for the general axis "
            f"({type(error).__name__}: {error}). It is read from the Hugging Face Hub, so this is "
            "usually no network or an offline cache that does not hold it. Pass "
            "`--n-windows none` (or `n_windows=None`) to skip the general axis and score the "
            "domain alone."
        ) from error
    text = "\n\n".join(dataset["text"])
    # The whole split is deliberately tokenized as one stream and then cut into windows below,
    # so transformers' "Token indices sequence length is longer than the specified maximum ...
    # will result in indexing errors" is wrong here and alarming in a run whose only job is to
    # produce two trustworthy numbers. Silenced for this one call, by name, so that any other
    # message from that logger still comes through.
    tokenization = logging.getLogger("transformers.tokenization_utils_base")
    previous = tokenization.level
    tokenization.setLevel(max(previous, logging.ERROR) if previous else logging.ERROR)
    try:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)
    finally:
        tokenization.setLevel(previous)

    if n_windows > 0:
        n_tokens = n_windows * stride
        if n_tokens < input_ids.size(0):
            input_ids = input_ids[:n_tokens]
    return input_ids


def wikitext2_perplexity(model, tokenizer, n_windows: int = 100, stride: int = 512,
                         max_length: int = 2048, device: str = DEFAULT_DEVICE) -> float:
    """WikiText-2 test perplexity by sliding window -- the paper's general-preservation number.

    Each step feeds a window of up to ``max_length`` tokens but scores only the ``stride`` tokens
    that no earlier window scored, so every token is scored exactly once with as much left
    context as the window allows. The per-window mean NLL is weighted by that stride, matching
    the research code's ``_run_perplexity_check`` term for term; do not "fix" the weighting,
    because the published seed-drift figures are this quantity.

    Args:
        model: any causal LM; left where it is and restored to its previous train/eval mode.
        tokenizer: the model's tokenizer.
        n_windows: how many stride-sized steps of the split to score (``0`` = all of it).
        stride: tokens scored per window.
        max_length: window size, i.e. the context each scored token may see.
        device: fallback placement for the batches; the model's own device wins (see
            :func:`_model_device`).
    """
    tokens = _wikitext2_tokens(tokenizer, n_windows, stride)
    seq_len = tokens.size(0)
    target_device = _model_device(model, device)

    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    with _eval_mode(model):
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            n_new = end - prev_end                     # tokens this window is responsible for

            window = tokens[begin:end].unsqueeze(0).to(target_device)
            labels = window.clone()
            labels[:, : -n_new] = -100                 # already scored by an earlier window

            mean_nll, _ = _mean_nll(model, window, torch.ones_like(window), labels)
            total_nll += mean_nll * n_new
            total_tokens += n_new

            prev_end = end
            if end == seq_len:
                break

    avg_nll = total_nll / max(total_tokens, 1)
    perplexity = _exp(avg_nll)
    logger.info("WikiText-2 perplexity %.2f (loss %.4f over %d tokens, window %d / stride %d)",
                perplexity, avg_nll, total_tokens, max_length, stride)
    return perplexity


# ------------------------------------------------------------------------------------------
# Domain: held-out corpus
# ------------------------------------------------------------------------------------------

def domain_perplexity(model, tokenizer, heldout: ChunkedCorpus,
                      device: str = DEFAULT_DEVICE) -> float:
    """Held-out perplexity on the new domain -- the paper's adaptation number.

    The corpus is walked in order (``shuffle=False``, so the number is reproducible) with the
    training loader's left-padding collation, and each batch contributes its mean NLL weighted
    by how many tokens it actually scored. Padding is labelled ``-100`` and never counted.

    Args:
        model: any causal LM; left where it is and restored to its previous train/eval mode.
        tokenizer: only its pad token is used, to match the training collation.
        heldout: documents held out of training, already chunked.
        device: fallback placement for the batches; the model's own device wins.
    """
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    loader = make_dataloader(heldout, DOMAIN_BATCH_SIZE, False, 42, pad_token_id)
    target_device = _model_device(model, device)

    total_nll = 0.0
    total_tokens = 0

    with _eval_mode(model):
        for batch in loader:
            mean_nll, n_scored = _mean_nll(
                model,
                batch["input_ids"].to(target_device),
                batch["attention_mask"].to(target_device),
                batch["labels"].to(target_device),
            )
            if n_scored == 0:
                continue
            total_nll += mean_nll * n_scored
            total_tokens += n_scored

    if total_tokens == 0:
        logger.warning("Held-out corpus scored 0 tokens (%d chunks); domain perplexity is inf.",
                       len(heldout))
        return float("inf")

    avg_nll = total_nll / total_tokens
    perplexity = _exp(avg_nll)
    logger.info("Domain perplexity %.2f (loss %.4f over %d tokens in %d chunks)",
                perplexity, avg_nll, total_tokens, len(heldout))
    return perplexity


# ------------------------------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------------------------------

def _delta_pct(before: float, after: float) -> str:
    """``after`` against ``before`` as a signed percentage, or ``n/a`` if ``before`` is 0."""
    if before == 0:
        return "n/a"
    return f"{(after - before) / before * 100:+.1f}%"


def _require(values: dict, key: str, which: str) -> float:
    if key not in values:
        raise KeyError(f"{which} perplexities are missing the {key!r} row; "
                       f"got {sorted(values)}")
    return float(values[key])


def perplexity_table(before: dict, after: dict, unanchored: dict | None = None) -> str:
    """A Markdown table of the run's two axes, with the change on each.

    ``before``, ``after`` and the optional ``unanchored`` are each keyed ``"general"`` (WikiText-2)
    and ``"domain"``. ``before`` is the model as it arrived, ``after`` the anchored run, and
    ``unanchored`` the same recipe with the anchoring switched off -- the column that shows what
    the anchor bought, since an unanchored run reaches the domain by giving up the general axis.

    Both Δ columns are read against ``before``, and both are signed: on the general row a
    positive Δ is forgetting, on the domain row a negative Δ is learning.
    """
    header = ["metric", "before", "after", "Δ%"]
    if unanchored is not None:
        header += ["unanchored", "Δ%"]

    rows = []
    for key, label in (("general", GENERAL_ROW), ("domain", DOMAIN_ROW)):
        b = _require(before, key, "before")
        a = _require(after, key, "after")
        row = [label, f"{b:.2f}", f"{a:.2f}", _delta_pct(b, a)]
        if unanchored is not None:
            u = _require(unanchored, key, "unanchored")
            row += [f"{u:.2f}", _delta_pct(b, u)]
        rows.append(row)

    widths = [max(len(cell) for cell in column) for column in zip(header, *rows)]
    # The label column reads left to right; every number column is right-aligned under its header.
    rule = ["-" * widths[0]] + ["-" * (w - 1) + ":" for w in widths[1:]]

    def line(cells: list[str]) -> str:
        padded = [cells[0].ljust(widths[0])]
        padded += [cell.rjust(width) for cell, width in zip(cells[1:], widths[1:])]
        return "| " + " | ".join(padded) + " |"

    return "\n".join([line(header), line(rule), *(line(row) for row in rows)])
