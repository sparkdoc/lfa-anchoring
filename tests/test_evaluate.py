"""Tests for the perplexity evaluator.

Three things have to hold. (1) ``sequence_perplexity`` must agree with the number Hugging Face
itself computes -- it is the shared primitive under both public evaluators, and if its shift or
its ``-100`` handling drifts, every reported perplexity drifts with it silently. It is therefore
checked against ``exp(model(..., labels=...).loss)`` on an unpadded sequence AND on a partially
masked one. (2) ``domain_perplexity`` must be a real, finite perplexity over a chunked corpus,
and on a corpus that produces exactly one chunk it must reduce to ``sequence_perplexity`` of that
chunk -- the token-weighted mean over one item is that item. (3) ``perplexity_table`` must render
the deltas the paper quotes, sign included.

The WikiText-2 case downloads a dataset, so it is marked ``slow`` (deselected by default) and
skips cleanly when the Hub is unreachable -- this machine's network is flaky and offline runs of
the suite must stay green.
"""

import math

import pytest
import torch

from lfa.corpus import ChunkedCorpus
from lfa.evaluate import (
    domain_perplexity,
    perplexity_table,
    sequence_perplexity,
    wikitext2_perplexity,
)


def _batch(tokenizer, text: str) -> dict[str, torch.Tensor]:
    """One unpadded sequence, labelled the way the corpus labels a chunk."""
    encoded = tokenizer(text, add_special_tokens=True, return_tensors="pt")
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": encoded["input_ids"].clone(),
    }


# --------------------------------------------------------------------------------------
# sequence_perplexity
# --------------------------------------------------------------------------------------

def test_sequence_perplexity_matches_hf_loss(tiny_model):
    """The shared primitive is exp() of the loss transformers computes from the same logits."""
    model, tokenizer = tiny_model
    batch = _batch(tokenizer, "anchoring functions on sampled hidden states")

    with torch.no_grad():
        expected = math.exp(model(**batch).loss.item())

    assert sequence_perplexity(model, batch["input_ids"], batch["attention_mask"],
                               batch["labels"]) == pytest.approx(expected, rel=1e-4)


def test_sequence_perplexity_scores_only_labelled_tokens(tiny_model):
    """Masking a token with -100 removes it from the mean, exactly as the HF loss does."""
    model, tokenizer = tiny_model
    batch = _batch(tokenizer, "anchoring functions on sampled hidden states")
    labels = batch["labels"].clone()
    labels[:, : labels.size(1) // 2] = -100

    with torch.no_grad():
        expected = math.exp(
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                  labels=labels).loss.item()
        )

    got = sequence_perplexity(model, batch["input_ids"], batch["attention_mask"], labels)
    assert got == pytest.approx(expected, rel=1e-4)
    # The masked half really is excluded: the fully-labelled score differs.
    assert got != pytest.approx(
        sequence_perplexity(model, batch["input_ids"], batch["attention_mask"],
                            batch["labels"]), rel=1e-3)


def test_sequence_perplexity_accepts_unbatched_tensors(tiny_model):
    """A 1-D sequence is scored as a batch of one, so callers need not unsqueeze."""
    model, tokenizer = tiny_model
    batch = _batch(tokenizer, "sampled hidden states")

    flat = sequence_perplexity(model, batch["input_ids"][0], batch["attention_mask"][0],
                               batch["labels"][0])
    assert flat == pytest.approx(
        sequence_perplexity(model, batch["input_ids"], batch["attention_mask"],
                            batch["labels"]), rel=1e-6)


def test_sequence_perplexity_without_labelled_tokens_is_infinite(tiny_model):
    """Nothing to score is not a crash and not a small number."""
    model, tokenizer = tiny_model
    batch = _batch(tokenizer, "sampled hidden states")
    labels = torch.full_like(batch["labels"], -100)

    assert sequence_perplexity(model, batch["input_ids"], batch["attention_mask"],
                               labels) == float("inf")


# --------------------------------------------------------------------------------------
# domain_perplexity
# --------------------------------------------------------------------------------------

def test_domain_perplexity_is_a_finite_perplexity(tiny_model, tiny_texts):
    model, tokenizer = tiny_model
    heldout = ChunkedCorpus(tiny_texts, tokenizer, max_length=32)
    assert len(heldout) > 4                                   # several batches, some padded

    ppl = domain_perplexity(model, tokenizer, heldout, device="cpu")
    assert isinstance(ppl, float)
    assert math.isfinite(ppl)
    assert ppl > 1.0


def test_domain_perplexity_over_one_chunk_is_that_chunk(tiny_model, tiny_texts):
    """The token-weighted mean over a single chunk is that chunk's own perplexity."""
    model, tokenizer = tiny_model
    heldout = ChunkedCorpus([tiny_texts[0]], tokenizer, max_length=512)
    assert len(heldout) == 1

    only = heldout[0]
    expected = sequence_perplexity(model, only["input_ids"], only["attention_mask"],
                                   only["labels"])
    assert domain_perplexity(model, tokenizer, heldout, device="cpu") == pytest.approx(
        expected, rel=1e-5)


def test_domain_perplexity_restores_training_mode(tiny_model_fresh, tiny_texts):
    """Evaluating mid-training must not leave the model in eval mode."""
    model, tokenizer = tiny_model_fresh
    model.train()
    heldout = ChunkedCorpus(tiny_texts[:3], tokenizer, max_length=32)

    domain_perplexity(model, tokenizer, heldout, device="cpu")
    assert model.training is True


# --------------------------------------------------------------------------------------
# perplexity_table
# --------------------------------------------------------------------------------------

def test_perplexity_table_renders_signed_deltas():
    table = perplexity_table({"general": 10, "domain": 20}, {"general": 10.1, "domain": 8})

    assert "+1.0%" in table
    assert "-60.0%" in table
    assert "general (WikiText-2)" in table
    assert "domain" in table


def test_perplexity_table_adds_the_unanchored_columns():
    table = perplexity_table({"general": 10, "domain": 20}, {"general": 10.1, "domain": 8},
                             {"general": 18.0, "domain": 7.0})

    assert "unanchored" in table
    assert "+80.0%" in table                                  # 10 -> 18.0 general
    assert "-65.0%" in table                                  # 20 -> 7.0 domain
    # The anchored columns survive the extra pair.
    assert "+1.0%" in table and "-60.0%" in table
    assert table.count("\n") >= 3                             # header, rule, two rows


def test_perplexity_table_survives_a_zero_baseline():
    """A zero baseline has no percentage change; it must not divide by zero."""
    table = perplexity_table({"general": 0, "domain": 20}, {"general": 10.1, "domain": 8})

    assert "n/a" in table
    assert "-60.0%" in table


def test_perplexity_table_reports_a_missing_row():
    with pytest.raises(KeyError, match="domain"):
        perplexity_table({"general": 10}, {"general": 10.1})


# --------------------------------------------------------------------------------------
# WikiText-2 (network)
# --------------------------------------------------------------------------------------

@pytest.mark.slow
def test_wikitext2_perplexity_sliding_window(tiny_model):
    """The sliding window runs end to end on real WikiText-2 text (needs the Hub)."""
    datasets = pytest.importorskip("datasets")
    try:
        datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as exc:                                  # offline, or a flaky Hub
        pytest.skip(f"WikiText-2 unavailable: {exc}")

    model, tokenizer = tiny_model
    ppl = wikitext2_perplexity(model, tokenizer, n_windows=3, stride=64, max_length=128,
                               device="cpu")
    assert math.isfinite(ppl)
    assert ppl > 1.0


@pytest.mark.parametrize("stride, max_length, seq_len", [(512, 2048, 1000), (100, 250, 1000)])
def test_the_sliding_window_scores_every_token_exactly_once(tiny_model, monkeypatch, stride,
                                                            max_length, seq_len):
    """The window loop's contract, on a fixed token stream and with no model in the way.

    Each window feeds up to ``max_length`` tokens for context but is *responsible* for only the
    tokens no earlier window scored, and its mean NLL is weighted by exactly that count. The two
    ways that goes wrong are both silent: a token scored twice quietly over-weights whatever it
    is, and a token never scored quietly drops out of the average. Neither changes the shape of
    the answer, and the published seed-drift figures are this quantity -- so what is asserted
    here is the partition itself, over two geometries (one window covering the whole stream, and
    an overlapping walk).
    """
    import lfa.evaluate as evaluate_module

    tokens = torch.arange(seq_len, dtype=torch.long)
    monkeypatch.setattr(evaluate_module, "_wikitext2_tokens",
                        lambda tokenizer, n_windows, stride: tokens)

    scored: list[torch.Tensor] = []

    def record(model, window, attention_mask, labels):
        kept = window[labels != -100]
        scored.append(kept)
        return 1.0, kept.numel()

    monkeypatch.setattr(evaluate_module, "_mean_nll", record)

    model, tokenizer = tiny_model
    perplexity = evaluate_module.wikitext2_perplexity(
        model, tokenizer, n_windows=0, stride=stride, max_length=max_length, device="cpu")

    counts = torch.zeros(seq_len, dtype=torch.long)
    for kept in scored:
        counts[kept] += 1
    assert counts.tolist() == [1] * seq_len, (
        f"{int((counts == 0).sum())} tokens scored never and {int((counts > 1).sum())} more than "
        f"once over {len(scored)} windows (stride {stride}, window {max_length})")
    assert sum(kept.numel() for kept in scored) == seq_len
    # Every window reported an NLL of 1.0, so the stride-weighted mean is 1.0 whatever the
    # partition -- unless the weights and the counts have come apart.
    assert math.isclose(perplexity, math.e, rel_tol=1e-9)
