"""Tests for the corpus loader.

The load-bearing behaviour is the chunking stream: a chunk is ``max_length`` tokens, the
per-epoch offset is deterministic in ``(seed, epoch)``, and epoch 0 is always offset 0. The
companion's defaults (``keep_short_whole=True``, ``rotate_offset=True``) keep every token of
every document in every epoch; the legacy settings (``False``) are what the research record was
produced under and must stay reproducible, so both branches are exercised here.

The **coverage** tests below are the evidence for the offset fix, and they are exhaustive rather
than illustrative: for a spread of document lengths across the damaged band, and for every offset
the epoch schedule can produce, the chunks are asserted to tile the document. The same assertion
run against ``rotate_offset=False`` fails, and a test says so.

``tiny_texts`` tokenizes to 69-280 tokens under the char-level fixture tokenizer, so
``max_length=128`` gives both short documents (kept whole / droppable) and multi-chunk ones.
The tokenizer is lossy on decode, so nothing here asserts on decoded text.
"""

import json

import pytest
import torch

from lfa.corpus import (
    MIN_CHUNK_TOKENS,
    ChunkedCorpus,
    load_corpus,
    load_texts,
    make_dataloader,
)


# --------------------------------------------------------------------------------------
# Coverage: what the epoch offset does to a document
# --------------------------------------------------------------------------------------

class ExactTokens:
    """A tokenizer whose document ``"600"`` is exactly the 600 distinct token ids ``0..599``.

    The chunker's contract is over token ids, so a real tokenizer would only add a
    length-perturbing indirection between the assertion and the thing asserted. Distinct ids make
    "these chunks tile that document" checkable by equality rather than by counting.
    """

    def __call__(self, text, add_special_tokens=True, truncation=False, return_tensors="pt"):
        n = int(text)
        return {"input_ids": torch.arange(n).unsqueeze(0),
                "attention_mask": torch.ones(1, n, dtype=torch.long)}


#: Document lengths in tokens, at ``max_length=512``: just over one chunk, the two lengths the
#: defect was measured at, an awkward one, just over four chunks, and the long tail where the
#: truncating loader was nearly harmless.
COVERAGE_LENGTHS = (513, 600, 700, 1024, 1500, 2049, 5000)


def chunks_at(dataset, offset):
    """The chunks this corpus cuts at ``offset``, as tensors, in the order it emits them."""
    dataset._chunk_all(offset=offset, epoch=1)
    return [ex["input_ids"] for ex in dataset]


def assert_tiles_the_document(chunks, n_tokens, *, where):
    """The chunks are the document, in order, once each -- bar a sub-minimum final remainder.

    This is the whole claim in one assertion. Concatenating the chunks in emission order and
    demanding equality with a prefix of the document rules out, together: a lost leading segment,
    a lost middle, a duplicated token, and a reordering. What it allows is the pre-existing
    minimum-chunk rule, which drops a document's final segment when it is under ten tokens -- and
    it pins that to fewer than ten, at the end, rather than taking it on trust.
    """
    document = torch.arange(n_tokens)
    emitted = torch.cat(chunks) if chunks else torch.empty(0, dtype=document.dtype)
    missing = n_tokens - len(emitted)
    assert 0 <= missing < MIN_CHUNK_TOKENS, (
        f"{where}: {missing} of {n_tokens} tokens were not emitted; the minimum-chunk rule "
        f"accounts for at most {MIN_CHUNK_TOKENS - 1}"
    )
    assert torch.equal(emitted, document[:n_tokens - missing]), (
        f"{where}: the chunks do not tile the document in order"
    )
    for chunk in chunks:
        assert MIN_CHUNK_TOKENS <= len(chunk) <= 512, f"{where}: chunk of {len(chunk)} tokens"


def coverage(n_tokens, *, rotate, max_length=512):
    """Fraction of the document emitted, at every offset the epoch schedule can produce."""
    ds = ChunkedCorpus([str(n_tokens)], ExactTokens(), max_length=max_length,
                       keep_short_whole=False, rotate_offset=rotate)
    return [sum(len(c) for c in chunks_at(ds, off)) / n_tokens for off in range(max_length)]


def test_every_offset_emits_every_token():
    """THE coverage test. Every length in the damaged band, every offset in ``[0, stride)``."""
    for n_tokens in COVERAGE_LENGTHS:
        ds = ChunkedCorpus([str(n_tokens)], ExactTokens(), max_length=512)
        for offset in range(512):
            assert_tiles_the_document(chunks_at(ds, offset), n_tokens,
                                      where=f"{n_tokens} tokens at offset {offset}")


def test_the_same_check_fails_on_the_historical_offset():
    """The negative control: the assertion above is not vacuous.

    Under ``rotate_offset=False`` the document starts at the offset, so its first ``offset``
    tokens are not emitted at all -- and the identical helper, on the identical documents, raises.
    """
    for n_tokens in COVERAGE_LENGTHS:
        ds = ChunkedCorpus([str(n_tokens)], ExactTokens(), max_length=512, rotate_offset=False)
        with pytest.raises(AssertionError, match="tokens were not emitted"):
            assert_tiles_the_document(chunks_at(ds, 200), n_tokens, where=f"{n_tokens} at 200")

        # ...and what is missing is exactly the leading 200 tokens, not a rounding effect.
        emitted = torch.cat(chunks_at(ds, 200))
        assert emitted[0].item() == 200


def test_the_coverage_table_the_fix_is_for():
    """The measured damage, and its removal, as numbers rather than as a property.

    The left column is what the truncating loader costs a document of each length, averaged over
    all 512 offsets and at its worst offset; it is the table in the defect report. The right
    column is the same corpus under the default: everything, bar the final under-ten-token
    remainder that the minimum-chunk rule has always dropped.
    """
    measured = {600: (57.40, 14.83), 1024: (75.04, 50.00),
                5000: (94.89, 89.78), 18000: (98.58, 97.16)}
    for n_tokens, (mean_pct, worst_pct) in measured.items():
        old = coverage(n_tokens, rotate=False)
        assert 100 * sum(old) / len(old) == pytest.approx(mean_pct, abs=0.05)
        assert 100 * min(old) == pytest.approx(worst_pct, abs=0.05)

        new = coverage(n_tokens, rotate=True)
        assert min(new) >= 1 - (MIN_CHUNK_TOKENS - 1) / n_tokens
        assert sum(new) / len(new) > 0.9998


def test_the_epoch_schedule_covers_every_token_too():
    """The offsets the sweep above enumerates are reached through ``rechunk``, not only injected.

    One document per corpus, so the chunks in ``examples`` are unambiguously that document's.
    """
    for n_tokens in (600, 1024, 5000):
        ds = ChunkedCorpus([str(n_tokens)], ExactTokens(), max_length=512)
        for epoch in range(60):
            ds.rechunk(epoch=epoch)
            assert_tiles_the_document([ex["input_ids"] for ex in ds], n_tokens,
                                      where=f"{n_tokens} tokens in epoch {epoch}")


def test_a_leading_segment_under_the_minimum_is_folded_not_dropped():
    """Offsets 1-9 cannot make a chunk of their own; the document takes the epoch-0 cut instead.

    The alternative -- dropping the leading segment because it is under the minimum -- would be
    the same defect in miniature on 9 of every 512 offsets. Nothing is dropped and no chunk below
    the minimum is emitted; the epoch simply gets the offset-0 boundaries.
    """
    ds = ChunkedCorpus(["600"], ExactTokens(), max_length=512)
    at_zero = [c.tolist() for c in chunks_at(ds, 0)]
    for offset in range(1, MIN_CHUNK_TOKENS):
        assert [c.tolist() for c in chunks_at(ds, offset)] == at_zero
    assert [c.tolist() for c in chunks_at(ds, MIN_CHUNK_TOKENS)] != at_zero


def test_rotation_leaves_the_offset_zero_chunking_alone():
    """Epoch 0 is the stream both modes share, which is why the recorded chunk COUNTS -- taken at
    construction -- are the same under either setting."""
    rotating = ChunkedCorpus([str(n) for n in COVERAGE_LENGTHS], ExactTokens(), max_length=512)
    legacy = ChunkedCorpus([str(n) for n in COVERAGE_LENGTHS], ExactTokens(), max_length=512,
                           rotate_offset=False)
    assert rotating.report == legacy.report
    assert ([ex["input_ids"].tolist() for ex in rotating]
            == [ex["input_ids"].tolist() for ex in legacy])


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------

def test_short_docs_present_every_epoch(tiny_model, tiny_texts):
    _, tok = tiny_model
    ds = ChunkedCorpus(tiny_texts, tok, max_length=128)
    n0 = len(ds)
    ds.rechunk(epoch=5)
    n5 = len(ds)
    short = ds.report["n_short_docs"]
    assert short > 0
    assert n0 > 0
    assert n5 >= short                       # every short doc contributes a chunk in epoch 5
    assert ds.report["n_dropped_short_chunks"] == 0


def test_legacy_dropout_reproducible(tiny_model, tiny_texts):
    """The research frame is both legacy flags together: the short-document dropout exists only
    because the offset truncates, so ``keep_short_whole=False`` needs ``rotate_offset=False``."""
    _, tok = tiny_model
    ds = ChunkedCorpus(tiny_texts, tok, max_length=128, keep_short_whole=False,
                       rotate_offset=False)
    ds.rechunk(epoch=5)
    assert ds.report["n_dropped_short_chunks"] >= 0   # counter exists; value depends on offset

    # Same (seed, epoch) => same stream, in a second instance as well as a second call.
    other = ChunkedCorpus(tiny_texts, tok, max_length=128, keep_short_whole=False,
                          rotate_offset=False)
    other.rechunk(epoch=5)
    assert other.report == ds.report
    assert [ex["input_ids"].tolist() for ex in other] == [ex["input_ids"].tolist() for ex in ds]


def test_legacy_dropout_actually_drops_and_default_does_not(tiny_model, tiny_texts):
    """The historical dropout is real: some epochs lose whole short documents under the research
    frame, and none ever do under the companion's defaults.

    Rotation alone would also prevent it -- a document shorter than the offset is emitted as the
    leading segment -- so the frame that reproduces the dropout is both legacy flags.
    """
    _, tok = tiny_model
    legacy = ChunkedCorpus(tiny_texts, tok, max_length=128, keep_short_whole=False,
                           rotate_offset=False)
    kept = ChunkedCorpus(tiny_texts, tok, max_length=128, keep_short_whole=True)

    legacy_drops, kept_drops = [], []
    for epoch in range(1, 21):
        legacy.rechunk(epoch=epoch)
        kept.rechunk(epoch=epoch)
        legacy_drops.append(legacy.report["n_dropped_short_chunks"])
        kept_drops.append(kept.report["n_dropped_short_chunks"])

    assert max(legacy_drops) > 0
    assert set(kept_drops) == {0}


def test_epoch_zero_is_offset_zero(tiny_model, tiny_texts):
    _, tok = tiny_model
    ds = ChunkedCorpus(tiny_texts, tok, max_length=128)
    first = [ex["input_ids"].tolist() for ex in ds]
    ds.rechunk(epoch=3)
    assert [ex["input_ids"].tolist() for ex in ds] != first   # a non-zero offset moved the stream
    ds.rechunk(epoch=0)
    assert [ex["input_ids"].tolist() for ex in ds] == first


def test_seed_changes_the_offset(tiny_model, tiny_texts):
    _, tok = tiny_model
    a = ChunkedCorpus(tiny_texts, tok, max_length=128)
    b = ChunkedCorpus(tiny_texts, tok, max_length=128)
    a.rechunk(epoch=1, seed=42)
    b.rechunk(epoch=1, seed=1337)
    assert [ex["input_ids"].tolist() for ex in a] != [ex["input_ids"].tolist() for ex in b]


def test_chunks_are_well_formed(tiny_model, tiny_texts):
    _, tok = tiny_model
    ds = ChunkedCorpus(tiny_texts, tok, max_length=128)
    for ex in ds:
        assert set(ex) == {"input_ids", "attention_mask", "labels"}
        n = len(ex["input_ids"])
        assert 10 <= n <= 128                                  # short chunks are skipped
        assert ex["attention_mask"].shape == ex["input_ids"].shape
        assert torch.equal(ex["labels"], ex["input_ids"])      # causal LM: labels are the inputs
        assert ex["input_ids"].dtype == torch.long


def test_stride_controls_overlap(tiny_model, tiny_texts):
    _, tok = tiny_model
    no_overlap = ChunkedCorpus(tiny_texts[:4], tok, max_length=128)
    overlapping = ChunkedCorpus(tiny_texts[:4], tok, max_length=128, stride=64)
    assert len(overlapping) > len(no_overlap)


def test_report_tracks_the_current_chunking(tiny_model, tiny_texts):
    _, tok = tiny_model
    ds = ChunkedCorpus(tiny_texts, tok, max_length=128)
    assert ds.report["n_docs"] == len(tiny_texts)
    assert ds.report["n_chunks"] == len(ds)
    assert ds.report["n_short_docs"] == sum(
        len(tok(t, add_special_tokens=True)["input_ids"]) <= 128 for t in tiny_texts
    )
    ds.rechunk(epoch=7)
    assert ds.report["n_chunks"] == len(ds)


def test_blank_documents_are_dropped(tiny_model):
    _, tok = tiny_model
    ds = ChunkedCorpus(["   ", "", "a real document with enough characters to survive"], tok,
                       max_length=128)
    assert ds.report["n_docs"] == 1
    assert len(ds) == 1


def test_total_tokens(tiny_model, tiny_texts):
    _, tok = tiny_model
    ds = ChunkedCorpus(tiny_texts, tok, max_length=128)
    assert ds.total_tokens() == sum(len(ex["input_ids"]) for ex in ds)
    assert ds.total_tokens() > 0


def test_empty_epoch_is_reported_and_warned_not_raised(tiny_model, caplog):
    """A corpus of only short documents can chunk to nothing under the legacy offset."""
    _, tok = tiny_model
    ds = ChunkedCorpus(["a short document"], tok, max_length=512, keep_short_whole=False,
                       rotate_offset=False)
    assert len(ds) == 1

    empty_epoch = None
    for epoch in range(1, 21):
        ds.rechunk(epoch=epoch)
        if ds.report["n_chunks"] == 0:
            empty_epoch = epoch
            break
    assert empty_epoch is not None, "expected some epoch to chunk to nothing"

    assert len(ds) == 0
    assert ds.report["n_dropped_short_chunks"] == 1
    with caplog.at_level("WARNING", logger="lfa.corpus"):
        ds.rechunk(epoch=empty_epoch)
    assert any(r.levelname == "WARNING" for r in caplog.records)


# --------------------------------------------------------------------------------------
# load_texts
# --------------------------------------------------------------------------------------

def test_load_texts_formats(tmp_path):
    (tmp_path / "a.txt").write_text("alpha text")
    (tmp_path / "b.jsonl").write_text('{"text":"beta"}\n{"prompt":"q","response":"r"}\n')
    t = load_texts(tmp_path)
    assert "alpha text" in t and "beta" in t and any("q" in x and "r" in x for x in t)


def test_load_texts_is_recursive_and_reads_md_and_json(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.md").write_text("# gamma")
    (tmp_path / "d.json").write_text(json.dumps([{"text": "delta"}, {"content": "epsilon"}]))
    (tmp_path / "e.json").write_text(json.dumps({"text": "zeta"}))
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
    t = load_texts(tmp_path)
    assert set(t) == {"# gamma", "delta", "epsilon", "zeta"}


def test_load_texts_accepts_a_single_file(tmp_path):
    f = tmp_path / "only.txt"
    f.write_text("just this one")
    assert load_texts(f) == ["just this one"]


def test_load_texts_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_texts(tmp_path / "nope")


def test_load_texts_skips_blank_and_unusable_records(tmp_path):
    (tmp_path / "a.txt").write_text("   \n  ")
    (tmp_path / "b.jsonl").write_text('{"nothing":"useful"}\n\n{"text":"kept"}\n')
    assert load_texts(tmp_path) == ["kept"]


# --------------------------------------------------------------------------------------
# load_corpus
# --------------------------------------------------------------------------------------

def test_load_corpus_no_validation_split_by_default(tmp_path, tiny_model, tiny_texts):
    _, tok = tiny_model
    for i, text in enumerate(tiny_texts):
        (tmp_path / f"doc{i:02d}.txt").write_text(text)
    train, val = load_corpus(tmp_path, tok, max_length=128)
    assert val is None
    assert train.report["n_docs"] == len(tiny_texts)


def test_load_corpus_validation_split_is_deterministic(tmp_path, tiny_model, tiny_texts):
    _, tok = tiny_model
    for i, text in enumerate(tiny_texts):
        (tmp_path / f"doc{i:02d}.txt").write_text(text)
    train, val = load_corpus(tmp_path, tok, max_length=128, val_fraction=0.25)
    assert val is not None
    assert train.report["n_docs"] == 9 and val.report["n_docs"] == 3

    again, again_val = load_corpus(tmp_path, tok, max_length=128, val_fraction=0.25)
    assert [ex["input_ids"].tolist() for ex in again] == [ex["input_ids"].tolist() for ex in train]
    assert [ex["input_ids"].tolist() for ex in again_val] == [ex["input_ids"].tolist() for ex in val]


def test_a_split_that_holds_out_every_document_is_refused(tmp_path, tiny_model):
    """The shipped recipe holds a tenth out by default, and a one-document corpus rounds that up
    to all of it -- which would otherwise train on nothing and report a plausible loss for it."""
    _, tok = tiny_model
    (tmp_path / "only.txt").write_text("the only document in this corpus " * 20)
    with pytest.raises(ValueError, match="nothing to train on"):
        load_corpus(tmp_path, tok, max_length=128, val_fraction=0.1)


def test_load_corpus_keeps_short_docs_by_default(tmp_path, tiny_model):
    _, tok = tiny_model
    (tmp_path / "short.txt").write_text("a short document")
    train, _ = load_corpus(tmp_path, tok, max_length=512)
    train.rechunk(epoch=4)
    assert len(train) == 1
    assert train.report["n_dropped_short_chunks"] == 0


def test_load_corpus_empty_directory(tmp_path, tiny_model):
    _, tok = tiny_model
    with pytest.raises(ValueError):
        load_corpus(tmp_path, tok)


# --------------------------------------------------------------------------------------
# make_dataloader
# --------------------------------------------------------------------------------------

def test_make_dataloader_left_pads(tiny_model, tiny_texts):
    _, tok = tiny_model
    ds = ChunkedCorpus(tiny_texts, tok, max_length=128, stride=100)
    loader = make_dataloader(ds, batch_size=4, shuffle=False, pad_token_id=tok.pad_token_id)
    batch = next(iter(loader))

    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[0] == 4

    padded = (batch["attention_mask"] == 0)
    assert padded.any(), "the fixture should produce at least one ragged batch"
    for row_pad, row_ids, row_labels in zip(padded, batch["input_ids"], batch["labels"]):
        n_pad = int(row_pad.sum())
        assert bool(row_pad[:n_pad].all())                      # padding is on the LEFT
        assert torch.equal(row_ids[:n_pad],
                           torch.full((n_pad,), tok.pad_token_id, dtype=torch.long))
        assert torch.equal(row_labels[:n_pad], torch.full((n_pad,), -100, dtype=torch.long))


def test_make_dataloader_shuffle_is_seeded(tiny_model, tiny_texts):
    _, tok = tiny_model
    ds = ChunkedCorpus(tiny_texts, tok, max_length=128)

    def first_batch(seed):
        loader = make_dataloader(ds, batch_size=4, shuffle=True, seed=seed)
        return next(iter(loader))["input_ids"].tolist()

    assert first_batch(42) == first_batch(42)
    assert first_batch(42) != first_batch(7)


def test_load_texts_renders_instruction_pairs_with_the_chat_template(tmp_path):
    (tmp_path / "pairs.jsonl").write_text('{"prompt":"q","response":"r"}\n')

    class Templating:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "<chat>" + "|".join(m["content"] for m in messages) + "</chat>"

    assert load_texts(tmp_path, tokenizer=Templating()) == ["<chat>q|r</chat>"]


def test_load_texts_falls_back_when_the_template_refuses(tmp_path):
    (tmp_path / "pairs.jsonl").write_text('{"prompt":"q","response":"r"}\n')

    class NoTemplate:
        def apply_chat_template(self, *args, **kwargs):
            raise ValueError("this tokenizer has no chat template")

    assert load_texts(tmp_path, tokenizer=NoTemplate()) == ["q\nr"]
