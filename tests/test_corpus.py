"""Tests for the corpus loader.

The load-bearing behaviour is the chunking stream: a chunk is ``max_length`` tokens, the
per-epoch offset is deterministic in ``(seed, epoch)``, and epoch 0 is always offset 0. The offset
rotates the chunk boundaries, so every token of every document is trained on in every epoch.

The **coverage** tests below are exhaustive rather than illustrative: for a spread of document
lengths across the band a truncating offset damages, and for every offset the epoch schedule can
produce, the chunks are asserted to tile the document. :func:`truncating_bounds` is the defect
that assertion exists to rule out, written out here as a local reference rather than kept in the
loader; the same helper run against it fails, and a test says so.

The other two groups are the shapes a corpus can have that train badly without failing
(:meth:`ChunkedCorpus.shape_warnings`) and the memory guard, and both are tested in both
directions: the shape that must warn, and the sound corpus that must not.

``tiny_texts`` tokenizes to 69-280 tokens under the char-level fixture tokenizer, so
``max_length=128`` gives both short documents (whole, or cut in two) and multi-chunk ones.
The tokenizer is lossy on decode, so nothing here asserts on decoded text.
"""

import json

import pytest
import torch

from lfa.corpus import (
    MEMORY_LIMIT_ENV,
    MIN_CHUNK_TOKENS,
    ChunkedCorpus,
    CorpusTooLarge,
    load_corpus,
    load_supplement,
    load_texts,
    make_dataloader,
    render_pair,
    select_supplement_prefix,
    split_documents,
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


def one(notes, fragment):
    """The single warning in ``notes`` containing ``fragment`` -- and the assertion that it is
    single, so that a message split in two, or emitted twice, is a failure rather than a pass."""
    matching = [note for note in notes if fragment in note]
    assert len(matching) == 1, f"{fragment!r} in {len(matching)} of {notes}"
    return matching[0]


def chunks_at(dataset, offset):
    """The chunks this corpus cuts at ``offset``, as tensors, in the order it emits them."""
    dataset._chunk_all(offset=offset, epoch=1)
    return [ex["input_ids"] for ex in dataset]


def truncating_bounds(n_tokens, offset, max_length=512):
    """The defect the loader used to have, as a local reference implementation.

    ``range(offset, n, stride)`` and nothing before it: the document's first ``offset`` tokens are
    not emitted at all. It lives here, in the tests, because it is what the coverage assertions
    exist to rule out -- a loader that offers it as a setting is a loader that can still do it.
    """
    return [(start, start + max_length) for start in range(offset, n_tokens, max_length)]


def truncating_chunks(n_tokens, offset, max_length=512):
    """The chunks that defect produces, subject to the same minimum-chunk rule."""
    document = torch.arange(n_tokens)
    return [document[start:end] for start, end in truncating_bounds(n_tokens, offset, max_length)
            if len(document[start:end]) >= MIN_CHUNK_TOKENS]


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


def coverage(n_tokens, max_length=512):
    """Fraction of the document emitted, at every offset the epoch schedule can produce."""
    ds = ChunkedCorpus([str(n_tokens)], ExactTokens(), max_length=max_length,
                       keep_short_whole=False)
    return [sum(len(c) for c in chunks_at(ds, off)) / n_tokens for off in range(max_length)]


def truncating_coverage(n_tokens, max_length=512):
    """The same fractions under :func:`truncating_bounds`."""
    return [sum(len(c) for c in truncating_chunks(n_tokens, off, max_length)) / n_tokens
            for off in range(max_length)]


def test_every_offset_emits_every_token():
    """THE coverage test. Every length in the damaged band, every offset in ``[0, stride)``."""
    for n_tokens in COVERAGE_LENGTHS:
        ds = ChunkedCorpus([str(n_tokens)], ExactTokens(), max_length=512)
        for offset in range(512):
            assert_tiles_the_document(chunks_at(ds, offset), n_tokens,
                                      where=f"{n_tokens} tokens at offset {offset}")


def test_the_same_check_fails_on_a_truncating_offset():
    """The negative control: the assertion above is not vacuous.

    Under :func:`truncating_bounds` the document starts at the offset, so its first ``offset``
    tokens are not emitted at all -- and the identical helper, on the identical documents, raises.
    """
    for n_tokens in COVERAGE_LENGTHS:
        with pytest.raises(AssertionError, match="tokens were not emitted"):
            assert_tiles_the_document(truncating_chunks(n_tokens, 200), n_tokens,
                                      where=f"{n_tokens} at 200")

        # ...and what is missing is exactly the leading 200 tokens, not a rounding effect.
        emitted = torch.cat(truncating_chunks(n_tokens, 200))
        assert emitted[0].item() == 200


def test_the_coverage_table_the_fix_is_for():
    """The measured damage, and its removal, as numbers rather than as a property.

    The left column is what a truncating offset costs a document of each length, averaged over
    all 512 offsets and at its worst offset; it is the table in the defect report, and it is why
    the loader rotates. The right column is the same corpus as the loader actually cuts it:
    everything, bar the final under-ten-token remainder that the minimum-chunk rule has always
    dropped.
    """
    measured = {600: (57.40, 14.83), 1024: (75.04, 50.00),
                5000: (94.89, 89.78), 18000: (98.58, 97.16)}
    for n_tokens, (mean_pct, worst_pct) in measured.items():
        old = truncating_coverage(n_tokens)
        assert 100 * sum(old) / len(old) == pytest.approx(mean_pct, abs=0.05)
        assert 100 * min(old) == pytest.approx(worst_pct, abs=0.05)

        new = coverage(n_tokens)
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


def test_epoch_zero_is_the_document_cut_end_to_end():
    """At offset 0 the chunks are simply the document in ``max_length`` pieces, with no leading
    segment to rotate -- the stream a freshly constructed corpus starts with."""
    ds = ChunkedCorpus([str(n) for n in COVERAGE_LENGTHS], ExactTokens(), max_length=512)
    assert ds.report["n_chunks"] == sum(n // 512 + (n % 512 >= MIN_CHUNK_TOKENS)
                                        for n in COVERAGE_LENGTHS)
    # ...and every chunk starts on a multiple of max_length, which is what "no leading segment"
    # means: the second document's two chunks start at 0 and 512, not at an offset.
    assert [c["input_ids"][0].item() for c in ds][1:3] == [0, 512]


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


def test_keep_short_whole_false_splits_a_short_document_rather_than_dropping_it(tiny_model,
                                                                                 tiny_texts):
    """What the remaining setting actually chooses between, in both directions.

    Under the default a document that fits in one chunk arrives whole in every epoch. Under
    ``False`` it is cut at the epoch offset like a longer one -- into a chunk and a fragment --
    and neither setting drops it: rotation emits the leading segment either way.
    """
    _, tok = tiny_model
    whole = ChunkedCorpus(tiny_texts, tok, max_length=128)
    split = ChunkedCorpus(tiny_texts, tok, max_length=128, keep_short_whole=False)
    assert whole.report["n_short_docs"] > 0

    whole_counts, split_counts = [], []
    for epoch in range(1, 21):
        whole.rechunk(epoch=epoch)
        split.rechunk(epoch=epoch)
        assert whole.report["n_dropped_short_chunks"] == 0
        assert split.report["n_dropped_short_chunks"] == 0
        whole_counts.append(whole.report["n_chunks"])
        split_counts.append(split.report["n_chunks"])

    # Cutting short documents too can only add chunks, never remove them, and on the epochs
    # whose offset is past the ten-token minimum it does add them. (An offset under ten folds to
    # the epoch-0 cut for every document, which is why this is not true epoch by epoch.)
    assert all(s >= w for s, w in zip(split_counts, whole_counts))
    assert any(s > w for s, w in zip(split_counts, whole_counts))


def test_keep_short_whole_is_deterministic_in_seed_and_epoch(tiny_model, tiny_texts):
    """Same ``(seed, epoch)`` => same stream, in a second instance as well as a second call."""
    _, tok = tiny_model
    ds = ChunkedCorpus(tiny_texts, tok, max_length=128, keep_short_whole=False)
    other = ChunkedCorpus(tiny_texts, tok, max_length=128, keep_short_whole=False)
    ds.rechunk(epoch=5)
    other.rechunk(epoch=5)
    assert other.report == ds.report
    assert [ex["input_ids"].tolist() for ex in other] == [ex["input_ids"].tolist() for ex in ds]


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


def test_a_corpus_of_sub_minimum_documents_is_reported_and_warned_not_raised(tiny_model, caplog):
    """The one chunking that can still come out empty: every document under the ten-token minimum.

    Since the offset rotates, a document of ten tokens or more always yields at least one chunk at
    every offset, so an empty epoch is now a statement about the corpus rather than about the
    epoch -- and it says so at construction, not on epoch seven.
    """
    _, tok = tiny_model
    with caplog.at_level("WARNING", logger="lfa.corpus"):
        ds = ChunkedCorpus(["tiny", "also"], tok, max_length=512)

    assert len(ds) == 0
    assert ds.report["n_dropped_short_chunks"] == 2
    assert any("chunked to 0 examples" in r.message for r in caplog.records)

    for epoch in range(1, 21):                    # and no offset rescues it
        ds.rechunk(epoch=epoch)
        assert ds.report["n_chunks"] == 0


# --------------------------------------------------------------------------------------
# Corpus shapes that train badly without failing
#
# Every one of these is tested in both directions, because a warning that also fires on a
# sound corpus is a warning a user learns to scroll past. `ExactTokens` makes the chunk
# arithmetic exact: a document "60000" is 60,000 tokens, so 118 chunks of it at 512.
# --------------------------------------------------------------------------------------

def sound_corpus():
    """Ten documents of 60,000 tokens: 1,180 chunks, no document over a tenth, 600 k tokens.

    Clear of all three thresholds -- 197 optimizer steps an epoch at the recipe's batch, a 10 %
    largest document, and above `SMALL_CORPUS_TOKENS` -- and it is the corpus every "must not
    fire" assertion below is made against.
    """
    return ChunkedCorpus(["60000"] * 10, ExactTokens(), max_length=512)


def test_a_sound_corpus_produces_no_shape_warnings():
    assert sound_corpus().shape_warnings(batch_size=6, epochs=15) == []


def test_too_few_chunks_for_the_batch_is_named_with_its_arithmetic():
    """Three 600-token documents: six chunks, one optimizer step an epoch at a batch of six."""
    ds = ChunkedCorpus(["600"] * 3, ExactTokens(), max_length=512)
    assert len(ds) == 6
    note = one(ds.shape_warnings(batch_size=6, epochs=15), "optimizer step(s) an epoch")

    assert "6 chunk(s)" in note and "1 optimizer step(s) an epoch" in note
    assert "15 in the whole 15-epoch run" in note
    assert "100% of the corpus" in note                     # what each step's gradient is
    assert "Add documents" in note                          # ...and what to do about it


def test_the_batch_size_is_taken_from_the_run_when_there_is_one():
    """The same corpus is fine at a batch of one and not at six, and the message says which."""
    ds = ChunkedCorpus(["600"] * 6, ExactTokens(), max_length=512)      # 12 chunks
    assert ds.shape_warnings(batch_size=1) == []                       # 12 steps an epoch
    assert one(ds.shape_warnings(batch_size=6), "optimizer step(s)")
    assert one(ds.shape_warnings(batch_size=2, gradient_accumulation_steps=4),
               "2 x 4 accumulation step(s)")


def test_without_a_batch_size_the_threshold_is_stated_against_the_shipped_one():
    ds = ChunkedCorpus(["600"] * 3, ExactTokens(), max_length=512)
    note = one(ds.shape_warnings(), "optimizer step(s) an epoch")
    assert "the shipped batch of 6 (this corpus was not told the run's own)" in note
    assert "in the whole" not in note                       # no epoch count, no epoch claim


def test_one_document_dominating_is_named_and_quantified():
    """40 of 44 chunks from one document: the gradient is that document, and it is identified."""
    ds = ChunkedCorpus(["20000", "600", "600"], ExactTokens(), max_length=512)
    note = one(ds.shape_warnings(), "of the 44 chunk(s)")

    assert "91% of the 44 chunk(s)" in note
    assert "20,000 of 21,200 tokens" in note
    assert "document 1 of 3" in note
    assert "beginning '20000'" in note                      # which document, not just that one
    assert "Split it" in note


def test_an_evenly_split_corpus_is_not_called_lopsided():
    """Two equal documents make one of them half the corpus by arithmetic. Half is not more than
    half, and the threshold is strict for exactly this case."""
    ds = ChunkedCorpus(["5000", "5000"], ExactTokens(), max_length=512)
    assert not [n for n in ds.shape_warnings(batch_size=6) if "chunk(s), " in n]
    assert sound_corpus().shape_warnings(batch_size=6) == []


def test_too_many_epochs_for_the_amount_of_text_is_warned_before_the_run():
    """200 k tokens at the recipe's 15 epochs -- the shape on which a weaker anchor's held-out
    curve turns, said before the run rather than after it."""
    ds = ChunkedCorpus(["20000"] * 10, ExactTokens(), max_length=512)
    note = one(ds.shape_warnings(batch_size=6, epochs=15), "epochs over")

    assert "15 epochs over 200,000 training token(s)" in note
    assert "400 chunk(s), 10 document(s)" in note
    assert "held-out minimum at epoch 10-11" in note        # the measurements it rests on
    assert "turned by epoch 4-5" in note
    assert "val_fraction" in note                           # and how to choose the dose

    assert ds.shape_warnings(batch_size=6, epochs=4) == []  # the same corpus at a sane dose
    assert ds.shape_warnings(batch_size=6) == []            # and no epochs, no epoch claim


def test_a_large_corpus_is_not_told_to_train_for_fewer_epochs():
    assert sound_corpus().shape_warnings(batch_size=6, epochs=15) == []


def test_an_empty_chunking_says_nothing_about_its_shape(tiny_model):
    """It has already been logged as empty; three more sentences about its shape are noise."""
    _, tok = tiny_model
    assert ChunkedCorpus([], tok, max_length=128).shape_warnings(batch_size=6, epochs=15) == []


# --------------------------------------------------------------------------------------
# The memory ceiling
# --------------------------------------------------------------------------------------

def test_a_corpus_too_large_for_memory_is_refused_before_it_is_tokenized(monkeypatch):
    """The refusal that replaces an OOM kill part-way through tokenizing.

    A 20,000-character corpus against a budget of ~1 KiB. The tokenizer is one that would raise on
    anything it were handed, which is what says the refusal happens before tokenization.
    """
    monkeypatch.setenv(MEMORY_LIMIT_ENV, "0.000001")

    def never_called(*args, **kwargs):
        raise AssertionError("the corpus was tokenized despite being refused")

    with pytest.raises(CorpusTooLarge) as excinfo:
        ChunkedCorpus(["x" * 20_000], never_called, max_length=512)

    message = str(excinfo.value)
    assert "5,000 tokens" in message                        # what it estimated, and from what
    assert "GiB is available" in message
    assert "successive domains" in message                  # the answer that is not "more RAM"
    assert MEMORY_LIMIT_ENV in message                      # ...and the one that is


def test_an_ordinary_corpus_is_not_refused(tiny_model, tiny_texts):
    _, tok = tiny_model
    assert len(ChunkedCorpus(tiny_texts, tok, max_length=128)) > 0


def test_the_guard_can_be_turned_off(monkeypatch, tiny_model, tiny_texts):
    _, tok = tiny_model
    monkeypatch.setenv(MEMORY_LIMIT_ENV, "0.000001")
    with pytest.raises(CorpusTooLarge):
        ChunkedCorpus(tiny_texts, tok, max_length=128)

    monkeypatch.setenv(MEMORY_LIMIT_ENV, "off")
    assert len(ChunkedCorpus(tiny_texts, tok, max_length=128)) > 0


def test_an_unreadable_limit_is_a_clear_error_not_a_silent_default(monkeypatch, tiny_model,
                                                                   tiny_texts):
    _, tok = tiny_model
    monkeypatch.setenv(MEMORY_LIMIT_ENV, "lots")
    with pytest.raises(ValueError, match="expected a number of GiB"):
        ChunkedCorpus(tiny_texts, tok, max_length=128)


def test_the_guard_is_silent_where_the_budget_cannot_be_read(monkeypatch, tiny_model, tiny_texts):
    """On a platform with no /proc/meminfo the ceiling is documented rather than enforced."""
    import lfa.corpus as corpus_module
    _, tok = tiny_model
    monkeypatch.delenv(MEMORY_LIMIT_ENV, raising=False)
    monkeypatch.setattr(corpus_module, "_available_memory_bytes", lambda: None)
    assert len(ChunkedCorpus(tiny_texts, tok, max_length=128)) > 0


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
    """Two documents at ``val_fraction=0.6`` round the training side down to none -- which would
    otherwise train on nothing and report a plausible loss for it. (A one-document corpus is the
    exception: it trains on its document and holds nothing out; see the test below.)"""
    _, tok = tiny_model
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text(f"document {name} of this corpus " * 20)
    with pytest.raises(ValueError, match="nothing to train on"):
        load_corpus(tmp_path, tok, max_length=128, val_fraction=0.6)
    with pytest.raises(ValueError, match="nothing to train on"):
        split_documents(tmp_path, 0.6, seed=0)


def test_load_corpus_keeps_short_docs_whole_by_default(tmp_path, tiny_model):
    _, tok = tiny_model
    (tmp_path / "short.txt").write_text("a short document, of well over ten tokens but under one "
                                        "chunk of them")
    train, _ = load_corpus(tmp_path, tok, max_length=512)
    for epoch in range(1, 6):
        train.rechunk(epoch=epoch)
        assert len(train) == 1                    # whole, in every epoch
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
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                                enable_thinking=None):
            assert enable_thinking is False           # the non-thinking turn, as render_pair
            return "<chat>" + "|".join(m["content"] for m in messages) + "</chat>"

    assert load_texts(tmp_path, tokenizer=Templating()) == ["<chat>q|r</chat>"]


def test_load_texts_falls_back_when_the_template_refuses(tmp_path):
    (tmp_path / "pairs.jsonl").write_text('{"prompt":"q","response":"r"}\n')

    class NoTemplate:
        def apply_chat_template(self, *args, **kwargs):
            raise ValueError("this tokenizer has no chat template")

    assert load_texts(tmp_path, tokenizer=NoTemplate()) == ["q\nr"]


# --------------------------------------------------------------------------------------
# The written supplement: selection, rendering, mixing
# --------------------------------------------------------------------------------------

def test_zero_fraction_selects_no_pairs():
    r = select_supplement_prefix([100, 100], [30, 30, 30], 0.0)
    assert (r.n_used, r.achieved_fraction, r.under_target) == (0, 0.0, False)


def test_picks_the_count_closest_to_the_target():
    # total_raw=200, target=0.2 -> needed=50; n=1: 30/230=0.130; n=2: 60/260=0.231 -> 2
    r = select_supplement_prefix([100, 100], [30, 30, 30], 0.2)
    assert r.n_used == 2 and r.achieved_fraction == pytest.approx(60 / 260, abs=1e-4)
    assert r.under_target is False


def test_a_pool_too_small_takes_all_and_flags_under_target():
    r = select_supplement_prefix([100], [10, 10], 0.5)
    assert r.n_used == 2 and r.under_target is True
    assert r.achieved_fraction == pytest.approx(20 / 120, abs=1e-4)


def test_an_ample_pool_is_not_under_target():
    # needed=50: n=2 gives 60/260=0.231, closest to 0.2, with 18 pairs to spare
    r = select_supplement_prefix([100, 100], [30] * 20, 0.2)
    assert r.n_used == 2 and r.under_target is False
    # needed=32.6: n=1 gives 30/230=0.130, closest to 0.14 and BELOW it, with 19 pairs to spare --
    # a rounding-to-nearest shortfall, not a pool too small, so no flag
    r = select_supplement_prefix([100, 100], [30] * 20, 0.14)
    assert r.n_used == 1 and r.achieved_fraction < 0.14 and r.under_target is False


def test_an_empty_pool_with_a_positive_target_is_under():
    r = select_supplement_prefix([100], [], 0.25)
    assert (r.n_used, r.achieved_fraction, r.under_target) == (0, 0.0, True)


def test_a_fraction_at_or_above_one_is_rejected():
    with pytest.raises(ValueError):
        select_supplement_prefix([100], [10], 1.0)


def test_a_pool_larger_than_the_target_uses_a_prefix_and_reports_the_rest(tmp_path, tiny_model):
    _, tokenizer = tiny_model
    docs = tmp_path / "docs"; docs.mkdir()
    for i in range(4):
        (docs / f"d{i}.txt").write_text("raw document text " * 20)
    supp = tmp_path / "supplement.jsonl"
    supp.write_text("".join('{"prompt": "q%d?", "response": "a%d."}\n' % (i, i) for i in range(40)))

    train, _ = load_corpus(docs, tokenizer, max_length=64, val_fraction=0.0, seed=0,
                           supplement=supp, supplement_fraction=0.1)

    report = train.supplement_report
    assert report["n_available"] == 40 and 0 < report["n_used"] < 40
    assert abs(report["achieved_fraction"] - 0.1) < 0.05 and report["under_target"] is False


def test_the_held_out_split_is_taken_from_raw_documents_before_mixing(tmp_path, tiny_model):
    _, tokenizer = tiny_model
    docs = tmp_path / "docs"; docs.mkdir()
    for i in range(10):
        (docs / f"d{i}.txt").write_text(f"raw document {i} " * 20)
    supp = tmp_path / "supplement.jsonl"
    supp.write_text('{"prompt": "q?", "response": "PAIRTEXT."}\n' * 5)

    train_docs, held = split_documents(docs, 0.2, seed=3)
    train, val = load_corpus(docs, tokenizer, max_length=64, val_fraction=0.2, seed=3,
                             supplement=supp, supplement_fraction=0.3)

    assert len(held) == 2 and val.report["n_docs"] == 2
    assert not any("PAIRTEXT" in t for t in held)
    assert train.report["n_docs"] == 8 + train.supplement_report["n_used"]


def test_a_single_document_corpus_trains_on_it_and_holds_nothing_out(tmp_path, tiny_model):
    _, tokenizer = tiny_model
    docs = tmp_path / "docs"; docs.mkdir()
    (docs / "only.txt").write_text("one long document " * 50)
    train, val = load_corpus(docs, tokenizer, max_length=64, val_fraction=0.1, seed=0)
    assert train.report["n_docs"] == 1 and val is None     # never an empty held-out corpus


def test_render_pair_uses_the_non_thinking_template_when_there_is_one():
    class T:
        def apply_chat_template(self, messages, tokenize=False, enable_thinking=None, **_):
            assert enable_thinking is False
            return "<u>" + messages[0]["content"] + "</u><a>" + messages[1]["content"] + "</a>"
    assert render_pair(T(), "q", "a") == "<u>q</u><a>a</a>"

    class NoTemplate:
        def apply_chat_template(self, *a, **k):
            raise ValueError("none")
    assert render_pair(NoTemplate(), "q", "a") == "q\na"


def test_load_supplement_renders_pairs_in_file_order_and_skips_blank_and_promptless_lines(
        tmp_path):
    supp = tmp_path / "supplement.jsonl"
    supp.write_text('{"prompt": "q1", "response": "a1"}\n\n{"response": "orphan"}\n'
                    '{"prompt": "q2", "response": "a2"}\n')

    class NoTemplate:
        def apply_chat_template(self, *a, **k):
            raise ValueError("none")

    assert load_supplement(supp, NoTemplate()) == ["q1\na1", "q2\na2"]
