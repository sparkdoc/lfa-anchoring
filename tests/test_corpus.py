"""Tests for the corpus loader.

The load-bearing behaviour is the chunking stream: a chunk is ``max_length`` tokens, the
per-epoch offset is deterministic in ``(seed, epoch)``, and epoch 0 is always offset 0. The
companion's default (``keep_short_whole=True``) keeps a document that fits in one chunk present
in every epoch; the legacy default (``False``) is what the the research code record was produced under
and must stay reproducible, so both branches are exercised here.

``tiny_texts`` tokenizes to 69-280 tokens under the char-level fixture tokenizer, so
``max_length=128`` gives both short documents (kept whole / droppable) and multi-chunk ones.
The tokenizer is lossy on decode, so nothing here asserts on decoded text.
"""

import json

import pytest
import torch

from lfa.corpus import ChunkedCorpus, load_corpus, load_texts, make_dataloader


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
    _, tok = tiny_model
    ds = ChunkedCorpus(tiny_texts, tok, max_length=128, keep_short_whole=False)
    ds.rechunk(epoch=5)
    assert ds.report["n_dropped_short_chunks"] >= 0   # counter exists; value depends on offset

    # Same (seed, epoch) => same stream, in a second instance as well as a second call.
    other = ChunkedCorpus(tiny_texts, tok, max_length=128, keep_short_whole=False)
    other.rechunk(epoch=5)
    assert other.report == ds.report
    assert [ex["input_ids"].tolist() for ex in other] == [ex["input_ids"].tolist() for ex in ds]


def test_legacy_dropout_actually_drops_and_default_does_not(tiny_model, tiny_texts):
    """The historical dropout is real: some epochs lose whole short documents under
    ``keep_short_whole=False``, and none ever do under the companion's default."""
    _, tok = tiny_model
    legacy = ChunkedCorpus(tiny_texts, tok, max_length=128, keep_short_whole=False)
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
    ds = ChunkedCorpus(["a short document"], tok, max_length=512, keep_short_whole=False)
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
