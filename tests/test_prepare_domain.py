"""Splitting long files into documents, and the corpus too small to hold anything out.

A downloaded book is the commonest first corpus, and prepared as it stands it is ONE document:
the trainer holds out whole documents, so it holds out nothing and there is no per-epoch curve to
tune against. ``split_chars`` cuts each file into documents at paragraph boundaries; a corpus
still too small for a held-out split is warned about, or refused under ``require_held_out`` (what
``prepare-domain --supplement`` sets) before anything is written. Everything here is offline and
CPU-only; the CLI's half of the refusal is in ``test_cli.py``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from lfa.corpus import (
    MIN_CHUNK_TOKENS,
    ChunkedCorpus,
    min_documents_for_held_out,
    split_documents,
)
from lfa.prepare_domain import (
    DEFAULT_VAL_FRACTION,
    MIN_SPLIT_CHARS,
    OVERLAP_MIN_SENTENCE_CHARS,
    REPREPARATION_SHARE,
    SUGGESTED_SPLIT_CHARS,
    held_out_shortfall,
    prepare_domain,
    split_into_documents,
)
from lfa.recipe import Recipe


def paragraph(i: int, length: int = 1000) -> str:
    """A prose paragraph of exactly ``length`` characters, ending a sentence, whose every sentence
    names its paragraph -- so no two paragraphs share a sentence, as no two books' prose does."""
    parts, j = [], 0
    while len(" ".join(parts)) < length:
        parts.append(f"Sentence {j} of paragraph {i} says the words here are its own and none "
                     f"other's.")
        j += 1
    return " ".join(parts)[: length - 1].rstrip().ljust(length - 1, "x") + "."


def sentences(n: int, length: int = 100) -> str:
    """``n`` sentences of ``length`` characters each, joined by single spaces."""
    return " ".join(("s%04d " % i + "a" * length)[: length - 1] + "." for i in range(n))


# ----------------------------------------------------------------------------------- splitting

def test_paragraphs_are_packed_into_documents_of_about_the_target():
    paragraphs = [paragraph(i) for i in range(10)]
    documents = split_into_documents("\n\n".join(paragraphs), 2500)

    # 1000-character paragraphs at a 2,500 target: a document closes on its third paragraph.
    assert [d.count("\n\n") + 1 for d in documents] == [3, 3, 4]       # 4th: remainder of 1 joins
    assert all(len(d) >= 2500 for d in documents)
    # Cut only at paragraph boundaries, and nothing lost or reordered.
    assert [p for d in documents for p in d.split("\n\n")] == paragraphs


def test_a_text_shorter_than_the_target_is_one_document():
    text = "\n\n".join(paragraph(i) for i in range(3))
    assert split_into_documents(text, 10_000) == [text]


def test_a_paragraph_over_twice_the_target_is_cut_at_sentence_ends():
    long_paragraph = sentences(100)                       # 100 x 100 characters, one paragraph
    documents = split_into_documents(long_paragraph, 1000)

    assert len(documents) > 1
    assert all(d.endswith(".") for d in documents)       # every cut falls after a sentence
    assert all(1000 <= len(d) <= 2000 for d in documents[:-1])
    assert " ".join(documents) == long_paragraph         # no paragraph break was invented
    assert not any("\n" in d for d in documents)


def test_a_paragraph_with_no_sentence_end_in_reach_is_cut_at_whitespace():
    long_paragraph = " ".join(["word"] * 2000)            # 9,999 characters, no full stop
    documents = split_into_documents(long_paragraph, 1000)

    assert len(documents) > 1
    assert all(not d.startswith(" ") and not d.endswith(" ") for d in documents)
    assert all(set(d.split(" ")) == {"word"} for d in documents)   # never mid-word
    assert " ".join(documents) == long_paragraph


def test_a_run_with_no_whitespace_at_all_is_cut_at_the_target():
    blob = "x" * 5000
    documents = split_into_documents(blob, 1000)
    assert "".join(documents) == blob
    assert [len(d) for d in documents[:-1]] == [1000] * (len(documents) - 1)


def test_a_short_remainder_joins_the_document_before_it():
    paragraphs = [paragraph(0, 3000), paragraph(1, 3000), paragraph(2, 200)]
    documents = split_into_documents("\n\n".join(paragraphs), 2500)
    assert len(documents) == 2
    assert documents[-1].endswith("\n\n" + paragraphs[2])          # kept, not dropped


def test_the_continuation_of_a_cut_paragraph_rejoins_with_a_space():
    """The tail of a cut paragraph joining the document before it is the same paragraph, so it
    is joined as one -- a paragraph break there would be invented."""
    long_paragraph = sentences(31)                        # 3,130 characters at a 1,000 target
    documents = split_into_documents(long_paragraph, 1000)
    assert " ".join(documents) == long_paragraph


def test_splitting_is_deterministic():
    text = "\n\n".join([paragraph(i, 700) for i in range(40)] + [sentences(90)])
    assert split_into_documents(text, 1500) == split_into_documents(text, 1500)


def test_a_split_size_below_one_is_refused():
    with pytest.raises(ValueError, match="positive"):
        split_into_documents("text", 0)


def test_the_suggested_size_is_the_walkthroughs():
    assert SUGGESTED_SPLIT_CHARS == 3500


# ------------------------------------------------------------------------- prepare_domain

def book(path: Path, n_paragraphs: int = 30, first: int = 0) -> Path:
    """A file of ``n_paragraphs`` distinct 1,000-character paragraphs; ``first`` makes another
    book's text differ from this one's."""
    path.write_text("\n\n".join(paragraph(i) for i in range(first, first + n_paragraphs)),
                    encoding="utf-8")
    return path


def test_split_chars_writes_numbered_documents_under_the_files_stem(tmp_path):
    src = book(tmp_path / "wells.txt")
    out = tmp_path / "out"

    written = prepare_domain(src, out, split_chars=3500)

    names = [p.name for p in written]
    assert names == [f"wells-{i:04d}.txt" for i in range(1, len(names) + 1)]
    assert len(names) == 8                                  # 30 x 1,000 at 3,500: 4 paragraphs each
    pieces = [p.read_text(encoding="utf-8") for p in written]
    assert "\n\n".join(pieces) == src.read_text(encoding="utf-8")


def test_without_split_chars_a_file_is_one_document(tmp_path, caplog):
    src = book(tmp_path / "wells.txt")
    with caplog.at_level(logging.WARNING, logger="lfa.prepare_domain"):
        written = prepare_domain(src, tmp_path / "out")
    assert [p.name for p in written] == ["wells.txt"]
    assert any("--split-chars 3500" in r.getMessage() for r in caplog.records
               if r.levelno == logging.WARNING)                 # and it says how to split it


def test_one_document_is_written_and_warned_about_with_the_fix(tmp_path, caplog):
    src = book(tmp_path / "wells.txt")
    with caplog.at_level(logging.WARNING, logger="lfa.prepare_domain"):
        written = prepare_domain(src, tmp_path / "out")

    assert len(written) == 1 and written[0].exists()
    [warning] = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "holds 1 document(s)" in warning and "at least 2" in warning
    assert "--split-chars 3500" in warning
    assert "fresh --out directory" in warning               # re-running into it would add to it


def test_a_split_corpus_is_not_warned_about(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="lfa.prepare_domain"):
        prepare_domain(book(tmp_path / "wells.txt"), tmp_path / "out", split_chars=3500)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_require_held_out_refuses_before_anything_is_written(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValueError) as refusal:
        prepare_domain(book(tmp_path / "wells.txt"), out, require_held_out=True)

    message = str(refusal.value)
    assert "would hold at most 1 document(s)" in message
    assert "--split-chars 3500" in message and "add more files" in message
    assert "Nothing was read or written" in message
    assert not out.exists()                                 # so the re-run doubles nothing


def test_require_held_out_passes_a_split_book(tmp_path):
    written = prepare_domain(book(tmp_path / "wells.txt"), tmp_path / "out", split_chars=3500,
                             require_held_out=True)
    assert len(written) == 8


def test_documents_already_in_the_directory_count(tmp_path, caplog):
    """The trainer reads the whole directory, so a second file prepared beside a first is a
    two-document corpus and is not refused."""
    out = tmp_path / "out"
    prepare_domain(book(tmp_path / "a.txt"), out)               # warned: one document so far
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="lfa.prepare_domain"):
        prepare_domain(book(tmp_path / "b.txt", first=100), out, require_held_out=True)
    assert sorted(p.name for p in out.iterdir()) == ["a.txt", "b.txt"]
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_combine_is_one_document_and_its_fix_says_so(tmp_path):
    a, b = book(tmp_path / "a.txt"), book(tmp_path / "b.txt", first=100)
    with pytest.raises(ValueError, match="--combine writes one file"):
        prepare_domain([a, b], tmp_path / "out", combine=True, require_held_out=True)


def test_split_chars_and_combine_are_refused_together(tmp_path):
    with pytest.raises(ValueError, match="--split-chars and --combine"):
        prepare_domain(book(tmp_path / "a.txt"), tmp_path / "out", combine=True, split_chars=3500)


def test_a_recipe_holding_nothing_out_is_not_warned(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="lfa.prepare_domain"):
        prepare_domain(book(tmp_path / "a.txt"), tmp_path / "out", val_fraction=0.0,
                       require_held_out=True)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_min_length_applies_to_the_input_before_splitting(tmp_path):
    src = book(tmp_path / "wells.txt", n_paragraphs=3)                  # 3,004 characters
    written = prepare_domain(src, tmp_path / "out", split_chars=500, min_length=2000)
    assert len(written) == 3                                # pieces under 2,000 are kept


# ------------------------------------------------------------- the minimum is the trainer's

def test_the_default_fraction_is_the_bundled_recipes():
    assert DEFAULT_VAL_FRACTION == Recipe.__dataclass_fields__["val_fraction"].default


@pytest.mark.parametrize("fraction", [0.0, 0.05, 0.1, 0.25, 0.5, 0.6, 0.9])
def test_the_minimum_is_where_split_documents_starts_holding_out(tmp_path, fraction):
    needed = min_documents_for_held_out(fraction)
    corpus = tmp_path / "c"
    corpus.mkdir()
    for i in range(needed):
        (corpus / f"{i}.txt").write_text(f"document {i}")
    if fraction > 0:
        _, held = split_documents(corpus, fraction, seed=0)
        assert held                                         # at the minimum, something is held
        (corpus / "0.txt").unlink()
        if needed - 1 >= 1:
            try:
                _, held = split_documents(corpus, fraction, seed=0)
            except ValueError:
                held = []                                   # refused: holds out everything
            assert not held                                 # one fewer, and nothing is
    else:
        assert needed == 1


def test_held_out_shortfall():
    assert min_documents_for_held_out(0.1) == 2
    assert held_out_shortfall(1) == 1 and held_out_shortfall(2) == 0
    assert held_out_shortfall(0) == 2 and held_out_shortfall(1, 0.0) == 0


def test_the_trial_book_case_end_to_end(tmp_path):
    """A Gutenberg-sized single book (~730 k characters): unsplit it is refused under
    ``require_held_out``; split at the suggested size it is a corpus of ~200 documents."""
    src = tmp_path / "pg35461.txt"
    src.write_text("\n\n".join(paragraph(i, 700) for i in range(1050)), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape("--split-chars 3500")):
        prepare_domain(src, tmp_path / "refused", require_held_out=True)
    written = prepare_domain(src, tmp_path / "out", split_chars=SUGGESTED_SPLIT_CHARS,
                             require_held_out=True)
    assert 190 <= len(written) <= 215


# ----------------------------------------------------------- the same text prepared twice

def licence(n: int = 2) -> list[str]:
    """``n`` 1,000-character paragraphs of boilerplate that two different books both carry."""
    return [("Licence clause %d: this ebook is for the use of anyone anywhere at no cost. " % i
             * 20)[:1000] for i in range(n)]


def write(path: Path, paragraphs: list[str]) -> Path:
    path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    return path


def test_the_thresholds_are_the_measured_ones():
    assert OVERLAP_MIN_SENTENCE_CHARS == 60 and REPREPARATION_SHARE == 0.5


def test_a_split_rerun_beside_the_unsplit_file_is_refused(tmp_path, caplog):
    """The up-arrow re-run: prepared unsplit and warned, then prepared again into the same --out
    with the split the warning named. Written, the corpus would hold the whole book beside its
    own pieces, and every held-out piece would be on the training side verbatim."""
    src, out = book(tmp_path / "wells.txt"), tmp_path / "out"
    with caplog.at_level(logging.WARNING, logger="lfa.prepare_domain"):
        prepare_domain(src, out)
    assert any("--split-chars 3500" in r.getMessage() for r in caplog.records)

    with pytest.raises(ValueError) as refusal:
        prepare_domain(src, out, split_chars=3500, require_held_out=True)

    message = str(refusal.value)
    assert f"{src}: 100% of its sentence text" in message
    assert f"the largest shares are in {out / 'wells.txt'} (100%)" in message
    assert "fresh --out directory" in message
    assert "delete the earlier preparation of this text" in message
    assert [p.name for p in out.iterdir()] == ["wells.txt"]           # nothing was added


def test_a_rerun_with_the_header_stripped_is_refused(tmp_path):
    """What the docs ask for -- strip the boilerplate -- must not slip a re-run past the check."""
    body = [paragraph(i) for i in range(30)]
    out = tmp_path / "out"
    prepare_domain(write(tmp_path / "wells.txt", licence() + body), out)
    (tmp_path / "edited").mkdir()
    stripped = write(tmp_path / "edited" / "wells.txt", body)
    with pytest.raises(ValueError, match="100% of its sentence text"):
        prepare_domain(stripped, out, split_chars=3500)
    assert [p.name for p in out.iterdir()] == ["wells.txt"]


def test_a_rerun_with_one_word_changed_is_refused(tmp_path):
    body = [paragraph(i) for i in range(30)]
    out = tmp_path / "out"
    prepare_domain(write(tmp_path / "wells.txt", body), out)
    edited = body[:]
    edited[7] = edited[7].replace("words", "WORDS", 1)                # one sentence of 390
    (tmp_path / "edited").mkdir()
    with pytest.raises(ValueError) as refusal:
        prepare_domain(write(tmp_path / "edited" / "wells.txt", edited), out, split_chars=3500)
    assert "100% of its sentence text" in str(refusal.value)         # 389 of 390, rounded


def test_the_same_file_prepared_twice_is_refused(tmp_path):
    src, out = book(tmp_path / "wells.txt"), tmp_path / "out"
    prepare_domain(src, out)
    with pytest.raises(ValueError, match="100% of its sentence text"):
        prepare_domain(src, out)
    assert [p.name for p in out.iterdir()] == ["wells.txt"]           # no wells_1.txt


def test_the_same_split_prepared_twice_is_refused_naming_the_files(tmp_path):
    src, out = book(tmp_path / "wells.txt"), tmp_path / "out"
    first = prepare_domain(src, out, split_chars=3500)
    with pytest.raises(ValueError) as refusal:
        prepare_domain(src, out, split_chars=3500)
    message = str(refusal.value)
    assert f"the largest shares are in {out / 'wells-000'}" in message
    assert ".txt (13%)" in message                                    # 4 of 30 paragraphs
    assert "and 5 more file(s)" in message                            # 8 files, 3 named
    assert sorted(out.iterdir()) == sorted(first)


def test_a_different_text_sharing_a_licence_warns_and_is_prepared(tmp_path, caplog):
    """Two Gutenberg books share the licence and nothing else: a warning, not a refusal."""
    out = tmp_path / "out"
    prepare_domain(write(tmp_path / "a.txt", licence() + [paragraph(i) for i in range(30)]),
                   out, split_chars=3500)
    caplog.clear()
    other = write(tmp_path / "b.txt", [paragraph(i) for i in range(100, 130)] + licence())
    with caplog.at_level(logging.WARNING, logger="lfa.prepare_domain"):
        written = prepare_domain(other, out, split_chars=3500)

    assert written and all(p.exists() for p in written)
    [warning] = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert f"{other} shares 6.1% of its sentence text" in warning    # 2 of 32 paragraphs
    assert "licence or front matter" in warning
    assert "Strip the boilerplate first" in warning


def test_the_threshold_is_inclusive(tmp_path, caplog):
    """Half of the paragraph text already there is refused; a quarter is warned about."""
    out = tmp_path / "out"
    # Two-digit paragraph numbers throughout, so every paragraph's sentences are the same length.
    prepare_domain(write(tmp_path / "a.txt", [paragraph(i) for i in range(10, 14)]), out)
    (tmp_path / "half").mkdir()
    with pytest.raises(ValueError, match="50% of its sentence text"):
        prepare_domain(write(tmp_path / "half" / "b.txt",
                             [paragraph(10), paragraph(11), paragraph(50), paragraph(51)]), out)
    with caplog.at_level(logging.WARNING, logger="lfa.prepare_domain"):
        prepare_domain(write(tmp_path / "c.txt",
                             [paragraph(10), paragraph(60), paragraph(61), paragraph(62)]), out)
    assert any("shares 25% of its sentence text" in r.getMessage() for r in caplog.records)


def test_short_sentences_are_not_compared(tmp_path, caplog):
    """Headings and short lines recur across unrelated texts; under 60 characters they do not
    count, so two texts sharing only those are prepared without a word."""
    headings = ["CHAPTER ONE", "Contents", "x" * (OVERLAP_MIN_SENTENCE_CHARS - 1),
                "It was the best of times. It was the worst of times."]
    out = tmp_path / "out"
    prepare_domain(write(tmp_path / "a.txt", headings + [paragraph(i) for i in range(5)]), out)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="lfa.prepare_domain"):
        prepare_domain(write(tmp_path / "b.txt", headings + [paragraph(i) for i in range(9, 14)]),
                       out)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_two_identical_inputs_in_one_call_are_refused(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    book(src / "a.txt")
    book(src / "b.txt")
    with pytest.raises(ValueError) as refusal:
        prepare_domain(src, tmp_path / "out", split_chars=3500)
    message = str(refusal.value)
    assert f"{src / 'b.txt'}: 100% of its sentence text" in message
    assert (f"is already in this call; the largest shares are in the earlier input "
            f"{src / 'a.txt'}") in message
    assert "Give each text once" in message
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------- refusing before reading anything

def test_a_certain_refusal_is_made_before_any_file_is_read(tmp_path, monkeypatch):
    """One PDF and no split can only ever be one document: the refusal comes before marker's
    models load (several GB of VRAM) and before the file is extracted."""
    import lfa.prepare_domain as module

    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 not really")
    monkeypatch.setattr(module, "make_pdf_converter",
                        lambda: pytest.fail("marker was loaded for a certain refusal"))
    monkeypatch.setattr(module, "extract", lambda *a, **k: pytest.fail("a file was extracted"))
    with pytest.raises(ValueError, match="would hold at most 1 document") as refusal:
        prepare_domain(pdf, tmp_path / "out", require_held_out=True)
    assert "Nothing was read or written" in str(refusal.value)
    assert not (tmp_path / "out").exists()


# -------------------------------------------------------------------- bounds on the inputs

@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
def test_a_held_out_fraction_outside_the_recipe_range_is_refused(tmp_path, fraction):
    """At 1.0 or above no document count leaves anything to train on; the search for one must
    refuse rather than run for ever."""
    with pytest.raises(ValueError, match=r"val_fraction must be in \[0, 1\)"):
        min_documents_for_held_out(fraction)
    with pytest.raises(ValueError, match=r"val_fraction must be in \[0, 1\)"):
        prepare_domain(book(tmp_path / "a.txt"), tmp_path / "out", val_fraction=fraction)


def test_split_chars_has_a_floor_that_names_the_suggested_size(tmp_path):
    with pytest.raises(ValueError) as refusal:
        prepare_domain(book(tmp_path / "a.txt"), tmp_path / "out",
                       split_chars=MIN_SPLIT_CHARS - 1)
    message = str(refusal.value)
    assert f"below the floor of {MIN_SPLIT_CHARS}" in message and "3500" in message
    assert f"{MIN_CHUNK_TOKENS} tokens" in message
    assert not (tmp_path / "out").exists()
    assert prepare_domain(book(tmp_path / "a.txt"), tmp_path / "out",
                          split_chars=MIN_SPLIT_CHARS)


def test_at_the_floor_no_split_document_is_under_half_of_it(tmp_path):
    """The floor's arithmetic: every split document is at least half the size (bar a whole file
    shorter than that), so at 500 characters each one is 250 or more -- ten tokens only at 25
    characters a token, which no prose comes near."""
    text = "\n\n".join([paragraph(i, 120) for i in range(30)] + [sentences(40, 60)]
                     + [" ".join(["w"] * 900), "z" * 1300, paragraph(99, 260)])
    documents = split_into_documents(text, MIN_SPLIT_CHARS)
    assert min(len(d) for d in documents) >= MIN_SPLIT_CHARS / 2


def test_at_the_floor_the_chunker_drops_no_document(tmp_path, tiny_model):
    _, tokenizer = tiny_model
    written = prepare_domain(book(tmp_path / "a.txt"), tmp_path / "out",
                             split_chars=MIN_SPLIT_CHARS)
    corpus = ChunkedCorpus([p.read_text() for p in written], tokenizer, max_length=128)
    assert corpus.report["n_docs"] == len(written)
    assert corpus.report["n_dropped_short_chunks"] == 0


# ------------------------------------- short-paragraph texts, and re-runs of a small split

def faq_corpus(directory: Path, n_files: int = 12) -> Path:
    """Question-and-answer files: many paragraphs, none near 200 characters, one sentence each."""
    directory.mkdir()
    for f in range(n_files):
        write(directory / f"faq{f:02d}.txt",
              [f"Question {q} of file {f}: how is the answer to this one found, and by whom?"
               for q in range(15)])
    return directory


def test_an_faq_corpus_prepared_twice_is_refused(tmp_path):
    """No paragraph here is long; every one is a sentence of 60 characters or more, so a second
    preparation into the same --out is seen and refused rather than written as `_1` copies."""
    src, out = faq_corpus(tmp_path / "faq"), tmp_path / "out"
    assert len(prepare_domain(src, out)) == 12
    with pytest.raises(ValueError, match="100% of its sentence text"):
        prepare_domain(src, out)
    assert len(list(out.iterdir())) == 12


def test_a_one_file_play_re_split_into_the_same_out_is_refused(tmp_path):
    """The D1 case for a text of short speeches: unsplit and warned, then re-split beside itself."""
    play = write(tmp_path / "play.txt",
                 [f"HAMLET. Speech {i}: to say this line of the play is to say it once, and only "
                  f"once." for i in range(400)])
    out = tmp_path / "out"
    prepare_domain(play, out)
    with pytest.raises(ValueError, match="100% of its sentence text"):
        prepare_domain(play, out, split_chars=3500)
    assert [p.name for p in out.iterdir()] == ["play.txt"]


@pytest.mark.parametrize("rerun_chars", [500, 3500])
def test_a_rerun_against_an_earlier_small_split_is_refused(tmp_path, rerun_chars):
    """An earlier split at 500 cut every paragraph over 1,000 characters at its sentence ends, so
    the paragraphs a re-run holds are not in --out -- but each of their sentences is."""
    src = write(tmp_path / "book.txt", [paragraph(i, 2500) for i in range(10, 22)])
    out = tmp_path / "out"
    first = prepare_domain(src, out, split_chars=MIN_SPLIT_CHARS)
    assert len(first) > 12                                  # the long paragraphs were cut
    with pytest.raises(ValueError) as refusal:
        prepare_domain(src, out, split_chars=rerun_chars)
    share = int(re.search(r": (\d+)% of its sentence text", str(refusal.value)).group(1))
    assert share >= 98
    assert sorted(out.iterdir()) == sorted(first)
