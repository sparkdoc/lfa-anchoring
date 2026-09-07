"""Seed-corpus preparation and domain-document preparation.

Everything here is offline. The Hugging Face downloads are exercised through an injected
``loader``; the real ones are a manual check (Task 18), not a test.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

from lfa.prepare_domain import clean_text, find_input_files, prepare_domain
from lfa.seed_corpus import (
    INSTRUCTION_SOURCES,
    MIN_PRETRAINING_CHARS,
    PRETRAINING_SOURCES,
    REDPAJAMA_PATH,
    SHIPPED_COMPOSITION,
    allocate,
    download_instruction,
    download_pretraining,
    extract_alpaca,
    extract_code_alpaca,
    extract_dolly,
    extract_oasst2_pairs,
    extract_ultrachat,
    is_arxiv,
    is_book,
    is_github,
    is_stackexchange,
    is_web,
    is_wikipedia,
    parse_meta,
    prepare_seed_corpus,
    weighted_mix,
)

HAVE_BS4 = importlib.util.find_spec("bs4") is not None and importlib.util.find_spec("markdownify") is not None
HAVE_MARKER = importlib.util.find_spec("marker") is not None

PARAGRAPH = "Anchoring prices the function a sub-module computes on the states it actually sees. "


# ==============================================================================================
# prepare_domain
# ==============================================================================================

def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_prepare_domain_writes_one_txt_per_input(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    _write(src / "a.txt", PARAGRAPH * 20)
    _write(src / "b.md", "# Heading\n\n" + PARAGRAPH * 20)

    written = prepare_domain([src], out)

    assert [p.name for p in written] == ["a.txt", "b.txt"]
    assert all(p.exists() and p.suffix == ".txt" for p in written)
    assert "Anchoring prices" in written[0].read_text(encoding="utf-8")
    assert written[1].read_text(encoding="utf-8").startswith("# Heading")


def test_prepare_domain_accepts_explicit_files_and_a_bare_path(tmp_path):
    out = tmp_path / "out"
    a = _write(tmp_path / "a.txt", PARAGRAPH * 20)
    b = _write(tmp_path / "b.md", PARAGRAPH * 20)

    assert len(prepare_domain([a, b], out)) == 2
    assert len(prepare_domain(a, tmp_path / "out2")) == 1


def test_prepare_domain_drops_a_file_under_min_length(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    _write(src / "long.txt", PARAGRAPH * 20)
    _write(src / "short.txt", "too short to train on")

    written = prepare_domain([src], out, min_length=1000)

    assert [p.name for p in written] == ["long.txt"]
    assert not (out / "short.txt").exists()


def test_prepare_domain_min_length_is_characters_after_cleaning(tmp_path):
    # A link farm: long on disk, almost nothing once the URLs and blank lines are gone.
    raw = "[see also](https://example.com/a/very/long/tracking/url/that/carries/no/prose)\n\n" * 40
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    _write(src / "links.md", raw)

    assert len(raw) > 1000 > len(clean_text(raw))          # raw would pass; cleaned must not

    assert prepare_domain([src], out, min_length=1000) == []
    assert prepare_domain([src], tmp_path / "out2", min_length=100) != []


def test_prepare_domain_combine_writes_one_file(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    _write(src / "a.txt", PARAGRAPH * 20)
    _write(src / "b.md", PARAGRAPH * 20)

    written = prepare_domain([src], out, combine=True)

    assert len(written) == 1
    combined = written[0].read_text(encoding="utf-8")
    assert written[0].name == "combined_domain_data.txt"
    assert "# Source: a.txt" in combined and "# Source: b.md" in combined
    assert combined.count("=" * 80) == 1                  # one separator between two documents


def test_prepare_domain_recursive_flag(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    (src / "nested").mkdir(parents=True)
    _write(src / "top.txt", PARAGRAPH * 20)
    _write(src / "nested" / "deep.txt", PARAGRAPH * 20)

    assert len(prepare_domain([src], out, recursive=True)) == 2
    assert [p.name for p in prepare_domain([src], tmp_path / "out2", recursive=False)] == ["top.txt"]


def test_prepare_domain_disambiguates_duplicate_stems(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    (src / "nested").mkdir(parents=True)
    _write(src / "doc.txt", PARAGRAPH * 20)
    _write(src / "nested" / "doc.md", PARAGRAPH * 20)

    written = prepare_domain([src], out)

    assert sorted(p.name for p in written) == ["doc.txt", "doc_1.txt"]


def test_prepare_domain_ignores_unsupported_extensions(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    _write(src / "keep.txt", PARAGRAPH * 20)
    _write(src / "skip.csv", PARAGRAPH * 20)

    assert [p.name for p in prepare_domain([src], out)] == ["keep.txt"]


def test_prepare_domain_raises_when_nothing_matches(tmp_path):
    out = tmp_path / "out"
    (tmp_path / "empty").mkdir()

    with pytest.raises(ValueError, match="No supported"):
        prepare_domain([tmp_path / "empty"], out)


def test_find_input_files_is_sorted_and_deduplicated(tmp_path):
    a = _write(tmp_path / "a.txt", "x")
    b = _write(tmp_path / "b.txt", "x")

    assert find_input_files([tmp_path, a, b]) == [a, b]


def test_find_input_files_rejects_a_path_that_does_not_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_input_files([tmp_path / "nowhere"])


@pytest.mark.skipif(not HAVE_BS4, reason="needs the [html] extra")
def test_prepare_domain_extracts_html(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    body = "".join(f"<p>{PARAGRAPH}</p>" for _ in range(20))
    _write(src / "c.html", f"<html><head><title>t</title></head><body><nav>menu</nav>{body}"
                           f"<script>evil()</script></body></html>")

    written = prepare_domain([src], out)

    text = written[0].read_text(encoding="utf-8")
    assert [p.name for p in written] == ["c.txt"]
    assert "Anchoring prices" in text
    assert "evil()" not in text and "menu" not in text and "<p>" not in text


@pytest.mark.skipif(HAVE_BS4, reason="only meaningful without the [html] extra")
def test_prepare_domain_names_the_html_extra_when_bs4_is_missing(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    _write(src / "c.html", "<html><body>" + PARAGRAPH * 20 + "</body></html>")

    with pytest.raises(ImportError, match=r"\[html\]"):
        prepare_domain([src], out)


@pytest.mark.skipif(HAVE_MARKER, reason="only meaningful without the [pdf] extra")
def test_prepare_domain_names_the_pdf_extra_when_marker_is_missing(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    (src / "d.pdf").write_bytes(b"%PDF-1.4 not really a pdf")

    with pytest.raises(ImportError, match=r"\[pdf\]"):
        prepare_domain([src], out)


def test_clean_text_removes_markup_artifacts_and_unwraps_paragraphs(tmp_path):
    raw = (
        "# Title\n"
        "A sentence with a <sup>3</sup> footnote and a [link](https://example.com).\n"
        "Continued on the next source line.\n"
        "\n"
        "![alt](figure.png)\n"
        "\n"
        "- a list item\n"
        "- another item\n"
    )

    cleaned = clean_text(raw)

    assert "<sup>" not in cleaned and "figure.png" not in cleaned
    assert "!alt" not in cleaned and "![" not in cleaned   # images go before links
    assert "a link." in cleaned and "(https://example.com)" not in cleaned
    assert "footnote and a link. Continued on the next source line." in cleaned
    assert "# Title" in cleaned
    assert "- a list item\n- another item" in cleaned          # structural lines are not unwrapped


# ==============================================================================================
# Seed corpus: source predicates
# ==============================================================================================

def test_parse_meta_accepts_a_dict_or_a_python_literal_string():
    assert parse_meta({"meta": {"url": "u"}}) == {"url": "u"}
    assert parse_meta({"meta": "{'url': 'u'}"}) == {"url": "u"}
    assert parse_meta({"meta": "not a literal"}) == {}
    assert parse_meta({}) == {}


@pytest.mark.parametrize("predicate, meta, expected", [
    (is_arxiv, {"arxiv_id": "2401.00001"}, True),
    (is_arxiv, {"url": "https://arxiv.org/abs/2401.00001"}, True),
    (is_arxiv, {"url": "https://example.com/page"}, False),
    (is_wikipedia, {"url": "https://en.wikipedia.org/wiki/Anchor"}, True),
    (is_wikipedia, {"url": "https://example.com/page"}, False),
    (is_github, {"url": "https://github.com/org/repo"}, True),
    (is_github, {"url": "https://example.com/page"}, False),
    (is_stackexchange, {"url": "https://stats.stackexchange.com/q/1"}, True),
    (is_stackexchange, {"url": "https://example.com/page"}, False),
    (is_book, {"short_book_title": "Moby Dick"}, True),
    (is_book, {"url": "https://example.com/page"}, False),
    (is_web, {"url": "https://example.com/page"}, True),
    (is_web, {"url": "https://en.wikipedia.org/wiki/Anchor"}, False),
    (is_web, {"url": "https://github.com/org/repo"}, False),
    (is_web, {"url": "https://stats.stackexchange.com/q/1"}, False),
    (is_web, {"url": "https://arxiv.org/abs/2401.00001"}, False),
    (is_web, {"short_book_title": "Moby Dick"}, False),         # no url at all
])
def test_redpajama_predicates(predicate, meta, expected):
    assert predicate({"text": "x", "meta": meta}) is expected


def test_pretraining_and_instruction_source_tables_are_the_shipped_ones():
    assert set(PRETRAINING_SOURCES) == {
        "redpajama_arxiv", "redpajama_book", "redpajama_wikipedia",
        "redpajama_stackexchange", "redpajama_web", "redpajama_github",
    }
    assert set(INSTRUCTION_SOURCES) == {"dolly", "alpaca", "code_alpaca", "ultrachat", "oasst2"}
    assert set(SHIPPED_COMPOSITION["pretraining"]) == set(PRETRAINING_SOURCES)
    assert set(SHIPPED_COMPOSITION["instruction"]) == set(INSTRUCTION_SOURCES)
    assert sum(SHIPPED_COMPOSITION["pretraining"].values()) == 9663
    assert sum(SHIPPED_COMPOSITION["instruction"].values()) == 966


# ==============================================================================================
# Seed corpus: instruction extractors
# ==============================================================================================

def test_extract_dolly_with_and_without_context():
    with_context = extract_dolly({"instruction": "Summarize.", "context": "A long passage.",
                                  "response": "A summary."})
    assert with_context == {"prompt": "Summarize.\n\nContext: A long passage.",
                            "response": "A summary."}

    assert extract_dolly({"instruction": "Summarize.", "context": "", "response": "A summary."}) \
        == {"prompt": "Summarize.", "response": "A summary."}
    assert extract_dolly({"instruction": "", "response": "A summary."}) is None
    assert extract_dolly({"instruction": "Summarize.", "response": "  "}) is None


def test_extract_alpaca_and_code_alpaca_fold_the_input_into_the_prompt():
    for extract in (extract_alpaca, extract_code_alpaca):
        assert extract({"instruction": "Reverse it.", "input": "abc", "output": "cba"}) == {
            "prompt": "Reverse it.\n\nInput: abc", "response": "cba"}
        assert extract({"instruction": "Reverse it.", "input": "", "output": "cba"}) == {
            "prompt": "Reverse it.", "response": "cba"}
        assert extract({"instruction": "Reverse it.", "output": ""}) is None


def test_extract_ultrachat_returns_every_turn_with_history():
    pairs = extract_ultrachat({"messages": [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
    ]})

    assert len(pairs) == 2
    assert pairs[0] == {"prompt": "First question", "response": "First answer"}
    assert pairs[1]["response"] == "Second answer"
    assert "Conversation history:" in pairs[1]["prompt"]
    assert "User: First question" in pairs[1]["prompt"]
    assert "Assistant: First answer" in pairs[1]["prompt"]
    assert pairs[1]["prompt"].endswith("User: Second question")

    assert extract_ultrachat({"messages": []}) is None


def test_extract_oasst2_pairs_walks_the_conversation_tree():
    rows = [
        {"message_id": "p1", "parent_id": None, "role": "prompter", "lang": "en",
         "text": "What does the anchor price?"},
        {"message_id": "a1", "parent_id": "p1", "role": "assistant", "lang": "en",
         "text": "The function the sub-module computes."},
        {"message_id": "a2", "parent_id": "p1", "role": "assistant", "lang": "en",
         "text": "Its outputs on sampled hidden states."},
        {"message_id": "p2", "parent_id": None, "role": "prompter", "lang": "de",
         "text": "Was misst der Anker hier eigentlich?"},
        {"message_id": "a3", "parent_id": "p2", "role": "assistant", "lang": "de",
         "text": "Die Funktion des Teilmoduls hier."},
        {"message_id": "p3", "parent_id": None, "role": "prompter", "lang": "en", "text": "short"},
    ]

    pairs = extract_oasst2_pairs(rows)

    assert len(pairs) == 2                                    # the German tree and "short" are out
    assert {p["response"] for p in pairs} == {
        "The function the sub-module computes.", "Its outputs on sampled hidden states."}
    assert all(p["prompt"] == "What does the anchor price?" for p in pairs)
    assert all(p["source"] == "oasst2" for p in pairs)

    assert len(extract_oasst2_pairs(rows, n_samples=1, seed=1)) == 1
    assert extract_oasst2_pairs(rows, n_samples=1, seed=1) == extract_oasst2_pairs(rows, n_samples=1, seed=1)


def test_extractors_are_registered_on_their_sources():
    assert INSTRUCTION_SOURCES["dolly"]["extract"] is extract_dolly
    assert INSTRUCTION_SOURCES["alpaca"]["extract"] is extract_alpaca
    assert INSTRUCTION_SOURCES["code_alpaca"]["extract"] is extract_code_alpaca
    assert INSTRUCTION_SOURCES["ultrachat"]["extract"] is extract_ultrachat
    assert INSTRUCTION_SOURCES["ultrachat"]["split"] == "train_sft"


# ==============================================================================================
# Seed corpus: allocation and the 10:1 mix
# ==============================================================================================

def test_allocate_splits_a_total_by_the_source_defaults():
    assert allocate({"a": 2000, "b": 2000}, 100) == {"a": 50, "b": 50}
    assert allocate({"a": 5000, "b": 2000}, 70) == {"a": 50, "b": 20}
    assert sum(allocate({s: c["default_samples"] for s, c in PRETRAINING_SOURCES.items()},
                        9663).values()) <= 9663


def _rows(n, kind):
    if kind == "pretraining":
        return [{"text": f"pretraining document {i}", "source": "redpajama_web"} for i in range(n)]
    return [{"prompt": f"question {i}", "response": f"answer {i}", "source": "dolly"}
            for i in range(n)]


def test_weighted_mix_holds_the_ten_to_one_ratio():
    mixed = weighted_mix(_rows(500, "pretraining"), _rows(50, "instruction"), ratio=(10, 1), seed=1)

    n_instruction = sum(1 for row in mixed if "prompt" in row)
    n_pretraining = len(mixed) - n_instruction
    assert n_instruction > 0
    assert abs(n_pretraining / n_instruction - 10) <= 1


def test_weighted_mix_is_limited_by_the_scarcer_side():
    # 500 pretraining rows can only carry 50 instruction rows at 10:1; the extra ones are dropped.
    mixed = weighted_mix(_rows(500, "pretraining"), _rows(900, "instruction"), seed=1)
    n_instruction = sum(1 for row in mixed if "prompt" in row)
    assert (len(mixed) - n_instruction, n_instruction) == (500, 50)

    # 30 instruction rows cap the pretraining side at 300 even though 500 are available.
    mixed = weighted_mix(_rows(500, "pretraining"), _rows(30, "instruction"), seed=1)
    n_instruction = sum(1 for row in mixed if "prompt" in row)
    assert (len(mixed) - n_instruction, n_instruction) == (300, 30)


def test_weighted_mix_is_deterministic_in_the_seed_and_shuffles():
    a = weighted_mix(_rows(500, "pretraining"), _rows(50, "instruction"), seed=1)
    b = weighted_mix(_rows(500, "pretraining"), _rows(50, "instruction"), seed=1)
    c = weighted_mix(_rows(500, "pretraining"), _rows(50, "instruction"), seed=2)

    assert a == b
    assert a != c
    assert any("prompt" in row for row in a[:100])            # interleaved, not concatenated


def test_weighted_mix_with_one_side_empty_returns_nothing():
    assert weighted_mix(_rows(100, "pretraining"), [], seed=1) == []
    assert weighted_mix([], _rows(10, "instruction"), seed=1) == []


# ==============================================================================================
# Seed corpus: downloads through an injected loader
# ==============================================================================================

class FakeLoader:
    """Stands in for ``datasets.load_dataset``; a table is any indexable, iterable sequence."""

    def __init__(self, tables: dict[str, list[dict]]):
        self.tables = tables
        self.calls: list[tuple[str, str, str | None]] = []

    def __call__(self, path, *, split, cache_dir=None):
        self.calls.append((path, split, cache_dir))
        if path not in self.tables:
            raise FileNotFoundError(path)
        return self.tables[path]


def _redpajama_rows(per_source: int = 50) -> list[dict]:
    metas = {
        "redpajama_arxiv": {"arxiv_id": "2401.00001"},
        "redpajama_wikipedia": {"url": "https://en.wikipedia.org/wiki/Anchor"},
        "redpajama_github": {"url": "https://github.com/org/repo"},
        "redpajama_stackexchange": {"url": "https://stats.stackexchange.com/q/1"},
        "redpajama_book": {"short_book_title": "Moby Dick"},
        "redpajama_web": {"url": "https://example.com/page"},
    }
    rows = []
    for name, meta in metas.items():
        for i in range(per_source):
            rows.append({"text": f"{name} document {i}. " + PARAGRAPH * 3, "meta": meta})
    return rows


def _instruction_tables(n: int = 30) -> dict[str, list[dict]]:
    long = " and the sub-module function it prices"
    return {
        "databricks/databricks-dolly-15k": [
            {"instruction": f"Dolly question {i}{long}", "context": "",
             "response": f"Dolly answer {i}{long}"} for i in range(n)],
        "tatsu-lab/alpaca": [
            {"instruction": f"Alpaca question {i}{long}", "input": "",
             "output": f"Alpaca answer {i}{long}"} for i in range(n)],
        "sahil2801/CodeAlpaca-20k": [
            {"instruction": f"CodeAlpaca question {i}{long}", "input": "",
             "output": f"CodeAlpaca answer {i}{long}"} for i in range(n)],
        "HuggingFaceH4/ultrachat_200k": [
            {"messages": [{"role": "user", "content": f"UltraChat question {i}{long}"},
                          {"role": "assistant", "content": f"UltraChat answer {i}{long}"}]}
            for i in range(n)],
        "OpenAssistant/oasst2": [
            row for i in range(n) for row in (
                {"message_id": f"p{i}", "parent_id": None, "role": "prompter", "lang": "en",
                 "text": f"OASST question {i}{long}"},
                {"message_id": f"a{i}", "parent_id": f"p{i}", "role": "assistant", "lang": "en",
                 "text": f"OASST answer {i}{long}"})],
    }


def _fake_loader() -> FakeLoader:
    tables = {REDPAJAMA_PATH: _redpajama_rows()}
    tables.update(_instruction_tables())
    return FakeLoader(tables)


class CountingTable(list):
    """A dataset table that records every ``table[i]`` -- how the scan reads is what is tested."""

    def __init__(self, rows):
        super().__init__(rows)
        self.reads: list[int] = []

    def __getitem__(self, index):
        self.reads.append(index)
        return super().__getitem__(index)


def _per_source_scan(rows, n_total, max_length=2048, seed=42):
    """The six-pass scan the single-pass one replaced, kept here as the reference.

    One independent pass per source, each stopping once it has three times its allocation, then
    a seeded sample from that shortlist. Restating it here is the only way to assert that the
    optimization changed the *reading* and not the corpus.
    """
    allocations = allocate({name: cfg["default_samples"]
                            for name, cfg in PRETRAINING_SOURCES.items()}, n_total)
    expected = []
    for name, config in PRETRAINING_SOURCES.items():
        n_samples = allocations[name]
        if n_samples <= 0:
            continue
        candidates = []
        for index in range(len(rows)):
            if config["predicate"](rows[index]):
                candidates.append(index)
                if len(candidates) >= n_samples * 3:
                    break
        rng = random.Random(seed)
        for index in rng.sample(candidates, min(n_samples, len(candidates))):
            text = str(rows[index].get("text", "")).strip()
            if len(text) > MIN_PRETRAINING_CHARS:
                expected.append({"text": text[:max_length], "source": name})
    return expected


def _redpajama_rows_the_scan_finds_awkward(per_source: int = 50) -> list[dict]:
    """RedPajama rows carrying the two cases the single-pass scan handles differently.

    The equal, disjoint blocks of `_redpajama_rows` are the happy path: every source fills its
    shortlist and no row belongs to two of them. The real split is not like that, and neither case
    is cosmetic for this rewrite:

    * a **sparse source** (here GitHub, five rows) never fills its shortlist, so the single pass's
      "stop when everyone is full" break must not fire -- the real `redpajama_github` is why;
    * an **overlapping row** (a `facebook.com` URL: `is_book` matches the metadata string, `is_web`
      matches the URL) must be taken by *both* sources, as it was when each scanned on its own. It
      is placed first, so a scan that let one source consume it would shift the other's shortlist
      by a row and change what is sampled.
    """
    rows = [{"text": "facebook.com page. " + PARAGRAPH * 3,
             "meta": {"url": "https://facebook.com/page"}}]
    metas = {
        "redpajama_arxiv": {"arxiv_id": "2401.00001"},
        "redpajama_wikipedia": {"url": "https://en.wikipedia.org/wiki/Anchor"},
        "redpajama_github": {"url": "https://github.com/org/repo"},
        "redpajama_stackexchange": {"url": "https://stats.stackexchange.com/q/1"},
        "redpajama_book": {"short_book_title": "Moby Dick"},
        "redpajama_web": {"url": "https://example.com/page"},
    }
    for name, meta in metas.items():
        count = 5 if name == "redpajama_github" else per_source
        for i in range(count):
            rows.append({"text": f"{name} document {i}. " + PARAGRAPH * 3, "meta": meta})
    return rows


@pytest.mark.parametrize("build_rows", [_redpajama_rows, _redpajama_rows_the_scan_finds_awkward],
                         ids=["disjoint-and-plentiful", "sparse-source-and-overlapping-row"])
def test_download_pretraining_scans_once_and_keeps_the_six_pass_result(build_rows):
    """One scan instead of six, with the corpus unchanged: the rows matter, the I/O is the win.

    A RedPajama row is decoded on access, so the six per-source passes cost about six times what
    one costs -- and the six predicates are cheap beside a decode. The shortlists are collected in
    ascending index order either way and each source still stops at three times its allocation,
    so the sampled rows are identical; this pins both halves of that claim, on the happy path and
    on the two shapes where the single pass's control flow actually differs.
    """
    rows = build_rows()
    table = CountingTable(rows)

    produced = download_pretraining(60, max_length=40, seed=42,
                                    loader=FakeLoader({REDPAJAMA_PATH: table}))

    assert produced == _per_source_scan(rows, 60, max_length=40, seed=42)
    # One pass over the rows the scan reaches, plus one re-read of each row actually sampled.
    assert len(table.reads) == len(set(table.reads)) + len(produced)

    six_pass = CountingTable(rows)
    _per_source_scan(six_pass, 60, max_length=40, seed=42)
    assert len(table.reads) < len(six_pass.reads)


def test_a_source_that_runs_out_comes_up_short_and_does_not_end_the_scan_early():
    """The GitHub case: a source with fewer rows than its allocation, and a full scan regardless."""
    rows = _redpajama_rows_the_scan_finds_awkward()
    table = CountingTable(rows)

    produced = download_pretraining(60, max_length=40, seed=42,
                                    loader=FakeLoader({REDPAJAMA_PATH: table}))

    counts = {name: sum(1 for r in produced if r["source"] == name) for name in PRETRAINING_SOURCES}
    assert counts["redpajama_github"] == 5           # allocated 10, only five exist
    assert counts["redpajama_arxiv"] == 10
    # The sparse source never fills its shortlist, so the scan cannot stop early: every row is read.
    assert set(table.reads) == set(range(len(rows)))


def test_download_pretraining_allocates_across_sources_and_truncates():
    rows = download_pretraining(60, max_length=40, seed=42, loader=_fake_loader())

    counts = {name: sum(1 for r in rows if r["source"] == name) for name in PRETRAINING_SOURCES}
    assert counts == {name: 10 for name in PRETRAINING_SOURCES}
    assert all(set(r) == {"text", "source"} for r in rows)
    assert all(len(r["text"]) <= 40 for r in rows)


def test_download_pretraining_is_deterministic_in_the_seed():
    a = download_pretraining(60, seed=42, loader=_fake_loader())
    b = download_pretraining(60, seed=42, loader=_fake_loader())
    c = download_pretraining(60, seed=7, loader=_fake_loader())
    assert a == b and a != c


def test_download_instruction_returns_prompt_response_rows_from_every_source():
    loader = _fake_loader()
    rows = download_instruction(22, seed=42, loader=loader)

    counts = {name: sum(1 for r in rows if r["source"] == name) for name in INSTRUCTION_SOURCES}
    assert counts == {"dolly": 5, "alpaca": 5, "code_alpaca": 2, "ultrachat": 5, "oasst2": 5}
    assert all(set(r) == {"prompt", "response", "source"} for r in rows)
    assert ("HuggingFaceH4/ultrachat_200k", "train_sft", None) in loader.calls


def test_download_instruction_truncates_to_max_length():
    rows = download_instruction(22, max_length=25, seed=42, loader=_fake_loader())
    assert all(len(r["prompt"]) <= 25 and len(r["response"]) <= 25 for r in rows)


def test_downloads_pass_the_cache_dir_through(tmp_path):
    loader = _fake_loader()
    download_pretraining(6, seed=42, loader=loader, cache_dir=str(tmp_path))
    assert all(call[2] == str(tmp_path) for call in loader.calls)


def test_download_raises_when_a_source_cannot_be_loaded():
    loader = FakeLoader({})
    with pytest.raises(RuntimeError, match="RedPajama|redpajama"):
        download_pretraining(6, seed=42, loader=loader)


# ==============================================================================================
# Seed corpus: end to end
# ==============================================================================================

def test_prepare_seed_corpus_writes_one_jsonl_with_both_row_shapes(tmp_path):
    out = tmp_path / "seed_corpus_10to1.jsonl"

    returned = prepare_seed_corpus(out, n_pretraining=220, n_instruction=22, seed=42,
                                   loader=_fake_loader())

    assert returned == out
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    pretraining = [r for r in rows if "text" in r]
    instruction = [r for r in rows if "prompt" in r]

    assert all(set(r) == {"text", "source"} for r in pretraining)
    assert all(set(r) == {"prompt", "response", "source"} for r in instruction)
    assert abs(len(pretraining) / len(instruction) - 10) <= 1
    assert len(pretraining) <= 220 and len(instruction) <= 22
    assert {r["source"] for r in pretraining} == set(PRETRAINING_SOURCES)
    assert {r["source"] for r in instruction} == set(INSTRUCTION_SOURCES)


def test_prepare_seed_corpus_output_is_readable_by_the_corpus_loader(tmp_path):
    from lfa.corpus import load_texts

    out = tmp_path / "seed.jsonl"
    prepare_seed_corpus(out, n_pretraining=220, n_instruction=22, seed=42, loader=_fake_loader())

    texts = load_texts(out)
    assert len(texts) == sum(1 for _ in out.read_text(encoding="utf-8").splitlines())
    assert any("question" in t and "answer" in t for t in texts)


def test_prepare_seed_corpus_writes_a_stats_sidecar(tmp_path):
    out = tmp_path / "seed.jsonl"
    prepare_seed_corpus(out, n_pretraining=220, n_instruction=22, seed=42, loader=_fake_loader())

    stats = json.loads((tmp_path / "seed.stats.json").read_text(encoding="utf-8"))
    n_rows = len(out.read_text(encoding="utf-8").splitlines())
    assert stats["total_samples"] == n_rows
    assert sum(stats["source_distribution"].values()) == n_rows
    assert stats["seed"] == 42
    assert stats["estimated_tokens"] > 0


def test_prepare_seed_corpus_is_deterministic_in_the_seed(tmp_path):
    a, b, c = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    for path, seed in ((a, 42), (b, 42), (c, 7)):
        prepare_seed_corpus(path, n_pretraining=220, n_instruction=22, seed=seed,
                            loader=_fake_loader())

    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() != c.read_bytes()


def test_prepare_seed_corpus_defaults_are_the_download_targets():
    """The defaults are what to ASK for, not what lands: 12,000/6 = 2,000 caps each pretraining
    source and 20,000 instruction pairs are cut to 966 by the mix. What lands is the shipped
    9,663 + 966, which the test below realizes end to end."""
    import inspect

    defaults = {p.name: p.default for p in inspect.signature(prepare_seed_corpus).parameters.values()}
    assert defaults["n_pretraining"] == 12_000
    assert defaults["n_instruction"] == 20_000
    assert defaults["n_pretraining"] // len(PRETRAINING_SOURCES) == 2000    # the per-source cap
    assert defaults["seed"] == 42 and defaults["max_length"] == 2048


def _shipped_availability_loader() -> FakeLoader:
    """A loader whose sources hold exactly what the shipped corpus' sources held.

    Pretraining availability is the shipped composition itself (arXiv 1,524 and GitHub 147 below
    the 2,000 cap, the other four at or just under it); instruction availability is each source's
    full allocation, of which the 10:1 mix keeps 966.
    """
    metas = {
        "redpajama_arxiv": {"arxiv_id": "2401.00001"},
        "redpajama_wikipedia": {"url": "https://en.wikipedia.org/wiki/Anchor"},
        "redpajama_github": {"url": "https://github.com/org/repo"},
        "redpajama_stackexchange": {"url": "https://stats.stackexchange.com/q/1"},
        "redpajama_book": {"short_book_title": "Moby Dick"},
        "redpajama_web": {"url": "https://example.com/page"},
    }
    redpajama = [{"text": f"{name} document {i}. " + PARAGRAPH, "meta": metas[name]}
                 for name, available in SHIPPED_COMPOSITION["pretraining"].items()
                 for i in range(available)]

    allocations = allocate({n: c["default_samples"] for n, c in INSTRUCTION_SOURCES.items()}, 20_000)
    instruction = _instruction_tables(n=5000)
    tables = {REDPAJAMA_PATH: redpajama}
    for name, config in INSTRUCTION_SOURCES.items():
        tables[config["path"]] = instruction[config["path"]][:allocations[name]]
    return FakeLoader(tables)


def test_prepare_seed_corpus_defaults_realize_the_shipped_composition(tmp_path):
    out = tmp_path / "seed_corpus_10to1.jsonl"

    prepare_seed_corpus(out, seed=42, loader=_shipped_availability_loader())

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    pretraining = [r for r in rows if "text" in r]
    instruction = [r for r in rows if "prompt" in r]
    per_source = {name: sum(1 for r in pretraining if r["source"] == name)
                  for name in PRETRAINING_SOURCES}

    assert len(pretraining) == 9663
    assert per_source == SHIPPED_COMPOSITION["pretraining"]
    assert len(instruction) == 966
    # The instruction side is a uniform draw from ~20,000 pairs, not a per-source cap, so only its
    # total is fixed; every source is represented, in proportion to its share of the pool.
    assert {r["source"] for r in instruction} == set(INSTRUCTION_SOURCES)
    assert len(rows) == 10_629
