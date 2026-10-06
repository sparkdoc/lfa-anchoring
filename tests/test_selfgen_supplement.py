"""The supplement writer's text logic: passages, the template, the forgiving parser."""
import json

import pytest

from lfa.selfgen.supplement import (NoPairsWritten, SupplementOptions, chunk_document,
                                    chunk_passages, parse_qa_pairs, render_template,
                                    write_supplement)


def test_parses_a_clean_json_array():
    out = parse_qa_pairs('[{"question": "What is qualia?", "answer": "Subjective experience."}]')
    assert out == [{"question": "What is qualia?", "answer": "Subjective experience."}]


def test_parses_json_embedded_in_chatter():
    text = 'Sure! Here you go:\n[{"question": "Q1?", "answer": "A1."}]\nHope that helps.'
    assert parse_qa_pairs(text) == [{"question": "Q1?", "answer": "A1."}]


def test_parses_question_answer_markers_when_json_fails():
    text = "Question: What is the hard problem?\nAnswer: Explaining why experience exists.\n"
    assert parse_qa_pairs(text) == [
        {"question": "What is the hard problem?", "answer": "Explaining why experience exists."}]


def test_parses_multiple_marker_pairs():
    text = "Q: First question?\nA: First answer.\n\nQ: Second question?\nA: Second answer."
    out = parse_qa_pairs(text)
    assert [p["question"] for p in out] == ["First question?", "Second question?"]
    assert out[1]["answer"] == "Second answer."


def test_rejects_empty_or_malformed():
    assert parse_qa_pairs("") == []
    assert parse_qa_pairs("just some prose with no pairs at all") == []
    assert parse_qa_pairs('[{"question": "", "answer": "orphan"}]') == []


def test_salvages_complete_objects_from_a_truncated_array():
    text = '[{"question": "Q1?", "answer": "A1."}, {"question": "Q2?", "answer": "A2."}, {"ques'
    assert [p["question"] for p in parse_qa_pairs(text)] == ["Q1?", "Q2?"]


def test_an_assistant_turn_is_cut_at_the_templates_own_turn_end():
    from lfa.selfgen.supplement import parse_assistant_turn
    assert parse_assistant_turn("Q: a\nA: b<|eot_id|><|start_header_id|>", "<|eot_id|>") == "Q: a\nA: b"
    assert parse_assistant_turn("Q: a\nA: b<|im_end|>\n", "<|im_end|>") == "Q: a\nA: b"
    assert parse_assistant_turn("Q: a\nA: b", None) == "Q: a\nA: b"


def test_chunk_document_splits_and_merges_on_paragraphs():
    split = chunk_document("A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100, 150)
    assert [c[0] for c in split] == ["A", "B", "C"]
    merged = chunk_document("A" * 50 + "\n\n" + "B" * 50, 150)
    assert len(merged) == 1 and "A" * 50 in merged[0] and "B" * 50 in merged[0]


def test_chunk_passages_drops_runts():
    assert chunk_passages("y" * 150, size=1000, min_size=200) == []


def test_the_template_names_the_domain_and_keeps_the_rules():
    prompt = render_template("Victorian cookery", 6, "PASSAGE")
    assert "about a text on Victorian cookery" in prompt
    assert "write 6 diverse question-answer pairs" in prompt
    assert "Keep answers to 2-5 sentences" in prompt and "PASSAGE" in prompt
    assert "philosophy" not in prompt
    assert "Chalmers" not in prompt and "thought experiment" not in prompt
    assert "e.g." not in prompt and "argues that" not in prompt   # no example for a small writer to copy


class _Tok:
    pad_token_id = 0
    eos_token_id = 1
    chat_template = "stub"

    def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=None, **_):
        return "<u>" + messages[0]["content"] + "</u><a>"

    def convert_tokens_to_ids(self, token):
        return 2


class _PlainTok:
    """A tokenizer with no chat template at all."""
    pad_token_id = 0
    eos_token_id = 1

    def convert_tokens_to_ids(self, token):
        return 2


class _Model:
    device = "cpu"


class _PlainCloseTok:
    """A template whose assistant turn closes with plain text: no special token to stop on."""
    pad_token_id = 0
    eos_token_id = 1
    chat_template = "stub"
    all_special_tokens = ["</s>"]

    def apply_chat_template(self, messages, **_):
        return "".join(f"{m['role']}: {m['content']}\n\n" for m in messages)

    def convert_tokens_to_ids(self, token):
        return 3


class _Llama3Tok:
    """A Llama-3-style writer: its assistant turn closes with <|eot_id|>, not <|im_end|>."""
    pad_token_id = 0
    eos_token_id = 1
    chat_template = "stub"
    all_special_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
                          "<|eot_id|>"]

    def apply_chat_template(self, messages, add_generation_prompt=False, **_):
        turns = "".join(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
                        for m in messages)
        opener = "<|start_header_id|>assistant<|end_header_id|>\n\n" if add_generation_prompt else ""
        return "<|begin_of_text|>" + turns + opener

    def convert_tokens_to_ids(self, token):
        return {"<|eot_id|>": 9}.get(token, 3)


def test_write_supplement_records_pairs_and_a_manifest(tmp_path, monkeypatch):
    hashed = []
    monkeypatch.setattr("lfa.selfgen.supplement.checkpoint_sha256",
                        lambda m: hashed.append(m) or "a" * 64)
    docs = ["para one " * 60 + "\n\n" + "para two " * 60, "short " * 80]

    def generate(model, tokenizer, prompts, **kwargs):
        assert hashed == ["stub"]                   # the writer is hashed before any sampling
        assert all(p.startswith("<u>") for p in prompts)
        assert kwargs["temperature"] == 0.7 and kwargs["top_p"] == 0.8
        return ['[{"question": "Why?", "answer": "' + "Because of the passage. " * 3 + '"},'
                ' {"question": "Tiny?", "answer": "no"}]<|im_end|>'] * len(prompts)

    manifest = write_supplement("stub", docs, tmp_path / "s.jsonl", domain_description="tests",
                                options=SupplementOptions(passage_chars=400, min_passage_chars=10,
                                                          batch_size=4),
                                corpus_sha256="b" * 64, generate=generate,
                                writer=(_Model(), _Tok()))

    rows = [json.loads(l) for l in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert rows and all(set(r) == {"prompt", "response", "source_index"} for r in rows)
    assert all(len(r["response"]) >= 40 for r in rows)          # the short answer was dropped
    assert manifest["n_passages"] >= 2 and manifest["n_pairs"] == len(rows)
    assert manifest["rejected"]["short_answer"] >= 1
    assert manifest["writer_sha256"] == "a" * 64 and manifest["corpus_sha256"] == "b" * 64
    assert manifest["template_sha256"] and "tests" in manifest["template"]
    assert manifest["chat_template_applied"] is True
    assert (tmp_path / "s.jsonl.manifest.json").is_file()


def test_no_pairs_at_all_is_a_refusal_naming_the_tally(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.supplement.checkpoint_sha256", lambda m: "a" * 64)

    def generate(model, tokenizer, prompts, **kwargs):
        return ["nothing parseable here"] * len(prompts)

    with pytest.raises(NoPairsWritten, match="1 passage"):
        write_supplement("stub", ["text " * 100], tmp_path / "s.jsonl", domain_description="d",
                         options=SupplementOptions(min_passage_chars=10), corpus_sha256="b" * 64,
                         generate=generate, writer=(_Model(), _Tok()))


def test_no_passages_refuses_before_any_model_loads(tmp_path):
    with pytest.raises(NoPairsWritten, match="0 passages"):
        write_supplement("stub", ["tiny"], tmp_path / "s.jsonl", domain_description="d",
                         corpus_sha256="b" * 64, writer=None)


def test_duplicates_are_keyed_on_the_question_alone(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.supplement.checkpoint_sha256", lambda m: "a" * 64)
    answers = iter(["The first answer, long enough to pass the filter.",
                    "A different answer, also long enough to pass the filter."])

    def generate(model, tokenizer, prompts, **kwargs):
        return ['[{"question": "Same?", "answer": "' + next(answers) + '"}]' for _ in prompts]

    manifest = write_supplement("stub", ["one " * 60, "two " * 60], tmp_path / "s.jsonl",
                                domain_description="d",
                                options=SupplementOptions(min_passage_chars=10),
                                corpus_sha256="b" * 64, generate=generate,
                                writer=(_Model(), _Tok()))
    rows = [json.loads(l) for l in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert manifest["n_passages"] == 2 and len(rows) == 1
    assert rows[0]["response"].startswith("The first answer")
    assert manifest["rejected"]["duplicate"] == 1


def test_a_writer_without_a_chat_template_warns_and_says_so(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("lfa.selfgen.supplement.checkpoint_sha256", lambda m: "a" * 64)

    def generate(model, tokenizer, prompts, **kwargs):
        assert all(p.startswith("You are creating training data") for p in prompts)
        return ['[{"question": "Why?", "answer": "' + "Because of the passage. " * 3 + '"}]'
                ] * len(prompts)

    with caplog.at_level("WARNING", logger="lfa.selfgen.supplement"):
        manifest = write_supplement("stub", ["text " * 100, "more " * 100], tmp_path / "s.jsonl",
                                    domain_description="d",
                                    options=SupplementOptions(min_passage_chars=10),
                                    corpus_sha256="b" * 64, generate=generate,
                                    writer=(_Model(), _PlainTok()))
    warnings = [r for r in caplog.records if r.name == "lfa.selfgen.supplement"
                and r.levelname == "WARNING"]
    assert len(warnings) == 1 and "plain text" in warnings[0].getMessage()
    assert manifest["chat_template_applied"] is False


def test_the_writer_stops_and_cuts_at_its_own_templates_turn_end(tmp_path, monkeypatch):
    """A non-ChatML writer: the stop id and the cut both come from its template, not <|im_end|>."""
    monkeypatch.setattr("lfa.selfgen.supplement.checkpoint_sha256", lambda m: "a" * 64)
    stops = []

    def generate(model, tokenizer, prompts, **kwargs):
        stops.append(kwargs["stop_token_ids"])
        return ['[{"question": "Why?", "answer": "' + "Because of the passage. " * 3 + '"}]'
                '<|eot_id|>[{"question": "After the turn?", "answer": "'
                + "Text past the turn end is not the answer. " * 2 + '"}]'] * len(prompts)

    write_supplement("stub", ["text " * 100], tmp_path / "s.jsonl", domain_description="d",
                     options=SupplementOptions(min_passage_chars=10), corpus_sha256="b" * 64,
                     generate=generate, writer=(_Model(), _Llama3Tok()))
    rows = [json.loads(l) for l in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert stops == [[1, 9]]                              # EOS, then the template's turn end
    assert [r["prompt"] for r in rows] == ["Why?"]


@pytest.mark.parametrize("tokenizer", [_Tok(), _PlainCloseTok()],
                         ids=["renders-no-assistant-turn", "plain-text-close"])
def test_a_template_with_no_turn_end_stops_at_eos_alone(tmp_path, monkeypatch, tokenizer):
    monkeypatch.setattr("lfa.selfgen.supplement.checkpoint_sha256", lambda m: "a" * 64)
    stops = []

    def generate(model, tokenizer, prompts, **kwargs):
        stops.append(kwargs["stop_token_ids"])
        return ['[{"question": "Why?", "answer": "' + "Because of the passage. " * 3 + '"}]'
                ] * len(prompts)

    write_supplement("stub", ["text " * 100], tmp_path / "s.jsonl", domain_description="d",
                     options=SupplementOptions(min_passage_chars=10), corpus_sha256="b" * 64,
                     generate=generate, writer=(_Model(), tokenizer))
    assert stops == [[1]]


# ------------------------------------------------------------ pairs carrying the writer's JSON

#: Three pairs from the two-domain walkthrough's Darwin supplement (Qwen3-0.6B writer,
#: ~/lfa-verify/models/data/darwin/train.supplement/d88afff85e82/supplement.jsonl, rows 1235,
#: 1427 and 1430 of 1,549; 12 of its pairs have this shape), and one from an earlier Darwin
#: supplement whose leak sits in the question: the writer's next field, and the next object,
#: parsed into this one.
LEAKED = [
    {"question": "What is the relationship between larger groups and their descendants?",
     "answer": "Larger groups tend to conquer smaller ones, reducing their numbers and leading to "
               "fewer variations and improvements. This process results in a decrease in the "
               "number of descendants.\", \"answer\": \"..."},
    {"question": "What is the relationship between the principle of natural selection and the "
                 "architectural powers of the hive-bee?",
     "answer": "The principle of gradation, which states that organisms evolve through "
               "successive, slight modifications, explains how the hive-bee's architectural "
               "powers develop. This principle shows how complex structures can emerge from "
               "simpler ones.\", \"answer\": \"..."},
    {"question": "How do the crossed offspring of acknowledged varieties behave in terms of "
                 "resemblance?",
     "answer": "The crossed offspring of acknowledged varieties follow the same complex laws in "
               "their resemblance to their parents, showing how genetic inheritance and selection "
               "influence their traits over time.\", \"answer\": \"..."},
    {"question": "Why does the passage mention the 'Himalaya' glaciers?', \"answer\": \"The "
                 "passage notes that Himalaya's glaciers are crucial for its flora and fauna, "
                 "reflecting the region's climatic and geological importance.\"}, {\"question\": "
                 "\"What is the role of 'Hewitt' in the passage?",
     "answer": "Hewitt, Mr., discusses the sterility of first crosses in plant populations, "
               "emphasizing their impact on genetic diversity."},
]

#: Pairs a domain can carry legitimately: quotations, braces in code and recipes, a JSON
#: example with other field names, and the words "question" and "answer" in quotes.
CLEAN = [
    {"question": "What does Mr. C. Noble say of his hybrid stocks?",
     "answer": 'He writes that the hybrids "seed as freely as it is possible to imagine", which '
               "the passage takes as evidence of their fertility."},
    {"question": "How is a Python dictionary written?",
     "answer": 'With braces around key-value pairs, as in {"name": "Ada", "year": 1843}; the '
               "keys are usually strings."},
    {"question": "What does the recipe for {Seed Cake} ask for?",
     "answer": "A pound of flour, {half a pound} of butter and an ounce of caraway seeds, beaten "
               "together for an hour."},
    {"question": 'Why does the author put the word "answer" in quotes?',
     "answer": 'Because the "answer" given by the critics was, in the author\'s view, no answer '
               'at all, and a "question" left open is better than one closed falsely.'},
    {"question": "What JSON does the API return?",
     "answer": 'An object such as {"status": "ok", "items": []}, with the items listed in the '
               "order they were created."},
]


@pytest.mark.parametrize("pair", LEAKED, ids=["darwin-1235", "darwin-1427", "darwin-1430",
                                              "leak-in-the-question"])
def test_a_pair_carrying_the_writers_json_syntax_is_caught(pair):
    from lfa.selfgen.supplement import leaks_json
    assert leaks_json(pair)


@pytest.mark.parametrize("pair", CLEAN, ids=["quotation", "code-braces", "recipe-braces",
                                             "quoted-field-words", "other-json"])
def test_braces_and_quotes_alone_are_ordinary_text(pair):
    from lfa.selfgen.supplement import leaks_json
    assert not leaks_json(pair)


def _as_writer_output(pairs):
    """A writer's JSON array whose parse gives back exactly ``pairs``."""
    return "[" + ", ".join('{"question": "%s", "answer": "%s"}' % (p["question"], p["answer"])
                           for p in pairs) + "]<|im_end|>"


def test_leaked_pairs_are_dropped_and_counted(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("lfa.selfgen.supplement.checkpoint_sha256", lambda m: "a" * 64)
    # The second passage repeats a leaked pair: counted as leaked, not as a duplicate.
    outputs = iter([_as_writer_output(LEAKED[:3] + CLEAN[:2]),
                    _as_writer_output([LEAKED[0]] + CLEAN[2:])])

    def generate(model, tokenizer, prompts, **kwargs):
        return [next(outputs) for _ in prompts]

    assert parse_qa_pairs(_as_writer_output(LEAKED[:3])) == LEAKED[:3]   # as the writer's were
    with caplog.at_level("INFO", logger="lfa.selfgen.supplement"):
        manifest = write_supplement("stub", ["one " * 60, "two " * 60], tmp_path / "s.jsonl",
                                    domain_description="d",
                                    options=SupplementOptions(min_passage_chars=10),
                                    corpus_sha256="b" * 64, generate=generate,
                                    writer=(_Model(), _Tok()))

    rows = [json.loads(l) for l in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert [r["prompt"] for r in rows] == [p["question"] for p in CLEAN]
    assert manifest["rejected"] == {"unparseable_passage": 0, "leaked_json": 4, "short_answer": 0,
                                    "long_answer": 0, "duplicate": 0}
    assert manifest["n_pairs"] == 5
    [line] = [r.getMessage() for r in caplog.records
              if r.getMessage().startswith("supplement: 5 pairs from 2 passages written to ")]
    assert line.endswith("; rejected: unparseable_passage 0, leaked_json 4, short_answer 0, "
                         "long_answer 0, duplicate 0")
    assert "Himalaya" not in caplog.text and "descendants" not in caplog.text   # no samples


def test_the_manifest_records_the_filters_it_was_written_under(tmp_path, monkeypatch):
    from lfa.selfgen.supplement import REJECTION_REASONS, filters_sha256
    monkeypatch.setattr("lfa.selfgen.supplement.checkpoint_sha256", lambda m: "a" * 64)

    def generate(model, tokenizer, prompts, **kwargs):
        return [_as_writer_output(CLEAN)] * len(prompts)

    manifest = write_supplement("stub", ["text " * 100], tmp_path / "s.jsonl",
                                domain_description="d",
                                options=SupplementOptions(min_passage_chars=10),
                                corpus_sha256="b" * 64, generate=generate,
                                writer=(_Model(), _Tok()))
    assert manifest["filters_sha256"] == filters_sha256() and len(filters_sha256()) == 64
    assert tuple(manifest["rejected"]) == REJECTION_REASONS
    assert "leaked_json" in REJECTION_REASONS
