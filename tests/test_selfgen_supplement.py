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
