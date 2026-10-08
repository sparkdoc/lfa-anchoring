"""The supplement writer's text logic: passages, the two templates, the forgiving parser, the
two passes and their filters."""
import json
import re

import pytest

from lfa.selfgen.supplement import (NoPairsWritten, SupplementOptions, chunk_document,
                                    chunk_passages, parse_questions, refers_to_source,
                                    render_answer_prompt, render_question_prompt,
                                    write_supplement)


def test_parses_a_json_array_of_strings():
    assert parse_questions('["What is qualia?", "Why does it matter?"]') == [
        "What is qualia?", "Why does it matter?"]


def test_parses_question_objects_and_an_array_inside_chatter():
    assert parse_questions('Sure! Here:\n["Q1?", "Q2?"]\nHope that helps.') == ["Q1?", "Q2?"]
    assert parse_questions('[{"question": "Q1?"}, {"question": "Q2?", "answer": "A2."}]') == [
        "Q1?", "Q2?"]


def test_salvages_the_complete_strings_of_an_array_that_does_not_parse():
    truncated = '["Q1?", "Why \\"Q2\\"?", "Q3 is cut off'
    assert parse_questions(truncated) == ["Q1?", 'Why "Q2"?']
    assert parse_questions('["Q1?" "Q2?", "Q3?"]') == ["Q1?", "Q2?", "Q3?"]   # a missing comma
    assert parse_questions('[{"question": "Q1?", "answer": "A1."}, {"question": "Q2?", "ans') == [
        "Q1?", "Q2?"]


def test_an_array_is_read_to_its_own_end_and_chatter_after_it_is_ignored():
    """A ``]`` or a quoted word in chatter after the array is not part of it (the review's
    adversarial strings), and a ``[1]`` before it holds no question."""
    assert parse_questions('["What is X?", "Why does Y?"]\n\n(Based on [1], "Origin" ch. 4)') == [
        "What is X?", "Why does Y?"]
    assert parse_questions('["What is X?", "Why Y?"]\nNote: the term "species" [sic] is used '
                           'loosely.') == ["What is X?", "Why Y?"]
    assert parse_questions('From [1] and [2]: ["What is X?"]') == ["What is X?"]
    assert parse_questions('["What is X?",\n "Why does\nY?"]') == ["What is X?", "Why does Y?"]


def test_a_broken_arrays_salvage_keeps_only_questions():
    """Salvaged literals must end in ``?``: a quoted word after a broken array is not a question."""
    assert parse_questions('["What is X?" "Why Y?"]\n(Based on [1], "Origin")') == [
        "What is X?", "Why Y?"]
    assert parse_questions('["What is X?", "Why Y?", "Cut off\nSee "Origin", ch. 4') == [
        "What is X?", "Why Y?"]


def test_of_several_arrays_the_one_that_asks_wins():
    """An echoed placeholder array or a list of topics before the real one does not win (the
    re-review's holes); with no asking array, the first usable one is taken."""
    assert parse_questions('Format: ["...", "..."]\n["What is X?", "Why Y?"]') == [
        "What is X?", "Why Y?"]
    topics_first = 'Topics: ["natural selection", "variation"]\n["What is X?", "Why Y?"]'
    assert parse_questions(topics_first) == ["What is X?", "Why Y?"]
    assert parse_questions('["Explain X.", "Describe Y."]') == ["Explain X.", "Describe Y."]


def test_parses_a_single_quoted_array():
    assert parse_questions("['What is X?', 'Why does Y?']") == ["What is X?", "Why does Y?"]
    assert parse_questions("Here: ['What is Darwin's view?', 'Why Y?'] done") == [
        "What is Darwin's view?", "Why Y?"]


def test_falls_back_to_lines_that_ask_something():
    text = "1. What is the hard problem?\n2) Why does it matter?\n- How?\nQ: Who?\nA remark."
    assert parse_questions(text) == ["What is the hard problem?", "Why does it matter?", "How?",
                                     "Who?"]


def test_nothing_parseable_and_the_templates_own_placeholders_give_no_question():
    assert parse_questions("") == []
    assert parse_questions("just some prose with no questions at all") == []
    assert parse_questions('["...", "..."]') == []
    assert parse_questions('["", "  "]') == []


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


def test_the_question_template_names_the_domain_and_keeps_the_question_rules():
    prompt = render_question_prompt("Victorian cookery", 6, "PASSAGE")
    assert "about a text on Victorian cookery" in prompt
    assert "write 6 diverse questions" in prompt and "JSON array of 6 strings" in prompt
    assert "information-seeking" in prompt and "exam-style" in prompt
    assert 'Prefer "What is...", "Why does...", "What is the relationship between...", ' \
           '"What does the author mean by..."' in prompt
    assert 'never mention "the passage" or "the text"' in prompt and "PASSAGE" in prompt
    assert "answer" not in prompt.lower().replace("answer questions", "")   # questions only
    assert "philosophy" not in prompt
    assert "Chalmers" not in prompt and "thought experiment" not in prompt
    assert "e.g." not in prompt and "argues that" not in prompt   # no example for a small writer to copy


def test_the_answer_prompt_is_the_prototypes_and_asks_for_no_length_and_no_format():
    """The measured frame (the format-drift diagnostic's N2): the passage, then "Using the
    passage, answer:" and the question, in one user turn. Nothing else constrains the answer: an
    instruction not to mention the passage suppressed the model's own format."""
    prompt = render_answer_prompt("Victorian cookery", "XYZ", "Why is the seed cake beaten?")
    assert prompt == ('Here is a passage from a text on Victorian cookery:\n"""\nXYZ\n"""\n\n'
                      "Using the passage, answer: Why is the seed cake beaten?")
    lowered = prompt.lower()
    for instruction in ("sentence", "short", "brief", "concise", "word", "length", "paragraph",
                        "markdown", "bullet", "list", "format", "json", "third person", "plain",
                        "assistant", "mention", "refer", "do not", "don't", "never", "source"):
        assert instruction not in lowered, instruction


class _Tok:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "</s>"
    chat_template = "stub"

    def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=None, **_):
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<a>"

    def convert_tokens_to_ids(self, token):
        return 2


class _PlainTok:
    """A tokenizer with no chat template at all."""
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "</s>"

    def convert_tokens_to_ids(self, token):
        return 2


class _Model:
    device = "cpu"


class _PlainCloseTok:
    """A template whose assistant turn closes with plain text: no special token to stop on."""
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "</s>"
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
    eos_token = "<|end_of_text|>"
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


LONG = "A full answer, long enough to clear the forty-character floor. "


def _is_question_prompt(prompt):
    return "diverse questions" in prompt


def _question_in(prompt):
    """The question an answer prompt asks (every test question ends with its only "?")."""
    return re.search(r"Using the passage, answer: (.*?\?)", prompt).group(1)


def _passage_in(prompt):
    return re.search(r'"""\n(.*?)\n"""', prompt, re.S).group(1)


def _two_pass(questions, answer, end="</s>"):
    """A fake ``generate``: ``questions(prompt)`` writes pass 1's output for a passage,
    ``answer(question, prompt)`` pass 2's; ``calls`` records each call's pass and keyword args."""
    calls = []

    def generate(model, tokenizer, prompts, **kwargs):
        kinds = {_is_question_prompt(p) for p in prompts}
        assert len(kinds) == 1, "a generate call mixes the two passes"
        kind = "questions" if kinds == {True} else "answers"
        calls.append(dict(kwargs, kind=kind, prompts=list(prompts)))
        if kind == "questions":
            return [questions(p) + end for p in prompts]
        return [answer(_question_in(p), p) for p in prompts]
    generate.calls = calls
    return generate


def _write(tmp_path, generate, docs=("text " * 100,), tokenizer=None, **options):
    options.setdefault("min_passage_chars", 10)
    return write_supplement("stub", list(docs), tmp_path / "s.jsonl", domain_description="d",
                            options=SupplementOptions(**options), corpus_sha256="b" * 64,
                            generate=generate, writer=(_Model(), tokenizer or _Tok()))


def _rows(tmp_path):
    return [json.loads(l) for l in (tmp_path / "s.jsonl").read_text().splitlines()]


@pytest.fixture
def unhashed(monkeypatch):
    monkeypatch.setattr("lfa.selfgen.supplement.checkpoint_sha256", lambda m: "a" * 64)


def test_write_supplement_asks_then_answers_and_records_a_manifest(tmp_path, monkeypatch):
    hashed = []
    monkeypatch.setattr("lfa.selfgen.supplement.checkpoint_sha256",
                        lambda m: hashed.append(m) or "a" * 64)
    # Three passages: document 0 splits in two, document 1 is one. Each passage's questions name
    # its first word, so an answer prompt can be checked against the passage it holds.
    docs = ["alpha " * 60 + "\n\n" + "beta " * 60, "gamma " * 80]

    def questions(prompt):
        assert hashed == ["stub"]                   # the writer is hashed before any sampling
        word = _passage_in(prompt).split()[0]
        return json.dumps([f"What is {word} one?", f"What is {word} two?", f"Tiny {word}?"])

    def answer(question, prompt):
        word = question.rstrip("?").split()[-2 if question.startswith("What") else -1]
        assert _passage_in(prompt).split()[0] == word     # answered with its own passage in hand
        return ("no" if question.startswith("Tiny") else f"**{word}** -- " + LONG * 3) + "</s>"

    generate = _two_pass(questions, answer)
    manifest = write_supplement("stub", docs, tmp_path / "s.jsonl", domain_description="tests",
                                options=SupplementOptions(passage_chars=400, min_passage_chars=10,
                                                          batch_size=4),
                                corpus_sha256="b" * 64, generate=generate,
                                writer=(_Model(), _Tok()))

    rows = _rows(tmp_path)
    assert [(r["source_index"], r["passage_index"]) for r in rows] == [
        (0, 0), (0, 0), (0, 1), (0, 1), (1, 0), (1, 0)]
    assert [r["prompt"].split()[2] for r in rows] == ["alpha"] * 2 + ["beta"] * 2 + ["gamma"] * 2
    assert all(set(r) == {"prompt", "response", "source_index", "passage_index"} for r in rows)
    assert all(r["response"].startswith("**") for r in rows)      # the answer kept as written
    # Pass 1: one call over the three passages; pass 2: nine questions in batches of four, with
    # sampling seeds that continue from pass 1's rather than reuse them.
    assert [c["kind"] for c in generate.calls] == ["questions", "answers", "answers", "answers"]
    assert [len(c["prompts"]) for c in generate.calls] == [3, 4, 4, 1]
    assert [c["batch_index"] for c in generate.calls] == [0, 1, 2, 3]
    assert all(c["temperature"] == 0.7 and c["top_p"] == 0.8 and c["max_new_tokens"] == 1024
               for c in generate.calls)
    assert manifest["n_passages"] == 3 and manifest["n_questions"] == 9
    assert manifest["n_pairs"] == len(rows) == 6
    assert manifest["rejected"]["short_answer"] == 3 and manifest["rejected"]["duplicate"] == 0
    assert manifest["writer_sha256"] == "a" * 64 and manifest["corpus_sha256"] == "b" * 64
    assert manifest["question_template_sha256"] and manifest["answer_template_sha256"]
    assert "a text on tests" in manifest["question_template"]
    assert manifest["answer_template"] == ('Here is a passage from a text on tests:\n"""\n{passage}'
                                           '\n"""\n\nUsing the passage, answer: {question}')
    assert manifest["chat_template_applied"] is True
    assert manifest["decoding"]["enable_thinking"] is False
    assert (tmp_path / "s.jsonl.manifest.json").is_file()


def test_an_answer_that_never_reached_its_turn_end_is_dropped_and_counted(tmp_path, unhashed):
    generate = _two_pass(lambda p: '["Finished?", "Cut off?"]',
                         lambda q, p: LONG + ("</s>" if q == "Finished?" else "and then"))
    manifest = _write(tmp_path, generate)
    assert [r["prompt"] for r in _rows(tmp_path)] == ["Finished?"]
    assert manifest["rejected"]["truncated_answer"] == 1


def test_a_question_that_cites_its_source_is_never_answered(tmp_path, unhashed):
    generate = _two_pass(
        lambda p: '["What does the passage say about seed cake?", "Why is seed cake beaten?"]',
        lambda q, p: LONG + "</s>")
    manifest = _write(tmp_path, generate)
    answered = [_question_in(p) for c in generate.calls if c["kind"] == "answers"
                for p in c["prompts"]]
    assert answered == ["Why is seed cake beaten?"]
    assert manifest["rejected"]["refers_to_source"] == 1 and manifest["n_questions"] == 1


def test_an_answer_that_cites_its_source_is_dropped_and_counted(tmp_path, unhashed):
    answers = {"Why is seed cake beaten?": "According to the passage, " + LONG,
               "What is caraway?": "Caraway is a seed. " + LONG}
    generate = _two_pass(lambda p: json.dumps(list(answers)),
                         lambda q, p: answers[q] + "</s>")
    manifest = _write(tmp_path, generate)
    assert [r["prompt"] for r in _rows(tmp_path)] == ["What is caraway?"]
    assert manifest["rejected"]["refers_to_source"] == 1


def test_no_question_at_all_is_a_refusal_naming_the_tally(tmp_path, unhashed):
    generate = _two_pass(lambda p: "nothing parseable here", lambda q, p: pytest.fail("answered"))
    with pytest.raises(NoPairsWritten, match="1 passage.*unparseable_passage': 1"):
        _write(tmp_path, generate)


def test_no_answer_surviving_is_a_refusal_too(tmp_path, unhashed):
    generate = _two_pass(lambda p: '["Why?"]', lambda q, p: "no</s>")
    with pytest.raises(NoPairsWritten, match="short_answer': 1"):
        _write(tmp_path, generate)
    assert not (tmp_path / "s.jsonl").exists()


def test_no_passages_refuses_before_any_model_loads(tmp_path):
    with pytest.raises(NoPairsWritten, match="0 passages"):
        write_supplement("stub", ["tiny"], tmp_path / "s.jsonl", domain_description="d",
                         corpus_sha256="b" * 64, writer=None)


def test_duplicates_are_keyed_on_the_question_and_never_answered_twice(tmp_path, unhashed):
    generate = _two_pass(lambda p: '["Same?"]', lambda q, p: LONG + "</s>")
    manifest = _write(tmp_path, generate, docs=["one " * 60, "two " * 60])
    rows = _rows(tmp_path)
    assert manifest["n_passages"] == 2 and len(rows) == 1
    assert rows[0]["source_index"] == 0
    assert manifest["rejected"]["duplicate"] == 1
    assert sum(len(c["prompts"]) for c in generate.calls if c["kind"] == "answers") == 1


def test_a_writer_without_a_chat_template_warns_and_says_so(tmp_path, unhashed, caplog):
    generate = _two_pass(lambda p: '["Why?"]', lambda q, p: LONG + "</s>")
    with caplog.at_level("WARNING", logger="lfa.selfgen.supplement"):
        manifest = _write(tmp_path, generate, docs=["text " * 100, "more " * 100],
                          tokenizer=_PlainTok())
    question_prompts, answer_prompts = (c["prompts"] for c in generate.calls)
    assert all(p.startswith("You are creating training data") for p in question_prompts)
    assert all(p.startswith('Here is a passage from a text on d:\n"""\n') for p in answer_prompts)
    assert all(p.endswith('"""\n\nUsing the passage, answer: Why?\n') for p in answer_prompts)
    warnings = [r for r in caplog.records if r.name == "lfa.selfgen.supplement"
                and r.levelname == "WARNING"]
    assert len(warnings) == 1 and "plain text" in warnings[0].getMessage()
    assert manifest["chat_template_applied"] is False


def test_the_answer_prompt_is_one_user_turn_holding_the_passage_and_the_question(tmp_path,
                                                                                 unhashed):
    generate = _two_pass(lambda p: '["Why is the seed cake beaten?"]', lambda q, p: LONG + "</s>")
    _write(tmp_path, generate)
    [prompt] = generate.calls[1]["prompts"]
    assert prompt == ('<user>Here is a passage from a text on d:\n"""\n' + "text " * 100 + '\n"""'
                      "\n\nUsing the passage, answer: Why is the seed cake beaten?</user><a>")


def test_the_writer_stops_and_cuts_at_its_own_templates_turn_end(tmp_path, unhashed):
    """A non-ChatML writer: the stop id and the cut both come from its template, not <|im_end|>,
    in both passes."""
    generate = _two_pass(
        lambda p: '["Why?"]<|eot_id|>["After the turn?"]',
        lambda q, p: LONG + "<|eot_id|>Text past the turn end is not the answer.",
        end="")
    _write(tmp_path, generate, tokenizer=_Llama3Tok())
    rows = _rows(tmp_path)
    assert [c["stop_token_ids"] for c in generate.calls] == [[1, 9], [1, 9]]
    assert [(r["prompt"], r["response"]) for r in rows] == [("Why?", LONG.strip())]


@pytest.mark.parametrize("tokenizer", [_Tok(), _PlainCloseTok()],
                         ids=["renders-no-assistant-turn", "plain-text-close"])
def test_a_template_with_no_turn_end_stops_at_eos_alone(tmp_path, unhashed, tokenizer):
    generate = _two_pass(lambda p: '["Why?"]', lambda q, p: LONG + "</s>")
    _write(tmp_path, generate, tokenizer=tokenizer)
    assert [c["stop_token_ids"] for c in generate.calls] == [[1], [1]]
    assert [r["response"] for r in _rows(tmp_path)] == [LONG.strip()]


def test_an_empty_think_block_is_not_part_of_the_answer(tmp_path, unhashed):
    generate = _two_pass(lambda p: '["Why?"]',
                         lambda q, p: "<think>\n\n</think>\n\n" + LONG + "</s>")
    _write(tmp_path, generate)
    assert [r["response"] for r in _rows(tmp_path)] == [LONG.strip()]


# ------------------------------------------------------------ questions and answers citing a source

#: Phrasings from the format-drift prototype's native answers and the one-pass writer's
#: questions (Qwen3-0.6B, Darwin), each speaking of a source the training pair does not hold.
CITES_SOURCE = [
    {"question": "What does the passage say about the arrangement of species?", "answer": ""},
    {"question": "What is the relationship between instinct and natural selection according to "
                 "the passage?", "answer": ""},
    {"question": "What is an example given in the passage to illustrate intermediate forms?",
     "answer": ""},
    {"question": "What is the main point of the passage regarding marine life?", "answer": ""},
    {"question": "Why do hybrids fail?",
     "answer": "The passage suggests that the sterility of hybrids is incidental."},
    {"question": "Why do hybrids fail?",
     "answer": "Sterility is common, as stated in the passage, among first crosses."},
    {"question": "What is rarity?",
     "answer": "Rarity precedes extinction. The text explains that unperceived causes act."},
    {"question": "Who wrote it?",
     "answer": "This is supported by the text, which discusses how variation arises."},
    {"question": "Who wrote Pride and Prejudice?",
     "answer": "The text provided does not mention who wrote Pride and Prejudice."},
    {"question": "Why are some forms rare?",
     "answer": "The author states that rare forms are the first to go extinct."},
    # Forms the format-drift run of this writer kept (N3 rows 323, 417, 876, 266, 1157, 979, 690).
    {"question": "Are hybrids fertile?",
     "answer": "**Fertility**: The text also states that the fertility of hybrids is not fixed."},
    {"question": "How old is selection?",
     "answer": "**Comparative Evidence**: The text references how ancient societies bred animals."},
    {"question": "Was the cold period simultaneous?",
     "answer": "The text explicitly states that the cold period was worldwide."},
    {"question": "What is generative variability?",
     "answer": "In the context of the text, the author argues that variability is generative."},
    {"question": "Why do ranges matter?",
     "answer": "A sharper range aligns with the text's argument about species relations."},
    {"question": "What does the author mean by 'homologous construction' in the context of the "
                 "text?", "answer": ""},
]

#: The same words used for something else, or "the author" when the question asked about one.
DOES_NOT = [
    {"question": "How is geological time measured?",
     "answer": "Organic change can be used to gauge the passage of time, though not always."},
    {"question": "Why did ships seek a northern route?",
     "answer": "The passage through the strait was shorter than the route around the cape."},
    {"question": "What did the treaty say?",
     "answer": "The text of the treaty fixed the border along the river for a century."},
    {"question": "What does Article 5 provide?",
     "answer": "In the text of the treaty, Article 5 sets out collective defence."},
    {"question": "Who declares war?",
     "answer": "According to the text of the Constitution, Congress declares war."},
    {"question": "What is created first?",
     "answer": "In the passage from Genesis 1, God creates light first."},
    {"question": "How did abbreviations spread?",
     "answer": "In the context of the text messaging era, abbreviations spread quickly."},
    {"question": "What does the author mean by propinquity of descent?",
     "answer": "The author means that similarity reflects common ancestry."},
    {"question": "Who is the author of On the Origin of Species?",
     "answer": "The author of On the Origin of Species is Charles Darwin, writing in 1859."},
]


@pytest.mark.parametrize("pair", CITES_SOURCE, ids=[f"cites-{i}" for i in range(len(CITES_SOURCE))])
def test_a_reference_to_the_source_is_caught(pair):
    assert refers_to_source(pair)


@pytest.mark.parametrize("pair", DOES_NOT, ids=["passage-of-time", "passage-through",
                                                "text-of-a-treaty", "in-the-text-of",
                                                "according-to-the-text-of", "passage-from",
                                                "context-of-the-text-messaging",
                                                "author-asked-about", "author-of"])
def test_the_same_words_used_otherwise_are_kept(pair):
    assert not refers_to_source(pair)


@pytest.mark.parametrize("answer", [
    "The passage in bars 12-16 modulates to the dominant.",
    "The text is not considered canonical by most scholars.",
    "Within the text, Hamlet says 'to be or not to be'.",
    "The text's earliest surviving manuscript dates from 1200.",
    "The text shows a clear chiastic structure."],
    ids=["music", "canonical", "hamlet", "philology", "chiasmus"])
def test_the_known_limits_are_still_read_as_the_source(answer):
    """No pattern tells these domain phrasings from a reference to the writer's passage or text;
    the documentation lists them as pairs that may be dropped."""
    assert refers_to_source({"question": "What is it?", "answer": answer})


def test_the_filters_hash_moves_with_the_source_pattern(monkeypatch):
    from lfa.selfgen import supplement
    before = supplement.filters_sha256()
    monkeypatch.setattr(supplement, "_SOURCE_TEXT", re.compile(r"\bthe passage\b"))
    assert supplement.filters_sha256() != before


# ------------------------------------------------------------ pairs carrying the writer's JSON

#: Three pairs from the two-domain walkthrough's Darwin supplement (Qwen3-0.6B writer,
#: ~/lfa-verify/models/data/darwin/train.supplement/d88afff85e82/supplement.jsonl, rows 1235,
#: 1427 and 1430 of 1,549; 12 of its pairs have this shape), and one from an earlier Darwin
#: supplement whose leak sits in the question: the writer's next field, and the next object,
#: parsed into this one. The three answers end on the writer's next field and its value, the
#: shape a pass-2 answer is checked for.
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
               "he takes as evidence of their fertility."},
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
    {"question": "What does the quiz endpoint return?",
     "answer": 'Each item is an object such as {"question": "What is 2 + 2?", "answer": "4"}, '
               "returned in the order the quiz was written."},
]


#: Pass-2 answers written as the writer's pair object: the whole answer, fenced, or as an array.
JSON_SHAPED = [
    {"question": "Why is seed cake beaten?",
     "answer": '{"question": "Why is seed cake beaten?", "answer": "To lighten it."}'},
    {"question": "Why is seed cake beaten?",
     "answer": '```json\n[{"answer": "To lighten it with air."}]\n```'},
    {"question": "Why is seed cake beaten?",
     "answer": 'To lighten it with air.", "answer": "To lighten it."}'},
]


@pytest.mark.parametrize("pair", LEAKED + JSON_SHAPED,
                         ids=["darwin-1235", "darwin-1427", "darwin-1430", "leak-in-the-question",
                              "answer-is-an-object", "answer-is-a-fenced-array",
                              "answer-ends-on-a-field"])
def test_a_pair_carrying_the_writers_json_syntax_is_caught(pair):
    from lfa.selfgen.supplement import leaks_json
    assert leaks_json(pair)


def test_an_answer_whose_closing_code_block_ends_on_an_answer_field_is_dropped():
    """The documented cost of the end anchor: a JSON-API answer whose last code block ends on an
    ``"answer"`` field reads as the writer's JSON shape."""
    from lfa.selfgen.supplement import leaks_json
    assert leaks_json({"question": "What does the endpoint return?",
                       "answer": 'It returns:\n```json\n{"id": 1, "answer": "42"}\n```'})


@pytest.mark.parametrize("pair", CLEAN, ids=["quotation", "code-braces", "recipe-braces",
                                             "quoted-field-words", "other-json",
                                             "field-names-mid-answer"])
def test_braces_and_quotes_alone_are_ordinary_text(pair):
    from lfa.selfgen.supplement import leaks_json
    assert not leaks_json(pair)


def test_leaked_questions_and_answers_are_dropped_and_counted(tmp_path, unhashed, caplog):
    """A leak in a question is caught in pass 1, before it is answered; one in an answer in pass
    2. The second passage repeats the leaked question: counted as leaked, not as a duplicate."""
    answers = {p["question"]: p["answer"] for p in LEAKED[:3] + CLEAN}
    passages = iter([[p["question"] for p in LEAKED + CLEAN[:2]],
                     [LEAKED[3]["question"]] + [p["question"] for p in CLEAN[2:]]])
    generate = _two_pass(lambda p: json.dumps(next(passages)),
                         lambda q, p: answers[next(k for k in answers if k.startswith(q))]
                         + "</s>")

    with caplog.at_level("INFO", logger="lfa.selfgen.supplement"):
        manifest = _write(tmp_path, generate, docs=["one " * 60, "two " * 60])

    rows = _rows(tmp_path)
    assert [r["prompt"] for r in rows] == [p["question"] for p in CLEAN]
    assert [r["response"] for r in rows] == [p["answer"] for p in CLEAN]
    assert manifest["rejected"] == {"unparseable_passage": 0, "leaked_json": 5,
                                    "refers_to_source": 0, "duplicate": 0,
                                    "truncated_answer": 0, "short_answer": 0}
    assert manifest["n_questions"] == 3 + len(CLEAN) and manifest["n_pairs"] == len(CLEAN) == 6
    [line] = [r.getMessage() for r in caplog.records
              if r.getMessage().startswith("supplement: 6 pairs from 2 passages written to ")]
    assert line.endswith("; rejected: unparseable_passage 0, leaked_json 5, refers_to_source 0, "
                         "duplicate 0, truncated_answer 0, short_answer 0")
    assert "Himalaya" not in caplog.text and "descendants" not in caplog.text   # no samples


def test_the_manifest_records_the_filters_it_was_written_under(tmp_path, unhashed):
    from lfa.selfgen.supplement import REJECTION_REASONS, filters_sha256
    generate = _two_pass(lambda p: json.dumps([c["question"] for c in CLEAN]),
                         lambda q, p: LONG + "</s>")
    manifest = _write(tmp_path, generate)
    assert manifest["filters_sha256"] == filters_sha256() and len(filters_sha256()) == 64
    assert tuple(manifest["rejected"]) == REJECTION_REASONS
    assert {"leaked_json", "refers_to_source", "truncated_answer"} <= set(REJECTION_REASONS)
    assert "long_answer" not in REJECTION_REASONS
