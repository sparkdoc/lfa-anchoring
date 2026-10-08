"""The domain supplement, written by the entry model from the domain's own passages.

The model contributes the form, not the knowledge: the writer reads each passage in context, so
the domain content comes from the corpus and only question-forming and the answer's construction
come from the model. It writes in two passes over each passage. First it writes questions from a
questions-only template, returned as a JSON array of strings. Then it answers each question in one
ordinary user turn holding the passage and "Using the passage, answer: <question>" -- no length,
sentence-count, format or voice instruction, so the answer comes in the model's own default chat
format. The training pair is the question and that answer; the passage is not in it, so a
question or answer that cites its source is dropped (``refers_to_source``), and under this
prompt many answers do (on the Darwin text, 395 of 1,235).

Why this answer turn. The one-pass writer this replaces asked for question-answer pairs as JSON,
answers in the third person in 2-5 sentences. In the format-drift diagnostic (Qwen3-0.6B, the
walkthrough's Darwin data, four epochs, one seed, one greedy decode per prompt) a model trained on
it answered in that style everywhere, under the anchor: 0 of 40 general prompts with any
markdown, against 39 of 40 for the base model; 0 of 20 Darwin prompts, against 19 of 20; 0 of the
judge's 80 Darwin questions, against 78 of 80; and 6 of 40 general prompts refused as not in "the
text". Trained on this writer's supplement instead, the model answered 36 of 40, 15 of 20 and 66
of 80 with markdown, refused none, wrote 625-character general answers (base 1,095, one-pass
223), and moved less from the base model's answers (NLL shift +0.106 against +0.222), at held-out
Darwin perplexity 19.36 against the one-pass writer's 19.26 and WikiText-2 perplexity 16.06
against 16.18. On correctness and completeness (gpt-5.6-luna at medium effort, 80 Darwin questions, blinded,
frozen rubric, 0-4) it scored 1.337 against the one-pass writer's 1.575 (-0.237, paired p =
0.071: not separable at this n; zero-rates 0.350 and 0.338); the base model's level was 1.050. A variant with the passage as
a system message and the question alone kept more of the format but was behind the one-pass
writer on the judged score (correctness and completeness, p = 0.010; correctness alone
p = 0.007), which is why this turn ships. The measured arms mixed their
supplement's unshuffled prefix, under the filter as it stood before the "of"/"from" guards and
the extended text patterns.

What the research measured. Its supplement was the one-pass template's (written for philosophy
texts): it scored about 0.1 below a frontier-written one on the judge, and its effect was on
*reachability* -- the new knowledge becomes answerable when the model is asked about it. It does
not protect skills: a supplement written in a skill's mode left that skill no better (instruction
following and reasoning); keeping skills is the anchor's job. One model (Qwen3-0.6B), one seed,
one domain. This writer departs from that frame in four ways, so the research's judged result is
not measured for it: the template's wording is domain-neutral ("a text on {domain}" where the
research said "a philosophy text", no thought experiments named in the rule against exam-style
questions, no named example -- a small writer copies whatever name an example gives); the
question template adds a rule against citing the passage, and pairs that cite it anyway are
dropped; the answers are the model's own chat answers to the passage and the question rather
than third-person 2-5-sentence ones; and nothing screens the pairs against an
evaluation set (a user has none). The comparison-only ``--enforce-spec`` reminder and the
reasoning/instruction modes are not ported.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import __version__
from .generate import chat_turn_end, checkpoint_sha256, generate_texts, load_writer

logger = logging.getLogger(__name__)

__all__ = ["QUESTION_TEMPLATE", "ANSWER_TEMPLATE", "REJECTION_REASONS", "SupplementOptions",
           "NoPairsWritten", "render_question_prompt", "render_answer_prompt",
           "question_template_sha256", "answer_template_sha256", "filters_sha256", "leaks_json",
           "refers_to_source", "chunk_document", "chunk_passages", "parse_assistant_turn",
           "parse_questions", "write_supplement"]

# Pass 1. The research template's question rules, made domain-neutral as recorded in the module
# docstring, asking for questions only; a JSON array of strings is the form a 0.6B writer fills
# reliably.
QUESTION_TEMPLATE = """You are creating training data that teaches a small language model to \
ANSWER QUESTIONS about a text on {domain}.

From the passage below, write {n} diverse questions.

Rules:
- Natural, information-seeking questions a curious student might ask. Do NOT write exam-style \
questions that only make sense with the passage in hand (avoid "How does the author use the \
example of X to argue Y"). Prefer "What is...", "Why does...", "What is the relationship \
between...", "What does the author mean by...".
- Each question must make sense on its own to someone who has not read the passage: never \
mention "the passage" or "the text".
- Vary the difficulty and type.

Passage:
\"\"\"
{passage}
\"\"\"

Return ONLY a JSON array of {n} strings: ["...", "..."]"""

# Pass 2. One ordinary user turn per question, in the format-drift prototype's own words: the
# passage, then "Using the passage, answer:". No length, sentence-count, format or voice
# instruction. An instruction not to mention the passage suppressed the model's own format in
# the diagnostic, and the passage as a system message fell behind on correctness (see the module
# docstring), so source references are left to ``refers_to_source``.
ANSWER_TEMPLATE = """Here is a passage from a text on {domain}:
\"\"\"
{passage}
\"\"\"

Using the passage, answer: {question}"""

#: The writer's own JSON field syntax -- ``"question":`` / ``"answer":``, or an object opening
#: on one of those names -- inside a question or answer. It is there when the writer botched an
#: object's quoting and the parser's field ran on into the next field or object
#: (``... the answer.", "answer": "..."``). Field-name syntax only: a brace or a quote alone is
#: ordinary text (a recipe, a code snippet, a quotation), and is kept. Applied to every question,
#: since pass 1's questions are parsed out of JSON.
_LEAKED_JSON = re.compile(r'"(?:question|answer)"\s*:|\{\s*"(?:question|answer)"')
#: An answer that IS the writer's JSON shape: it opens on an object (or an array of them) whose
#: first field is ``"question"``/``"answer"``, or it ends on such a field and its string value
#: (closed or not, with or without the closing brace). Pass 2's answers are free chat text, never
#: parsed out of JSON, so the run-on leak above cannot arise inside them; what is left is the
#: model writing its answer as a pair object. The field names in mid-answer (an example in a text
#: about a JSON API) are kept.
_JSON_SHAPED_ANSWER = re.compile(
    r'\A\s*(?:```\w*\s*)?\[?\s*\{\s*"(?:question|answer)"\s*:'
    r'|"(?:question|answer)"\s*:\s*"(?:[^"\\]|\\.)*"?\s*\}?\s*\]?\s*(?:```)?\s*\Z')

_DET = r"(?:the|this|that)"
_QUALIFIER = r"(?:(?:given|provided|above|original|same)\s+)?"
#: A reference to the passage or the text as a source. Specific rather than any mention of the
#: words: "the passage" not followed by ``of``/``through``/``to``/... (so "the passage of time"
#: and "the passage through the strait" are ordinary text), the passage after a preposition that
#: places something in it ("in the passage", "according to the passage"; not "in the passage
#: of/from X", a domain's own passage), "the text" only after such a preposition (not "in the
#: text of X", a law's or a treaty's), with "given"/"provided", before a verb of reporting, an
#: adverb allowed ("the text states", "the text also notes", "the text does not mention"), as
#: "the text's", or in "in the context of the text" not followed by a word ("... text messaging"
#: is ordinary). Tuned on the 1,537 native-format answers of the format-drift prototype and the
#: kept pairs of its system-message arm (1,158) and its shipped arm (840) (Qwen3-0.6B, Darwin).
#: Still read as a source reference, since no pattern tells them apart: a musical or scriptural
#: "the passage in bars 12-16", "the text is not considered canonical", "within the text, Hamlet
#: says", "the text's earliest manuscript", "the text shows a chiastic structure".
_SOURCE_TEXT = re.compile(
    r"\b" + _DET + r"\s+" + _QUALIFIER + r"(?:passage|excerpt)\b"
    r"(?!\s+(?:of|through|to|from|between|across|into|way)\b)"
    r"|\b(?:in|from|according\s+to|based\s+on|within)\s+" + _DET + r"\s+" + _QUALIFIER
    + r"(?:passage|excerpt)\b(?!\s+(?:of|from)\b)"
    r"|\b(?:according\s+to|in|from|based\s+on|within|by)\s+(?:the|this)\s+" + _QUALIFIER
    + r"text\b(?!\s+of\b)"
    r"|\b(?:the|this)\s+(?:given|provided|above)\s+text\b"
    r"|\b(?:the|this)\s+text\s+(?:(?:also|explicitly|clearly|further)\s+)?(?:provided|given|"
    r"above|says|said|states|stated|mentions|mentioned|describes|described|suggests|suggested|"
    r"explains|explained|notes|noted|implies|implied|discusses|discussed|highlights|"
    r"emphasi[sz]es|indicates|argues|presents|references|referenced|contrasts|contrasted|"
    r"provides|gives|lists|shows|outlines|does\s+not|doesn't|did\s+not|is\s+not|isn't)\b"
    r"|\bthe\s+text(?:'|\u2019)s\b"
    r"|\bin\s+the\s+context\s+of\s+the\s+text\b(?!\s*\w)",
    re.I)
#: "The author" before a verb of reporting ("the author states", "the author does not mention").
#: Counted only in an answer whose question does not itself name the author (_NAMES_AUTHOR): the
#: question template invites "What does the author mean by...", and "The author means..." then
#: answers it in its own terms.
_AUTHOR_REPORTS = re.compile(
    r"\bthe\s+author\s+(?:also\s+)?(?:states|stated|mentions|mentioned|says|said|notes|noted|"
    r"writes|wrote|describes|described|explains|explained|suggests|suggested|argues|argued|"
    r"claims|claimed|believes|implies|implied|means|meant|concludes|concluded|emphasi[sz]es|"
    r"highlights|points\s+out|pointed\s+out|discusses|discussed|refers|uses|used|compares|"
    r"indicates|presents|does\s+not|doesn't|did\s+not)\b",
    re.I)
_NAMES_AUTHOR = re.compile(r"\bthe\s+author\b", re.I)

#: Every reason a question, an answer or a whole passage's output is dropped; the manifest's
#: ``rejected`` counts each. Each pass tries its reasons in the order given for it below. The
#: tuple is the union of the two, not an order across them (``truncated_answer`` comes after
#: ``duplicate`` here and is pass 2's first check). Pass 1 (per passage, then per question):
#: ``unparseable_passage`` (no question parsed), ``leaked_json``, ``refers_to_source``,
#: ``duplicate`` (a question already queued). Pass 2 (per answer): ``truncated_answer`` (no
#: turn end before ``max_new_tokens``), ``leaked_json`` (an answer in the writer's JSON shape),
#: ``refers_to_source``, ``short_answer``.
REJECTION_REASONS = ("unparseable_passage", "leaked_json", "refers_to_source", "duplicate",
                     "truncated_answer", "short_answer")

_QUESTION_FIELD = re.compile(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"')
_STRING = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
_QUESTION_LINE = re.compile(
    r"^\s*(?:\d+\s*[.):]|[-*\u2022]|Q(?:uestion)?\s*\d*\s*[:.)])?\s*(.+?\?)\s*$", re.M | re.I)
_EMPTY_THINK = re.compile(r"^\s*<think>\s*</think>\s*")


class NoPairsWritten(ValueError):
    """Raised when the corpus gives no passages, or the writer produced no usable pair."""


@dataclass
class SupplementOptions:
    """The recorded frame of the self-written supplement (one model, Qwen3-0.6B; one seed; one
    domain).

    ``pairs_per_passage`` is how many questions pass 1 asks for per passage; ``max_new_tokens``
    caps both passes (an answer that reaches it is dropped as ``truncated_answer``: 1,024 left
    none of this writer's 1,235 Darwin answers in the format-drift diagnostic unfinished);
    ``batch_size`` is prompts per
    ``generate`` call in both passes.
    """
    pairs_per_passage: int = 6
    passage_chars: int = 4000
    min_passage_chars: int = 200
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.8
    batch_size: int = 16
    min_answer_chars: int = 40
    seed: int = 42


def render_question_prompt(domain_description: str, n: int, passage: str) -> str:
    return QUESTION_TEMPLATE.format(domain=domain_description, n=n, passage=passage)


def render_answer_prompt(domain_description: str, passage: str, question: str) -> str:
    return ANSWER_TEMPLATE.format(domain=domain_description, passage=passage, question=question)


def question_template_sha256() -> str:
    return hashlib.sha256(QUESTION_TEMPLATE.encode("utf-8")).hexdigest()


def answer_template_sha256() -> str:
    return hashlib.sha256(ANSWER_TEMPLATE.encode("utf-8")).hexdigest()


def filters_sha256() -> str:
    """The hash of what decides which questions and answers are kept: the rejection reasons, in
    their per-pass order, the two leaked-JSON patterns and the refers-to-source patterns.

    Recorded in the manifest and compared before a supplement is reused
    (:func:`lfa.supplements.supplement_for`), so a supplement written under other filters --
    one that kept pairs these drop -- is written again rather than trained on. The lengths
    are :class:`SupplementOptions` fields, recorded beside it.
    """
    return hashlib.sha256(json.dumps({
        "reasons": REJECTION_REASONS,
        "leaked_json": [_LEAKED_JSON.pattern, _JSON_SHAPED_ANSWER.pattern],
        "refers_to_source": [_SOURCE_TEXT.pattern, _AUTHOR_REPORTS.pattern,
                             _NAMES_AUTHOR.pattern]}).encode()).hexdigest()


def leaks_json(pair: dict) -> bool:
    """Whether a pair carries the writer's JSON: its field syntax anywhere in the question
    (questions are parsed out of JSON), or an answer that is itself in the writer's JSON shape."""
    return bool(_LEAKED_JSON.search(pair["question"])
                or _JSON_SHAPED_ANSWER.search(pair["answer"]))


def refers_to_source(pair: dict) -> bool:
    """Whether a pair's question or answer speaks of its source, which the training pair lacks.

    The question or the answer citing the passage or the text ("according to the passage", "the
    text states", "in the passage"); or the answer reporting what "the author" says when the
    question did not itself ask about the author. A pair with an empty ``answer`` checks the
    question alone, as pass 1 does.
    """
    question, answer = pair["question"], pair["answer"]
    if _SOURCE_TEXT.search(question) or _SOURCE_TEXT.search(answer):
        return True
    return bool(_AUTHOR_REPORTS.search(answer)) and not _NAMES_AUTHOR.search(question)


def chunk_document(text: str, target_chars: int) -> list[str]:
    """Split on blank-line (paragraph) boundaries into ~``target_chars`` sections; ported verbatim."""
    paras = [p for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > target_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur.strip():
        chunks.append(cur)
    return chunks


def chunk_passages(text: str, size: int = 4000, min_size: int = 200) -> list[str]:
    return [c for c in chunk_document(text, size) if len(c) >= min_size]


def parse_assistant_turn(decoded: str, turn_end: str | None) -> str:
    """The assistant's text up to ``turn_end``; the whole string when it is ``None`` or absent.

    ``turn_end`` is :func:`lfa.selfgen.generate.chat_turn_end` of the writer's tokenizer.
    """
    return decoded.split(turn_end, 1)[0] if turn_end else decoded


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip('"')


def _unescape(literal: str) -> str:
    try:
        return json.loads(f'"{literal}"')
    except ValueError:
        return literal


#: ``strict=False``: a raw newline inside a question string is kept rather than failing the array.
_DECODER = json.JSONDecoder(strict=False)


def _questions_in(value) -> list[str]:
    """The questions of a decoded value: a list of strings, or of ``{"question": ...}`` objects."""
    if not isinstance(value, list):
        return []
    return [item if isinstance(item, str) else item["question"] for item in value
            if isinstance(item, str)
            or (isinstance(item, dict) and isinstance(item.get("question"), str))]


def _single_quoted(text: str, start: int) -> list[str]:
    """``['What is X?', 'Why Y?']`` at ``start``: a Python-style array of single-quoted strings,
    split on the quote-comma-quote between items (so an apostrophe inside one is kept)."""
    end = text.find("]", start)
    inner = text[start + 1:end].strip() if end != -1 else ""
    if '"' in inner or len(inner) < 2 or inner[0] != "'" or inner[-1] != "'":
        return []
    return [item.replace("\\'", "'") for item in re.split(r"'\s*,\s*'", inner[1:-1])]


def _usable(items: list[str]) -> list[str]:
    """Items cleaned, without the ones holding no letter or digit (an echoed ``"..."``)."""
    return [q for q in map(_clean, items) if re.search(r"\w", q)]


def _asks(items: list[str]) -> bool:
    return bool(items) and all(q.endswith("?") for q in items)


def parse_questions(text: str) -> list[str]:
    """The questions in a pass-1 output, forgivingly.

    A JSON array first, decoded from its own ``[`` to its own ``]`` (chatter after it, brackets
    and quoted words included, is ignored): of strings, or of objects with a ``"question"``
    field. Of several arrays, the first whose every item ends in ``?`` wins, else the first with
    any usable item -- so a ``[1]``, an echoed ``["...", "..."]`` or a list of topics before the
    real array does not. Then a single-quoted Python-style array, chosen the same way. An array
    that does not parse whole (truncated, or a missing comma) gives up its ``"question"`` fields
    when it has any, else its complete string literals that end in ``?`` -- a quoted word in
    chatter after a broken array is not a question. With no array, lines ending in ``?``
    (numbered, bulleted or ``Q:``-marked or bare). A "question" with no letter or digit in it
    (the template's own ``"..."``, echoed) is not one.
    """
    text = text or ""
    starts = [match.start() for match in re.finditer(r"\[", text)]

    def first_asking(candidates):
        candidates = [c for c in candidates if c]
        return next((c for c in candidates if _asks(c)), candidates[0] if candidates else [])

    def decoded():
        for start in starts:
            try:
                value, _ = _DECODER.raw_decode(text, start)
            except ValueError:
                continue
            yield _usable(_questions_in(value))

    found = first_asking(decoded())
    if not found:
        found = first_asking(_usable(_single_quoted(text, start)) for start in starts)
    if not found and starts:
        body = text[starts[0]:]
        fields = _QUESTION_FIELD.findall(body)
        found = _usable([_unescape(literal) for literal in fields] if fields else
                        [q for q in map(_unescape, _STRING.findall(body))
                         if q.rstrip().endswith("?")])
    if not found:
        found = _usable(_QUESTION_LINE.findall(text))
    return found


def _answer_text(output: str, turn_end: str | None, eos_token: str | None) -> tuple[str, bool]:
    """``(answer, finished)``: the assistant's text cut at its turn end (or EOS), and whether the
    turn ended at all before the token cap."""
    stops = [t for t in (turn_end, eos_token) if t]
    finished = any(t in output for t in stops)
    for t in stops:
        output = output.split(t, 1)[0]
    return _EMPTY_THINK.sub("", output).strip(), finished


def _as_prompt(tokenizer, body: str, chat_template: bool) -> str:
    if not chat_template:
        return body + "\n"
    return tokenizer.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def write_supplement(model_id: str, documents: list[str], out_path, *, domain_description: str,
                     options: SupplementOptions | None = None, corpus_sha256: str,
                     generate=generate_texts, writer=None, device: str = "cuda:0") -> dict:
    """Write question-and-answer pairs from ``documents`` (the TRAINING side only) to ``out_path``.

    Each row is ``{"prompt", "response", "source_index", "passage_index"}``: ``source_index`` is
    the DOCUMENT's index in ``documents``, ``passage_index`` the passage's index within that
    document's :func:`chunk_passages` (with the manifest's ``passage_chars`` and
    ``min_passage_chars``), so ``chunk_passages(documents[s], ...)[p]`` is the passage the pair
    was written from.

    Raises:
        NoPairsWritten: no passage of at least ``min_passage_chars`` (before any model loads),
            or no question or no answer survived parsing and the filters.
    """
    options = options or SupplementOptions()
    out_path = Path(out_path)
    passages = [(source_index, passage_index, passage)
                for source_index, text in enumerate(documents)
                for passage_index, passage in enumerate(
                    chunk_passages(text, options.passage_chars, options.min_passage_chars))]
    if not passages:
        raise NoPairsWritten(
            f"0 passages of at least {options.min_passage_chars} characters in "
            f"{len(documents)} training document(s): nothing to write a supplement from.")

    writer_sha256 = checkpoint_sha256(model_id)          # before the sampling, not after
    model, tokenizer = writer if writer is not None else load_writer(model_id, device)
    chat_template = bool(getattr(tokenizer, "chat_template", None))
    if not chat_template:
        logger.warning("%s has no chat template: supplement prompts are sent as plain text, "
                       "outside the recorded frame", model_id)
    # The turn end comes from the writer's own template (<|im_end|> under ChatML, <|eot_id|>
    # under Llama 3); no template, or one that names none, leaves EOS as the only stop.
    turn_end = chat_turn_end(tokenizer)
    eos_token = getattr(tokenizer, "eos_token", None)
    stop = [tokenizer.eos_token_id]
    if turn_end is not None:
        end_id = tokenizer.convert_tokens_to_ids(turn_end)
        if isinstance(end_id, int) and end_id >= 0 and end_id not in stop:
            stop.append(end_id)
    batch_size = options.batch_size
    rejected = dict.fromkeys(REJECTION_REASONS, 0)

    def no_pairs() -> NoPairsWritten:
        return NoPairsWritten(
            f"{len(passages)} passage(s) were sent to {model_id} and no usable pair came back "
            f"(rejected: {rejected}). Check that the model follows the template; a chat model "
            "with a chat template is expected.")

    # -- pass 1: questions, per passage
    queued, seen = [], set()
    for start in range(0, len(passages), batch_size):
        batch = passages[start:start + batch_size]
        prompts = [_as_prompt(tokenizer, render_question_prompt(
            domain_description, options.pairs_per_passage, passage), chat_template)
            for _, _, passage in batch]
        outputs = generate(model, tokenizer, prompts, max_new_tokens=options.max_new_tokens,
                           temperature=options.temperature, top_p=options.top_p,
                           stop_token_ids=stop, seed=options.seed,
                           batch_index=start // batch_size)
        for (source_index, passage_index, passage), output in zip(batch, outputs):
            questions = parse_questions(parse_assistant_turn(output, turn_end))
            if not questions:
                rejected["unparseable_passage"] += 1
            for question in questions:
                # First: a leaked field's text is the writer's syntax, whatever else it is.
                if _LEAKED_JSON.search(question):
                    rejected["leaked_json"] += 1
                    continue
                if refers_to_source({"question": question, "answer": ""}):
                    rejected["refers_to_source"] += 1
                    continue
                if question in seen:                     # keyed on the question, as recorded
                    rejected["duplicate"] += 1
                    continue
                seen.add(question)
                queued.append((source_index, passage_index, passage, question))
        logger.info("supplement questions: %d/%d passages -> %d questions",
                    min(start + len(batch), len(passages)), len(passages), len(queued))
    if not queued:
        raise no_pairs()

    # -- pass 2: one answer per question, many questions per generate call. The batch indices
    # continue from pass 1's, so no batch of either pass reuses another's sampling seed.
    first_answer_batch = -(-len(passages) // batch_size)
    rows = []
    for start in range(0, len(queued), batch_size):
        batch = queued[start:start + batch_size]
        prompts = [_as_prompt(tokenizer, render_answer_prompt(domain_description, passage,
                                                              question), chat_template)
                   for _, _, passage, question in batch]
        outputs = generate(model, tokenizer, prompts, max_new_tokens=options.max_new_tokens,
                           temperature=options.temperature, top_p=options.top_p,
                           stop_token_ids=stop, seed=options.seed,
                           batch_index=first_answer_batch + start // batch_size)
        for (source_index, passage_index, _, question), output in zip(batch, outputs):
            answer, finished = _answer_text(output, turn_end, eos_token)
            if not finished:
                rejected["truncated_answer"] += 1
                continue
            pair = {"question": question, "answer": answer}
            if leaks_json(pair):
                rejected["leaked_json"] += 1
                continue
            if refers_to_source(pair):
                rejected["refers_to_source"] += 1
                continue
            if len(answer) < options.min_answer_chars:
                rejected["short_answer"] += 1
                continue
            rows.append({"prompt": question, "response": answer, "source_index": source_index,
                         "passage_index": passage_index})
        logger.info("supplement answers: %d/%d questions -> %d pairs",
                    min(start + len(batch), len(queued)), len(queued), len(rows))
    if not rows:
        raise no_pairs()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "kind": "supplement",
        "model_id": model_id,
        "writer_sha256": writer_sha256,
        "corpus_sha256": corpus_sha256,
        "domain_description": domain_description,
        "question_template": render_question_prompt(domain_description,
                                                    options.pairs_per_passage, "{passage}"),
        "question_template_sha256": question_template_sha256(),
        "answer_template": render_answer_prompt(domain_description, "{passage}", "{question}"),
        "answer_template_sha256": answer_template_sha256(),
        "filters_sha256": filters_sha256(),
        "chat_template_applied": chat_template,
        "options": asdict(options),
        "decoding": {"temperature": options.temperature, "top_p": options.top_p, "top_k": 0,
                     "min_p": 0.0, "enable_thinking": False},
        "n_documents": len(documents),
        "n_passages": len(passages),
        "n_questions": len(queued),
        "n_pairs": len(rows),
        "rejected": rejected,
        "lfa_version": __version__,
    }
    Path(str(out_path) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("supplement: %d pairs from %d passages written to %s; rejected: %s", len(rows),
                len(passages), out_path,
                ", ".join(f"{reason} {count}" for reason, count in rejected.items()))
    return manifest
