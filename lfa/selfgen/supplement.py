"""The domain supplement, written by the entry model from the domain's own passages.

The model contributes the form, not the knowledge: the writer reads each passage in context, so
the domain content comes from the corpus and only question-forming, answer construction and
third-person voice come from the model. In the research runs the resulting supplement scored
about 0.1 below a frontier-written one on the judge, and its effect was on *reachability* -- the
new knowledge becomes answerable when the model is asked about it. It does not protect skills: a
supplement written in a skill's mode left that skill no better (instruction following and
reasoning); keeping skills is the anchor's job. One model (Qwen3-0.6B), one seed, one domain.

Two deliberate deviations from the research frame: the template says "about a text on
{domain}" where the research one said "about a philosophy text", and there is no contamination
screen against an evaluation set (a user has none). The comparison-only `--enforce-spec`
reminder and the reasoning/instruction modes are not ported.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import __version__
from .generate import checkpoint_sha256, generate_texts, load_writer

logger = logging.getLogger(__name__)

__all__ = ["GENERATE_TEMPLATE", "SupplementOptions", "NoPairsWritten", "render_template",
           "template_sha256", "chunk_document", "chunk_passages", "parse_assistant_turn",
           "parse_qa_pairs", "write_supplement"]

# The research code's template, with the one change recorded in the module docstring.
GENERATE_TEMPLATE = """You are creating training data that teaches a small language model to \
ANSWER QUESTIONS about a text on {domain} in a helpful assistant's voice.

From the passage below, write {n} diverse question-answer pairs.

Rules:
- QUESTIONS: natural, information-seeking questions a curious student might ask. Do NOT write \
exam-style questions that name specific thought experiments or arguments (avoid "How does the \
author use the X thought experiment to argue Y"). Prefer "What is...", "Why does...", "What is \
the relationship between...", "What does the author mean by...".
- ANSWERS: written in the THIRD PERSON as a knowledgeable assistant, referring to the author by \
name where the passage names them (e.g., "Chalmers argues that..."). NEVER answer in the first \
person as the author ("I argue..."). Ground every answer strictly in the passage; do not invent. \
Keep answers to 2-5 sentences.
- Vary the difficulty and type.

Passage:
\"\"\"
{passage}
\"\"\"

Return ONLY a JSON array of objects: [{{"question": "...", "answer": "..."}}, ...]"""

_OBJ = re.compile(r'\{\s*"question"\s*:\s*"(.*?)"\s*,\s*"answer"\s*:\s*"(.*?)"\s*\}', re.S)
_MARK = re.compile(r'(?:^|\n)\s*(?:Question|Q)\s*[:.\)]\s*(.+?)\n\s*(?:Answer|A)\s*[:.\)]\s*(.+?)'
                   r'(?=\n\s*(?:Question|Q)\s*[:.\)]|\Z)', re.S | re.I)
_TURN_END = "<|im_end|>"


class NoPairsWritten(ValueError):
    """Raised when the corpus gives no passages, or the writer produced no usable pair."""


@dataclass
class SupplementOptions:
    """The recorded frame of the self-written supplement (one model, Qwen3-0.6B; one seed; one
    domain)."""
    pairs_per_passage: int = 6
    passage_chars: int = 4000
    min_passage_chars: int = 200
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.8
    batch_size: int = 16
    min_answer_chars: int = 40
    max_answer_chars: int = 100_000
    seed: int = 42


def render_template(domain_description: str, n: int, passage: str) -> str:
    return GENERATE_TEMPLATE.format(domain=domain_description, n=n, passage=passage)


def template_sha256() -> str:
    return hashlib.sha256(GENERATE_TEMPLATE.encode("utf-8")).hexdigest()


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


def parse_assistant_turn(decoded: str) -> str:
    """The assistant's text up to the turn end; the whole string when there is no turn end."""
    return decoded.split(_TURN_END, 1)[0] if _TURN_END in decoded else decoded


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip('"')


def parse_qa_pairs(text: str) -> list[dict]:
    """JSON objects first (salvaging complete ones from a truncated array), then Q/A markers."""
    pairs = [{"question": _clean(q), "answer": _clean(a)} for q, a in _OBJ.findall(text or "")]
    if not pairs:
        pairs = [{"question": _clean(q), "answer": _clean(a)} for q, a in _MARK.findall(text or "")]
    return [p for p in pairs if p["question"] and p["answer"]]


def _prompt_for(tokenizer, passage: str, n: int, domain_description: str,
                chat_template: bool) -> str:
    body = render_template(domain_description, n, passage)
    if not chat_template:
        return body + "\n"
    return tokenizer.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def write_supplement(model_id: str, documents: list[str], out_path, *, domain_description: str,
                     options: SupplementOptions | None = None, corpus_sha256: str,
                     generate=generate_texts, writer=None, device: str = "cuda:0") -> dict:
    """Write question-and-answer pairs from ``documents`` (the TRAINING side only) to ``out_path``.

    Raises:
        NoPairsWritten: no passage of at least ``min_passage_chars`` (before any model loads),
            or no pair survived parsing and the length filter.
    """
    options = options or SupplementOptions()
    out_path = Path(out_path)
    passages = [(index, passage) for index, text in enumerate(documents)
                for passage in chunk_passages(text, options.passage_chars,
                                              options.min_passage_chars)]
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
    stop = [tokenizer.eos_token_id]
    end_id = tokenizer.convert_tokens_to_ids(_TURN_END)
    if isinstance(end_id, int) and end_id >= 0 and end_id not in stop:
        stop.append(end_id)

    rows, rejected, seen = [], {"short_answer": 0, "long_answer": 0, "duplicate": 0,
                                "unparseable_passage": 0}, set()
    for batch_index in range(0, len(passages), options.batch_size):
        batch = passages[batch_index:batch_index + options.batch_size]
        prompts = [_prompt_for(tokenizer, passage, options.pairs_per_passage, domain_description,
                               chat_template) for _, passage in batch]
        outputs = generate(model, tokenizer, prompts, max_new_tokens=options.max_new_tokens,
                           temperature=options.temperature, top_p=options.top_p,
                           stop_token_ids=stop, seed=options.seed,
                           batch_index=batch_index // options.batch_size)
        for (source_index, _), output in zip(batch, outputs):
            pairs = parse_qa_pairs(parse_assistant_turn(output))
            if not pairs:
                rejected["unparseable_passage"] += 1
            for pair in pairs:
                length = len(pair["answer"])
                if length < options.min_answer_chars:
                    rejected["short_answer"] += 1
                    continue
                if length > options.max_answer_chars:
                    rejected["long_answer"] += 1
                    continue
                if pair["question"] in seen:             # keyed on the question, as recorded
                    rejected["duplicate"] += 1
                    continue
                seen.add(pair["question"])
                rows.append({"prompt": pair["question"], "response": pair["answer"],
                             "source_index": source_index})
        logger.info("supplement: %d/%d passages -> %d pairs", min(batch_index + len(batch),
                    len(passages)), len(passages), len(rows))

    if not rows:
        raise NoPairsWritten(
            f"{len(passages)} passage(s) were sent to {model_id} and no usable pair came back "
            f"(rejected: {rejected}). Check that the model follows the template; a chat model "
            "with a chat template is expected.")

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
        "template": render_template(domain_description, options.pairs_per_passage, "{passage}"),
        "template_sha256": template_sha256(),
        "chat_template_applied": chat_template,
        "options": asdict(options),
        "decoding": {"temperature": options.temperature, "top_p": options.top_p, "top_k": 0,
                     "min_p": 0.0},
        "n_documents": len(documents),
        "n_passages": len(passages),
        "n_pairs": len(rows),
        "rejected": rejected,
        "lfa_version": __version__,
    }
    Path(str(out_path) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
