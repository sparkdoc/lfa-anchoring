"""The seed corpus: the general-purpose text p(h) is estimated on.

Layerwise Function Anchoring (LFA) prices a sub-module's function on the hidden states the model
actually visits, so the artifact behind an anchoring run is only as good as the corpus those
states were collected from. This module rebuilds that corpus: it downloads pretraining text
(RedPajama, filtered into six sources) and instruction data (five datasets, each flattened to
prompt/response pairs), mixes them at **10:1 pretraining:instruction**, and writes one JSONL that
:func:`lfa.corpus.load_texts` reads directly.

Two row shapes go into that one file, matching what the loader understands::

    {"text": "...", "source": "redpajama_arxiv"}          # pretraining
    {"prompt": "...", "response": "...", "source": "dolly"}   # instruction

The instruction rows are rendered with the model's chat template when the corpus is read for
p(h) estimation -- the hidden states of a formatted exchange are not those of the same words run
together -- which is why the pair is kept in two fields rather than joined here.

**The recorded corpus.** The real-text artifact the paper's operating point was tuned against was
estimated on 10,629 documents in exactly this 10:1 shape (its 1.54 M samples per site count
hidden states, not documents). Two levels are involved, and they carry different numbers:

* the **download targets** -- 12,000 pretraining documents and 20,000 instruction pairs, the
  defaults of :func:`prepare_seed_corpus`. The pretraining target is a per-source *cap* of
  ``n/6`` = 2,000 (:func:`allocate`), applied before any filtering; a source with fewer rows
  within reach of the scan simply comes up short.
* the **realized corpus** -- 9,663 pretraining documents and 966 instruction pairs, after two
  pretraining sources land below their cap and the 10:1 mix trims the instruction side
  (:func:`weighted_mix`). This is :data:`SHIPPED_COMPOSITION`, a RECORD of what the February
  2026 datasets yielded, not a target a rebuild must hit::

    pretraining (9,663 documents)        instruction (966 pairs)
      redpajama_stackexchange  2,000         dolly          203
      redpajama_web            1,999         alpaca         231
      redpajama_book           1,998         code_alpaca     96
      redpajama_wikipedia      1,995         ultrachat      212
      redpajama_arxiv          1,524         oasst2         224
      redpajama_github           147

GitHub is far short of its 2,000 cap because GitHub entries are sparse in the head of the
RedPajama sample the scan reaches (see :func:`download_pretraining`). Why arXiv landed at 1,524
is NOT established: a 2,000-row streaming spot-check in September 2026 found the head of that
split roughly three-quarters arXiv, so scarcity cannot be the explanation there. Treat both
counts as a record of the corpus that was built, and expect a rebuild to drift as the upstream
sample moves. The instruction side is cut from ~20,000 available pairs to the 966 the 10:1 ratio
allows, so its per-source counts are a uniform draw from the pool rather than a cap. A rebuild
will not reproduce the recorded corpus byte for byte -- the upstream datasets move, and the
sampling RNG is this module's own -- but it reproduces its *composition*, which is what p(h)
depends on.
"""

from __future__ import annotations

import ast
import json
import logging
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REDPAJAMA_PATH = "ZengXiangyu/RedPajama-Data-1T-Sample"


class SourceUnavailable(RuntimeError):
    """Raised when a seed-corpus dataset cannot be loaded.

    The usual cause is the ordinary one -- no network, an offline cache that does not hold this
    dataset, or an upstream id that has moved -- so the message names the dataset and the split and
    reaches the user as one line (it is in :data:`lfa.cli.USER_FACING_ERRORS`) rather than as the
    last line of a traceback.
    """

# The composition of the corpus behind the real-text artifact the paper's operating point was
# tuned against (documents per source).
SHIPPED_COMPOSITION: dict[str, dict[str, int]] = {
    "pretraining": {
        "redpajama_arxiv": 1524,
        "redpajama_book": 1998,
        "redpajama_wikipedia": 1995,
        "redpajama_stackexchange": 2000,
        "redpajama_web": 1999,
        "redpajama_github": 147,
    },
    "instruction": {
        "dolly": 203,
        "alpaca": 231,
        "code_alpaca": 96,
        "ultrachat": 212,
        "oasst2": 224,
    },
}

# A document shorter than this many characters is not worth a row.
MIN_PRETRAINING_CHARS = 50
MIN_INSTRUCTION_CHARS = 20


# ----------------------------------------------------------------------------------------------
# RedPajama source predicates
# ----------------------------------------------------------------------------------------------

def parse_meta(example: dict) -> dict:
    """The ``meta`` field of a RedPajama row, which may be a dict or a Python-literal string."""
    meta = example.get("meta", {})
    if isinstance(meta, str):
        try:
            meta = ast.literal_eval(meta)
        except (ValueError, SyntaxError):
            return {}
    return meta if isinstance(meta, dict) else {}


def is_arxiv(example: dict) -> bool:
    """An arXiv paper: it carries an ``arxiv_id``, or an arXiv URL."""
    meta = parse_meta(example)
    return "arxiv_id" in meta or "arxiv" in str(meta.get("url", "")).lower()


def is_wikipedia(example: dict) -> bool:
    return "wikipedia" in str(parse_meta(example).get("url", "")).lower()


def is_github(example: dict) -> bool:
    return "github" in str(parse_meta(example).get("url", "")).lower()


def is_stackexchange(example: dict) -> bool:
    return "stackexchange" in str(parse_meta(example).get("url", "")).lower()


def is_book(example: dict) -> bool:
    """Any row whose metadata mentions "book" anywhere -- book titles, but a ``facebook.com`` URL
    too, which therefore matches :func:`is_web` as well. Kept exactly as the recorded corpus was
    built: tightening it would change the mixture p(h) was estimated on."""
    return "book" in str(parse_meta(example)).lower()


def is_web(example: dict) -> bool:
    """Web text: it has a URL, and that URL is none of the other four sources'."""
    url = str(parse_meta(example).get("url", "")).lower()
    return bool(url) and not any(marker in url
                                 for marker in ("arxiv", "wikipedia", "github", "stackexchange"))


PRETRAINING_SOURCES: dict[str, dict[str, Any]] = {
    "redpajama_arxiv": {"predicate": is_arxiv, "default_samples": 2000,
                        "description": "RedPajama arXiv academic papers"},
    "redpajama_wikipedia": {"predicate": is_wikipedia, "default_samples": 2000,
                            "description": "RedPajama Wikipedia articles"},
    "redpajama_github": {"predicate": is_github, "default_samples": 2000,
                         "description": "RedPajama GitHub code"},
    "redpajama_stackexchange": {"predicate": is_stackexchange, "default_samples": 2000,
                                "description": "RedPajama StackExchange Q&A"},
    "redpajama_book": {"predicate": is_book, "default_samples": 2000,
                       "description": "RedPajama books"},
    "redpajama_web": {"predicate": is_web, "default_samples": 2000,
                      "description": "RedPajama web text"},
}


# ----------------------------------------------------------------------------------------------
# Instruction extractors
# ----------------------------------------------------------------------------------------------

def _instruction_pair(instruction: str, extra: str, response: str, extra_label: str
                      ) -> dict[str, str] | None:
    """One prompt/response pair, with ``extra`` folded into the prompt under ``extra_label``."""
    instruction, extra, response = instruction.strip(), extra.strip(), response.strip()
    if not instruction or not response:
        return None
    prompt = f"{instruction}\n\n{extra_label}: {extra}" if extra else instruction
    return {"prompt": prompt, "response": response}


def extract_dolly(example: dict) -> dict[str, str] | None:
    """A Dolly row; its retrieval ``context``, when present, is folded into the prompt."""
    return _instruction_pair(example.get("instruction", ""), example.get("context", ""),
                             example.get("response", ""), "Context")


def extract_alpaca(example: dict) -> dict[str, str] | None:
    """A Stanford Alpaca row; its ``input``, when present, is folded into the prompt."""
    return _instruction_pair(example.get("instruction", ""), example.get("input", ""),
                             example.get("output", ""), "Input")


def extract_code_alpaca(example: dict) -> dict[str, str] | None:
    """A CodeAlpaca row; same shape as Alpaca."""
    return _instruction_pair(example.get("instruction", ""), example.get("input", ""),
                             example.get("output", ""), "Input")


def extract_ultrachat(example: dict) -> list[dict[str, str]] | None:
    """Every user->assistant turn of an UltraChat conversation, or ``None`` if it has none.

    A turn after the first carries the preceding conversation in its prompt, so the pair is
    self-contained: the hidden states of a reply that depends on unseen history are not the ones
    the model would visit in use.
    """
    messages = example.get("messages", [])
    pairs: list[dict[str, str]] = []

    for i in range(len(messages) - 1):
        if messages[i].get("role") != "user" or messages[i + 1].get("role") != "assistant":
            continue
        prompt = messages[i].get("content", "").strip()
        response = messages[i + 1].get("content", "").strip()
        if not prompt or not response:
            continue

        history = [f"{messages[j].get('role', '').capitalize()}: "
                   f"{messages[j].get('content', '').strip()}"
                   for j in range(i) if messages[j].get("content", "").strip()]
        if history:
            prompt = "Conversation history:\n" + "\n\n".join(history) + f"\n\nUser: {prompt}"
        pairs.append({"prompt": prompt, "response": response})

    return pairs or None


def extract_oasst2_pairs(rows: Iterable[dict], n_samples: int | None = None,
                         max_length: int = 2048, seed: int = 42) -> list[dict[str, str]]:
    """English prompter->assistant pairs from OASST2's message forest.

    OASST2 ships one row per message, not per exchange, so the tree has to be rebuilt: every
    English ``prompter`` message pairs with each of its ``assistant`` children (a prompt with
    three replies yields three pairs). ``n_samples`` takes a seeded random subset.
    """
    messages: dict[Any, dict] = {}
    children: dict[Any, list] = defaultdict(list)

    for row in rows:
        message_id, parent_id = row.get("message_id"), row.get("parent_id")
        messages[message_id] = {"text": row.get("text", "").strip(),
                                "role": row.get("role", ""),
                                "lang": row.get("lang", "en")}
        if parent_id:
            children[parent_id].append(message_id)

    pairs: list[dict[str, str]] = []
    for message_id, message in messages.items():
        if message["role"] != "prompter" or message["lang"] != "en":
            continue
        for child_id in children.get(message_id, []):
            child = messages.get(child_id)
            if not child or child["role"] != "assistant":
                continue
            prompt, response = message["text"], child["text"]
            if len(prompt) > MIN_INSTRUCTION_CHARS and len(response) > MIN_INSTRUCTION_CHARS:
                pairs.append({"prompt": prompt[:max_length], "response": response[:max_length],
                              "source": "oasst2"})

    if n_samples is not None and len(pairs) > n_samples:
        pairs = random.Random(seed).sample(pairs, n_samples)
    return pairs


INSTRUCTION_SOURCES: dict[str, dict[str, Any]] = {
    "dolly": {"path": "databricks/databricks-dolly-15k", "split": "train",
              "extract": extract_dolly, "default_samples": 5000,
              "description": "Databricks Dolly instructions"},
    "alpaca": {"path": "tatsu-lab/alpaca", "split": "train",
               "extract": extract_alpaca, "default_samples": 5000,
               "description": "Stanford Alpaca instructions"},
    "code_alpaca": {"path": "sahil2801/CodeAlpaca-20k", "split": "train",
                    "extract": extract_code_alpaca, "default_samples": 2000,
                    "description": "Code instruction-following"},
    "ultrachat": {"path": "HuggingFaceH4/ultrachat_200k", "split": "train_sft",
                  "extract": extract_ultrachat, "default_samples": 5000,
                  "description": "UltraChat multi-turn conversations"},
    "oasst2": {"path": "OpenAssistant/oasst2", "split": "train",
               "extract": None, "default_samples": 5000,
               "description": "OpenAssistant human conversations"},
}


# ----------------------------------------------------------------------------------------------
# Downloading
# ----------------------------------------------------------------------------------------------

def allocate(defaults: dict[str, int], total: int) -> dict[str, int]:
    """Split ``total`` rows across sources in proportion to their default sample counts."""
    denominator = sum(defaults.values())
    return {name: int(total * default / denominator) for name, default in defaults.items()}


def _load(loader: Callable, path: str, split: str, cache_dir: str | Path | None, what: str):
    """Call the injected loader, turning any failure into a clear :class:`SourceUnavailable`."""
    try:
        return loader(path, split=split, cache_dir=cache_dir)
    except Exception as exc:
        raise SourceUnavailable(
            f"Could not load {what} from '{path}' (split '{split}'): {exc}") from exc


def download_pretraining(n_total: int, *, max_length: int = 2048, seed: int = 42,
                         loader: Callable | None = None,
                         cache_dir: str | Path | None = None) -> list[dict[str, str]]:
    """``{"text", "source"}`` rows from RedPajama, allocated across the six sources.

    One pass over the split builds every source's shortlist at once: a source collects matching
    rows until it has ``3 x`` its allocation, and then samples its allocation from those
    candidates. The scan is a head scan, which is why a sparse source (GitHub) ends up short of
    its allocation -- see the composition table in the module docstring. Texts are truncated to
    ``max_length`` characters.

    Args:
        loader: ``datasets.load_dataset`` by default; anything with the signature
            ``loader(path, *, split, cache_dir)`` returning an indexable sequence of rows.
    """
    loader = loader if loader is not None else _default_loader()
    allocations = allocate({name: cfg["default_samples"]
                            for name, cfg in PRETRAINING_SOURCES.items()}, n_total)

    logger.info("Loading %s ...", REDPAJAMA_PATH)
    dataset = _load(loader, REDPAJAMA_PATH, "train", cache_dir, "RedPajama pretraining text")

    # ONE scan, not one per source. A RedPajama row is decoded on access and the six predicates
    # are cheap beside that, so six passes cost about six times what one does. Each source still
    # stops collecting at three times its allocation, and the shortlists are still built in
    # ascending index order, so the candidate lists -- and therefore the sampled rows -- are
    # exactly the ones six passes gave; only the reading changed. A row matching two sources is
    # still taken by both, which is how `book` and `web` overlap.
    wanted = {name: allocations[name] * 3 for name in PRETRAINING_SOURCES
              if allocations[name] > 0}
    candidates: dict[str, list[int]] = {name: [] for name in wanted}
    for index in range(len(dataset)):
        if all(len(found) >= wanted[name] for name, found in candidates.items()):
            break
        example = dataset[index]
        for name, found in candidates.items():
            if len(found) < wanted[name] and PRETRAINING_SOURCES[name]["predicate"](example):
                found.append(index)

    rows: list[dict[str, str]] = []
    for name, found in candidates.items():
        n_samples = allocations[name]
        rng = random.Random(seed)
        n_before = len(rows)
        for index in rng.sample(found, min(n_samples, len(found))):
            text = str(dataset[index].get("text", "")).strip()
            if len(text) > MIN_PRETRAINING_CHARS:
                rows.append({"text": text[:max_length], "source": name})
        logger.info("  %s: %d documents (allocated %d)", name, len(rows) - n_before, n_samples)

    return rows


def download_instruction(n_total: int, *, max_length: int = 2048, seed: int = 42,
                         loader: Callable | None = None,
                         cache_dir: str | Path | None = None) -> list[dict[str, str]]:
    """``{"prompt", "response", "source"}`` rows from the five instruction datasets.

    Allocation is proportional to each source's default sample count. Each source (except OASST2,
    whose whole message forest is needed to rebuild exchanges) samples ``3 x`` its allocation of
    rows and extracts pairs from them until the allocation is met; prompts and responses are
    truncated to ``max_length`` characters and pairs shorter than a sentence are dropped.
    """
    loader = loader if loader is not None else _default_loader()
    allocations = allocate({name: cfg["default_samples"]
                            for name, cfg in INSTRUCTION_SOURCES.items()}, n_total)

    rows: list[dict[str, str]] = []
    for name, config in INSTRUCTION_SOURCES.items():
        n_samples = allocations[name]
        if n_samples <= 0:
            continue

        logger.info("Loading %s: %s ...", name, config["description"])
        dataset = _load(loader, config["path"], config["split"], cache_dir, f"{name} instructions")

        if name == "oasst2":
            pairs = extract_oasst2_pairs(dataset, n_samples, max_length, seed)
        else:
            pairs = _extract_pairs(dataset, config["extract"], name, n_samples, max_length, seed)

        rows.extend(pairs)
        logger.info("  %s: %d pairs (allocated %d)", name, len(pairs), n_samples)

    return rows


def _extract_pairs(dataset: Sequence[dict], extract: Callable, source: str, n_samples: int,
                   max_length: int, seed: int) -> list[dict[str, str]]:
    """Extract up to ``n_samples`` pairs from a seeded random subset of ``dataset``."""
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n_samples * 3, len(dataset)))

    pairs: list[dict[str, str]] = []
    for index in indices:
        if len(pairs) >= n_samples:
            break
        extracted = extract(dataset[index])
        if not extracted:
            continue
        for pair in (extracted if isinstance(extracted, list) else [extracted]):
            prompt, response = pair["prompt"][:max_length], pair["response"][:max_length]
            if len(prompt) > MIN_INSTRUCTION_CHARS and len(response) > MIN_INSTRUCTION_CHARS:
                pairs.append({"prompt": prompt, "response": response, "source": source})
    return pairs[:n_samples]


def _default_loader() -> Callable:
    """``datasets.load_dataset``, imported on use so the module stays importable without it."""
    from datasets import load_dataset

    def loader(path, *, split, cache_dir=None):
        return load_dataset(path, split=split, cache_dir=cache_dir)

    return loader


# ----------------------------------------------------------------------------------------------
# Mixing and writing
# ----------------------------------------------------------------------------------------------

def weighted_mix(pretraining_rows: Sequence[dict], instruction_rows: Sequence[dict],
                 ratio: tuple[float, float] = (10, 1), seed: int = 42) -> list[dict]:
    """Mix the two row sets at ``ratio`` and shuffle, using as much data as the ratio allows.

    The scarcer side sets the scale: with 10:1 and 500 pretraining rows only 50 instruction rows
    fit, and with only 30 instruction rows only 300 pretraining rows are used. Each side is then
    sampled uniformly (no score weighting: the recorded corpus used none) and the result shuffled,
    so the two kinds are interleaved rather than concatenated. Deterministic in ``seed``.
    """
    rng = random.Random(seed)
    sides = {"pretraining": (list(pretraining_rows), ratio[0]),
             "instruction": (list(instruction_rows), ratio[1])}

    scale = min(len(rows) / weight for rows, weight in sides.values())
    mixed: list[dict] = []
    for name, (rows, weight) in sides.items():
        n_samples = min(int(scale * weight), len(rows))
        mixed.extend(rng.sample(rows, n_samples))
        logger.info("  %s: %d of %d rows", name, n_samples, len(rows))

    rng.shuffle(mixed)
    return mixed


def corpus_statistics(rows: Sequence[dict]) -> dict[str, Any]:
    """Row counts by source, character total and a rough token estimate (~4 characters/token)."""
    source_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        source_counts[row.get("source", "unknown")] += 1

    lengths = [len(row["text"]) if "text" in row else len(row["prompt"]) + len(row["response"])
               for row in rows]
    total_characters = sum(lengths)
    return {
        "total_samples": len(rows),
        "source_distribution": dict(sorted(source_counts.items())),
        "total_characters": total_characters,
        "estimated_tokens": total_characters // 4,
        "avg_length_chars": total_characters / len(lengths) if lengths else 0.0,
    }


def prepare_seed_corpus(
    out_path: str | Path,
    *,
    n_pretraining: int = 12_000,
    n_instruction: int = 20_000,
    max_length: int = 2048,
    seed: int = 42,
    cache_dir: str | Path | None = None,
    loader: Callable | None = None,
) -> Path:
    """Download, mix at 10:1 and write the seed corpus; return the JSONL path.

    The defaults are the **download targets** that realized the corpus behind the real-text
    artifact the paper's operating point was tuned against: 12,000 pretraining documents (a
    per-source cap of 2,000) and 20,000 instruction pairs. What lands is smaller -- two pretraining sources
    came up below the cap, and the 10:1 mix then trims the instruction side -- giving the
    recorded 9,663 + 966 of
    :data:`SHIPPED_COMPOSITION`, which is a record rather than a target. Read the
    realized per-source counts off the ``.stats.json`` sidecar written beside the corpus, not off
    the targets.

    Args:
        out_path: the JSONL to write. Its parent is created if absent.
        n_pretraining: pretraining documents to *aim for*; ``n_pretraining/6`` caps each source,
            and the 10:1 mix trims the result.
        n_instruction: instruction pairs to *aim for*, split across the five sources in
            proportion to their default sample counts; the 10:1 mix trims the result.
        max_length: truncate each document, prompt and response to this many characters.
        seed: seeds sampling and the shuffle; the same seed gives the same file.
        cache_dir: passed to the loader, for a non-default Hugging Face cache.
        loader: ``datasets.load_dataset`` by default; injectable for tests, with the signature
            ``loader(path, *, split, cache_dir)``.
    """
    out_path = Path(out_path)
    loader = loader if loader is not None else _default_loader()

    logger.info("Downloading up to %d pretraining documents ...", n_pretraining)
    pretraining_rows = download_pretraining(n_pretraining, max_length=max_length, seed=seed,
                                            loader=loader, cache_dir=cache_dir)
    logger.info("Downloading up to %d instruction pairs ...", n_instruction)
    instruction_rows = download_instruction(n_instruction, max_length=max_length, seed=seed,
                                            loader=loader, cache_dir=cache_dir)

    logger.info("Mixing at 10:1 pretraining:instruction ...")
    rows = weighted_mix(pretraining_rows, instruction_rows, ratio=(10, 1), seed=seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    stats = corpus_statistics(rows)
    stats["seed"] = seed
    stats["max_length"] = max_length
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logger.info("Wrote %d rows to %s (~%d tokens); statistics in %s",
                stats["total_samples"], out_path, stats["estimated_tokens"], stats_path)
    for source, count in stats["source_distribution"].items():
        logger.info("  %-24s %d", source, count)
    return out_path
