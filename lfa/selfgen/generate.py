"""One batched sampling loop, and the text-level helpers around it.

Ported from the research generator (mr-fusion ``scripts/prepare_selfgen_corpus.py``). The one
rule that matters most is in :func:`generate_texts`: **every truncation knob is passed
explicitly**. ``model.generate`` inherits any knob it is not given from the checkpoint's
``generation_config.json``, and Qwen3's ships ``top_k: 20`` -- so passing only temperature and
top-p yields top-20 sampling out of a 151,936-token vocabulary while looking untruncated
(measured in the research record, 2026-09-09: lifting it took distinct sampled tokens from 817
to 1,326 at a fixed seed). ``top_k=0`` and ``min_p=0.0`` disable truncation;
``repetition_penalty=1.0`` keeps the chain a true sample of the model's distribution.

Reproducibility: ``transformers.generate`` takes no private generator, so each batch seeds
torch's global RNG from ``(seed, batch_index)``; a rebuild with the same seed and batch size
draws the same text.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch

from ..models import load_tokenizer, resolve_model_path

__all__ = [
    "load_writer", "pick_seed_prefix", "chat_user_header", "boundary_markers", "clean_raw",
    "drop_burn_in", "passes_filters", "generate_texts", "checkpoint_sha256", "sha256_text",
]

_HEADER_MARK = "␟"   # characters no template contains, to find where each turn's content goes
_USER_MARK = "␝"
_ASSISTANT_MARK = "␞"


def load_writer(model_id: str, device: str = "cuda:0"):
    """The model that writes: bf16 on an accelerator, fp32 on CPU, in eval mode, left-padded."""
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(model_id)
    tokenizer.padding_side = "left"                       # batched generation needs left padding
    dtype = torch.float32 if device.startswith("cpu") else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(resolve_model_path(model_id), dtype=dtype,
                                                 device_map=device).eval()
    return model, tokenizer


def pick_seed_prefix(tokenizer, model=None) -> str:
    """The token a free-running sample starts from, chosen without assuming a packing convention.

    Precedence: the model's DECLARED sequence-start id (Qwen3 sets
    ``generation_config.bos_token_id`` to ``<|endoftext|>`` while ``tok.bos_token`` is None),
    then the tokenizer's BOS, then its EOS, then a bare newline.
    """
    if model is not None:
        bid = getattr(getattr(model, "generation_config", None), "bos_token_id", None)
        if isinstance(bid, (list, tuple)):
            bid = bid[0] if bid else None
        if bid is not None:
            return tokenizer.decode([bid])
    for attr in ("bos_token", "eos_token"):
        token = getattr(tokenizer, attr, None)
        if token:
            return token
    return "\n"


def chat_user_header(tokenizer) -> str | None:
    """What a user turn opens with under the tokenizer's chat template, or ``None`` without one.

    Isolated by difference, not read off a lone user message: a template may prepend BOS or a
    default system turn (Qwen2.5-Instruct, Llama-3.2), and that preamble is not the opener. Two
    conversations are rendered, ``[user, assistant]`` and ``[user, assistant, user]``. The text
    after the assistant's content in the first is the assistant turn's close; in the second, what
    follows that same close up to the last user's content is the opener. This is anchored on the
    assistant content rather than on the first render being a prefix of the second, because
    Qwen3's template drops an earlier assistant turn's empty think block once a later user turn
    exists, so the prefix property fails there.
    """
    def render(messages):
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=False)

    exchange = [{"role": "user", "content": _USER_MARK},
                {"role": "assistant", "content": _ASSISTANT_MARK}]
    try:
        short = render(exchange)
        full = render(exchange + [{"role": "user", "content": _HEADER_MARK}])
    except Exception:                                     # no template, or one that rejects it
        return None
    before_mark, sep, _ = full.partition(_HEADER_MARK)
    if not sep or _ASSISTANT_MARK not in short or _ASSISTANT_MARK not in before_mark:
        return None
    close = short[short.rfind(_ASSISTANT_MARK) + 1:]
    after_assistant = before_mark[before_mark.rfind(_ASSISTANT_MARK) + 1:]
    if not after_assistant.startswith(close):
        return None
    return after_assistant[len(close):] or None


def boundary_markers(tokenizer, seed_prefix: str) -> tuple[str, ...]:
    """Where a raw continuation ends: the document boundary, EOS, and any chat-turn opening."""
    markers = [seed_prefix]
    for token in (getattr(tokenizer, "eos_token", None),):
        if token and token not in markers:
            markers.append(token)
    header = chat_user_header(tokenizer)
    if header:
        # The opener's leading special token, e.g. "<|im_start|>", not the whole header: cut at
        # the role word or the first newline, whichever comes first.
        cut = min((i for i in (header.find("user"), header.find("\n")) if i != -1),
                  default=len(header))
        candidate = header[:cut].strip() or header.strip()
        if candidate and candidate not in markers:
            markers.append(candidate)
    return tuple(markers)


def clean_raw(text: str, markers: Iterable[str]) -> str:
    """Truncate a raw continuation at the first boundary marker."""
    cut = len(text)
    for marker in markers:
        index = text.find(marker)
        if index != -1:
            cut = min(cut, index)
    return text[:cut].strip()


def drop_burn_in(text: str, tokenizer, n_tokens: int) -> str:
    """Discard the first ``n_tokens`` of a sample so the record does not depend on the prefix.

    An empirical decorrelation, not a mixing guarantee; the recorded frame uses 0.
    """
    if n_tokens <= 0:
        return text
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= n_tokens:
        return ""
    return tokenizer.decode(ids[n_tokens:])


def passes_filters(text: str, *, min_chars: int, max_repeat_ratio: float) -> bool:
    """Length floor plus a degeneracy filter: the repeated fraction of 8-grams, 1 - distinct/total."""
    if len(text) < min_chars:
        return False
    tokens = text.split()
    if len(tokens) < 8:
        return False
    grams = Counter(tuple(tokens[i:i + 8]) for i in range(len(tokens) - 7))
    total = sum(grams.values())
    return (1.0 - len(grams) / total) <= max_repeat_ratio


def generate_texts(model, tokenizer, prompts: list[str], *, max_new_tokens: int,
                   temperature: float, top_p: float, stop_token_ids: list[int], seed: int,
                   batch_index: int = 0) -> list[str]:
    """Sample one continuation per prompt; returns the NEW tokens, special tokens kept."""
    torch.manual_seed(seed * 1_000_003 + batch_index)
    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
    # transformers may log "right-padding was detected": Qwen3's pad token is <|endoftext|>, the
    # seed prefix itself, so a one-token prompt ends in the pad id. The attention mask is correct.
    with torch.no_grad():
        out = model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, top_k=0, min_p=0.0, repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_token_ids,
        )
    new = out[:, encoded["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new, skip_special_tokens=False)


def checkpoint_sha256(model_id: str) -> str:
    """One hash over every ``*.safetensors`` file of the checkpoint, in sorted order.

    A Hub id is resolved to its local cache snapshot (``local_files_only=True``): resolved online,
    it stays a bare id, which is no directory, and the hash would be that of empty input.

    Raises:
        ValueError: when no ``*.safetensors`` file is found under the resolved directory.
    """
    root = Path(resolve_model_path(model_id, local_files_only=True))
    files = sorted(root.rglob("*.safetensors"))
    if not files:
        raise ValueError(f"No *.safetensors files under {root} (resolved from {model_id!r}).")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def sha256_text(parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()
