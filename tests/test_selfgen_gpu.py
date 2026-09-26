"""Self-generation on the card: Qwen3-0.6B writes, the package fits, trains and chains.

Sized for an 8 GB card (RTX 2070, 2026-09-26). Run with::

    LFA_SKIP_TOOLCHAIN_CHECK=1 HF_HUB_OFFLINE=1 pytest tests/test_selfgen_gpu.py -m gpu -q
"""
from __future__ import annotations

import pytest
import torch

from lfa.selfgen.generate import (boundary_markers, chat_user_header, clean_raw, generate_texts,
                                  load_writer, pick_seed_prefix)

pytestmark = pytest.mark.gpu

MODEL = "Qwen/Qwen3-0.6B"
DEVICE = "cuda:0"


@pytest.fixture(scope="module")
def writer():
    model, tokenizer = load_writer(MODEL, DEVICE)
    yield model, tokenizer
    del model
    torch.cuda.empty_cache()


def test_qwen3_writes_from_its_document_boundary_and_stops_at_the_next(writer):
    model, tokenizer = writer
    prefix = pick_seed_prefix(tokenizer, model)
    assert prefix == "<|endoftext|>"
    assert chat_user_header(tokenizer) == "<|im_start|>user\n"
    markers = boundary_markers(tokenizer, prefix)
    stop = [tokenizer.convert_tokens_to_ids(t) for t in ("<|endoftext|>", "<|im_start|>")]

    texts = generate_texts(model, tokenizer, [prefix] * 4, max_new_tokens=64, temperature=1.0,
                           top_p=1.0, stop_token_ids=stop, seed=42)
    again = generate_texts(model, tokenizer, [prefix] * 4, max_new_tokens=64, temperature=1.0,
                           top_p=1.0, stop_token_ids=stop, seed=42)

    assert len(texts) == 4 and all(clean_raw(t, markers) for t in texts)
    assert texts == again                                  # seeded per batch
