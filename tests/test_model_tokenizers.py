"""What the self-generation and supplement code reads off each bundled model's tokenizer.

    pytest tests/test_model_tokenizers.py -m slow -q      # downloads tokenizer + config files
"""
import types

import pytest
from transformers import AutoTokenizer, GenerationConfig

from lfa.selfgen.generate import (boundary_markers, chat_turn_end, chat_user_header,
                                  pick_seed_prefix)

pytestmark = pytest.mark.slow
MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]


@pytest.fixture(scope="module", params=MODELS)
def loaded(request):
    tok = AutoTokenizer.from_pretrained(request.param)
    model = types.SimpleNamespace(generation_config=GenerationConfig.from_pretrained(request.param))
    return request.param, tok, model


def test_the_document_start_token_is_endoftext(loaded):
    _, tok, model = loaded
    prefix = pick_seed_prefix(tok, model)
    assert prefix == "<|endoftext|>"
    assert len(tok(prefix, add_special_tokens=False)["input_ids"]) == 1


def test_the_user_opener_is_the_chatml_user_turn(loaded):
    _, tok, _ = loaded
    assert chat_user_header(tok) == "<|im_start|>user\n"


def test_the_turn_end_is_im_end(loaded):
    _, tok, _ = loaded
    assert chat_turn_end(tok) == "<|im_end|>"


def test_the_boundary_markers_hold_the_document_and_turn_markers(loaded):
    _, tok, model = loaded
    markers = boundary_markers(tok, pick_seed_prefix(tok, model))
    assert "<|endoftext|>" in markers and "<|im_start|>" in markers
