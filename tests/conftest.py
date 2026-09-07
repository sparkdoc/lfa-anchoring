import copy

import pytest, torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers

@pytest.fixture(scope="session")
def tiny_model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True)
    model = LlamaForCausalLM(cfg).eval()
    tok = Tokenizer(models.WordLevel({chr(i): i for i in range(256)}, unk_token=chr(0)))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token=chr(0), pad_token=chr(1), eos_token=chr(2))
    return model, tokenizer

@pytest.fixture
def tiny_model_fresh(tiny_model):
    """A private deep copy of the session model, for tests that wrap, mutate, train or move it."""
    model, tokenizer = tiny_model
    return copy.deepcopy(model), tokenizer


@pytest.fixture
def tiny_texts():
    return [f"document number {i} about anchoring functions on sampled hidden states " * (1 + i % 4) for i in range(12)]
