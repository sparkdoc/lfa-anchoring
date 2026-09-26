"""The pieces of the sampling loop that need no model: prefix precedence, boundaries, filters."""
from types import SimpleNamespace

import pytest

from lfa.selfgen.generate import (boundary_markers, chat_user_header, clean_raw, drop_burn_in,
                                  passes_filters, pick_seed_prefix, sha256_text)


class _Tok:
    """A stub tokenizer: decode by id, optional bos/eos, optional chat template."""

    def __init__(self, bos=None, eos=None, template=True):
        self.bos_token, self.eos_token = bos, eos
        self._template = template

    def decode(self, ids, **_):
        return {7: "<|endoftext|>", 8: "<s>"}.get(ids[0], "?")

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
        if not self._template:
            raise ValueError("no chat template")
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def test_seed_prefix_prefers_the_models_declared_start_token():
    model = SimpleNamespace(generation_config=SimpleNamespace(bos_token_id=7))
    assert pick_seed_prefix(_Tok(bos="<s>"), model) == "<|endoftext|>"


def test_seed_prefix_falls_back_to_bos_then_eos_then_newline():
    assert pick_seed_prefix(_Tok(bos="<s>", eos="</s>")) == "<s>"
    assert pick_seed_prefix(_Tok(eos="</s>")) == "</s>"
    assert pick_seed_prefix(_Tok()) == "\n"


def test_chat_user_header_is_what_precedes_the_user_content():
    assert chat_user_header(_Tok()) == "<|im_start|>user\n"


def test_a_tokenizer_without_a_chat_template_gives_no_header():
    assert chat_user_header(_Tok(template=False)) is None


def test_clean_raw_cuts_at_the_first_boundary_marker():
    markers = boundary_markers(_Tok(), "<|endoftext|>")
    assert "<|endoftext|>" in markers and "<|im_start|>" in markers
    assert clean_raw("alpha beta<|im_start|>user\ngamma<|endoftext|>", markers) == "alpha beta"
    assert clean_raw("no marker here", markers) == "no marker here"


def test_drop_burn_in_removes_the_first_n_tokens_and_empties_short_text():
    class T(_Tok):
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}
        def decode(self, ids, **_):
            return " ".join(ids)
    assert drop_burn_in("a b c d", T(), 2) == "c d"
    assert drop_burn_in("a b", T(), 2) == ""
    assert drop_burn_in("a b", T(), 0) == "a b"


def test_passes_filters_rejects_short_and_looping_text():
    prose = "the quick brown fox jumps over the lazy dog and keeps running far away"
    loop = "one two three four five six seven " * 20
    assert passes_filters(prose, min_chars=10, max_repeat_ratio=0.3)
    assert not passes_filters("too short", min_chars=200, max_repeat_ratio=1.0)
    assert not passes_filters(loop, min_chars=10, max_repeat_ratio=0.3)
    assert passes_filters(loop, min_chars=10, max_repeat_ratio=1.0)   # the unfiltered frame


def test_sha256_text_is_order_sensitive_and_stable():
    assert sha256_text(["a", "b"]) == sha256_text(["a", "b"])
    assert sha256_text(["a", "b"]) != sha256_text(["b", "a"])
