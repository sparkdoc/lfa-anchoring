"""The pieces of the sampling loop that need no model: prefix precedence, boundaries, filters."""
from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding

from lfa.selfgen.generate import (boundary_markers, chat_user_header, checkpoint_sha256, clean_raw,
                                  drop_burn_in, generate_texts, passes_filters, pick_seed_prefix,
                                  sha256_text)


class _Tok:
    """A stub tokenizer: decode by id, optional bos/eos, optional ChatML template and preamble."""

    def __init__(self, bos=None, eos=None, template=True, preamble=""):
        self.bos_token, self.eos_token = bos, eos
        self._template, self._preamble = template, preamble

    def decode(self, ids, **_):
        return {7: "<|endoftext|>", 8: "<s>"}.get(ids[0], "?")

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
        if not self._template:
            raise ValueError("no chat template")
        turns = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return self._preamble + turns

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


def test_a_template_preamble_is_not_mistaken_for_the_user_turn_opener():
    tok = _Tok(eos="<|im_end|>", preamble="<|im_start|>system\nYou are helpful<|im_end|>\n")
    assert chat_user_header(tok) == "<|im_start|>user\n"
    assert boundary_markers(tok, "<|endoftext|>") == ("<|endoftext|>", "<|im_end|>", "<|im_start|>")


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


def test_generate_texts_passes_every_truncation_knob_and_decodes_only_new_tokens():
    class Model:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            self.kwargs = kwargs
            new = torch.tensor([[50, 51, 52], [60, 61, 62]])
            return torch.cat([kwargs["input_ids"], new], dim=1)

    class Tok:
        pad_token_id = 0

        def __call__(self, prompts, **_):
            ids = torch.tensor([[0, 5], [6, 7]])
            # A BERT-style tokenizer also emits token_type_ids, which `generate` refuses.
            return BatchEncoding({"input_ids": ids, "attention_mask": (ids != 0).long(),
                                  "token_type_ids": torch.zeros_like(ids)})

        def batch_decode(self, ids, skip_special_tokens):
            self.decoded, self.skip = ids.tolist(), skip_special_tokens
            return [" ".join(map(str, row)) for row in self.decoded]

    model, tok = Model(), Tok()
    out = generate_texts(model, tok, ["a", "b"], max_new_tokens=3, temperature=1.0, top_p=1.0,
                         stop_token_ids=[9, 10], seed=1)
    kw = model.kwargs
    assert kw["top_k"] == 0 and kw["min_p"] == 0.0 and kw["repetition_penalty"] == 1.0
    assert kw["do_sample"] is True and kw["eos_token_id"] == [9, 10]
    assert kw["temperature"] == 1.0 and kw["top_p"] == 1.0 and kw["max_new_tokens"] == 3
    assert "token_type_ids" not in kw and "attention_mask" in kw
    assert tok.decoded == [[50, 51, 52], [60, 61, 62]] and tok.skip is False
    assert out == ["50 51 52", "60 61 62"]


def test_checkpoint_sha256_hashes_every_safetensors_file_and_refuses_an_empty_directory(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"alpha")
    (tmp_path / "b.safetensors").write_bytes(b"beta")
    (tmp_path / "config.json").write_text("{}")
    first = checkpoint_sha256(str(tmp_path))
    assert len(first) == 64 and all(c in "0123456789abcdef" for c in first)
    assert checkpoint_sha256(str(tmp_path)) == first
    (tmp_path / "b.safetensors").write_bytes(b"betb")
    assert checkpoint_sha256(str(tmp_path)) != first

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="empty"):
        checkpoint_sha256(str(empty))


def test_checkpoint_sha256_downloads_a_hub_id_and_hashes_the_snapshot(tmp_path, monkeypatch):
    import huggingface_hub

    snapshot = tmp_path / "snapshot"
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        snapshot.mkdir(exist_ok=True)                  # a cold cache: the files appear on download
        (snapshot / "model-00001.safetensors").write_bytes(b"alpha")
        (snapshot / "model-00002.safetensors").write_bytes(b"beta")
        (snapshot / "config.json").write_text("{}")
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    from_hub = checkpoint_sha256("some-org/not-a-local-path")
    assert calls == [("some-org/not-a-local-path",
                      {"allow_patterns": ["*.safetensors", "*.json"]})]
    assert from_hub == checkpoint_sha256(str(snapshot))   # the hash covers the downloaded files
    assert len(calls) == 1                                # a local directory never downloads

    (snapshot / "model-00002.safetensors").write_bytes(b"betb")
    assert checkpoint_sha256(str(snapshot)) != from_hub
    assert len(calls) == 1
