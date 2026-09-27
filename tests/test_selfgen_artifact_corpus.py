"""The artifact corpus writer, with the model replaced by an injected generator."""
import json

import pytest

from lfa.selfgen.artifact_corpus import DegenerateCorpus, SelfGenOptions, write_artifact_corpus


class _Tok:
    bos_token = None
    eos_token = "<|endoftext|>"
    pad_token_id = 0

    def decode(self, ids, **_):
        return "<|endoftext|>"

    def apply_chat_template(self, messages, **_):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)

    def convert_tokens_to_ids(self, token):
        return {"<|endoftext|>": 1, "<|im_start|>": 2}.get(token, 3)

    def __call__(self, text, **_):
        return {"input_ids": [0] * len(text.split())}


class _Model:
    device = "cpu"
    generation_config = None


def _gen_factory(script):
    """A stand-in for ``generate_texts``: like it, returns only the NEW text, never the prompt."""
    calls = []

    def generate(model, tokenizer, prompts, **kwargs):
        calls.append((list(prompts), kwargs))
        return [script(p, i) for i, p in enumerate(prompts)]
    generate.calls = calls
    return generate


def test_writes_raw_and_chat_shares_with_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    gen = _gen_factory(lambda p, i: f"some generated prose number {i} that goes on<|endoftext|>tail")
    options = SelfGenOptions(n_raw=5, n_chat=2, batch_size=4, min_docs=1)

    manifest = write_artifact_corpus("stub", tmp_path / "corpus.jsonl", options,
                                     generate=gen, writer=(_Model(), _Tok()))

    rows = [json.loads(l) for l in (tmp_path / "corpus.jsonl").read_text().splitlines()]
    assert [r["source"] for r in rows].count("selfgen_raw") == 5
    assert [r["source"] for r in rows].count("selfgen_chatfmt") == 2
    raw = [r for r in rows if r["source"] == "selfgen_raw"][0]["text"]
    assert raw.startswith("some generated") and "<|endoftext|>" not in raw    # no prefix, cut
    chat = [r for r in rows if r["source"] == "selfgen_chatfmt"][0]["text"]
    assert chat.startswith("<|im_start|>user\nsome generated")                 # header KEPT
    assert manifest["counts"] == {"raw": 5, "chat": 2, "empty": 0}
    assert manifest["frame"]["n_raw"] == 5 and manifest["writer_sha256"] == "c" * 64
    assert manifest["corpus_sha256"] and (tmp_path / "corpus.jsonl.manifest.json").is_file()
    # the prompts: the seed prefix for the raw share, the bare user header for the chat share
    assert {p for prompts, _ in gen.calls for p in prompts} == {"<|endoftext|>", "<|im_start|>user\n"}
    # the recorded decoding frame, on every call
    assert all(k["temperature"] == 1.0 and k["top_p"] == 1.0 for _, k in gen.calls)
    # the two shares are seeded apart
    seeds = {k["seed"] for _, k in gen.calls}
    assert seeds == {42, 43}


def test_the_recorded_frame_is_the_default():
    o = SelfGenOptions()
    # The C12 corpus held no chat-format documents (the record's audit); the share is an option.
    assert (o.n_raw, o.n_chat, o.max_new_tokens, o.seed, o.chat_seed) == (2500, 0, 2048, 42, 43)
    assert (o.max_samples, o.gmm_k, o.pca_variance) == (600_000, 32, 0.95)


def test_too_many_empty_documents_is_a_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    gen = _gen_factory(lambda p, i: "<|endoftext|>" if i % 2
                       else f"text {i} here now and it runs on long enough")
    options = SelfGenOptions(n_raw=8, n_chat=0, batch_size=8, min_docs=1, max_empty_fraction=0.2)

    with pytest.raises(DegenerateCorpus, match="empty"):
        write_artifact_corpus("stub", tmp_path / "c.jsonl", options, generate=gen,
                              writer=(_Model(), _Tok()))
    assert not (tmp_path / "c.jsonl").exists()


def test_too_few_documents_is_a_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    gen = _gen_factory(lambda p, i: f"fine text {i} that is long enough to pass the filters")
    options = SelfGenOptions(n_raw=3, n_chat=0, batch_size=3, min_docs=50)

    with pytest.raises(DegenerateCorpus, match="50"):
        write_artifact_corpus("stub", tmp_path / "c.jsonl", options, generate=gen,
                              writer=(_Model(), _Tok()))


def test_empty_draws_within_the_limit_are_replaced_by_further_draws(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    drawn = iter(range(100))                    # the first draw of the whole run is blank
    gen = _gen_factory(lambda p, i: "   " if next(drawn) == 0
                       else f"doc {i} with words enough to be kept here")
    options = SelfGenOptions(n_raw=10, n_chat=0, batch_size=10, min_docs=1, max_empty_fraction=0.2)

    manifest = write_artifact_corpus("stub", tmp_path / "c.jsonl", options, generate=gen,
                                     writer=(_Model(), _Tok()))

    assert manifest["counts"] == {"raw": 10, "chat": 0, "empty": 1}
    assert [k["batch_index"] for _, k in gen.calls] == [0, 1]         # a second, seeded batch


def test_a_writer_that_only_ever_draws_empties_stops_once_the_refusal_is_certain(tmp_path,
                                                                                 monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    gen = _gen_factory(lambda p, i: "<|endoftext|>")
    options = SelfGenOptions(n_raw=20, n_chat=5, batch_size=4, min_docs=1, max_empty_fraction=0.2)

    with pytest.raises(DegenerateCorpus, match="empty"):
        write_artifact_corpus("stub", tmp_path / "c.jsonl", options, generate=gen,
                              writer=(_Model(), _Tok()))
    assert len(gen.calls) == 2                  # 8 empties > 0.2 x 25: no chat share is drawn
