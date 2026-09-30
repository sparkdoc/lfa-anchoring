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
    # The recorded corpus held no chat-format documents (the research run's audit); the share is
    # an option.
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


from lfa.selfgen.artifact_corpus import (CorpusFrameMismatch, frame_sha256, partial_path,
                                         progress_path)


def _deterministic(p, i, *, seed=None, batch_index=None):
    return f"doc seed {seed} batch {batch_index} item {i} with some words<|endoftext|>"


def _gen_with_kwargs(fail_on_call=None):
    """Deterministic in (seed, batch_index, position); raises KeyboardInterrupt on one call."""
    calls = []

    def generate(model, tokenizer, prompts, **kwargs):
        calls.append(kwargs["batch_index"])
        if fail_on_call is not None and len(calls) == fail_on_call:
            raise KeyboardInterrupt
        return [_deterministic(p, i, seed=kwargs["seed"], batch_index=kwargs["batch_index"])
                for i, p in enumerate(prompts)]
    generate.calls = calls
    return generate


def _options(**kw):
    base = dict(n_raw=10, n_chat=0, batch_size=3, min_docs=1)
    base.update(kw)
    return SelfGenOptions(**base)


def test_an_interrupted_build_resumes_to_the_same_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    whole = tmp_path / "whole" / "corpus.jsonl"
    write_artifact_corpus("stub", whole, _options(), generate=_gen_with_kwargs(),
                          writer=(_Model(), _Tok()))

    resumed = tmp_path / "resumed" / "corpus.jsonl"
    with pytest.raises(KeyboardInterrupt):
        write_artifact_corpus("stub", resumed, _options(), generate=_gen_with_kwargs(3),
                              writer=(_Model(), _Tok()))
    assert partial_path(resumed).is_file() and progress_path(resumed).is_file()
    assert not resumed.exists()                         # nothing final until it is complete
    second = _gen_with_kwargs()
    manifest = write_artifact_corpus("stub", resumed, _options(batch_size=5), generate=second,
                                     writer=(_Model(), _Tok()))

    assert second.calls[0] == 2                         # batches 0 and 1 were kept, not redone
    assert resumed.read_text() == whole.read_text()     # drawn at the recorded batch size, 3
    assert manifest["frame"]["batch_size"] == 3         # and the manifest says so
    assert manifest["corpus_sha256"] == json.loads(
        (whole.parent / "corpus.jsonl.manifest.json").read_text())["corpus_sha256"]
    assert not partial_path(resumed).exists() and not progress_path(resumed).exists()


def test_rows_appended_after_the_last_progress_write_are_dropped_on_resume(tmp_path,
                                                                            monkeypatch):
    """A crash between the append and the progress write leaves extra lines; they go."""
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    whole = tmp_path / "whole" / "corpus.jsonl"
    write_artifact_corpus("stub", whole, _options(), generate=_gen_with_kwargs(),
                          writer=(_Model(), _Tok()))
    out = tmp_path / "torn" / "corpus.jsonl"
    with pytest.raises(KeyboardInterrupt):
        write_artifact_corpus("stub", out, _options(), generate=_gen_with_kwargs(3),
                              writer=(_Model(), _Tok()))
    with open(partial_path(out), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": "torn row", "source": "selfgen_raw"}) + "\n")
    write_artifact_corpus("stub", out, _options(), generate=_gen_with_kwargs(),
                          writer=(_Model(), _Tok()))
    assert out.read_text() == whole.read_text()


def test_empties_carry_across_a_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    calls, interrupt = [], {"on": True}

    def half_empty(model, tokenizer, prompts, **kwargs):
        calls.append(kwargs["batch_index"])
        if interrupt["on"] and len(calls) == 3:
            raise KeyboardInterrupt
        # kept texts need 8 words to pass the filters' floor; a one-prompt batch is not emptied,
        # or the tail of the share (batches of one) could never finish
        return ["<|endoftext|>" if i == 0 and len(prompts) > 1
                else f"text {kwargs['batch_index']} {i} ok with enough words to pass<|endoftext|>"
                for i in range(len(prompts))]
    out = tmp_path / "corpus.jsonl"
    with pytest.raises(KeyboardInterrupt):
        write_artifact_corpus("stub", out, _options(max_empty_fraction=1.0), generate=half_empty,
                              writer=(_Model(), _Tok()))
    empties_before = json.loads(progress_path(out).read_text())["shares"]["selfgen_raw"]["empties"]
    assert empties_before == 2
    interrupt["on"] = False
    manifest = write_artifact_corpus("stub", out, _options(max_empty_fraction=1.0),
                                     generate=half_empty, writer=(_Model(), _Tok()))
    # 2 before the interruption + 3 in the resumed tail (batches of 3, 3, 2 each empty one; the
    # last batch of one is kept)
    assert manifest["counts"]["empty"] == 5

    # The limit applies to the whole build: at 40% of 10 (4 empties) the resumed tail alone (3)
    # would pass, but with the 2 drawn before the interruption the build is refused.
    calls.clear()
    interrupt["on"] = True
    limited = tmp_path / "limited" / "corpus.jsonl"
    with pytest.raises(KeyboardInterrupt):
        write_artifact_corpus("stub", limited, _options(max_empty_fraction=0.4),
                              generate=half_empty, writer=(_Model(), _Tok()))
    interrupt["on"] = False
    with pytest.raises(DegenerateCorpus, match="empty"):
        write_artifact_corpus("stub", limited, _options(max_empty_fraction=0.4),
                              generate=half_empty, writer=(_Model(), _Tok()))
    assert not limited.exists() and partial_path(limited).is_file()


def test_a_changed_frame_refuses_to_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    out = tmp_path / "corpus.jsonl"
    with pytest.raises(KeyboardInterrupt):
        write_artifact_corpus("stub", out, _options(), generate=_gen_with_kwargs(2),
                              writer=(_Model(), _Tok()))
    with pytest.raises(CorpusFrameMismatch, match="rebuild"):
        write_artifact_corpus("stub", out, _options(n_raw=11), generate=_gen_with_kwargs(),
                              writer=(_Model(), _Tok()))


def test_a_complete_corpus_is_reused_without_generating(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    out = tmp_path / "corpus.jsonl"
    first = write_artifact_corpus("stub", out, _options(), generate=_gen_with_kwargs(),
                                  writer=(_Model(), _Tok()))
    unused = _gen_with_kwargs()
    again = write_artifact_corpus("stub", out, _options(batch_size=5), generate=unused,
                                  writer=(_Model(), _Tok()))
    assert unused.calls == [] and again == first     # batch size does not change the frame


def test_a_complete_corpus_from_another_writer_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    out = tmp_path / "corpus.jsonl"
    write_artifact_corpus("stub", out, _options(), generate=_gen_with_kwargs(),
                          writer=(_Model(), _Tok()))
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "d" * 64)
    with pytest.raises(CorpusFrameMismatch, match="writer"):
        write_artifact_corpus("stub", out, _options(), generate=_gen_with_kwargs(),
                              writer=(_Model(), _Tok()))


def test_progress_is_logged_per_batch(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    with caplog.at_level("INFO"):
        write_artifact_corpus("stub", tmp_path / "c.jsonl", _options(), generate=_gen_with_kwargs(),
                              writer=(_Model(), _Tok()))
    lines = [r.message for r in caplog.records if "self-generated corpus:" in r.message]
    assert lines[0].startswith("self-generated corpus: 3/10 documents")
    assert lines[-1].startswith("self-generated corpus: 10/10 documents")


def test_the_frames_and_their_hash():
    o = SelfGenOptions()
    assert o.corpus_frame() == {"n_raw": 2500, "n_chat": 0, "max_new_tokens": 2048, "seed": 42,
                                "chat_seed": 43, "min_chars": 1, "max_repeat_ratio": 1.0,
                                "burn_in_tokens": 0}
    assert o.artifact_frame() == {**o.corpus_frame(), "max_samples": 600_000, "gmm_k": 32,
                                  "pca_variance": 0.95, "reservoir_size": 200_000}
    assert frame_sha256(o) == frame_sha256(SelfGenOptions(batch_size=8, device="cpu"))
    assert frame_sha256(o) != frame_sha256(SelfGenOptions(n_raw=60))


def test_a_build_stopped_between_its_manifest_and_the_rename_finishes_on_rerun(tmp_path,
                                                                             monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    whole = tmp_path / "whole" / "corpus.jsonl"
    write_artifact_corpus("stub", whole, _options(), generate=_gen_with_kwargs(),
                          writer=(_Model(), _Tok()))
    out = tmp_path / "stopped" / "corpus.jsonl"
    import os as _os
    real_replace = _os.replace

    def replace(src, dst):
        if str(dst) == str(out):
            raise KeyboardInterrupt
        real_replace(src, dst)
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.os.replace", replace)
    with pytest.raises(KeyboardInterrupt):
        write_artifact_corpus("stub", out, _options(), generate=_gen_with_kwargs(),
                              writer=(_Model(), _Tok()))
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.os.replace", real_replace)
    assert not out.exists() and partial_path(out).is_file()
    unused = _gen_with_kwargs()
    write_artifact_corpus("stub", out, _options(), generate=unused, writer=(_Model(), _Tok()))
    assert unused.calls == [] and out.read_text() == whole.read_text()
    assert not partial_path(out).exists() and not progress_path(out).exists()
