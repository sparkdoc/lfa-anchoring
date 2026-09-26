"""Self-generation on the card: Qwen3-0.6B writes, the package fits, trains and chains.

Sized for an 8 GB card (RTX 2070, 2026-09-26). Run with::

    LFA_SKIP_TOOLCHAIN_CHECK=1 HF_HUB_OFFLINE=1 pytest tests/test_selfgen_gpu.py -m gpu -q
"""
from __future__ import annotations

import pytest
import torch

from lfa.artifact.build import build_artifact_self_generated
from lfa.artifact.schema import load_artifact
from lfa.sampler import Sampler
from lfa.selfgen.artifact_corpus import SelfGenOptions, write_artifact_corpus
from lfa.selfgen.generate import (boundary_markers, chat_user_header, clean_raw, generate_texts,
                                  load_writer, pick_seed_prefix)
from lfa.workspace import Workspace

pytestmark = pytest.mark.gpu

MODEL = "Qwen/Qwen3-0.6B"
DEVICE = "cuda:0"

SMALL = SelfGenOptions(n_raw=16, n_chat=4, max_new_tokens=128, batch_size=8, min_docs=10,
                       max_samples=20_000, gmm_k=4, layer_group_size=7, reservoir_size=5_000,
                       device=DEVICE)


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

    # Unfiltered draws at T=1.0 can be whitespace only (Task 4 counts them as "empty"), so a
    # few empties are allowed; at least half the prompts must still yield text.
    assert len(texts) == 4 and sum(bool(clean_raw(t, markers)) for t in texts) >= 2
    assert texts == again                                  # seeded per batch


@pytest.fixture(scope="module")
def small_corpus(tmp_path_factory, writer):
    out = tmp_path_factory.mktemp("selfgen") / "corpus.jsonl"
    manifest = write_artifact_corpus(MODEL, out, SMALL, writer=writer)
    return out, manifest


def test_the_artifact_corpus_has_both_shares_and_a_manifest(small_corpus):
    import json
    out, manifest = small_corpus
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert manifest["counts"]["raw"] == 16 and manifest["counts"]["chat"] == 4
    assert manifest["seed_prefix"] == "<|endoftext|>"
    assert all(r["text"].startswith("<|im_start|>user\n") for r in rows
               if r["source"] == "selfgen_chatfmt")
    assert not any("<|endoftext|>" in r["text"] for r in rows)
    assert len(manifest["writer_sha256"]) == 64


@pytest.fixture(scope="module")
def small_artifact(tmp_path_factory, writer):
    out = tmp_path_factory.mktemp("selfgen_art") / "art.pt"
    build_artifact_self_generated(MODEL, out, SMALL, writer=writer)
    return out


def test_a_self_generated_artifact_fits_carries_provenance_and_samples(small_artifact):
    params = load_artifact(small_artifact)
    meta = params["__meta__"]
    assert meta["provenance"] == "self-generated" and len(meta["corpus_sha256"]) == 64
    assert meta["model_id"] == MODEL and meta["num_layers"] == 28
    assert small_artifact.with_suffix(".corpus.jsonl").is_file()

    sampler = Sampler(small_artifact, device=DEVICE, seed=0)
    draw = sampler.sample_best(5, "pre_mlp", 8)
    assert draw is not None and draw.shape == (8, 1024)


def test_init_self_generated_on_qwen3_records_the_corpus_hash(tmp_path):
    ws = Workspace.init(tmp_path / "ws", MODEL, artifact="self-generated", selfgen=SMALL)
    assert ws.state["artifact_id"].startswith("self-generated:")
    assert (tmp_path / "ws" / "artifacts" / "v1.corpus.jsonl.manifest.json").is_file()
    assert ws._artifact_meta()["provenance"] == "self-generated"
