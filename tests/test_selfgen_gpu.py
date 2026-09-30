"""Self-generation on the card: Qwen3-0.6B writes, the package fits, trains and chains.

Sized for an 8 GB card (RTX 2070, 2026-09-26). Run with::

    LFA_SKIP_TOOLCHAIN_CHECK=1 HF_HUB_OFFLINE=1 pytest tests/test_selfgen_gpu.py -m gpu -q
"""
from __future__ import annotations

import pytest
import torch
import yaml

from lfa import Recipe
from lfa.artifact.build import build_artifact_self_generated
from lfa.artifact.schema import load_artifact
from lfa.artifact.store import STORE_ENV
from lfa.sampler import Sampler
from lfa.selfgen.artifact_corpus import SelfGenOptions, write_artifact_corpus
from lfa.selfgen.generate import (boundary_markers, chat_user_header, clean_raw, generate_texts,
                                  load_writer, pick_seed_prefix)
from lfa.selfgen.supplement import SupplementOptions, write_supplement
from lfa.workspace import Workspace

pytestmark = pytest.mark.gpu

MODEL = "Qwen/Qwen3-0.6B"
DEVICE = "cuda:0"

SMALL = SelfGenOptions(n_raw=16, n_chat=4, max_new_tokens=128, batch_size=8, min_docs=10,
                       max_samples=20_000, gmm_k=4, layer_group_size=7, reservoir_size=5_000,
                       device=DEVICE)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path_factory, monkeypatch):
    """Every test builds into a store of its own.

    ``Workspace.init(..., artifact="self-generated")`` goes through the local store, which is
    ``~/.cache/lfa/artifacts`` unless ``LFA_ARTIFACT_STORE`` says otherwise. Left there, a test
    would reuse the entry an earlier test built -- and a real one the user built -- instead of
    building, so the build it claims to check would never run.
    """
    monkeypatch.setenv(STORE_ENV, str(tmp_path_factory.mktemp("store")))


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


DARWIN = """On the Origin of Species was published in 1859. Darwin argued that species change over
time through a process he called natural selection, in which individuals better suited to their
environment leave more offspring.

He drew on his observations of finches in the Galapagos, whose beaks differed from island to
island according to the food available, and on the practice of animal breeders, who select for
traits deliberately.

The book provoked immediate controversy, but by the 1870s most naturalists accepted that
evolution had occurred, even where they doubted that natural selection was its main cause.
""" * 3


@pytest.fixture(scope="module")
def small_supplement(tmp_path_factory, writer):
    out = tmp_path_factory.mktemp("supp") / "supplement.jsonl"
    manifest = write_supplement(MODEL, [DARWIN], out, domain_description="natural history",
                                options=SupplementOptions(passage_chars=800, batch_size=4,
                                                          max_new_tokens=512),
                                corpus_sha256="0" * 64, writer=writer)
    return out, manifest


def test_qwen3_writes_parseable_pairs_from_a_real_passage(small_supplement):
    import json
    out, manifest = small_supplement
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert manifest["n_passages"] >= 2 and manifest["n_pairs"] == len(rows) >= 2
    assert all(len(r["response"]) >= 40 and r["prompt"].strip() for r in rows)
    assert "about a text on natural history" in manifest["template"]


def _small_recipe():
    return Recipe(name="qwen3-small", model_id=MODEL, artifact="self-generated",
                  lora_rank=4, lora_alpha=8, lambda_qkv=1000.0, lambda_mlp=1000.0, mu=0.05,
                  n_anchor_samples=4, epochs=1, checkpoint_mode="none", learning_rate=1e-4,
                  warmup_steps=1, batch_size=1, gradient_accumulation_steps=2,
                  sequence_length=128, seed=42, val_fraction=0.25, supplement_fraction=0.13,
                  calibrated_rank=4)


@pytest.fixture(scope="module")
def darwin_corpus(tmp_path_factory):
    directory = tmp_path_factory.mktemp("domain") / "natural_history"
    directory.mkdir()
    for i in range(4):
        (directory / f"doc_{i}.txt").write_text(DARWIN.replace("1859", str(1859 + i)))
    return directory


def test_a_stage_trains_with_the_self_written_supplement_mixed_in(tmp_path, darwin_corpus):
    ws = Workspace.init(tmp_path / "ws", MODEL, artifact="self-generated", selfgen=SMALL)

    entry = ws.train(darwin_corpus, recipe=_small_recipe(), device=DEVICE)

    supp = entry["supplement"]
    assert supp["n_pairs_available"] >= 2 and supp["n_pairs_used"] >= 1
    assert 0.0 < supp["achieved_fraction"] < 0.5
    assert len(supp["writer_sha256"]) == 64
    assert entry["n_val_docs"] == 1 and entry["n_train_docs"] == 3 + supp["n_pairs_used"]
    assert (tmp_path / "ws" / "supplements").is_dir()


def test_a_two_stage_chain_under_regenerate_refits_v2_from_the_fused_model(tmp_path, darwin_corpus):
    second = tmp_path / "cookery"
    second.mkdir()
    for i in range(4):
        (second / f"recipe_{i}.txt").write_text(
            ("Take a pound of flour and rub in the butter. Add the eggs one at a time, beating "
             "well, and bake in a moderate oven for forty minutes.\n\n") * 12)
    recipe_path = _small_recipe().save(tmp_path / "small.yaml")
    ws = Workspace.init(tmp_path / "ws", MODEL, artifact="self-generated", selfgen=SMALL,
                        recipe=str(recipe_path))                # a chain uses the workspace's recipe
    spec = tmp_path / "domains.yaml"
    spec.write_text(yaml.safe_dump({"artifact": "regenerate", "domains": [
        {"name": "darwin", "corpus": str(darwin_corpus)}, {"name": "cookery", "corpus": str(second)}]}))

    entries = ws.chain(spec, device=DEVICE, selfgen=SMALL)

    assert [e["stage"] for e in entries] == [1, 2]
    assert entries[1]["artifact_route"] == "regenerate" and entries[1]["artifact_version"] == 2
    meta = ws._artifact_meta()
    assert meta["provenance"] == "self-generated"
    assert meta["model_id"] == str(tmp_path / "ws" / "models" / "stage1_fused")
    assert (tmp_path / "ws" / "artifacts" / "v2.corpus.jsonl.manifest.json").is_file()
