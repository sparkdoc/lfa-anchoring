"""Artifact build pipeline: collection hooks, per-site PCA+GMM fit, and the end-to-end build.

Everything here runs on CPU against the 2-layer `tiny_model`, with the sample budget and the
reservoir cut down so the whole file stays inside a couple of seconds.
"""

import copy
import inspect
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from lfa.adapters import get_adapter
from lfa.artifact.build import build_artifact
from lfa.artifact.collect import SiteStats, collect_hidden_states
from lfa.artifact.fit import TorchGMM, fit_site
from lfa.artifact.schema import (
    ArtifactModelMismatch,
    load_artifact,
    validate_against_model,
)
from lfa.losses import anchor_loss
from lfa.sampler import Sampler

EXPECTED_KEYS = {"0_pre_o", "0_pre_mlp", "1_pre_qkv", "1_pre_o", "1_pre_mlp", "2_pre_lm_head"}


@pytest.fixture(scope="module")
def build_texts():
    """40 short documents -- enough tokens for a covariance, small enough to collect in a second."""
    return [f"document {i} " + "anchoring hidden states on sampled vectors " * (2 + i % 4)
            for i in range(40)]


@pytest.fixture(scope="module")
def collected(tiny_model, build_texts):
    model, tokenizer = tiny_model
    stats, freqs = collect_hidden_states(
        model, tokenizer, get_adapter(model), build_texts,
        max_samples=2000, seq_len=128, batch_size=8, reservoir_size=500,
        device="cpu", progress=False,
    )
    return stats, freqs


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, tiny_model):
    """The tiny model and tokenizer saved to disk, so `build_artifact` can load them by path."""
    model, tokenizer = tiny_model
    path = tmp_path_factory.mktemp("tiny_model_dir")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def corpus_file(tmp_path_factory, build_texts):
    path = tmp_path_factory.mktemp("corpus") / "texts.jsonl"
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in build_texts), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def built(tmp_path_factory, model_dir, corpus_file, tiny_model):
    out = tmp_path_factory.mktemp("built") / "distribution_stats.pt"
    build_artifact(str(model_dir), corpus_file, out, max_samples=2000, seq_len=128,
                   pca_variance=0.9, gmm_k=2, quantize=True, device="cpu", seed=0)
    return out


# --------------------------------------------------------------------------- collect

def test_collect_covers_every_site_but_layer_0_pre_qkv(collected):
    stats, _ = collected
    assert set(stats) == EXPECTED_KEYS


def test_collect_accumulates_enough_samples(collected):
    stats, _ = collected
    for key, s in stats.items():
        assert isinstance(s, SiteStats)
        assert s.n >= 1000, key
        assert s.reservoir.shape == (500, 32)


def test_collect_covariance_is_symmetric_and_psd(collected):
    stats, _ = collected
    for key, s in stats.items():
        cov = s.cov
        assert cov.shape == (32, 32) and cov.dtype is torch.float32, key
        assert torch.equal(cov, cov.T), key
        evals = torch.linalg.eigvalsh(cov)
        assert evals.min() >= -1e-6 * evals.max().clamp(min=1.0), key
        assert s.mean.shape == (32,) and s.std.shape == (32,)


def test_collect_excludes_padding_positions(collected):
    """Every counted token contributes exactly one hidden vector at every site."""
    stats, freqs = collected
    assert freqs is not None and freqs.shape[0] >= 256
    assert int(freqs.sum()) == stats["1_pre_mlp"].n


def test_collect_takes_the_width_from_the_hooked_input(build_texts):
    """`pre_o`'s width is num_heads*head_dim, not the model's hidden size."""
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=128, tie_word_embeddings=True)
    model = LlamaForCausalLM(cfg).eval()

    tok = Tokenizer(models.WordLevel({chr(i): i for i in range(64)}, unk_token=chr(0)))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token=chr(0), pad_token=chr(1),
                                        eos_token=chr(2))

    stats, _ = collect_hidden_states(model, tokenizer, get_adapter(model), build_texts[:8],
                                     max_samples=200, seq_len=64, batch_size=4,
                                     reservoir_size=100, device="cpu", progress=False)
    assert stats["0_pre_o"].mean.shape == (64,)
    assert stats["0_pre_mlp"].mean.shape == (32,)


def test_collect_layer_subset_and_no_lm_head(tiny_model, build_texts):
    model, tokenizer = tiny_model
    stats, freqs = collect_hidden_states(model, tokenizer, get_adapter(model), build_texts[:8],
                                         max_samples=500, seq_len=128, batch_size=4,
                                         reservoir_size=100, layers=[1], include_lm_head=False,
                                         token_frequencies=False, device="cpu", progress=False)
    assert set(stats) == {"1_pre_qkv", "1_pre_o", "1_pre_mlp"}
    assert freqs is None


# --------------------------------------------------------------------------- fit

def test_torch_gmm_recovers_blob_means():
    g = torch.Generator().manual_seed(0)
    centres = torch.tensor([[-8.0, 0.0], [8.0, 0.0], [0.0, 9.0]])
    x = torch.cat([c + 0.2 * torch.randn(400, 2, generator=g) for c in centres])
    gmm = TorchGMM(n_components=3, covariance_type="diag", random_state=0, device="cpu").fit(x)
    assert gmm.weights_.shape == (3,) and torch.allclose(gmm.weights_.sum(), torch.tensor(1.0))
    found = gmm.means_
    for c in centres:
        assert (found - c).norm(dim=1).min() < 0.1


def test_fit_site_entry_matches_the_shipped_conventions(collected):
    stats, _ = collected
    entry = fit_site(stats["1_pre_mlp"], pca_variance=0.9, gmm_k=2, seed=0, device="cpu")

    n = entry["pca_n_components"]
    assert entry["pca_components"].shape == (32, n)
    assert entry["pca_eigenvalues"].shape == (n,)
    assert entry["mean"].dtype is torch.float32 and entry["std"].dtype is torch.float32
    assert entry["pca_components"].dtype is torch.float16
    assert entry["pca_eigenvalues"].dtype is torch.float32

    basis = entry["pca_components"].float()
    assert torch.allclose(basis.T @ basis, torch.eye(n), atol=2e-3)
    assert (entry["pca_eigenvalues"][:-1] >= entry["pca_eigenvalues"][1:]).all()

    assert entry["gmm_n_components"] == 2
    assert entry["gmm_covariance_type"] == "diag"
    assert "gmm_whitened" not in entry                      # fitted on un-whitened PCA coords
    assert torch.allclose(entry["gmm_weights"].sum(), torch.tensor(1.0), atol=1e-5)
    assert entry["gmm_means"].shape == (2, n)
    assert entry["gmm_covariances"].shape == (2, n)
    assert (entry["gmm_covariances"] > 0).all()
    assert entry["n_samples"] == stats["1_pre_mlp"].n

    p10, p50, p90 = (entry[f"gmm_log_likelihood_p{p}"] for p in (10, 50, 90))
    assert isinstance(p50, float) and p10 <= p50 <= p90


def test_fit_site_pca_variance_threshold_controls_width(collected):
    stats, _ = collected
    narrow = fit_site(stats["1_pre_o"], pca_variance=0.5, gmm_k=2, seed=0, device="cpu")
    wide = fit_site(stats["1_pre_o"], pca_variance=0.99, gmm_k=2, seed=0, device="cpu")
    assert 1 <= narrow["pca_n_components"] < wide["pca_n_components"] <= 32


def test_fit_site_is_deterministic(collected):
    stats, _ = collected
    a = fit_site(stats["0_pre_mlp"], pca_variance=0.9, gmm_k=2, seed=3, device="cpu")
    b = fit_site(stats["0_pre_mlp"], pca_variance=0.9, gmm_k=2, seed=3, device="cpu")
    assert torch.equal(a["gmm_means"], b["gmm_means"])
    assert torch.equal(a["gmm_weights"], b["gmm_weights"])


# --------------------------------------------------------------------------- build

def test_build_artifact_round_trips(built, tiny_model):
    model, _ = tiny_model
    params = load_artifact(built)

    assert {k for k in params if k[0].isdigit()} == EXPECTED_KEYS
    validate_against_model(params, model, get_adapter(model))

    meta = params["__meta__"]
    assert meta["num_layers"] == 2 and meta["hidden_size"] == 32
    assert meta["sites"] == ["pre_qkv", "pre_o", "pre_mlp", "pre_lm_head"]
    site_counts = [params[k]["n_samples"] for k in EXPECTED_KEYS]
    assert meta["n_samples_total"] == max(site_counts)      # per site, not a cross-site sum
    assert meta["n_samples_total"] < sum(site_counts)        # ... and six sites saw it each

    lookup = params["embedding_lookup"]
    assert set(lookup) == {"token_frequencies"}               # the table is rebuilt at load
    assert torch.allclose(lookup["token_frequencies"].sum(), torch.tensor(1.0), atol=1e-5)


def test_build_artifact_samples(built):
    sampler = Sampler(built, device="cpu", seed=0)
    assert sampler.sample_best(1, "pre_mlp", 4).shape == (4, 32)
    assert sampler.sample_best(2, "pre_lm_head", 4).shape == (4, 32)


def test_build_artifact_layer_groups_cover_the_same_sites(
        tmp_path, model_dir, corpus_file, tiny_model):
    model, _ = tiny_model
    out = tmp_path / "grouped.pt"
    build_artifact(str(model_dir), corpus_file, out, max_samples=1000, seq_len=128,
                   pca_variance=0.9, gmm_k=2, layer_group_size=1, quantize=False,
                   device="cpu", seed=0)
    params = load_artifact(out)
    assert {k for k in params if k[0].isdigit()} == EXPECTED_KEYS
    validate_against_model(params, model, get_adapter(model))


def test_build_artifact_meta_records_the_layer_group_size(tmp_path, model_dir, corpus_file,
                                                          built):
    """The value used is recorded for the record (it changes the memory bill, not the fit)."""
    assert load_artifact(built)["__meta__"]["layer_group_size"] is None
    out = tmp_path / "grouped.pt"
    build_artifact(str(model_dir), corpus_file, out, max_samples=1000, seq_len=128,
                   pca_variance=0.9, gmm_k=2, layer_group_size=1, quantize=False,
                   device="cpu", seed=0)
    assert load_artifact(out)["__meta__"]["layer_group_size"] == 1


# ------------------------------------------------------- meta / model agreement

def test_validate_rejects_meta_that_contradicts_the_sites(built, tiny_model):
    model, _ = tiny_model
    params = load_artifact(built)

    wrong_width = copy.deepcopy(params)
    wrong_width["__meta__"]["hidden_size"] = 64
    with pytest.raises(ArtifactModelMismatch, match="hidden size"):
        validate_against_model(wrong_width, model, get_adapter(model))

    wrong_depth = copy.deepcopy(params)
    wrong_depth["__meta__"]["num_layers"] = 7
    with pytest.raises(ArtifactModelMismatch, match="layers"):
        validate_against_model(wrong_depth, model, get_adapter(model))


def test_validate_accepts_a_site_wider_than_the_hidden_size(tiny_model):
    """`pre_o` is num_heads*head_dim wide; validation must read the width off the site module."""
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=128, tie_word_embeddings=True)
    model = LlamaForCausalLM(cfg).eval()
    params = {
        "0_pre_o": {"mean": torch.zeros(64), "std": torch.ones(64)},
        "0_pre_mlp": {"mean": torch.zeros(32), "std": torch.ones(32)},
    }
    validate_against_model(params, model, get_adapter(model))

    params["0_pre_o"]["mean"] = torch.zeros(32)
    with pytest.raises(ArtifactModelMismatch, match="hidden size|width"):
        validate_against_model(params, model, get_adapter(model))


def test_collected_stats_match_an_independent_forward(tiny_model):
    """`pre_lm_head` is the final-norm output, which the model also reports as its last hidden
    state -- so the collected mean must equal that, over non-padding positions only."""
    model, tokenizer = tiny_model
    texts = ["short one", "a considerably longer document, padded against the first"]

    stats, freqs = collect_hidden_states(model, tokenizer, get_adapter(model), texts,
                                         max_samples=10_000, seq_len=128, batch_size=2,
                                         reservoir_size=10_000, device="cpu", progress=False)

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        hidden = model(**inputs, output_hidden_states=True).hidden_states[-1]
    reference = hidden[inputs["attention_mask"].bool()]

    site = stats["2_pre_lm_head"]
    assert site.n == reference.shape[0] < inputs["input_ids"].numel()   # padding really dropped
    assert torch.allclose(site.mean, reference.mean(0), atol=1e-5)
    assert torch.allclose(site.reservoir, reference, atol=1e-5)
    assert int(freqs.sum()) == reference.shape[0]


def test_built_gmm_head_reproduces_the_site_distribution(built):
    """The head is fitted on un-whitened PCA coordinates; sampling it must land back on the
    site's own first two moments (a whitened fit, or a basis mismatch, would not)."""
    params = load_artifact(built)
    entry = params["1_pre_mlp"]
    x = Sampler(built, device="cpu", seed=5).sample_gmm(1, "pre_mlp", 4000)
    assert ((x.mean(0) - entry["mean"]).abs() < 0.25 * entry["std"]).all()
    ratio = x.std(0) / entry["std"]
    assert 0.85 < float(ratio.mean()) < 1.15


def test_built_artifact_anchors_once_the_lookup_is_rebuilt(built, tiny_model):
    """Layer-0 pre_qkv is unavailable until the table is rebuilt from the teacher, and available
    -- not raising -- after; `anchor_loss` must run over a freshly built artifact either way."""
    model, _ = tiny_model
    adapter = get_adapter(model)

    sampler = Sampler(built, device="cpu", seed=0)
    assert sampler.has_embedding_lookup() is False
    assert sampler.sample_best(0, "pre_qkv", 4) is None

    sampler.build_embedding_lookup_from_model(model, adapter)
    assert sampler.sample_best(0, "pre_qkv", 4).shape == (4, 32)

    student = copy.deepcopy(model)
    with torch.no_grad():
        student.model.layers[1].mlp.down_proj.weight.add_(0.05)
    out = anchor_loss(model, student, sampler, adapter, n_samples=4)
    assert out["total"].item() > 0 and torch.isfinite(out["total"])


def test_build_defaults_to_an_fp16_reservoir(built):
    """fp32 reservoirs are ~92 GB for Qwen3-0.6B in one group; fp16 is the shipped default."""
    assert inspect.signature(build_artifact).parameters["dtype"].default is torch.float16


def test_build_honours_reservoir_size(tmp_path, model_dir, corpus_file):
    """A reservoir of 3 caps the mixture at 3 components, whatever `gmm_k` asks for."""
    out = tmp_path / "small_reservoir.pt"
    build_artifact(str(model_dir), corpus_file, out, max_samples=1000, seq_len=128,
                   reservoir_size=3, pca_variance=0.9, gmm_k=8, quantize=False,
                   device="cpu", seed=0)
    params = load_artifact(out)
    assert all(params[k]["gmm_n_components"] == 3 for k in EXPECTED_KEYS)


def test_collect_reservoir_is_seeded(tiny_model, build_texts):
    model, tokenizer = tiny_model
    adapter = get_adapter(model)

    def run(seed):
        stats, _ = collect_hidden_states(model, tokenizer, adapter, build_texts,
                                         max_samples=10_000, seq_len=128, batch_size=8,
                                         reservoir_size=64, device="cpu", progress=False,
                                         token_frequencies=False, seed=seed)
        return stats["1_pre_mlp"].reservoir

    assert torch.equal(run(11), run(11))
    assert not torch.equal(run(11), run(12))


def test_build_artifact_self_generated_writes_the_corpus_then_fits_with_provenance(tmp_path,
                                                                                  monkeypatch):
    """The corpus lands beside the artifact, and the meta names it."""
    import lfa.artifact.build as build_module
    from lfa.artifact.build import build_artifact_self_generated
    from lfa.selfgen.artifact_corpus import SelfGenOptions

    seen = {}

    def fake_corpus(model_id, out_path, options, *, generate, writer):
        Path(out_path).write_text('{"text": "generated"}\n')
        return {"corpus_sha256": "d" * 64}

    def fake_build(model_id, corpus_path, out_path, **kwargs):
        seen.update(kwargs, corpus=str(corpus_path))
        Path(out_path).write_bytes(b"pt")
        return Path(out_path)

    monkeypatch.setattr(build_module, "write_artifact_corpus", fake_corpus)
    monkeypatch.setattr(build_module, "build_artifact", fake_build)
    # No layer_group_size given: the group is chosen from the model's config and host RAM.
    monkeypatch.setattr(build_module, "_auto_layer_group_size",
                        lambda model_id, reservoir_size, itemsize: 5)

    out = build_artifact_self_generated("m", tmp_path / "art.pt",
                                        SelfGenOptions(max_samples=123, gmm_k=4))

    assert out == tmp_path / "art.pt"
    assert seen["corpus"] == str(tmp_path / "art.corpus.jsonl")
    assert seen["max_samples"] == 123 and seen["gmm_k"] == 4
    assert seen["provenance"] == "self-generated" and seen["corpus_sha256"] == "d" * 64
    assert seen["layer_group_size"] == 5
    assert seen["selfgen_frame"] == SelfGenOptions(max_samples=123, gmm_k=4).artifact_frame()


def test_build_artifact_self_generated_keeps_a_given_layer_group_size(tmp_path, monkeypatch):
    import lfa.artifact.build as build_module
    from lfa.artifact.build import build_artifact_self_generated
    from lfa.selfgen.artifact_corpus import SelfGenOptions

    seen = {}

    def never(*args, **kwargs):
        raise AssertionError("a given layer_group_size must not be re-chosen")

    monkeypatch.setattr(build_module, "write_artifact_corpus",
                        lambda *a, **k: {"corpus_sha256": "d" * 64})
    monkeypatch.setattr(build_module, "build_artifact",
                        lambda model_id, corpus_path, out_path, **kwargs: seen.update(kwargs))
    monkeypatch.setattr(build_module, "_auto_layer_group_size", never)

    build_artifact_self_generated("m", tmp_path / "art.pt", SelfGenOptions(layer_group_size=3))
    assert seen["layer_group_size"] == 3


QWEN3_WIDTHS = dict(hidden_size=1024, pre_o_width=2048, num_layers=28, reservoir_size=200_000,
                    itemsize=2)


@pytest.mark.parametrize("available, expected", [
    (25 * 2**30, 8),       # 0.5 * 25 GiB / (200k * 4096 * 2 B = 1.64 GB per layer) = 8.19
    (1, 1),                # never below one layer
    (2**50, 28),           # never above the model's depth
    (None, 7),             # unreadable: the documented Qwen3 setting
])
def test_choose_layer_group_size_fits_the_reservoirs_in_half_the_available_ram(available,
                                                                               expected):
    from lfa.artifact.build import choose_layer_group_size

    assert choose_layer_group_size(**QWEN3_WIDTHS, available_bytes=available) == expected


def test_auto_layer_group_size_reads_the_config_not_the_model(monkeypatch, caplog):
    import logging
    from types import SimpleNamespace

    import lfa.artifact.build as build_module

    configs = {
        "with-head-dim": SimpleNamespace(hidden_size=1024, num_attention_heads=16, head_dim=128,
                                         num_hidden_layers=28),
        # No head_dim: the pre_o width is hidden_size, as for a Llama.
        "no-head-dim": SimpleNamespace(hidden_size=1024, num_attention_heads=16,
                                       num_hidden_layers=28),
    }
    monkeypatch.setattr(build_module, "AutoConfig",
                        SimpleNamespace(from_pretrained=lambda path: configs[path]))
    monkeypatch.setattr(build_module, "_available_memory_bytes", lambda: 25 * 2**30)

    with caplog.at_level(logging.INFO, logger="lfa.artifact.build"):
        assert build_module._auto_layer_group_size("with-head-dim", 200_000, 2) == 8
    assert "layer_group_size=8" in caplog.text and "GiB" in caplog.text
    # 200k * (2*1024 + 1024) * 2 B = 1.23 GB per layer: 0.5 * 25 GiB / that = 10.9
    assert build_module._auto_layer_group_size("no-head-dim", 200_000, 2) == 10


# ------------------------------------------------- per-site reservoir generators

def test_collect_reservoirs_do_not_depend_on_which_layers_share_a_pass(tiny_model, build_texts):
    """A site's reservoir is the same whether its layer is collected alone or with the others.

    The group size is chosen from host RAM, so if the draws depended on it, two machines would
    build different artifacts from the same model, corpus and seed.
    """
    model, tokenizer = tiny_model
    adapter = get_adapter(model)

    def run(layers, include_lm_head):
        stats, _ = collect_hidden_states(model, tokenizer, adapter, build_texts,
                                         max_samples=10_000, seq_len=128, batch_size=8,
                                         reservoir_size=64, device="cpu", progress=False,
                                         token_frequencies=False, seed=11, layers=layers,
                                         include_lm_head=include_lm_head)
        return stats

    together = run(None, True)
    apart = {**run([0], False), **run([1], True)}
    assert set(together) == set(apart) == EXPECTED_KEYS
    for key in EXPECTED_KEYS:
        assert together[key].seen > 64                          # replacement draws were taken
        assert torch.equal(together[key].reservoir, apart[key].reservoir), key


def test_build_artifact_is_the_same_at_any_layer_group_size(tmp_path, model_dir, corpus_file):
    """Same seed, different group sizes: the same fitted artifact, tensor for tensor."""
    def build(name, group):
        out = tmp_path / name
        build_artifact(str(model_dir), corpus_file, out, max_samples=1000, seq_len=128,
                       reservoir_size=100, pca_variance=0.9, gmm_k=2, layer_group_size=group,
                       quantize=False, device="cpu", seed=0)
        return load_artifact(out)

    one_pass, grouped = build("one_pass.pt", None), build("grouped.pt", 1)
    for key in EXPECTED_KEYS:
        assert one_pass[key]["n_samples"] > 100                 # the reservoir was resampled
        assert set(one_pass[key]) == set(grouped[key]), key
        for field_name, value in one_pass[key].items():
            other = grouped[key][field_name]
            if torch.is_tensor(value):
                assert torch.equal(value, other), (key, field_name)
            else:
                assert value == other, (key, field_name)
    assert torch.equal(one_pass["embedding_lookup"]["token_frequencies"],
                       grouped["embedding_lookup"]["token_frequencies"])
    assert one_pass["__meta__"]["n_samples_total"] == grouped["__meta__"]["n_samples_total"]


def test_reservoir_seeds_are_per_site_and_apart_from_the_fit_seeds():
    """Each site's reservoir stream is its own, and is not the stream its GMM fit starts from."""
    from lfa.artifact.build import _site_seed as build_site_seed
    from lfa.artifact.collect import _reservoir_seed, _site_seed

    assert build_site_seed is _site_seed
    assert _site_seed(0, 3, "pre_o") == 31 and _site_seed(7, 28, "pre_lm_head") == 292
    sites = [(layer, site) for layer in range(28) for site in ("pre_qkv", "pre_o", "pre_mlp")]
    sites.append((28, "pre_lm_head"))
    reservoir = {_reservoir_seed(0, layer, site) for layer, site in sites}
    fit = {_site_seed(0, layer, site) for layer, site in sites}
    assert len(reservoir) == len(sites)
    assert not reservoir & fit
    assert _reservoir_seed(0, 1, "pre_o") != _reservoir_seed(1, 1, "pre_o")


def test_reservoir_replacement_resolves_a_repeated_slot_to_the_later_vector():
    """Two vectors of one batch drawing the same slot: the later one is kept, every run.

    An indexed assignment with a repeated index leaves the winner unspecified, and on a
    multi-threaded CPU it varied from run to run, so a seeded build was not reproducible even
    on one machine. The reference here is the same draws applied one at a time. The old code
    fails this only on a multi-threaded CPU (single-threaded index writes were already
    last-wins), so a green run on one core is no evidence that the resolution is unneeded.
    """
    size, width, n = 1000, 32, 20_000
    stats = SiteStats(reservoir_size=size, generator=torch.Generator().manual_seed(3))
    stats.update(torch.zeros(size, width))                      # fill: no draw is taken
    x = torch.randn(n, width, generator=torch.Generator().manual_seed(4))
    stats.update(x)

    slots = torch.randint(0, size + n, (n,), generator=torch.Generator().manual_seed(3))
    expected = torch.zeros(size, width)
    for i in range(n):
        if slots[i] < size:
            expected[slots[i]] = x[i]
    assert torch.bincount(slots[slots < size]).max() > 1        # the case is exercised
    assert torch.equal(stats.reservoir, expected)
