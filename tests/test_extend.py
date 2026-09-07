"""Extending a p(h) artifact with a new domain: fit in the base's basis, merge by union.

Everything runs on CPU against the 2-layer `tiny_model` and the `tiny_artifact` fixture, whose
entries carry `n_samples=1000` and a 3-component GMM. `NEED` activations per site are collected
through the "fused" model and fitted as `K_DOMAIN` extra components, so a site's mixture grows by
exactly that many and its count by exactly `NEED`.

`NEED` is 400 rather than the 300 of the task brief because the fit caps K at one component per
200 samples (the reference implementation's rule, kept): at 300 samples K would be capped to 1 and
the union could not be observed. `test_k_domain_is_capped_by_the_sample_count` pins that cap.
"""

import copy
import json

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from lfa.artifact.extend import extend_artifact, fit_domain_gmm
from lfa.artifact.schema import ArtifactModelMismatch, load_artifact
from lfa.merge import alpha_from_counts, annotate_count, merge_stats
from lfa.sampler import Sampler

NEED = 400
K_DOMAIN = 2
SITE_KEYS = {"0_pre_o", "0_pre_mlp", "1_pre_qkv", "1_pre_o", "1_pre_mlp", "2_pre_lm_head"}


@pytest.fixture(scope="module")
def fused_dir(tmp_path_factory, tiny_model):
    """The tiny model as the "fused" base+A model the new domain is collected through."""
    model, tokenizer = tiny_model
    path = tmp_path_factory.mktemp("fused_model")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def corpus_file(tmp_path_factory):
    texts = [f"paper {i} on anchoring the function of a sub-module " * (2 + i % 3)
             for i in range(12)]
    path = tmp_path_factory.mktemp("domain") / "domain_b.jsonl"
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
    return path


def _extend(out_path, base_path, fused_dir, corpus_file, **kwargs):
    return extend_artifact(str(fused_dir), base_path, corpus_file, out_path,
                           k_domain=K_DOMAIN, need=NEED, seq_len=64, seed=42,
                           device="cpu", **kwargs)


@pytest.fixture(scope="module")
def extended(tmp_path_factory, tiny_artifact, fused_dir, corpus_file):
    _, base_path = tiny_artifact
    out = tmp_path_factory.mktemp("extended") / "distribution_stats.pt"
    _extend(out, base_path, fused_dir, corpus_file)
    return out


# --------------------------------------------------------------------------- extend_artifact

def test_extend_unions_the_domain_components_into_every_gmm_site(extended, tiny_artifact):
    base, _ = tiny_artifact
    params = load_artifact(extended)

    assert {k for k in params if k[0].isdigit()} == SITE_KEYS
    for key in SITE_KEYS:
        entry = params[key]
        assert entry["gmm_n_components"] == base[key]["gmm_n_components"] + K_DOMAIN, key
        assert entry["gmm_means"].shape == (5, base[key]["gmm_means"].shape[1]), key
        assert torch.isclose(entry["gmm_weights"].sum(), torch.tensor(1.0), atol=1e-4), key
        assert entry["n_samples"] == 1000 + NEED, key
        # the base keeps its own frame: the components carry the domain, not a shifted mean
        assert torch.allclose(entry["mean"], base[key]["mean"].float(), atol=1e-6), key


def test_extend_weights_the_domain_by_its_sample_share(extended, tiny_artifact):
    base, _ = tiny_artifact
    entry = load_artifact(extended)["1_pre_mlp"]
    share = NEED / (1000 + NEED)
    assert torch.isclose(entry["gmm_weights"][-K_DOMAIN:].sum(), torch.tensor(share), atol=1e-3)
    assert torch.isclose(entry["gmm_weights"][:3].sum(), torch.tensor(1 - share), atol=1e-3)


def test_extend_records_its_provenance_in_the_meta(extended, tiny_artifact):
    params = load_artifact(extended)
    meta = params["__meta__"]
    assert meta["version"] == 2                      # absent counts as 1, so the first bump is 2
    assert meta["extended_with"] == ["domain_b.jsonl"]
    assert meta["num_layers"] == 2 and meta["hidden_size"] == 32     # carried forward
    assert meta["n_samples_total"] == 1000 + NEED           # per site, not a cross-site sum
    assert all(params[k]["n_samples"] == meta["n_samples_total"] for k in SITE_KEYS)


def test_the_extended_artifact_still_samples(extended):
    sampler = Sampler(extended, device="cpu", seed=0)
    assert sampler.sample_gmm(1, "pre_mlp", 8).shape == (8, 32)
    assert sampler.sample_best(2, "pre_lm_head", 8).shape == (8, 32)
    assert torch.isfinite(sampler.sample_best(1, "pre_o", 64)).all()


def test_a_second_extension_unions_again(tmp_path, extended, fused_dir, corpus_file):
    out = tmp_path / "twice.pt"
    _extend(out, extended, fused_dir, corpus_file)

    params = load_artifact(out)
    entry = params["1_pre_mlp"]
    assert entry["gmm_n_components"] == 3 + 2 * K_DOMAIN
    assert torch.isclose(entry["gmm_weights"].sum(), torch.tensor(1.0), atol=1e-4)
    assert entry["n_samples"] == 1000 + 2 * NEED
    assert params["__meta__"]["version"] == 3
    assert params["__meta__"]["extended_with"] == ["domain_b.jsonl", "domain_b.jsonl"]


# --------------------------------------------------------------------------- the base count

def _uncounted_base(tmp_path, base, meta=None):
    """The tiny artifact with its per-entry counts stripped -- the shipped-artifact shape."""
    params = copy.deepcopy(base)
    for key in SITE_KEYS:
        params[key].pop("n_samples")
    if meta is None:
        params.pop("__meta__")
    else:
        params["__meta__"] = meta
    path = tmp_path / "uncounted.pt"
    torch.save(params, path)
    return path


def test_an_uncounted_base_without_meta_requires_base_n(tmp_path, tiny_artifact, fused_dir,
                                                        corpus_file):
    base, _ = tiny_artifact
    path = _uncounted_base(tmp_path, base)

    with pytest.raises(ValueError, match="base_n"):
        _extend(tmp_path / "out.pt", path, fused_dir, corpus_file)

    out = _extend(tmp_path / "out.pt", path, fused_dir, corpus_file, base_n=500)
    assert load_artifact(out)["1_pre_mlp"]["n_samples"] == 500 + NEED


def test_meta_n_samples_total_supplies_the_base_count(tmp_path, tiny_artifact, fused_dir,
                                                      corpus_file):
    base, _ = tiny_artifact
    path = _uncounted_base(tmp_path, base, meta={"n_samples_total": 777})

    out = _extend(tmp_path / "from_meta.pt", path, fused_dir, corpus_file)
    assert load_artifact(out)["1_pre_mlp"]["n_samples"] == 777 + NEED


def test_per_entry_counts_win_over_the_meta_total(tmp_path, tiny_artifact, fused_dir,
                                                  corpus_file):
    """A block that carries its own count is believed over the meta, however the meta reads."""
    base, _ = tiny_artifact
    params = copy.deepcopy(base)
    params["__meta__"] = dict(params["__meta__"], n_samples_total=999_999)
    path = tmp_path / "stale_meta.pt"
    torch.save(params, path)

    out = _extend(tmp_path / "out.pt", path, fused_dir, corpus_file)
    assert load_artifact(out)["1_pre_mlp"]["n_samples"] == 1000 + NEED


# --------------------------------------------------------------------------- fit_domain_gmm

@pytest.fixture(scope="module")
def base_entry(tiny_artifact):
    base, _ = tiny_artifact
    return copy.deepcopy(base["1_pre_mlp"])


def _two_blobs(base_entry, n=600, seed=7):
    """Activations built as two blobs in the base entry's own basis, for a recoverable fit.

    The per-coordinate offsets are what separate the two conventions: whitening divides
    coordinate `j` by `sqrt(eig_j)`, and the fixture's eigenvalues run 1.0 down to 0.1, so a head
    fitted in the wrong frame lands at the wrong place on every coordinate but the first.
    """
    g = torch.Generator().manual_seed(seed)
    basis = base_entry["pca_components"].float()
    mean = base_entry["mean"].float()
    n_comp = basis.shape[1]
    z = torch.randn(n, n_comp, generator=g) * 0.3 + torch.linspace(0.5, 2.0, n_comp)
    z[: n // 2, 0] += 4.0
    z[n // 2:, 0] -= 4.0
    return mean + z @ basis.T


def test_fit_domain_gmm_returns_the_bases_unwhitened_convention(base_entry):
    """The base's head is un-whitened, so the domain head must be too -- they get concatenated."""
    activations = _two_blobs(base_entry)
    block = fit_domain_gmm(activations, base_entry, 2, seed=0, device="cpu")

    assert block["gmm_n_components"] == 2
    assert block["gmm_covariance_type"] == "diag"
    assert "gmm_whitened" not in block
    assert block["gmm_means"].shape == (2, base_entry["gmm_means"].shape[1])
    assert block["gmm_covariances"].shape == block["gmm_means"].shape
    assert (block["gmm_covariances"] > 0).all()
    assert block["n_samples"] == len(activations)
    assert torch.allclose(block["mean"], activations.mean(0), atol=1e-5)
    assert torch.allclose(block["std"], activations.std(0), atol=1e-5)

    # A mixture's mean is its weighted component means; in the base's un-whitened coordinates
    # that must be the projection of the domain's own mean. A whitened head fails this by a
    # factor of sqrt(eigenvalue) per coordinate.
    basis = base_entry["pca_components"].float()
    projected = (activations - base_entry["mean"].float()) @ basis
    mixture_mean = (block["gmm_weights"].unsqueeze(1) * block["gmm_means"]).sum(0)
    assert torch.allclose(mixture_mean, projected.mean(0), atol=1e-3)

    # ... and its second moment must be the domain's too, which is what pins the covariance's
    # un-whitening factor at `eig` rather than the means' `sqrt(eig)`.
    mixture_var = (block["gmm_weights"].unsqueeze(1)
                   * (block["gmm_covariances"] + block["gmm_means"] ** 2)).sum(0) - mixture_mean ** 2
    assert torch.allclose(mixture_var, projected.var(0, unbiased=False), rtol=0.05, atol=1e-3)


def test_fit_domain_gmm_recovers_the_domains_modes(base_entry):
    """The two blobs are 8 units apart on the first coordinate; the fit must find both."""
    activations = _two_blobs(base_entry)
    block = fit_domain_gmm(activations, base_entry, 2, seed=0, device="cpu")

    centres = sorted(block["gmm_means"][:, 0].tolist())          # offset 0.5, blobs at +-4
    assert centres[0] == pytest.approx(-3.5, abs=0.3)
    assert centres[1] == pytest.approx(4.5, abs=0.3)
    assert torch.allclose(block["gmm_weights"], torch.full((2,), 0.5), atol=0.05)


def test_fit_domain_gmm_follows_a_whitened_top_m_base_entry(base_entry):
    """Some shipped artifacts carry whitened top-m heads; the domain head must match the base's
    convention and width, or the union concatenates two different coordinate systems."""
    whitened = copy.deepcopy(base_entry)
    head_dim = 4
    whitened["gmm_means"] = whitened["gmm_means"][:, :head_dim]
    whitened["gmm_covariances"] = whitened["gmm_covariances"][:, :head_dim]
    whitened["gmm_whitened"] = True

    activations = _two_blobs(base_entry)
    block = fit_domain_gmm(activations, whitened, 2, seed=0, device="cpu")

    assert block["gmm_whitened"] is True
    assert block["gmm_means"].shape == (2, head_dim)

    basis = whitened["pca_components"].float()[:, :head_dim]
    eig = whitened["pca_eigenvalues"].float()[:head_dim]
    z = ((activations - whitened["mean"].float()) @ basis) / eig.sqrt()
    mixture_mean = (block["gmm_weights"].unsqueeze(1) * block["gmm_means"]).sum(0)
    assert torch.allclose(mixture_mean, z.mean(0), atol=1e-3)
    mixture_var = (block["gmm_weights"].unsqueeze(1)
                   * (block["gmm_covariances"] + block["gmm_means"] ** 2)).sum(0) - mixture_mean ** 2
    assert torch.allclose(mixture_var, z.var(0, unbiased=False), rtol=0.05, atol=1e-3)


def test_the_union_places_the_domain_where_the_domain_is(base_entry):
    """End of the round trip: after the merge, the mixture mean is the sample-weighted blend of
    the two pools' means -- which pins the domain head to the base's frame. A head left whitened
    would land at `mean / sqrt(eig)` on every coordinate and this recovery would miss."""
    activations = _two_blobs(base_entry)
    block = fit_domain_gmm(activations, base_entry, 2, seed=0, device="cpu")

    base_stats = annotate_count({"1_pre_mlp": copy.deepcopy(base_entry)}, 1000)
    merged = merge_stats(base_stats, {"1_pre_mlp": block})["1_pre_mlp"]

    share = alpha_from_counts(1000, len(activations))

    def mixture_mean(entry):
        return (entry["gmm_weights"].unsqueeze(1) * entry["gmm_means"]).sum(0)

    recovered = (mixture_mean(merged) - (1 - share) * mixture_mean(base_entry)) / share
    basis = base_entry["pca_components"].float()
    projected = ((activations - base_entry["mean"].float()) @ basis).mean(0)
    assert torch.allclose(recovered, projected, atol=1e-3)


def test_k_domain_is_capped_by_the_sample_count(base_entry):
    """One component per 200 samples: 300 activations buy exactly one, however many are asked."""
    activations = _two_blobs(base_entry, n=300)
    assert fit_domain_gmm(activations, base_entry, 8, seed=0, device="cpu")["gmm_n_components"] == 1
    assert fit_domain_gmm(_two_blobs(base_entry, n=1000), base_entry, 8, seed=0,
                          device="cpu")["gmm_n_components"] == 5


# --------------------------------------------------------------------------- guards

def test_extend_refuses_a_model_the_artifact_does_not_describe(tmp_path, tiny_artifact,
                                                               tiny_model, corpus_file):
    _, tokenizer = tiny_model
    _, base_path = tiny_artifact
    cfg = LlamaConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True)
    wrong = tmp_path / "one_layer"
    LlamaForCausalLM(cfg).save_pretrained(wrong)
    tokenizer.save_pretrained(wrong)

    with pytest.raises(ArtifactModelMismatch, match="layers"):
        _extend(tmp_path / "never.pt", base_path, wrong, corpus_file)


def test_extend_collects_through_the_fused_model_not_the_base(tmp_path, tiny_artifact,
                                                              tiny_model, fused_dir,
                                                              corpus_file):
    """The domain components describe the fused model's activations: a differently-weighted
    "fused" model must produce a different fit from the same corpus."""
    model, tokenizer = tiny_model
    moved = copy.deepcopy(model)
    with torch.no_grad():
        for param in moved.parameters():
            param.add_(0.02)
    moved_dir = tmp_path / "moved"
    moved.save_pretrained(moved_dir)
    tokenizer.save_pretrained(moved_dir)

    _, base_path = tiny_artifact
    a = load_artifact(_extend(tmp_path / "a.pt", base_path, fused_dir, corpus_file,
                              quantize=False))
    b = load_artifact(_extend(tmp_path / "b.pt", base_path, moved_dir, corpus_file,
                              quantize=False))
    assert not torch.allclose(a["1_pre_mlp"]["gmm_means"], b["1_pre_mlp"]["gmm_means"])


def test_a_corpus_shorter_than_need_yields_what_there_is(tmp_path, tiny_artifact, fused_dir,
                                                         caplog):
    """A short domain is fitted on what it has, and said so -- not padded, not silently truncated
    to a component count the data cannot support."""
    _, base_path = tiny_artifact
    corpus = tmp_path / "tiny_domain.jsonl"
    corpus.write_text(json.dumps({"text": "one short paper about anchoring "}) + "\n",
                      encoding="utf-8")

    with caplog.at_level("WARNING", logger="lfa.artifact.extend"):
        out = extend_artifact(str(fused_dir), base_path, corpus, tmp_path / "short.pt",
                              k_domain=8, need=10_000, seq_len=64, seed=42, device="cpu")

    entry = load_artifact(out)["1_pre_mlp"]
    assert 0 < entry["n_samples"] - 1000 < 10_000
    assert entry["gmm_n_components"] == 3 + 1                    # one component per 200 samples
    assert any("Corpus exhausted" in record.message for record in caplog.records)


def test_a_corpus_that_chunks_to_nothing_says_so(tmp_path, tiny_artifact, fused_dir):
    """Documents below the minimum chunk length leave no training stream to collect from."""
    _, base_path = tiny_artifact
    corpus = tmp_path / "too_short.jsonl"
    corpus.write_text(json.dumps({"text": "hi"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No training chunks"):
        _extend(tmp_path / "never.pt", base_path, fused_dir, corpus)
