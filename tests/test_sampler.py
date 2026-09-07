import torch
from lfa.sampler import Sampler
from lfa.adapters import get_adapter

def test_shapes_and_determinism(tiny_artifact):
    _, path = tiny_artifact
    a = Sampler(path, device="cpu", seed=7); b = Sampler(path, device="cpu", seed=7)
    x = a.sample_gmm(1, "pre_mlp", 5); y = b.sample_gmm(1, "pre_mlp", 5)
    assert x.shape == (5, 32) and torch.equal(x, y)
    assert a.sample_pca(1, "pre_qkv", 3).shape == (3, 32)
    assert a.sample_best(2, "pre_lm_head", 4).shape == (4, 32)
    assert a.sample_best(0, "pre_qkv", 4) is None          # no lookup built yet, no 0_pre_qkv key

def test_num_layers_excludes_lm_head(tiny_artifact):
    _, path = tiny_artifact
    # 2_pre_lm_head is the model-level site at index num_layers(), not a third transformer layer
    assert Sampler(path, device="cpu").num_layers() == 2

def test_embedding_lookup_from_model(tiny_artifact, tiny_model):
    _, path = tiny_artifact; model, _ = tiny_model
    s = Sampler(path, device="cpu", seed=1)
    s.build_embedding_lookup_from_model(model, get_adapter(model))
    assert s.has_embedding_lookup()
    h = s.sample_best(0, "pre_qkv", 6); assert h.shape == (6, 32)
    # a lookup row equals input_layernorm(embed_tokens(id)) for that id
    table = s.params["embedding_lookup"]["pre_qkv_table"]
    ref = model.model.layers[0].input_layernorm(model.model.embed_tokens(torch.tensor([5])))[0]
    assert torch.allclose(table[5], ref, atol=1e-6)

def test_gmm_marginal_mean_close(tiny_artifact):
    params, path = tiny_artifact; s = Sampler(path, device="cpu", seed=3)
    x = s.sample_gmm(1, "pre_mlp", 20000)
    e = params["1_pre_mlp"]; V = e["pca_components"]; mix_mean = (e["gmm_weights"][:, None] * e["gmm_means"]).sum(0)
    assert torch.allclose(x.mean(0), e["mean"] + mix_mean @ V.T, atol=0.05)

def test_frequencies_only_stub_is_not_a_lookup(tiny_artifact, tiny_model, tmp_path):
    """A built artifact ships token frequencies without the table; layer-0 pre_qkv is then simply
    unavailable, not a KeyError waiting inside `sample_best`."""
    params, _ = tiny_artifact; model, _ = tiny_model
    stub = {k: v for k, v in params.items()}
    stub["embedding_lookup"] = {"token_frequencies": torch.ones(256) / 256}
    path = tmp_path / "stub.pt"; torch.save(stub, path)

    s = Sampler(path, device="cpu", seed=0)
    assert s.has_embedding_lookup() is False
    assert s.sample_best(0, "pre_qkv", 4) is None

    s.build_embedding_lookup_from_model(model, get_adapter(model))
    assert s.has_embedding_lookup() is True
    assert s.sample_best(0, "pre_qkv", 4).shape == (4, 32)
    # the frequencies from the stub survive the rebuild and still drive weighted sampling
    assert "token_frequencies" in s.params["embedding_lookup"]

def test_seeding_changes_reproducibility_not_the_distribution(tiny_artifact, tiny_model):
    """A seeded sampler draws from the same p(h) as an unseeded one.

    Seeding gives the sampler a private generator so a run is reproducible without disturbing
    whatever the training loop is drawing from -- and that is ALL it is allowed to do. The check
    is on moments rather than on values: two RNG streams never agree draw for draw, but a private
    generator that had (say) skipped the residual-variance term or moved the component draw onto
    the other device would show up as a second moment that no longer matches.

    Both rungs that carry their own noise are covered: the PCA site (basis draw plus off-basis
    residual) and the layer-0 lookup, whose "distribution" is the frequency-weighted table.
    """
    _, path = tiny_artifact
    model, _ = tiny_model
    n = 40_000

    seeded = Sampler(path, device="cpu", seed=11)
    unseeded = Sampler(path, device="cpu", seed=None)
    torch.manual_seed(999)

    before = torch.random.get_rng_state()
    a = seeded.sample_pca(1, "pre_qkv", n)
    assert torch.equal(torch.random.get_rng_state(), before), \
        "a seeded sampler drew from the global RNG, which is the stream it exists to leave alone"

    b = unseeded.sample_pca(1, "pre_qkv", n)
    assert not torch.equal(a, b)                                    # different streams
    assert torch.allclose(a.mean(0), b.mean(0), atol=0.03)
    assert torch.allclose(a.std(0), b.std(0), rtol=0.05)

    frequencies = torch.rand(model.config.vocab_size, generator=torch.Generator().manual_seed(5))
    for sampler in (seeded, unseeded):
        sampler.build_embedding_lookup_from_model(model, get_adapter(model), frequencies)
    a = seeded.sample_embedding_lookup(n)
    b = unseeded.sample_embedding_lookup(n)
    assert not torch.equal(a, b)
    assert torch.allclose(a.mean(0), b.mean(0), atol=0.05)
    assert torch.allclose(a.std(0), b.std(0), rtol=0.05)
