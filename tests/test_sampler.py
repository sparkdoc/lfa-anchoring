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
