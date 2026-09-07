import torch
from lfa.merge import merge_stats, annotate_count, alpha_from_counts

def _block(K, n, seed):
    g = torch.Generator().manual_seed(seed); D = 8
    V, _ = torch.linalg.qr(torch.randn(D, n, generator=g))
    w = torch.rand(K, generator=g); w = w / w.sum()
    return {"mean": torch.zeros(D), "std": torch.ones(D), "pca_components": V, "pca_eigenvalues": torch.linspace(2, 1, n),
            "gmm_weights": w, "gmm_means": torch.randn(K, n, generator=g), "gmm_covariances": torch.rand(K, n, generator=g) + 0.1,
            "gmm_n_components": K, "gmm_covariance_type": "diag"}

def test_union_weights_and_counts():
    base = {"0_pre_mlp": _block(4, 3, 1)}; dom = {"0_pre_mlp": _block(2, 3, 2)}
    base = annotate_count(base, 300); dom = annotate_count(dom, 100)
    out = merge_stats(base, dom)
    e = out["0_pre_mlp"]
    assert e["gmm_n_components"] == 6 and e["gmm_weights"].shape == (6,)
    assert torch.isclose(e["gmm_weights"].sum(), torch.tensor(1.0))
    assert torch.isclose(e["gmm_weights"][:4].sum(), torch.tensor(1 - alpha_from_counts(300, 100)))
    assert e["n_samples"] == 400
    assert torch.equal(e["mean"], base["0_pre_mlp"]["mean"])   # base mean/basis kept (see source comment)

def test_passthrough_non_moment_keys():
    base = {"__meta__": {"a": 1}, "0_pre_mlp": annotate_count({"x": _block(2, 3, 3)}, 10)["x"]}
    out = merge_stats(base, {})
    assert out["__meta__"] == {"a": 1} and out["0_pre_mlp"] is base["0_pre_mlp"]
