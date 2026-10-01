import pytest
import torch
from lfa.merge import merge_stats, alpha_from_counts

def _block(K, n, seed):
    g = torch.Generator().manual_seed(seed); D = 8
    V, _ = torch.linalg.qr(torch.randn(D, n, generator=g))
    w = torch.rand(K, generator=g); w = w / w.sum()
    return {"mean": torch.zeros(D), "std": torch.ones(D), "pca_components": V, "pca_eigenvalues": torch.linspace(2, 1, n),
            "gmm_weights": w, "gmm_means": torch.randn(K, n, generator=g), "gmm_covariances": torch.rand(K, n, generator=g) + 0.1,
            "gmm_n_components": K, "gmm_covariance_type": "diag"}

def _counted(block, n):
    return {**block, "n_samples": n}

def test_union_weights_and_counts():
    base = {"0_pre_mlp": _counted(_block(4, 3, 1), 300)}
    dom = {"0_pre_mlp": _counted(_block(2, 3, 2), 100)}
    out = merge_stats(base, dom)
    e = out["0_pre_mlp"]
    assert e["gmm_n_components"] == 6 and e["gmm_weights"].shape == (6,)
    assert torch.isclose(e["gmm_weights"].sum(), torch.tensor(1.0))
    assert torch.isclose(e["gmm_weights"][:4].sum(), torch.tensor(1 - alpha_from_counts(300, 100)))
    assert e["n_samples"] == 400
    assert torch.equal(e["mean"], base["0_pre_mlp"]["mean"])   # base mean/basis kept (see source comment)

def test_passthrough_non_moment_keys():
    base = {"__meta__": {"a": 1}, "0_pre_mlp": _counted(_block(2, 3, 3), 10)}
    out = merge_stats(base, {})
    assert out["__meta__"] == {"a": 1} and out["0_pre_mlp"] is base["0_pre_mlp"]


def test_diagonal_only_merge_matches_the_concatenated_pools():
    """No GMM anywhere: the marginal moments must equal those of the two pools concatenated."""
    g = torch.Generator().manual_seed(5)
    pool_a = torch.randn(300, 4, generator=g) * 2.0 + 1.0
    pool_b = torch.randn(100, 4, generator=g) * 0.5 - 3.0

    def moments(pool):
        return {"mean": pool.mean(0), "std": pool.std(0, unbiased=False), "n_samples": len(pool)}

    out = merge_stats({"0_pre_mlp": moments(pool_a)}, {"0_pre_mlp": moments(pool_b)})["0_pre_mlp"]

    both = torch.cat([pool_a, pool_b])
    assert torch.allclose(out["mean"], both.mean(0), atol=1e-6)
    assert torch.allclose(out["std"], both.std(0, unbiased=False), atol=1e-6)
    assert out["n_samples"] == 400
    assert "gmm_weights" not in out          # nothing invented where there was no mixture


def test_n_weighted_merge_refuses_blocks_without_counts():
    with pytest.raises(ValueError, match="n_samples"):
        merge_stats({"0_pre_mlp": _block(4, 3, 1)}, {"0_pre_mlp": _block(2, 3, 2)})


def test_explicit_alpha_overrides_the_counts_but_still_accumulates_them():
    base = {"0_pre_mlp": _counted(_block(4, 3, 1), 300)}
    dom = {"0_pre_mlp": _counted(_block(2, 3, 2), 100)}

    out = merge_stats(base, dom, alpha=0.25)["0_pre_mlp"]

    assert torch.isclose(out["gmm_weights"][:4].sum(), torch.tensor(0.75))   # not 1 - 100/400
    assert out["n_samples"] == 400


def test_explicit_alpha_without_counts_leaves_the_block_uncounted():
    """The weighting is no longer sample-proportional, so an invented count would be a lie."""
    out = merge_stats({"0_pre_mlp": _block(4, 3, 1)},
                      {"0_pre_mlp": _block(2, 3, 2)}, alpha=0.5)["0_pre_mlp"]
    assert "n_samples" not in out
    assert torch.isclose(out["gmm_weights"][:4].sum(), torch.tensor(0.5))
