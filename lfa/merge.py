"""Data-free merge of diagonal-Gaussian p(h) moments, for continual anchoring.

Adding a domain to an existing p(h) artifact does NOT require the original corpus. The artifact
already carries the base distribution as per-sub-module (mean, std) over ``n`` samples; to account
for ``base+A`` we sample only the new domain the user owns (``m`` samples, collected through the
*fused* model) and superpose its moments. Because moments are a sufficient statistic, this makes
p(h) an *accumulating* artifact: each stage touches only the new domain, and the correction is a
fixed-size moment update — the compounding advantage over replay, which must rehearse every prior
corpus at every stage.

Correctness is exact: the merged (mean, std) equals the moments of the *concatenated* samples
(population moments, divide by N). The between-group variance term is what makes this hold when the
group means differ — which they always do when a domain shifts the distribution.

Two modes:
  * n-weighted (``alpha=None``): the statistically exact combination, weight = m / (n+m). Requires
    ``n_samples`` on each block; maintains the sufficient-statistic invariant across stages.
  * chosen ``alpha``: a tuning knob (the p(h)-space analog of a replay ratio) for probing how much
    the new domain should weigh. Does not preserve the exact-accumulation invariant — use for a
    single downstream run, not for chained merges.
"""
from __future__ import annotations


def alpha_from_counts(n_base: int, n_new: int) -> float:
    """Weight on the NEW moment set for an n-weighted merge: ``m / (n + m)``."""
    total = n_base + n_new
    if total <= 0:
        raise ValueError("alpha_from_counts: n_base + n_new must be positive")
    return n_new / total


def merge_diagonal_moments(mean_a, std_a, mean_b, std_b, alpha):
    """Merge two diagonal-Gaussian moment sets, weight ``alpha`` on set B, ``1-alpha`` on set A.

    Population-moment (law-of-total-variance) combination, elementwise:

        mean = (1-a)*mean_a + a*mean_b
        var  = (1-a)*var_a + a*var_b + a*(1-a)*(mean_b-mean_a)^2   <- between-group term

    With ``alpha = m/(n+m)`` this equals the population moments of the concatenated samples exactly
    (Chan's parallel algorithm). Returns ``(mean, std)`` as tensors like the inputs.
    """
    one_m = 1.0 - alpha
    delta = mean_b - mean_a
    mean = one_m * mean_a + alpha * mean_b
    var = one_m * std_a ** 2 + alpha * std_b ** 2 + alpha * one_m * delta ** 2
    return mean, var ** 0.5


def is_moment_block(value) -> bool:
    """True for a sampled-sub-module stats block (carries diagonal moments), False for the
    embedding-lookup table, scalar metadata, etc."""
    return isinstance(value, dict) and "mean" in value and "std" in value


def merge_gmm_blocks(base_val: dict, dom_val: dict, alpha: float) -> dict | None:
    """Exact n-weighted merge of two GMMs that live in the SAME PCA basis: the component UNION.

    A mixture p = (1-α)·p_base + α·p_dom is, by construction, the distribution of "draw from the
    base pool with prob (1-α), else the domain pool" — so merging is just concatenating the
    component sets and scaling their weights by (1-α) and α. This is exact (no refitting) and O(1)
    per domain, the accumulating-sufficient-statistic property extended from moments to mixtures.

    REQUIRES dom_val's GMM to have been fit in base_val's basis (`pca_components`) — the collector
    projects the new domain onto the shipped basis before fitting, so the coordinates are shared.
    Returns the merged gmm fields, or None if either block lacks a GMM.
    """
    import torch
    if "gmm_weights" not in base_val or "gmm_weights" not in dom_val:
        return None
    wb, wd = base_val["gmm_weights"].float(), dom_val["gmm_weights"].float()
    weights = torch.cat([(1.0 - alpha) * wb, alpha * wd])
    weights = weights / weights.sum().clamp_min(1e-12)
    return {
        "gmm_weights": weights,
        "gmm_means": torch.cat([base_val["gmm_means"].float(), dom_val["gmm_means"].float()], dim=0),
        "gmm_covariances": torch.cat(
            [base_val["gmm_covariances"].float(), dom_val["gmm_covariances"].float()], dim=0),
        "gmm_n_components": int(weights.shape[0]),
        "gmm_covariance_type": base_val.get("gmm_covariance_type", "diag"),
    }


def merge_stats(base_stats: dict, domain_stats: dict, alpha: float | None = None) -> dict:
    """Merge ``domain_stats`` into ``base_stats`` per sub-module, returning a new dict.

    ``alpha=None`` → n-weighted exact merge (needs ``n_samples`` on both blocks; accumulates the
    count). ``alpha`` given → weighted merge with that fixed weight on the domain (count set to the
    sum when both are present, else dropped, since the weighting is no longer sample-proportional).

    The base artifact's structure is preserved: sub-module blocks present in both are merged;
    everything else (embedding_lookup, ``"__meta__"``, base-only keys) is carried through unchanged.
    """
    out: dict = {}
    for key, base_val in base_stats.items():
        if not (is_moment_block(base_val) and key in domain_stats and is_moment_block(domain_stats[key])):
            out[key] = base_val  # non-moment entry, or no domain counterpart -> pass through
            continue

        dom_val = domain_stats[key]
        if alpha is None:
            if "n_samples" not in base_val or "n_samples" not in dom_val:
                raise ValueError(
                    f"n-weighted merge needs n_samples on both blocks (missing on '{key}'); "
                    "pass an explicit alpha to merge without counts."
                )
            n_b, n_d = base_val["n_samples"], dom_val["n_samples"]
            a = alpha_from_counts(n_b, n_d)
            n_out = n_b + n_d
        else:
            a = alpha
            n_out = (base_val.get("n_samples", 0) + dom_val.get("n_samples", 0)) or None

        merged = dict(base_val)  # keep diagnostics/other keys from the base block (incl. pca basis)
        gmm = merge_gmm_blocks(base_val, dom_val, a)
        if gmm is not None:
            # GMM path: the component union already encodes A's location and spread (A's components
            # were fit relative to the base mean, in the base basis). `sample_gmm` reconstructs
            # h = stored_mean + gmm_sample @ Vᵀ + residual, so we MUST keep stored_mean = base_mean
            # and base std/basis (off-basis residual ≈ unchanged) — overwriting the mean with the
            # n-weighted blend would double-shift every component. Marginal mean/cov come out correct
            # from the mixture itself (verified upstream against the proportional concatenation of the two
            # sample pools).
            merged.update(gmm)
        else:
            # pure-diagonal path: no mixture, so the marginal blend IS the update
            merged["mean"], merged["std"] = merge_diagonal_moments(
                base_val["mean"], base_val["std"], dom_val["mean"], dom_val["std"], a)
        if n_out is not None:
            merged["n_samples"] = n_out
        else:
            merged.pop("n_samples", None)
        out[key] = merged
    return out
