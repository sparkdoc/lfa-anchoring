"""Draw hidden-state vectors from a fitted p(h) artifact.

This is the input side of Layerwise Function Anchoring: the anchor compares teacher and student
outputs of a sub-module on vectors ``h ~ p(h)``, so the quality of the preservation is exactly the
quality of these samples. Per layer and site the artifact may carry, in increasing fidelity:
diagonal moments (``mean``/``std``), a PCA basis with eigenvalues, a GMM head fitted in that basis,
and -- for layer-0 ``pre_qkv`` only -- the exact ``input_layernorm(embed_tokens(id))`` lookup table.
:meth:`Sampler.sample_best` walks that ladder.

Two properties of the PCA and GMM paths are easy to lose and cost anchoring strength when lost:

* **Residual variance.** The stored basis spans only the retained (~95%) variance, so projecting
  back from it reproduces short marginals and a covariance trace ~5% low. Both paths add a
  diagonal off-basis residual, which makes them strictly better than the diagonal model (exact
  marginals *and* the real correlations) instead of merely different. Without it lambda is
  silently weaker, since the anchor loss scales with the second moment of p(h).
* **float32 throughout.** Artifacts store fp16, and the project-back ``z @ components.T`` is a
  matmul, which does not auto-promote -- fp16 basis with fp32 noise is a dtype error, not a
  silent cast.

Determinism: passing ``seed`` gives the sampler its own :class:`torch.Generator`, threaded through
every draw, so a run is reproducible without touching the global RNG. With ``seed=None`` every draw
goes to the global default generator on the device it would naturally use -- including the
component draw, which stays on CPU (see :meth:`Sampler._dev`).
"""

from __future__ import annotations

from pathlib import Path

import torch

from .artifact.schema import load_artifact, parse_site_key, site_key


class Sampler:
    """Samples ``h ~ p(h)`` for one model's anchoring sites, from a loaded artifact."""

    def __init__(self, artifact: str | Path | dict, device: str = "cuda", seed: int | None = None):
        """
        Args:
            artifact: path to an artifact file, or an already-loaded (and dequantized) params dict.
                A dict is used as given -- the caller keeps ownership of it, and the lookup table
                built by :meth:`build_embedding_lookup_from_model` is written into it.
            device: device to generate samples on.
            seed: if given, all draws use a private generator seeded with it. If ``None``, draws use
                the global RNG, matching the reference implementation call for call.
        """
        self.device = device
        self.params = artifact if isinstance(artifact, dict) else load_artifact(artifact)
        self.generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)

        # Device/dtype cache for the immutable per-site tensors (see `_dev`). The artifact is loaded
        # to CPU in fp16, but sampling runs in fp32 on `device`, so without this every draw
        # re-copies and re-converts tensors that never change -- `pca_components` alone is
        # [hidden, pca_dim]. The anchor calls the sampler once per layer per site, ~85 times per
        # training step, which made this ~170 MB of redundant host->device traffic per step and
        # ~70% of the anchor's forward cost at 0.6B. Populated lazily; entries are never invalidated
        # because the per-site stats are read-only after load. The one key written post-load is
        # `embedding_lookup` (build_embedding_lookup_from_model), which is NOT routed through this
        # cache -- it has its own lazy loader -- so no entry can go stale. If a future change starts
        # mutating per-site stats in place, it must clear this.
        self._dev_cache: dict[tuple[str, str], torch.Tensor] = {}

        # Which (layer, site) pairs have statistics. Reserved keys ("__meta__",
        # "embedding_lookup") do not parse as site keys and are skipped.
        self.available: set[tuple[int, str]] = set()
        self._embedding_lookup_loaded = False  # lazy-loading flag
        for key in self.params.keys():
            parsed = parse_site_key(key)
            if parsed is not None:
                self.available.add(parsed)

    # ------------------------------------------------------------------ RNG
    # `self.generator` is None on the default path, and every torch call below accepts
    # `generator=None` as "use the global default generator" -- so these helpers are transparent
    # when unseeded and keep the draw order identical in both modes.

    def _randn(self, *shape: int, device=None) -> torch.Tensor:
        return torch.randn(*shape, device=self.device if device is None else device,
                           generator=self.generator)

    def _randn_like(self, x: torch.Tensor) -> torch.Tensor:
        if self.generator is None:
            return torch.randn_like(x)
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=self.generator)

    def _multinomial(self, weights: torch.Tensor, n: int) -> torch.Tensor:
        if self.generator is None:
            return torch.multinomial(weights, n, replacement=True)
        # A generator draws only for its own device, so the weights must live there too.
        return torch.multinomial(weights.to(self.device), n, replacement=True,
                                 generator=self.generator)

    # -------------------------------------------------------------- topology

    def num_layers(self) -> int:
        """Number of layers with statistics."""
        return len({layer for layer, _ in self.available})

    def hidden_dim(self, layer: int = 0, site: str = "pre_mlp") -> int:
        """Hidden dimension of the sampled vectors."""
        return self.params[site_key(layer, site)]["mean"].shape[0]

    def has_site(self, layer: int, site: str) -> bool:
        """Whether the artifact carries any statistics for this layer and site."""
        return site_key(layer, site) in self.params

    def has_pca(self, layer: int, site: str) -> bool:
        """Whether a PCA basis is available for this layer and site."""
        key = site_key(layer, site)
        return key in self.params and "pca_components" in self.params[key]

    def has_gmm(self, layer: int, site: str) -> bool:
        """Whether GMM parameters are available for this layer and site."""
        key = site_key(layer, site)
        return (
            key in self.params
            and "gmm_n_components" in self.params[key]
            and self.params[key]["gmm_n_components"] > 0
        )

    # ----------------------------------------------------------------- cache

    def _cache(self) -> dict:
        """The device/dtype memo, created on demand.

        Not simply read from ``__init__``: samplers are also built by merge/extend/quantize helpers
        and by tests that populate ``.params`` directly without running ``__init__``, and those must
        not crash on a missing attribute.
        """
        cache = getattr(self, "_dev_cache", None)
        if cache is None:
            cache = {}
            self._dev_cache = cache
        return cache

    def _dev(self, key: str, name: str) -> torch.Tensor | None:
        """``self.params[key][name]`` on ``self.device`` in fp32, converted once and cached.

        The conversion is deterministic and the artifact is read-only after load, so caching is
        pure memoisation -- sample streams are bit-identical with or without it.

        NOTE: ``gmm_weights`` is deliberately NOT routed through here. ``torch.multinomial`` draws
        from the generator of the tensor's device, so moving it to CUDA would switch the component
        draw from the CPU RNG to the CUDA RNG and silently change every sample stream.
        """
        t = self.params[key].get(name)
        if t is None or not torch.is_tensor(t):
            return t
        cache = self._cache()
        hit = cache.get((key, name))
        if hit is None:
            hit = t.to(self.device).float()
            cache[(key, name)] = hit
        return hit

    def _derived(self, key: str, name: str, build) -> torch.Tensor:
        """Cache a tensor derived from the (immutable) stats, e.g. a sqrt or a residual std.

        Same bit-identity argument as :meth:`_dev`: the elementwise maps below commute with the
        gather that consumes them, so hoisting them out of the per-call path changes nothing
        numerically.
        """
        cache = self._cache()
        hit = cache.get((key, name))
        if hit is None:
            hit = build()
            cache[(key, name)] = hit
        return hit

    # --------------------------------------------------------------- drawing

    def sample(self, layer: int, site: str, n: int) -> torch.Tensor:
        """Sample from the diagonal Gaussian ``N(mean, std^2)`` for a layer and site.

        The lowest rung of the fidelity ladder: exact marginals, no correlations. Returns
        ``[n, hidden_dim]``.
        """
        key = site_key(layer, site)
        if key not in self.params:
            raise ValueError(f"No statistics for layer {layer}, site {site}")

        mean = self.params[key]["mean"].to(self.device)
        std = self.params[key]["std"].to(self.device)

        # Reparameterisation: x = mean + std * epsilon, epsilon ~ N(0, I)
        epsilon = self._randn(n, mean.shape[0])
        return mean + std * epsilon

    def sample_pca(self, layer: int, site: str, n: int) -> torch.Tensor:
        """Sample from a PCA-reduced Gaussian: draw in the basis, project back, add the residual.

        Returns ``[n, hidden_dim]``.

        Raises:
            ValueError: if the site is absent or has no PCA basis.
        """
        key = site_key(layer, site)
        if key not in self.params:
            raise ValueError(f"No statistics for layer {layer}, site {site}")

        if "pca_components" not in self.params[key]:
            raise ValueError(
                f"No PCA basis for layer {layer}, site {site}. Collect the statistics with "
                "micro-structure enabled to fit one."
            )

        mean = self._dev(key, "mean")
        pca_components = self._dev(key, "pca_components")          # [hidden, n_comp]
        pca_eigenvalues = self._dev(key, "pca_eigenvalues")        # [n_comp]

        n_components = pca_eigenvalues.shape[0]

        # Sample in PCA space: z ~ N(0, diag(eigenvalues)) == sqrt(eigenvalues) * epsilon
        epsilon = self._randn(n, n_components)
        z = epsilon * self._derived(key, "_eig_sqrt", lambda: torch.sqrt(pca_eigenvalues))

        # Project back to the full space: [n, n_comp] @ [n_comp, hidden]
        samples = mean + z @ pca_components.T

        # Restore the residual (off-basis) variance -- see the module docstring.
        # (Factor-analysis / probabilistic-PCA form: Sigma = V diag(lam) V^T + Psi.)
        std = self.params[key].get("std")
        if std is not None:
            # Cached for the same reason as in sample_gmm: a reduction over the whole
            # [hidden, n_comp] basis, rebuilding a constant on every draw.
            def _residual_std():
                explained = (pca_components ** 2 * pca_eigenvalues).sum(dim=1)      # [hidden]
                return (self._dev(key, "std") ** 2 - explained).clamp_min(0.0).sqrt()

            samples = samples + self._randn_like(samples) * self._derived(
                key, "_pca_residual_std", _residual_std)

        return samples

    def sample_gmm(self, layer: int, site: str, n: int) -> torch.Tensor:
        """Sample from the GMM fitted in this site's PCA basis, projected back to hidden space.

        Captures the multimodal structure a single Gaussian cannot. Returns ``[n, hidden_dim]``.

        Raises:
            ValueError: if the site is absent or has no GMM head.
        """
        key = site_key(layer, site)
        if key not in self.params:
            raise ValueError(f"No statistics for layer {layer}, site {site}")

        if "gmm_n_components" not in self.params[key] or self.params[key]["gmm_n_components"] == 0:
            raise ValueError(
                f"No GMM head for layer {layer}, site {site}. Collect the statistics with "
                "micro-structure enabled to fit one."
            )

        mean = self._dev(key, "mean")
        pca_components = self._dev(key, "pca_components")  # [hidden, pca_dim]
        gmm_weights = self.params[key]["gmm_weights"]  # [K] -- stays on CPU, see _dev
        gmm_means = self.params[key]["gmm_means"]  # [K, pca_dim]
        gmm_covariances = self.params[key]["gmm_covariances"]  # [K, pca_dim(, pca_dim)]
        covariance_type = self.params[key].get("gmm_covariance_type", "full")
        is_diagonal = gmm_covariances.ndim == 2 or covariance_type == "diag"

        pca_dim = gmm_means.shape[1]

        # Top-m head support: the GMM may cover only the top-m PCA coordinates (where the mixture
        # structure lives), with the remaining basis coordinates modelled as independent Gaussians
        # at their fitted eigenvalue variances -- keeps 100% of stored directions at a fraction of
        # the mixture-fitting cost. `gmm_whitened` marks heads fitted in whitened coordinates
        # (z/sqrt(eig), for EM conditioning); they are un-whitened at sample time.
        n_comp_total = pca_components.shape[1]
        gmm_whitened = bool(self.params[key].get("gmm_whitened", False))
        eig = self.params[key].get("pca_eigenvalues")
        if (gmm_whitened or pca_dim < n_comp_total) and eig is None:
            raise ValueError(
                f"GMM for {key} has a whitened or top-m head but no pca_eigenvalues -- "
                "cannot un-whiten the head or sample the Gaussian tail."
            )
        eig_d = self._dev(key, "pca_eigenvalues") if eig is not None else None

        component_indices = self._multinomial(gmm_weights, n)  # [n]

        # Vectorised mixture draw: GATHER the chosen component's parameters instead of looping over
        # components. A Python loop over K with a .item() sync inside it made cost grow linearly in
        # K (0.35 s/step at K=32, 15.4 s/step at K=2048, x84 sub-modules), which would make large-K
        # mixtures unusable. A gather is O(1) in K and the dominant matmul [n, pca_dim] @
        # [pca_dim, hidden] does not depend on K at all.
        gmm_means_d = self._dev(key, "gmm_means")                      # [K, pca_dim]
        gmm_cov_d = self._dev(key, "gmm_covariances")
        idx = component_indices                                        # [n]
        eps = self._randn(n, pca_dim)
        if is_diagonal:
            # sqrt of the GATHERED covariance == gather of the sqrt (elementwise ops commute with
            # indexing), so hoist it: this ran over [n, pca_dim] on every one of the ~85 calls per
            # step.
            cov_sqrt = self._derived(key, "_gmm_cov_sqrt", lambda: gmm_cov_d.clamp_min(0).sqrt())
            samples_pca = gmm_means_d[idx] + eps * cov_sqrt[idx]
        else:
            # Cached: factorising [K, pca_dim, pca_dim] on every draw dominated full-covariance
            # heads. The factorisation depends only on the (immutable) fitted covariance.
            L = self._derived(key, "_gmm_cov_chol",
                              lambda: torch.linalg.cholesky(gmm_cov_d))  # [K, pca_dim, pca_dim]
            samples_pca = gmm_means_d[idx] + torch.einsum("nij,nj->ni", L[idx], eps)

        if gmm_whitened:
            samples_pca = samples_pca * eig_d[:pca_dim].clamp_min(0).sqrt()
        if pca_dim < n_comp_total:
            # fitted Gaussian tail on the remaining basis coordinates
            tail = self._randn(n, n_comp_total - pca_dim) * eig_d[pca_dim:].clamp_min(0).sqrt()
            samples_pca = torch.cat([samples_pca, tail], dim=1)

        # Project back to the full space
        samples = mean + samples_pca.float() @ pca_components.T

        # Restore the residual (off-basis) variance -- see sample_pca. The GMM lives in the
        # truncated ~95%-variance basis, so without this the marginals and the covariance trace
        # come out short and lambda is silently weaker. Same fix, same reason.
        std = self.params[key].get("std")
        if std is not None:
            eig = self.params[key].get("pca_eigenvalues")
            if eig is not None:
                # Cached: this reduced over the whole [hidden, pca_dim] basis on EVERY draw, purely
                # to rebuild a constant. It is a function of the fitted stats alone.
                def _residual_std():
                    explained = (pca_components ** 2 * self._dev(key, "pca_eigenvalues")).sum(dim=1)
                    return (self._dev(key, "std") ** 2 - explained).clamp_min(0.0).sqrt()

                samples = samples + self._randn_like(samples) * self._derived(
                    key, "_residual_std", _residual_std)

        return samples

    # ------------------------------------------------ embedding lookup (L0)

    def build_embedding_lookup_from_model(
        self,
        model,
        adapter,
        token_frequencies: torch.Tensor | None = None,
        batch_size: int = 10000,
    ) -> None:
        """Reconstruct the layer-0 pre_qkv lookup table from the teacher's own weights.

        Layer-0 pre_qkv = ``input_layernorm(embed_tokens(token_id))`` is deterministic in the
        teacher's weights (no positional/context term -- RoPE is applied later, inside attention),
        so the vocab x hidden table need not be shipped: the tuner already holds the teacher and
        rebuilds it here in one layernorm pass. That flips :meth:`has_embedding_lookup` on and makes
        :meth:`sample_embedding_lookup` work with no table on disk -- exact p(h) for layer 0, at
        zero artifact bytes.

        If ``token_frequencies`` are given (or already present from a frequencies-only stub in the
        shipped stats) frequency-weighted sampling stays available; otherwise sampling falls back
        to uniform.
        """
        embed_tokens, input_layernorm = adapter.embed_modules(model)
        vocab_size = embed_tokens.num_embeddings
        hidden_dim = embed_tokens.embedding_dim

        device = next(embed_tokens.parameters()).device
        all_ids = torch.arange(vocab_size, device=device)
        chunks = []
        with torch.no_grad():
            for i in range(0, vocab_size, batch_size):
                pre_qkv = input_layernorm(embed_tokens(all_ids[i:i + batch_size]))
                chunks.append(pre_qkv.cpu())
        pre_qkv_table = torch.cat(chunks, dim=0)

        lookup = dict(self.params.get("embedding_lookup") or {})
        lookup["pre_qkv_table"] = pre_qkv_table
        lookup["vocab_size"] = vocab_size
        lookup["hidden_dim"] = hidden_dim
        if token_frequencies is not None:
            freqs = token_frequencies.float()
            lookup["token_frequencies"] = (freqs / freqs.sum()).cpu()
        self.params["embedding_lookup"] = lookup
        self._embedding_lookup_loaded = False  # force re-load onto device on the next sample

    def has_embedding_lookup(self) -> bool:
        """Whether the exact layer-0 pre_qkv lookup table is available."""
        return "embedding_lookup" in self.params

    def _load_embedding_lookup(self) -> None:
        """Lazily move the lookup table (and frequencies) onto the sampling device, once."""
        if self._embedding_lookup_loaded:
            return

        if not self.has_embedding_lookup():
            raise ValueError("No embedding lookup table available")

        lookup = self.params["embedding_lookup"]
        lookup["pre_qkv_table"] = lookup["pre_qkv_table"].to(self.device)
        if "token_frequencies" in lookup:
            lookup["token_frequencies"] = lookup["token_frequencies"].to(self.device)

        self._embedding_lookup_loaded = True

    def sample_embedding_lookup(self, n: int, weighted: bool = True) -> torch.Tensor:
        """Sample layer-0 pre_qkv vectors by token-ID lookup.

        Layer-0 pre_qkv is a deterministic function of the token ID, so this is perfect sampling
        coverage for layer 0 -- no fitted approximation at all.

        Args:
            n: number of vectors to sample.
            weighted: sample token IDs by corpus frequency (default) rather than uniformly over
                the vocabulary. Frequency weighting needs frequencies in the artifact; without
                them this falls back to uniform.

        Returns:
            ``[n, hidden_dim]``.

        Raises:
            ValueError: if no lookup table is available.
        """
        if not self.has_embedding_lookup():
            raise ValueError(
                "No embedding lookup table available. Build one from the teacher with "
                "build_embedding_lookup_from_model(), or collect statistics with the table enabled."
            )

        self._load_embedding_lookup()

        lookup = self.params["embedding_lookup"]
        pre_qkv_table = lookup["pre_qkv_table"]  # [vocab, hidden]
        vocab_size = lookup["vocab_size"]

        if weighted and "token_frequencies" in lookup:
            token_ids = self._multinomial(lookup["token_frequencies"], n)
        else:
            token_ids = torch.randint(0, vocab_size, (n,), device=self.device,
                                      generator=self.generator)

        return pre_qkv_table[token_ids]  # [n, hidden]

    # ----------------------------------------------------- isotropic control

    def _isotropic_moments(self, layer: int, site: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mean, std)`` of the true p(h) here, the isotropic control's target moments.

        Layer-0 pre_qkv has no collected moments when an embedding lookup is present (that site IS
        the table -- collection is skipped as redundant). Derive its true moments from the table
        itself, frequency-weighted to match how it is actually sampled, so the isotropic control is
        location- and energy-matched at EVERY layer rather than silently unmatched at layer 0.
        """
        key = site_key(layer, site)
        if key in self.params:
            p = self.params[key]
            return p["mean"].to(self.device), p["std"].to(self.device)

        if layer == 0 and site == "pre_qkv" and self.has_embedding_lookup():
            cached = getattr(self, "_embed_moments", None)
            if cached is None:
                self._load_embedding_lookup()
                lookup = self.params["embedding_lookup"]
                table = lookup["pre_qkv_table"].float()               # [vocab, hidden]
                freqs = lookup.get("token_frequencies")
                if freqs is not None:
                    w = freqs.float().to(table.device)
                    # The table can be longer than the frequency vector (embedding rows vs corpus
                    # tokens). multinomial() only ever draws ids < len(freqs), so the
                    # frequency-weighted moments must be taken over exactly that prefix -- matching
                    # how sample_embedding_lookup actually samples.
                    if w.shape[0] != table.shape[0]:
                        table = table[: w.shape[0]]
                    w = w / w.sum()
                    mean = (w.unsqueeze(1) * table).sum(0)
                    var = (w.unsqueeze(1) * (table - mean) ** 2).sum(0)
                else:
                    mean = table.mean(0)
                    var = table.var(0, unbiased=False)
                cached = (mean.to(self.device), var.clamp_min(0).sqrt().to(self.device))
                self._embed_moments = cached
            return cached

        raise KeyError(
            f"No moments for {key!r} and no embedding-lookup fallback -- cannot build the "
            f"isotropic control for this layer/site."
        )

    def sample_isotropic(self, layer: int, site: str, n: int, mode: str = "matched") -> torch.Tensor:
        """Ablation control: sample h from an ISOTROPIC distribution instead of the fitted p(h).

        This is the honest control for the "distribution-weighted" claim: the function-space loss
        and lambda are unchanged, so any performance difference is attributable to the *shape* of
        p(h) alone. (Frobenius weight anchoring is not this control -- it is L2-SP in disguise, in
        units where lambda cannot transfer.)

        Args:
            mode: ``"matched"`` -- ``N(mu, sigma_bar^2 I)`` with ``sigma_bar^2`` the mean of the
                true per-dimension variances. Location and total covariance energy (trace) are
                matched to the real p(h); ONLY the per-dimension shape is destroyed. The strong
                control. ``"unit"`` -- ``N(0, I)``, no distributional knowledge at all. The floor.

        Returns:
            ``[n, hidden_dim]``.
        """
        mean, std = self._isotropic_moments(layer, site)
        dim = mean.shape[0]

        if mode == "unit":
            return self._randn(n, dim)

        if mode != "matched":
            raise ValueError(f"unknown isotropic mode: {mode!r} (expected 'matched' or 'unit')")

        # trace-matched isotropic std: sigma_bar = sqrt(mean(sigma_i^2))
        sigma_bar = (std.float() ** 2).mean().sqrt()
        noise = self._randn(n, dim) * sigma_bar
        return (mean.float().unsqueeze(0) + noise).to(std.dtype)

    # ------------------------------------------------------------ the ladder

    def sample_best(self, layer: int, site: str, n: int, weighted: bool = True):
        """Sample with the highest-fidelity model this artifact has for the layer and site.

        Ladder: exact embedding lookup (layer-0 pre_qkv only) -> GMM -> PCA -> diagonal Gaussian.

        Args:
            weighted: for the embedding lookup, whether to use frequency weighting.

        Returns:
            ``[n, hidden_dim]``, or ``None`` if the artifact has nothing for this layer and site
            (the caller then leaves that sub-module unanchored rather than anchoring it on noise).
        """
        if layer == 0 and site == "pre_qkv" and self.has_embedding_lookup():
            return self.sample_embedding_lookup(n, weighted=weighted)

        if not self.has_site(layer, site):
            return None

        if self.has_gmm(layer, site):
            return self.sample_gmm(layer, site, n)

        if self.has_pca(layer, site):
            return self.sample_pca(layer, site, n)

        return self.sample(layer, site, n)
