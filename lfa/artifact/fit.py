"""Fit one site's collected statistics into an artifact entry: PCA basis, then a GMM in it.

The ladder the sampler walks (diagonal -> PCA -> GMM) is built here, in that order:

1. **PCA** comes from the eigendecomposition of the accumulated covariance, not from an SVD of
   stored samples -- which is what lets the basis be estimated over millions of vectors while only
   a bounded reservoir is kept. Components are the eigenvectors of the top ``pca_variance``
   fraction of the variance, stored ``[D, n]``.
2. **The GMM** is fitted with :class:`TorchGMM` on the reservoir *projected into that basis*, with
   diagonal covariances: in PCA coordinates the axes are already decorrelated globally, so a
   diagonal head buys the per-mode structure a single Gaussian misses at 1/D the parameters.

Two conventions are inherited from the real-text qwen3-0.6b artifact the paper's operating point
was tuned against, which is the oracle for this format, and both matter to the sampler:

* the GMM is fitted on **un-whitened** PCA coordinates ``(h - mean) @ V`` -- not divided by
  ``sqrt(eigenvalue)`` -- so no ``gmm_whitened`` key exists and none may be written;
* the basis is stored fp16, and the projection here uses that same rounded basis, so the fitted
  coordinates and the sampled ones live in exactly the same frame.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from .collect import SiteStats

__all__ = ["TorchGMM", "fit_site"]

logger = logging.getLogger(__name__)


class TorchGMM:
    """A diagonal-covariance Gaussian mixture fitted by EM on the GPU (or CPU), in torch.

    Diagonal is the only kind: it is the only head this package writes and reads.

    Ported from the reference implementation. Two things differ from ``sklearn``'s
    ``GaussianMixture``: parameters are exposed as tensors rather than numpy arrays, and the RNG is
    a private :class:`torch.Generator` seeded from ``random_state`` -- fitting a mixture never
    perturbs the global RNG stream a training run may be drawing from.

    Args:
        n_components: number of mixture components, K.
        max_iter: EM iterations per initialization.
        tol: stop when the mean log-likelihood moves by less than this.
        reg_covar: added to every variance, keeping every standard deviation positive.
        n_init: independent initializations; the best log-likelihood is kept.
        random_state: seed for the private generator (``None`` = global RNG).
        device: device to fit on.
        init_params: ``"kmeans"`` (k-means++ seeding with sklearn's local trials, Lloyd
            iterations, then a hard-assignment M-step -- every component starts with its own
            weight, mean and variance) or ``"kmeans++"`` (the historical initialization: seeded
            centres, uniform weights, covariance = the global variance). ``"kmeans"`` is the
            default because the historical one falls into collapsed-component optima on
            activation data.

    Attributes:
        weights_: ``[K]`` mixture weights, summing to 1.
        means_: ``[K, D]`` component means.
        covariances_: ``[K, D]`` per-dimension variances.
        converged_, n_iter_, lower_bound_: EM diagnostics.
    """

    def __init__(
        self,
        n_components: int = 4,
        max_iter: int = 100,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        n_init: int = 1,
        random_state: int | None = None,
        device: str | torch.device | None = None,
        verbose: bool = False,
        init_params: str = "kmeans",
    ):
        if init_params not in ("kmeans", "kmeans++"):
            raise ValueError(f"init_params must be 'kmeans' or 'kmeans++', got {init_params!r}")

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.init_params = init_params

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._generator: torch.Generator | None = None

        self.weights_: torch.Tensor | None = None
        self.means_: torch.Tensor | None = None
        self.covariances_: torch.Tensor | None = None
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: float = -float("inf")

        self._weights: torch.Tensor | None = None
        self._means: torch.Tensor | None = None
        self._covariances: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    # ---------------------------------------------------------------- RNG helpers

    def _randint(self, high: int, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(0, high, shape, device=self.device, generator=self._generator)

    def _multinomial(self, probs: torch.Tensor, n: int, replacement: bool = True) -> torch.Tensor:
        return torch.multinomial(probs, n, replacement=replacement, generator=self._generator)

    def _randn(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(shape, device=self.device, dtype=dtype, generator=self._generator)

    # ---------------------------------------------------------------- EM internals

    def _initialize_parameters(self, X: torch.Tensor) -> None:
        """Seed means with k-means++, then (``init_params="kmeans"``) refine with Lloyd steps."""
        n_samples, n_features = X.shape
        K = self.n_components
        means = torch.zeros(K, n_features, device=self.device, dtype=X.dtype)
        means[0] = X[self._randint(n_samples, (1,))]

        # sklearn's k-means++ takes 2 + log(K) candidate centres per step and keeps the one that
        # reduces the total squared distance most; the historical path takes a single draw.
        n_local_trials = (2 + int(np.log(K))) if self.init_params == "kmeans" else 1
        closest = ((X - means[0]) ** 2).sum(dim=1) if self.init_params == "kmeans" else None
        for k in range(1, K):
            if self.init_params == "kmeans":
                probs = closest / closest.sum().clamp_min(1e-30)
                cand = self._multinomial(probs.float().contiguous(), n_local_trials)
                d_cand = torch.cdist(X, X[cand]) ** 2                    # [N, trials]
                new_closest = torch.minimum(closest.unsqueeze(1), d_cand)
                best = new_closest.sum(dim=0).argmin()
                means[k] = X[cand[best]]
                closest = new_closest[:, best]
                continue
            min_dists = torch.cdist(X, means[:k]).min(dim=1).values
            probs = (min_dists ** 2)
            means[k] = X[self._multinomial((probs / probs.sum().clamp_min(1e-30)).float().contiguous(), 1)]

        self._means = means

        if self.init_params == "kmeans":
            for _ in range(20):
                assign = torch.cdist(X, means).argmin(dim=1)
                counts = torch.bincount(assign, minlength=K).to(X.dtype)
                sums = torch.zeros_like(means).index_add_(0, assign, X)
                new_means = means.clone()
                nonempty = counts > 0
                new_means[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
                if torch.allclose(new_means, means):
                    means = new_means
                    break
                means = new_means
            self._means = means
            assign = torch.cdist(X, means).argmin(dim=1)
            counts = torch.bincount(assign, minlength=K)
            for k in torch.nonzero(counts == 0).flatten().tolist():
                far = torch.cdist(X, means).min(dim=1).values.argmax()   # re-seed on the outlier
                means[k] = X[far]
                assign = torch.cdist(X, means).argmin(dim=1)
            self._m_step(X, torch.nn.functional.one_hot(assign, K).to(X.dtype))
            return

        self._weights = torch.ones(K, device=self.device, dtype=X.dtype) / K
        data_var = X.var(dim=0).mean()
        self._covariances = torch.ones(K, n_features, device=self.device, dtype=X.dtype) * data_var
        self._compute_std()

    def _compute_std(self) -> None:
        """Per-dimension standard deviations of the regularized variances."""
        self._std = torch.sqrt(self._covariances + self.reg_covar)

    def _compute_log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """``[N, K]`` log ``p(x | component)``."""
        n_samples, n_features = X.shape
        log_prob = torch.zeros(n_samples, self.n_components, device=self.device, dtype=X.dtype)
        for k in range(self.n_components):
            diff = X - self._means[k]
            mahalanobis = ((diff / self._std[k]) ** 2).sum(dim=1)
            log_det = 2 * torch.log(self._std[k]).sum()
            log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahalanobis)
        return log_prob

    def _e_step(self, X: torch.Tensor) -> tuple[torch.Tensor, float]:
        weighted = self._compute_log_prob(X) + torch.log(self._weights)
        log_norm = torch.logsumexp(weighted, dim=1, keepdim=True)
        return torch.exp(weighted - log_norm), log_norm.mean().item()

    def _m_step(self, X: torch.Tensor, resp: torch.Tensor) -> None:
        n_samples, n_features = X.shape
        nk = resp.sum(dim=0) + 1e-10
        self._weights = nk / n_samples
        self._means = (resp.T @ X) / nk.unsqueeze(1)

        self._covariances = torch.zeros(self.n_components, n_features,
                                        device=self.device, dtype=X.dtype)
        for k in range(self.n_components):
            self._covariances[k] = (resp[:, k:k + 1] * (X - self._means[k]) ** 2).sum(dim=0) / nk[k]

        self._compute_std()

    # ---------------------------------------------------------------- public API

    def fit(self, X: torch.Tensor | np.ndarray) -> "TorchGMM":
        """Fit by EM, keeping the best of ``n_init`` initializations. Returns ``self``."""
        X = torch.as_tensor(X).float().to(self.device)
        if X.shape[0] < self.n_components:
            raise ValueError(
                f"Cannot fit {self.n_components} components to {X.shape[0]} samples."
            )
        if self.random_state is not None:
            self._generator = torch.Generator(device=self.device).manual_seed(self.random_state)

        best_log_likelihood = -float("inf")
        best_params = None
        for _ in range(self.n_init):
            self._initialize_parameters(X)
            prev = -float("inf")
            log_likelihood = prev
            iteration = 0
            for iteration in range(self.max_iter):
                resp, log_likelihood = self._e_step(X)
                if abs(log_likelihood - prev) < self.tol:
                    break
                prev = log_likelihood
                self._m_step(X, resp)
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_params = (self._weights.clone(), self._means.clone(),
                               self._covariances.clone(), iteration + 1)

        self._weights, self._means, self._covariances, self.n_iter_ = best_params
        self._compute_std()
        self.lower_bound_ = best_log_likelihood
        self.converged_ = True

        self.weights_ = self._weights
        self.means_ = self._means
        self.covariances_ = self._covariances
        return self

    def sample(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(samples [n, D], component index [n])`` drawn from the fitted mixture."""
        self._require_fitted()
        components = self._multinomial(self._weights, n_samples)
        n_features = self._means.shape[1]
        samples = torch.zeros(n_samples, n_features, device=self.device, dtype=self._means.dtype)
        for k in range(self.n_components):
            mask = components == k
            n_k = int(mask.sum())
            if n_k == 0:
                continue
            z = self._randn((n_k, n_features), self._means.dtype)
            samples[mask] = self._means[k] + z * self._std[k]
        return samples, components

    def score_samples(self, X: torch.Tensor | np.ndarray) -> torch.Tensor:
        """``[N]`` log ``p(x)`` under the fitted mixture."""
        self._require_fitted()
        X = torch.as_tensor(X).float().to(self.device)
        return torch.logsumexp(self._compute_log_prob(X) + torch.log(self._weights), dim=1)

    def bic(self, X: torch.Tensor | np.ndarray) -> float:
        """Bayesian information criterion on ``X`` (lower is better)."""
        X = torch.as_tensor(X)
        n_samples, n_features = X.shape[0], X.shape[1]
        total_ll = float(self.score_samples(X).sum())
        K, D = self.n_components, n_features
        n_params = K * D + (K - 1) + K * D
        return -2.0 * total_ll + n_params * float(np.log(n_samples))

    def _require_fitted(self) -> None:
        if self._weights is None:
            raise RuntimeError("TorchGMM must be fitted before sampling or scoring.")


def fit_site(
    stats: SiteStats,
    *,
    pca_variance: float = 0.95,
    gmm_k: int = 32,
    seed: int = 0,
    device: str = "cuda:0",
    n_init: int = 3,
) -> dict:
    """Turn one site's collected statistics into an artifact entry.

    Args:
        stats: the site's accumulator, from :func:`~lfa.artifact.collect.collect_hidden_states`.
        pca_variance: fraction of the variance the stored basis must span.
        gmm_k: number of mixture components (clamped down if the reservoir is smaller).
        seed: seed for the GMM's initialization; vary it per site.
        device: device to fit the GMM on.
        n_init: GMM initializations to try, the best kept (the reference pipeline uses 3).

    Returns:
        The artifact entry: ``mean``/``std`` (fp32), ``pca_components`` ``[D, n]`` fp16 with
        ``pca_eigenvalues``/``pca_n_components``, the diagonal GMM head, its
        ``gmm_log_likelihood_p10/p50/p90`` reference percentiles, and ``n_samples`` -- the count
        this site was fitted from, which is what lets two artifacts be merged n-weighted.
    """
    mean, std, cov = stats.mean, stats.std, stats.cov

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    order = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[order].clamp(min=1e-10)
    eigenvectors = eigenvectors[:, order]

    cumulative = eigenvalues.cumsum(dim=0)
    n_components = int((cumulative < pca_variance * cumulative[-1]).sum()) + 1
    n_components = min(n_components, eigenvalues.shape[0])

    # Store the basis fp16 and project with that same rounded basis: the GMM then lives in exactly
    # the coordinates the sampler reconstructs from, rather than in a frame 1e-3 away from them.
    components = eigenvectors[:, :n_components].half()

    entry = {
        "mean": mean,
        "std": std,
        "pca_components": components,
        "pca_eigenvalues": eigenvalues[:n_components].float(),
        "pca_n_components": n_components,
        "n_samples": int(stats.n),
    }

    reservoir = stats.reservoir
    if reservoir is None or reservoir.shape[0] < 2:
        logger.warning("No reservoir samples to fit a GMM head on; entry keeps PCA only.")
        return entry

    coords = (reservoir.float() - mean) @ components.float()
    k = min(gmm_k, coords.shape[0])
    if k < gmm_k:
        logger.warning("Reservoir holds %d samples; fitting K=%d instead of %d.",
                       coords.shape[0], k, gmm_k)

    gmm = TorchGMM(n_components=k, n_init=n_init,
                   random_state=seed, device=device).fit(coords)

    # Reference percentiles of the fitted log-likelihood, as in the source: they are read as
    # "how typical is this vector for the distribution the head was fitted to", so they are taken
    # over the fitting coordinates themselves, not over a held-out slice.
    log_probs = gmm.score_samples(coords)
    quantiles = torch.quantile(log_probs.cpu().float(), torch.tensor([0.10, 0.50, 0.90]))

    entry.update({
        "gmm_n_components": k,
        "gmm_weights": gmm.weights_.detach().cpu().float(),
        "gmm_means": gmm.means_.detach().cpu().float(),
        "gmm_covariances": gmm.covariances_.detach().cpu().float(),
        "gmm_covariance_type": "diag",
        "gmm_log_likelihood_p10": float(quantiles[0]),
        "gmm_log_likelihood_p50": float(quantiles[1]),
        "gmm_log_likelihood_p90": float(quantiles[2]),
    })
    return entry
