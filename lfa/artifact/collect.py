"""Collect the hidden-state statistics a p(h) artifact is fitted from.

LFA anchors a sub-module's *function* on vectors drawn from the distribution its input actually
takes, so the artifact starts here: run a seed corpus through the frozen model and record, at each
anchoring site, enough to fit p(h) later. "Enough" is a running mean and covariance (exact, over
every token seen) plus a bounded reservoir of raw vectors (for the GMM head) -- so the memory cost
is fixed by the reservoir, not by the corpus.

Three details are load-bearing and easy to get wrong:

* **The site's width comes from the hooked input, never from the model's hidden size.** ``pre_o``
  is the concatenated attention head outputs, ``num_heads * head_dim`` wide, which on Qwen3-0.6B
  is 2048 against a hidden size of 1024. :class:`SiteStats` therefore takes its dimension from the
  first batch it sees.
* **Padding positions are excluded.** A padded batch is mostly pad at the tail, and pad rows are
  not vectors the model ever conditions on -- keeping them would pull every mean toward the
  pad embedding's trajectory.
* **Layer 0's ``pre_qkv`` is never collected.** It is ``input_layernorm(embed_tokens(id))``,
  deterministic in the teacher's own weights, so the exact table is rebuilt at load time
  (:meth:`lfa.sampler.Sampler.build_embedding_lookup_from_model`) rather than fitted here.

Accumulation runs in float64 on CPU: a covariance is a sum of ~10^6 outer products, and float32
loses the small eigenvalues that the PCA basis is precisely there to keep. That accumulation, not
the model's forward pass, is where the wall-clock goes -- one ``[D, B] @ [B, D]`` per site per
batch, in double precision -- so a build is measured in hours, once, and never again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import torch

from .schema import LM_HEAD_SITE, SITES, site_key

__all__ = ["SiteStats", "collect_hidden_states"]

logger = logging.getLogger(__name__)


@dataclass
class SiteStats:
    """Running mean/covariance plus a reservoir of raw vectors, for one anchoring site.

    Args:
        reservoir_size: how many raw vectors to keep for the GMM fit.
        dtype: storage dtype of the reservoir. float16 halves its memory (the shipped
            qwen3-0.6b artifact was collected that way); float32 keeps the samples exact.
        generator: RNG for the reservoir's replacement draws. ``None`` uses the global RNG.

    Attributes:
        n: number of vectors accumulated (padding excluded).
        reservoir: ``[<= reservoir_size, D]`` retained samples.
    """

    reservoir_size: int = 200_000
    dtype: torch.dtype = torch.float32
    generator: torch.Generator | None = None

    n: int = 0
    hidden_dim: int = 0
    reservoir: torch.Tensor | None = None
    seen: int = 0                                   # vectors offered to the reservoir so far

    _sum_x: torch.Tensor | None = field(default=None, repr=False)     # [D]      float64
    _sum_xx: torch.Tensor | None = field(default=None, repr=False)    # [D, D]   float64
    _m2: torch.Tensor | None = field(default=None, repr=False)        # [D]      float64

    def update(self, x: torch.Tensor) -> None:
        """Accumulate a batch of vectors ``[B, D]`` (already stripped of padding)."""
        if x.shape[0] == 0:
            return
        x = x.detach().to("cpu", torch.float64)
        n_batch, dim = x.shape

        batch_mean = x.mean(dim=0)
        if self._sum_x is None:
            self.hidden_dim = dim
            self._sum_x = x.sum(dim=0)
            self._sum_xx = x.T @ x
            self._m2 = ((x - batch_mean) ** 2).sum(dim=0)
            self.n = n_batch
        else:
            if dim != self.hidden_dim:
                raise ValueError(
                    f"Site width changed mid-collection: {self.hidden_dim} -> {dim}."
                )
            old_mean = self._sum_x / self.n
            self._sum_x += x.sum(dim=0)
            self._sum_xx += x.T @ x
            self.n += n_batch
            # Chan's parallel variance update: combine this batch's M2 with the running one.
            delta = batch_mean - old_mean
            batch_m2 = ((x - batch_mean) ** 2).sum(dim=0)
            self._m2 += batch_m2 + delta ** 2 * (self.n - n_batch) * n_batch / self.n

        self._update_reservoir(x.to(self.dtype))

    def _update_reservoir(self, x: torch.Tensor) -> None:
        """Algorithm R: fill, then replace, so every vector seen is equally likely to be held.

        The replacement draw is taken once per batch over the post-batch count rather than
        per item over the running count -- the vectorization the reference implementation uses.
        """
        if self.reservoir is None:
            self.reservoir = x.new_empty((0, x.shape[1]))

        room = self.reservoir_size - self.reservoir.shape[0]
        if room > 0:
            take = min(room, x.shape[0])
            self.reservoir = torch.cat([self.reservoir, x[:take]], dim=0)
            self.seen += take
            x = x[take:]

        n_batch = x.shape[0]
        if n_batch == 0:
            return
        slots = torch.randint(0, self.seen + n_batch, (n_batch,), generator=self.generator)
        hit = slots < self.reservoir_size
        if hit.any():
            self.reservoir[slots[hit]] = x[hit]
        self.seen += n_batch

    @property
    def mean(self) -> torch.Tensor:
        """``[D]`` float32 mean over every vector seen."""
        self._require(1)
        return (self._sum_x / self.n).float()

    @property
    def std(self) -> torch.Tensor:
        """``[D]`` float32 per-dimension standard deviation (sample, ``n-1``)."""
        self._require(2)
        return torch.sqrt(self._m2 / (self.n - 1) + 1e-10).float()

    @property
    def cov(self) -> torch.Tensor:
        """``[D, D]`` float32 covariance ``E[xx^T] - E[x]E[x]^T``, symmetrized.

        Symmetrizing costs nothing and guarantees the exact symmetry ``torch.linalg.eigh``
        assumes; without it the two triangles can differ in the last bits.
        """
        self._require(2)
        mean = self._sum_x / self.n
        cov = self._sum_xx / self.n - torch.outer(mean, mean)
        return (0.5 * (cov + cov.T)).float()

    def _require(self, k: int) -> None:
        if self.n < k:
            raise ValueError(f"Need at least {k} accumulated samples, have {self.n}.")


def collect_hidden_states(
    model,
    tokenizer,
    adapter,
    texts: Iterable[str],
    *,
    max_samples: int,
    seq_len: int = 512,
    batch_size: int = 8,
    reservoir_size: int = 200_000,
    sites: tuple[str, ...] = SITES,
    include_lm_head: bool = True,
    layers: list[int] | None = None,
    device: str = "cuda:0",
    token_frequencies: bool = True,
    progress: bool = True,
    dtype: torch.dtype = torch.float32,
) -> tuple[dict[str, SiteStats], torch.Tensor | None]:
    """Run ``texts`` through ``model`` and accumulate statistics at every anchoring site.

    Hooks are forward-*pre* hooks on the module the site feeds -- ``q_proj`` for ``pre_qkv``,
    ``o_proj`` for ``pre_o``, the whole ``mlp`` for ``pre_mlp``, ``lm_head`` for ``pre_lm_head``
    (resolved by ``adapter.site_module``) -- so what is recorded is exactly the tensor the anchored
    function is evaluated on at training time. Reading the module's input rather than the preceding
    norm's output also keeps this architecture-agnostic: whichever module produces the site's input,
    the adapter names it.

    Args:
        texts: documents, consumed in order and in batches of ``batch_size``.
        max_samples: stop once *every* site has this many vectors. The check is made between
            batches, so the final counts overshoot by up to one batch.
        seq_len: tokenizer truncation length.
        reservoir_size: raw vectors retained per site for the GMM fit.
        sites: which per-layer sites to collect.
        include_lm_head: also collect the model-level ``pre_lm_head`` site (stored at layer index
            ``num_layers``). Pass ``False`` for every layer group but the last.
        layers: restrict collection to these layer indices (``None`` = all), the memory bound
            behind ``build_artifact(layer_group_size=...)``.
        device: device the input batches are moved to; must be where ``model`` lives.
        token_frequencies: also count non-padding token ids, for frequency-weighted embedding
            sampling. Counts run to ``len(tokenizer)``, which can be shorter than the embedding
            table (Qwen3: 151669 vs 151936) -- the sampler handles the shorter prefix.
        dtype: reservoir storage dtype (see :class:`SiteStats`).

    Returns:
        ``(stats, token_counts)``: statistics keyed ``f"{layer}_{site}"``, and raw token counts
        ``[len(tokenizer)]`` (``None`` when ``token_frequencies=False``).
    """
    num_layers = adapter.num_layers(model)
    layer_indices = list(range(num_layers)) if layers is None else sorted(layers)

    targets: list[tuple[str, torch.nn.Module]] = []
    for layer in layer_indices:
        for site in sites:
            if layer == 0 and site == "pre_qkv":
                continue        # exact at load time from the embedding table; never fitted
            targets.append((site_key(layer, site), adapter.site_module(model, layer, site)))
    if include_lm_head:
        targets.append((site_key(num_layers, LM_HEAD_SITE),
                        adapter.site_module(model, num_layers, LM_HEAD_SITE)))

    stats: dict[str, SiteStats] = {
        key: SiteStats(reservoir_size=reservoir_size, dtype=dtype) for key, _ in targets
    }
    mask_holder: dict[str, torch.Tensor | None] = {"mask": None}

    def make_hook(key: str):
        def hook(module, args, kwargs):
            data = args[0] if args else next(iter(kwargs.values()))
            if isinstance(data, tuple):
                data = data[0]
            if not torch.is_tensor(data):
                return
            mask = mask_holder["mask"]
            if data.dim() == 3:
                if mask is not None and tuple(mask.shape) == tuple(data.shape[:2]):
                    data = data[mask.bool()]                       # [valid, D]
                else:
                    # No usable mask (or a sliced head input): keep every position rather than
                    # mis-aligning the filter.
                    data = data.reshape(-1, data.shape[-1])
            stats[key].update(data.float())
        return hook

    handles = [module.register_forward_pre_hook(make_hook(key), with_kwargs=True)
               for key, module in targets]

    texts = list(texts)
    vocab_size = len(tokenizer)
    token_counts = torch.zeros(vocab_size, dtype=torch.long) if token_frequencies else None

    was_training = model.training
    model.eval()
    iterator = range(0, len(texts), batch_size)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Collecting hidden states")
        except ImportError:                                        # pragma: no cover - optional
            pass

    try:
        with torch.no_grad():
            for start in iterator:
                batch = texts[start:start + batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                   max_length=seq_len).to(device)
                input_ids = inputs["input_ids"]
                mask = inputs.get("attention_mask")
                if mask is None:
                    mask = torch.ones_like(input_ids)
                mask_holder["mask"] = mask

                if token_counts is not None:
                    valid = input_ids[mask.bool()].flatten().cpu()
                    valid = valid[valid < vocab_size]
                    token_counts += torch.bincount(valid, minlength=vocab_size)

                model(**inputs)
                mask_holder["mask"] = None

                if stats and all(s.n >= max_samples for s in stats.values()):
                    break
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    logger.info("Collected %d sites, %d samples at the thinnest site",
                len(stats), min((s.n for s in stats.values()), default=0))
    return stats, token_counts
