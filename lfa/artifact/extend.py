"""Extend a p(h) artifact with a new domain, without the base corpus.

This is what makes LFA *continual*. Stage two anchors on ``base + A``, so its p(h) has to describe
that model -- but the base artifact describes the base alone, and the corpus it was built from is
not something a tuner has (nor wants to re-run: it is a multi-hour, million-vector collection).
The alternative to re-collecting is to add only what is new: run the domain the tuner *does* own
through the **fused** model, fit those activations as a small mixture **in the base's own PCA
basis**, and take the union of the two mixtures, weighted by their sample counts.

    p(h) = (1 - a) * p_base(h) + a * p_domain(h),    a = n_domain / (n_base + n_domain)

That union is exact -- it is the distribution of "draw from the base pool with probability
``1-a``, else from the domain pool" -- so no refitting is involved and the merged artifact is a
sufficient statistic for the next stage as well. Each round costs one pass over the new domain and
adds ``k_domain`` components per site: **O(1) in the number of rounds**, against replay's
requirement to rehearse every prior corpus at every stage. Only the new domain is ever sampled, so
the adaptation stays data-free with respect to everything that came before.

**The domain head is emitted in the base entry's own coordinates**, because
:func:`lfa.merge.merge_gmm_blocks` concatenates the two component sets and keeps the *base*
entry's mean and basis. The base head is diagonal over un-whitened PCA coordinates
``(h - mean) @ V``, the only kind :func:`lfa.artifact.fit.fit_site` writes. The fit itself runs on
whitened coordinates ``((h - mean) @ V) / sqrt(eig)`` -- EM on raw PCA coordinates is badly
conditioned when the eigenvalues span orders of magnitude -- and the result is then un-whitened
back into the base's frame. Left whitened, every domain component would be rescaled by
``sqrt(eigenvalue)`` per coordinate against the base's; nothing downstream would raise.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch

from ..adapters import get_adapter
from ..corpus import load_corpus
from ..merge import merge_stats
from ..models import load_teacher, load_tokenizer
from .fit import TorchGMM
from .schema import META_KEY, load_artifact, parse_site_key, save_artifact, validate_against_model

__all__ = ["fit_domain_gmm", "extend_artifact", "gmm_site_keys", "require_site_counts"]

logger = logging.getLogger(__name__)

#: Minimum activations per fitted component. A mixture fitted on fewer is noise given a shape.
_SAMPLES_PER_COMPONENT = 200


def fit_domain_gmm(
    H: torch.Tensor,
    base_entry: dict,
    k: int,
    seed: int = 0,
    device: str = "cuda:0",
) -> dict:
    """Fit ``H`` as a small diagonal mixture in ``base_entry``'s basis and coordinates.

    Args:
        H: ``[N, D]`` activations collected at this site through the fused model.
        base_entry: the base artifact's entry for the same site. Its ``mean``,
            ``pca_components`` and ``pca_eigenvalues`` define the coordinates.
        k: components to fit, capped at one per 200 samples.
        seed: EM/k-means seed.
        device: device to fit on.

    Returns:
        A stats block ready for :func:`lfa.merge.merge_stats`: the GMM fields, this domain's own
        diagonal moments (which is what makes it a moment block the merge recognizes), and
        ``n_samples``.
    """
    basis = base_entry["pca_components"].float()                    # [D, n_comp]
    eigenvalues = base_entry["pca_eigenvalues"].float().clamp_min(1e-8)
    mean = base_entry["mean"].float()

    z = ((H.float() - mean) @ basis) / eigenvalues.sqrt()           # whitened base coordinates
    k = min(k, max(1, len(z) // _SAMPLES_PER_COMPONENT))

    # n_init=3 as in the reference implementation: a domain mixture is fitted once per site per
    # stage, so the cheapest insurance against a bad k-means++ draw is worth taking.
    gmm = TorchGMM(n_components=k, max_iter=100, tol=1e-3,
                   reg_covar=1e-4, n_init=3, random_state=seed, device=device,
                   init_params="kmeans").fit(z.to(device))

    weights = gmm.weights_.detach().cpu().float()
    means = gmm.means_.detach().cpu().float()
    covariances = gmm.covariances_.detach().cpu().float()

    # Back to un-whitened base coordinates: a whitened mean scales by sqrt(eig), a whitened
    # variance by eig.
    return {
        "gmm_weights": weights,
        "gmm_means": means * eigenvalues.sqrt(),
        "gmm_covariances": covariances * eigenvalues,
        "gmm_n_components": int(k),
        "gmm_covariance_type": "diag",
        "mean": H.float().mean(0),
        "std": H.float().std(0),
        "n_samples": int(len(H)),
    }


def _collect_domain_activations(
    model,
    tokenizer,
    adapter,
    corpus_path,
    site_keys: list[str],
    *,
    need: int,
    seq_len: int,
    seed: int,
    device: str,
    keep_short_whole: bool,
) -> dict[str, torch.Tensor]:
    """Record ``need`` activations per site, from the stage's own chunked training stream.

    The chunks are the ones training saw -- same loader, same length, same
    ``keep_short_whole`` setting, no validation split -- taken
    in a seeded random order rather than in file order, so the sample spans the whole corpus at the
    proportions it is trained on. (The reference implementation's other mode, the first ~78 files
    each truncated to one chunk, is a thin order-dependent sample and is deliberately not ported.)
    """
    dataset, _ = load_corpus(corpus_path, tokenizer, max_length=seq_len, stride=0,
                             val_fraction=0.0, seed=seed, keep_short_whole=keep_short_whole)
    if len(dataset) == 0:
        raise ValueError(f"No training chunks in {corpus_path}: nothing to collect.")

    # One `need x width` float32 buffer per site, filled in place. The obvious implementation --
    # a list of per-chunk tensors, concatenated at the end -- costs TWICE this, and not only
    # transiently: the chunks are freed but glibc keeps their pages in the arena while every
    # concatenation asks for fresh ones, so the process's resident peak is the sum of both. That
    # is the 35.9 GiB a user measured against the ~18 GB docs/faq.md budgets. Allocating the
    # destination up front makes the budget the arithmetic and the measurement agree on.
    buffers: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {key: 0 for key in site_keys}

    def make_hook(key: str):
        def hook(module, args, kwargs):
            have = counts[key]
            if have >= need:
                return
            data = args[0] if args else next(iter(kwargs.values()))
            if isinstance(data, tuple):
                data = data[0]
            if not torch.is_tensor(data):
                return
            flat = data.detach().reshape(-1, data.shape[-1]).float().cpu()
            buffer = buffers.get(key)
            if buffer is None:
                # The site's width is not known until the first activation arrives.
                buffer = buffers[key] = torch.empty(need, flat.shape[1], dtype=torch.float32)
            take = min(need - have, flat.shape[0])
            buffer[have:have + take] = flat[:take]
            counts[key] = have + take
        return hook

    handles = []
    for key in site_keys:
        layer, site = parse_site_key(key)
        handles.append(adapter.site_module(model, layer, site)
                       .register_forward_pre_hook(make_hook(key), with_kwargs=True))

    # One chunk per forward: no padding, so every recorded position is a real activation.
    order = torch.randperm(len(dataset),
                           generator=torch.Generator().manual_seed(seed)).tolist()
    was_training = model.training
    model.eval()
    n_chunks = 0
    try:
        with torch.no_grad():
            for index in order:
                input_ids = dataset[index]["input_ids"].reshape(1, -1).to(device)
                model(input_ids)
                n_chunks += 1
                if all(count >= need for count in counts.values()):
                    break
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    thinnest = min(counts.values())
    if thinnest < need:
        logger.warning("Corpus exhausted at %d activations per site, %d were asked for",
                       thinnest, need)
    logger.info("Collected %d activations at each of %d sites from %d chunks",
                thinnest, len(site_keys), n_chunks)

    # A site that filled its buffer hands the buffer over; a short one copies out its prefix, so
    # the unfilled remainder is released rather than held by a view for the rest of the call.
    return {key: buffer if counts[key] == need else buffer[:counts[key]].clone()
            for key, buffer in buffers.items() if counts[key]}


def gmm_site_keys(params: dict) -> list[str]:
    """The artifact's sites that carry a mixture -- the ones an extension adds components to.

    A site whose reservoir was too small to fit a head on keeps its PCA basis only
    (:func:`lfa.artifact.fit.fit_site`). Anything asking whether the artifact carries its sample
    counts has to ask about *these* keys, since they are the only ones the merge touches.
    """
    return [key for key, entry in params.items()
            if parse_site_key(key) is not None and isinstance(entry, dict)
            and int(entry.get("gmm_n_components", 0)) > 0]


def require_site_counts(base: dict, gmm_keys: list[str], source) -> None:
    """Refuse a base whose mixture sites carry no ``n_samples``.

    The n-weighted merge reads each block's own count to weight the new domain by its sample
    share. Every artifact this package builds or extends records one per site, so a site without
    it means the file was not built here; the merge would refuse it anyway, but only after the
    collection and the fits.

    Raises:
        ValueError: a mixture site carries no ``n_samples``.
    """
    missing = [key for key in gmm_keys if "n_samples" not in base[key]]
    if missing:
        raise ValueError(
            f"{source} carries no n_samples on {len(missing)} of its {len(gmm_keys)} mixture "
            f"sites (first: {missing[0]}), so the new domain cannot be weighted by its sample "
            "share. Every artifact this package builds records one per site: build it with "
            "`lfa build-artifact` or `lfa init --artifact self-generated`."
        )


def extend_artifact(
    fused_model_path,
    base_artifact_path,
    corpus_path,
    out_path,
    *,
    k_domain: int = 8,
    need: int = 40_000,
    seq_len: int = 512,
    seed: int = 42,
    device: str = "cuda:0",
    quantize: bool = True,
    keep_short_whole: bool = True,
) -> Path:
    """Add a domain to ``base_artifact_path`` and write the extended artifact to ``out_path``.

    Args:
        fused_model_path: the model the domain is collected through -- the *fused* base+A model
            that stage two will anchor, not the base. Its activations are what the new components
            must describe.
        base_artifact_path: the artifact to extend, which this package must have built (itself
            possibly the output of an earlier extension: the merge accumulates).
        corpus_path: the new domain's training corpus, read exactly as training reads it.
        out_path: where to write the extended artifact.
        k_domain: components to fit per site for the new domain, capped at one per 200 samples.
        need: activations to collect per site.
        seq_len: chunk length, which should be the one the stage trains at.
        seed: seeds the chunk order and every site's mixture fit.
        device: device to run the collection and the fits on.
        quantize: store the extended artifact blockwise-int8.
        keep_short_whole: the corpus-chunking frame to collect under. It should be the one the
            stage trained under: the two frames differ in whether a document shorter than the
            epoch's chunk offset appears at all, so collecting under the other one describes a
            training stream the model was not trained on.

    Returns:
        The path written.

    Raises:
        lfa.artifact.schema.ForeignArtifact: the base artifact was not built by this package (no
            meta block, another builder, or a mixture head that is not diagonal).
        ValueError: the base carries no GMM sites, or a mixture site carries no ``n_samples``.
            All three refusals come before any model is loaded.
        lfa.artifact.schema.ArtifactModelMismatch: the base does not describe the fused model.
    """
    out_path = Path(out_path)
    base = load_artifact(base_artifact_path)
    gmm_keys = gmm_site_keys(base)
    if not gmm_keys:
        raise ValueError(f"{base_artifact_path} has no GMM sites to extend.")

    require_site_counts(base, gmm_keys, base_artifact_path)

    tokenizer = load_tokenizer(str(fused_model_path))
    # float32 for the same reason the build collects in it: these vectors become a second-moment
    # estimate, and bf16's 8-bit mantissa is a large error on a covariance.
    model = load_teacher(str(fused_model_path), device=device, dtype=torch.float32)
    adapter = get_adapter(model)
    # An artifact that does not describe this model would be extended with activations from the
    # wrong function -- silently, since every shape downstream still matches the base.
    validate_against_model(base, model, adapter)

    activations = _collect_domain_activations(
        model, tokenizer, adapter, corpus_path, gmm_keys,
        need=need, seq_len=seq_len, seed=seed, device=device,
        keep_short_whole=keep_short_whole,
    )
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    if not activations:
        raise ValueError(
            "no activations collected -- is the corpus empty or every chunk shorter than 10 "
            f"tokens? ({corpus_path})"
        )

    # Fitting is the long half of an extension and used to print nothing at all: on the shipped
    # artifact it is 84 mixtures and several minutes, which is indistinguishable from a hang.
    domain_stats = {}
    total = len(activations)
    # Insertion order, not sorted: each site's fit is seeded on its own, but changing the
    # order of a stream of fits is exactly the kind of silent difference this package
    # exists to avoid, and the port-verification fixtures were captured in this order.
    for position, (key, H) in enumerate(activations.items(), start=1):
        if position == 1 or position == total or position % 10 == 0:
            logger.info("Fitting the domain mixture: site %d/%d (%s)", position, total, key)
        domain_stats[key] = fit_domain_gmm(H, base[key], k_domain, seed=seed, device=device)

    merged = merge_stats(base, domain_stats)

    meta = dict(base[META_KEY])
    meta["version"] = int(meta.get("version", 1)) + 1
    meta["extended_with"] = list(meta.get("extended_with", [])) + [Path(corpus_path).name]
    # Recomputed rather than carried, and per-site (the base count plus this round's `need`), so
    # the record matches the blocks it describes.
    site_counts = [entry["n_samples"] for key, entry in merged.items()
                   if parse_site_key(key) is not None and isinstance(entry, dict)
                   and "n_samples" in entry]
    meta["n_samples_total"] = max(site_counts)
    merged[META_KEY] = meta

    components = [block["gmm_n_components"] for block in domain_stats.values()]
    n_domain = next(iter(domain_stats.values()))["n_samples"]
    logger.info("Extended %d sites with K %d-%d new components from %d activations "
                "(domain weight share %.4f); wrote version %d",
                len(domain_stats), min(components), max(components),
                n_domain, n_domain / merged[gmm_keys[0]]["n_samples"], meta["version"])
    return save_artifact(merged, out_path, quantize=quantize)
