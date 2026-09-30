"""Build a p(h) artifact end to end: load the model, stream a corpus, fit every site, save.

This is the whole offline half of LFA. It runs once per (model, seed corpus) pair and produces the
file the anchor samples from for the rest of the model's life -- no domain data enters it, which is
what makes adaptation itself data-free.

What the artifact does *not* contain is as deliberate as what it does. The layer-0 ``pre_qkv``
lookup table is ``input_layernorm(embed_tokens(id))``, exactly reconstructible from the teacher's
own weights, so only the corpus **token frequencies** are stored (~600 KB against ~300 MB for the
table). And the ``__meta__`` block records which model the statistics describe, so an artifact can
never be silently pointed at a different one.
"""

from __future__ import annotations

import gc
import logging
import math
from pathlib import Path

import torch
from transformers import AutoConfig

from ..adapters import get_adapter
from ..corpus import _available_memory_bytes, load_texts
from ..models import load_teacher, load_tokenizer, resolve_model_path
from ..selfgen.artifact_corpus import SelfGenOptions, write_artifact_corpus
from ..selfgen.generate import generate_texts
from .collect import collect_hidden_states
from .fit import fit_site
from .schema import (
    EMBEDDING_LOOKUP_KEY,
    LM_HEAD_SITE,
    META_KEY,
    SELF_GENERATED,
    SITES,
    make_meta,
    parse_site_key,
    save_artifact,
    validate_against_model,
)

__all__ = ["build_artifact", "build_artifact_self_generated", "choose_layer_group_size"]

logger = logging.getLogger(__name__)

# Per-site seed offsets, so no two sites start their k-means from a correlated draw.
_SITE_SEED_OFFSET = {"pre_qkv": 0, "pre_o": 1, "pre_mlp": 2, LM_HEAD_SITE: 5}


def _site_seed(base_seed: int, layer: int, site: str) -> int:
    return base_seed + layer * 10 + _SITE_SEED_OFFSET.get(site, 9)


def build_artifact(
    model_id: str,
    corpus_path,
    out_path,
    *,
    max_samples: int = 1_500_000,
    seq_len: int = 512,
    batch_size: int = 8,
    reservoir_size: int = 200_000,
    pca_variance: float = 0.95,
    gmm_k: int = 32,
    layer_group_size: int | None = None,
    quantize: bool = True,
    device: str = "cuda:0",
    seed: int = 0,
    dtype: torch.dtype = torch.float16,
    provenance: str | None = None,
    corpus_sha256: str | None = None,
    selfgen_frame: dict | None = None,
) -> Path:
    """Collect, fit and save the p(h) artifact for ``model_id`` over ``corpus_path``.

    Args:
        model_id: model to collect from -- a local path or a Hub id.
        corpus_path: seed corpus: a file or a directory of ``.txt``/``.md``/``.json``/``.jsonl``.
            Instruction records (``prompt``/``response``) are rendered with the tokenizer's chat
            template, so their hidden states reflect the format the model is actually used in.
        out_path: where to write ``distribution_stats.pt``.
        max_samples: hidden vectors to collect per site.
        seq_len: tokenizer truncation length.
        batch_size: documents per forward pass.
        reservoir_size: raw vectors retained per site for the GMM fit. This is what the build
            costs in host RAM -- see below.
        pca_variance: variance the stored basis must span.
        gmm_k: mixture components per site.
        layer_group_size: collect this many layers at a time instead of all at once, then fit and
            free before the next group, at the cost of one corpus pass per group. ``None`` keeps
            every site live at once, which is only viable when the arithmetic below fits. The
            value is recorded in the meta block: the reservoir draws depend on it, so a rebuild
            passes the same value.
        quantize: store the large fields blockwise-int8 (halves the file; dequantized on load).
        device: device to run collection on.
        seed: base seed -- the reservoir's draws and each site's GMM initialization derive from it.
        dtype: storage dtype of the retained reservoir samples. fp16 (the default, and what the
            shipped qwen3-0.6b artifact used) halves the figures below; fp32 keeps the samples
            exact and doubles them.
        provenance: recorded in the meta block. :data:`~lfa.artifact.schema.SELF_GENERATED` when
            ``corpus_path`` is text the model wrote (see :func:`build_artifact_self_generated`);
            ``None`` for real text.
        corpus_sha256: the hash of a generated corpus, recorded in the meta block beside
            ``provenance``.
        selfgen_frame: the generation and fit frame of a self-generated corpus
            (:meth:`lfa.selfgen.artifact_corpus.SelfGenOptions.artifact_frame`), recorded in the
            meta block so :meth:`lfa.recipe.Recipe.warnings` can compare it with the frame the
            recipe was calibrated at. ``None`` for real text.

    Returns:
        The path written.

    Note:
        **Host RAM is the binding constraint, and it is the reservoirs.** One site costs
        ``reservoir_size * D * itemsize`` bytes, where ``D`` is that site's own width -- which for
        ``pre_o`` is ``num_heads * head_dim``, twice the hidden size on Qwen3-0.6B. For that model
        (28 layers, hidden 1024, ``pre_o`` 2048) at the default 200k reservoir in fp16: 56 sites of
        width 1024 at 0.41 GB, plus 28 ``pre_o`` sites at 0.82 GB, is **~46 GB in one group** (~92 GB
        in fp32) -- so ``layer_group_size=None`` is *not* what to run it with. ``layer_group_size=7``
        holds 14 narrow sites and 7 wide ones, **~12 GB per group** in fp16, in four corpus passes.
        The float64 covariance accumulators add ``D * D * 8`` bytes per site (~1.6 GB across all 84
        sites of that model), which is small beside the reservoirs but not nothing.

    Note:
        The model is loaded in float32 rather than bf16: the artifact is a second-moment estimate,
        and bf16's 8-bit mantissa is a large error on a covariance. For a model too large to hold
        in fp32, call :func:`~lfa.artifact.collect.collect_hidden_states` directly with a model
        loaded as you please.
    """
    torch.manual_seed(seed)
    out_path = Path(out_path)

    tokenizer = load_tokenizer(model_id)
    model = load_teacher(model_id, device=device, dtype=torch.float32)
    adapter = get_adapter(model)
    num_layers = adapter.num_layers(model)

    texts = load_texts(corpus_path, tokenizer=tokenizer)
    if not texts:
        raise ValueError(f"No texts found in {corpus_path}")
    logger.info("Building p(h) artifact for %s from %d documents", model_id, len(texts))

    if layer_group_size is None:
        groups: list[list[int]] = [list(range(num_layers))]
    else:
        groups = [list(range(start, min(start + layer_group_size, num_layers)))
                  for start in range(0, num_layers, layer_group_size)]

    params: dict = {}
    token_counts: torch.Tensor | None = None

    for index, group in enumerate(groups):
        stats, counts = collect_hidden_states(
            model, tokenizer, adapter, texts,
            max_samples=max_samples, seq_len=seq_len, batch_size=batch_size,
            reservoir_size=reservoir_size, sites=SITES,
            include_lm_head=(index == len(groups) - 1), layers=group,
            device=device, token_frequencies=(index == 0), dtype=dtype, seed=seed,
        )
        if counts is not None:
            token_counts = counts

        for key, site_stats in stats.items():
            layer, site = parse_site_key(key)
            params[key] = fit_site(site_stats, pca_variance=pca_variance, gmm_k=gmm_k,
                                   seed=_site_seed(seed, layer, site), device=device)
        stats.clear()
        gc.collect()

    site_keys = [k for k in params if parse_site_key(k) is not None]
    # A PER-SITE count, not a cross-site sum: every site sees the same token stream, and this is
    # what a later extension reads back as each block's own count. Collection stops between
    # batches, so the counts can differ by up to one batch; the largest is the one to record.
    site_counts = [params[k]["n_samples"] for k in site_keys]
    if len(set(site_counts)) > 1:
        logger.info("Site sample counts differ (%d-%d, one batch of slack); recording %d",
                    min(site_counts), max(site_counts), max(site_counts))
    params[META_KEY] = make_meta(
        model_id=model_id,
        hidden_size=model.config.hidden_size,
        num_layers=num_layers,
        sites=list(SITES) + [LM_HEAD_SITE],
        n_samples_total=max(site_counts),
        provenance=provenance,
        corpus_sha256=corpus_sha256,
        layer_group_size=layer_group_size,
        selfgen_frame=selfgen_frame,
    )
    if token_counts is not None:
        frequencies = token_counts.float()
        params[EMBEDDING_LOOKUP_KEY] = {"token_frequencies": frequencies / frequencies.sum()}

    # Fail here rather than at the first training step: an artifact that does not describe this
    # model anchors toward the wrong function, and does so silently.
    validate_against_model(params, model, adapter, model_id=model_id)

    path = save_artifact(params, out_path, quantize=quantize)
    logger.info("Wrote %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


#: The group size documented for Qwen3-0.6B (~12 GB per group in fp16), used when host memory
#: cannot be read.
_DOCUMENTED_LAYER_GROUP_SIZE = 7


def choose_layer_group_size(hidden_size: int, pre_o_width: int, num_layers: int,
                            reservoir_size: int, itemsize: int, available_bytes: int | None,
                            share: float = 0.5) -> int:
    """The most layers whose reservoirs fit in ``share`` of ``available_bytes``, at least one.

    One layer holds two sites of width ``hidden_size`` (``pre_qkv``, ``pre_mlp``) and one of width
    ``pre_o_width``, so it costs ``reservoir_size * (2 * hidden_size + pre_o_width) * itemsize``
    bytes; the LM-head site rides on the last group and is inside the margin ``share`` leaves.
    With ``available_bytes`` unreadable (``None``) this returns 7, the Qwen3-0.6B setting
    documented on :func:`build_artifact`.

    The choice is not neutral: grouping changes the reservoir draws and the fitted mixtures, not
    the exact moments. So the value chosen here depends on the host's free memory at run time,
    and the artifact's meta records it (``layer_group_size``); to reproduce a build, pass
    ``--layer-group-size`` with the value in the artifact's meta.
    """
    if available_bytes is None:
        return _DOCUMENTED_LAYER_GROUP_SIZE
    per_layer = reservoir_size * (2 * hidden_size + pre_o_width) * itemsize
    return max(1, min(num_layers, math.floor(share * available_bytes / per_layer)))


def _auto_layer_group_size(model_id: str, reservoir_size: int, itemsize: int) -> int:
    """:func:`choose_layer_group_size` for ``model_id``, read from its config (no model load)."""
    config = AutoConfig.from_pretrained(resolve_model_path(model_id))
    hidden_size = config.hidden_size
    head_dim = getattr(config, "head_dim", None) or hidden_size // config.num_attention_heads
    pre_o_width = config.num_attention_heads * head_dim
    num_layers = config.num_hidden_layers
    available = _available_memory_bytes()
    size = choose_layer_group_size(hidden_size, pre_o_width, num_layers, reservoir_size,
                                   itemsize, available)
    per_group = min(size, num_layers) * reservoir_size * (2 * hidden_size + pre_o_width) * itemsize
    logger.info("layer_group_size=%d for %s: ~%.1f GiB of reservoirs per group against %s "
                "available", size, model_id, per_group / 2**30,
                "unreadable memory" if available is None else f"{available / 2**30:.1f} GiB")
    return size


def build_artifact_self_generated(
    model_id: str,
    out_path,
    options: SelfGenOptions | None = None,
    *,
    corpus_path=None,
    quantize: bool = True,
    seed: int = 0,
    writer=None,
    generate=generate_texts,
) -> Path:
    """Write a corpus with the model itself, then fit p(h) on it at the recorded frame.

    The corpus lands at ``corpus_path`` (default ``<out>.corpus.jsonl``) with its manifest, so
    what the artifact was fitted on is inspectable. The artifact's meta carries
    ``provenance = "self-generated"`` and the corpus hash; :meth:`lfa.recipe.Recipe.warnings`
    reads both.

    ``options`` defaults to :class:`SelfGenOptions`, the recorded frame
    (2,500 raw documents, no chat-format share; 600k samples per site; K=32). Scale it down for a
    smoke run. When ``options.layer_group_size`` is ``None`` the group is chosen from the model's
    config and the host's available memory (:func:`choose_layer_group_size`), and the choice is
    logged.
    Grouping changes the reservoir draws and the fitted mixtures, not the exact moments, so the
    value used (chosen or given) is recorded in the artifact's meta as ``layer_group_size``; to
    reproduce a build, pass ``--layer-group-size`` with that value.
    """
    options = options or SelfGenOptions()
    out_path = Path(out_path)
    corpus_path = Path(corpus_path) if corpus_path else out_path.with_suffix(".corpus.jsonl")
    manifest = write_artifact_corpus(model_id, corpus_path, options, generate=generate,
                                     writer=writer)
    build_kwargs = options.build_kwargs()
    if build_kwargs["layer_group_size"] is None:
        # The reservoirs are stored at build_artifact's default dtype, fp16.
        build_kwargs["layer_group_size"] = _auto_layer_group_size(
            model_id, options.reservoir_size, torch.float16.itemsize)
    return build_artifact(model_id, corpus_path, out_path, quantize=quantize, seed=seed,
                          provenance=SELF_GENERATED, corpus_sha256=manifest["corpus_sha256"],
                          selfgen_frame=options.artifact_frame(), **build_kwargs)
