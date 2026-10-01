"""The on-disk schema of an LFA p(h) artifact: keys, meta block, load/save, validation.

An artifact is a flat ``dict`` mapping a **site key** ``f"{layer}_{site}"`` to that site's fitted
statistics (``mean``, ``std``, optional PCA basis, optional GMM head), plus two reserved keys:

``"__meta__"``
    provenance -- which model the statistics were collected from, and how many samples went in.
    Never a site; skipped by every site-wise pass (:func:`lfa.quantize.quantize_params` and
    :func:`lfa.quantize.dequantize_params` both pass it through untouched).
``"embedding_lookup"``
    the layer-0 ``input_layernorm(embed_tokens(id))`` table, when one is present. It is exact
    rather than fitted, so it is not a site either.

The layer index of the model-level head site ``pre_lm_head`` is the model's layer *count* (a
2-layer model stores it under ``"2_pre_lm_head"``), which keeps every key parseable by the same
rule while leaving the per-layer sites at their own indices.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .. import __version__
from ..quantize import dequantize_params, quantize_params

__all__ = [
    "SITES",
    "LM_HEAD_SITE",
    "META_KEY",
    "EMBEDDING_LOOKUP_KEY",
    "SELF_GENERATED",
    "site_key",
    "parse_site_key",
    "make_meta",
    "load_artifact",
    "save_artifact",
    "ArtifactModelMismatch",
    "validate_against_model",
]

#: The per-layer sites an artifact can carry statistics for.
SITES = ("pre_qkv", "pre_o", "pre_mlp")
#: The model-level site, stored under layer index ``num_layers``.
LM_HEAD_SITE = "pre_lm_head"

META_KEY = "__meta__"
EMBEDDING_LOOKUP_KEY = "embedding_lookup"

#: ``__meta__["provenance"]`` of an artifact fitted on text the model wrote itself.
SELF_GENERATED = "self-generated"


def site_key(layer: int, site: str) -> str:
    """The artifact key holding ``site``'s statistics for ``layer``."""
    return f"{layer}_{site}"


def parse_site_key(key: str) -> tuple[int, str] | None:
    """``(layer, site)`` for a site key, or ``None`` for a reserved/unparseable key.

    Splits on the first underscore only, so site names may contain underscores.
    """
    layer, _, site = key.partition("_")
    if not site or not layer.isdigit():
        return None
    return int(layer), site


def make_meta(
    model_id: str,
    hidden_size: int,
    num_layers: int,
    sites: list[str],
    n_samples_total: int | None,
    built_with: str = "lfa-anchoring",
    provenance: str | None = None,
    corpus_sha256: str | None = None,
    layer_group_size: int | None = None,
    selfgen_frame: dict | None = None,
) -> dict:
    """The ``__meta__`` block: what the statistics describe, and what built them.

    Args:
        n_samples_total: hidden vectors collected **per site**, not summed across them. Every site
            sees the same token stream, so the per-site counts are equal up to the batch the
            collection stops on. It is what a continual extension reads back when the blocks carry
            no count of their own (:func:`lfa.artifact.extend.extend_artifact`).
        built_with: what collected and fitted these statistics. It defaults to this package, which
            is right for :func:`lfa.artifact.build.build_artifact`; a meta block added to a file
            something else built should say so.
            ``lfa_version`` records which version wrote the block either way, so the two fields do
            not have to answer the same question.
        provenance: :data:`SELF_GENERATED` for an artifact fitted on the model's own text;
            ``None`` for everything fitted on real text.
        corpus_sha256: the hash of the corpus file the statistics were collected on, when it is a
            generated one.
        layer_group_size: how many layers were collected per pass, for the record: it sets the
            build's memory bill, not its result. ``None`` when unknown, or when every layer was
            collected in one pass.
        selfgen_frame: the generation and fit frame of a self-generated artifact
            (:meth:`lfa.selfgen.artifact_corpus.SelfGenOptions.artifact_frame`), which
            :meth:`lfa.recipe.Recipe.warnings` compares with the recipe's calibrated frame.
            ``None`` leaves the key out, as it is for an artifact fitted on real text.
    """
    meta = {
        "model_id": model_id,
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "sites": list(sites),
        "n_samples_total": None if n_samples_total is None else int(n_samples_total),
        "built_with": built_with,
        "lfa_version": __version__,
        # What text the statistics were collected on. `None` is real text (the seed corpus, a
        # domain); SELF_GENERATED is text the model wrote, and `corpus_sha256` then names it.
        "provenance": provenance,
        "corpus_sha256": corpus_sha256,
        "layer_group_size": None if layer_group_size is None else int(layer_group_size),
    }
    if selfgen_frame is not None:
        meta["selfgen_frame"] = dict(selfgen_frame)
    return meta


def load_artifact(path) -> dict:
    """Load an artifact from ``path`` and dequantize it -- the only loader.

    Blockwise-int8 fields (see :mod:`lfa.quantize`) are reconstructed to their original dtype
    once, here, so nothing downstream has to know whether the file was quantized.
    """
    params = torch.load(Path(path), map_location="cpu", weights_only=False)
    dequantize_params(params)
    return params


def save_artifact(params: dict, path, quantize: bool = False) -> Path:
    """Write ``params`` to ``path`` (parents created), optionally blockwise-int8 quantized.

    Quantization is a storage format only and does not touch ``params``: it is applied to a
    shallow copy, and the meta block passes through by identity.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantize_params(params) if quantize else params, path)
    return path


class ArtifactModelMismatch(ValueError):
    """Raised when a p(h) artifact does not describe the model it is about to anchor."""


def _site_input_width(adapter, model, layer: int, site: str) -> int | None:
    """How wide the tensor entering ``site`` is, read off the module the site feeds.

    This is NOT always the model's hidden size: ``pre_o`` is the concatenated attention head
    outputs, ``num_heads * head_dim``, which on Qwen3-0.6B is 2048 against a hidden size of 1024.
    Returns ``None`` when the module cannot be resolved (an out-of-range layer, say) or exposes no
    input width, leaving that site unchecked rather than failed on a guess.
    """
    try:
        module = adapter.site_module(model, layer, site)
    except (ValueError, IndexError, AttributeError):
        return None
    width = getattr(module, "in_features", None)
    if width is not None:
        return int(width)
    for child in module.modules():                 # a composite site (the whole MLP): its first
        if isinstance(child, torch.nn.Linear):     # projection reads the site's input
            return int(child.in_features)
    return None


def validate_against_model(params: dict, model, adapter, model_id: str | None = None) -> None:
    """Check that ``params`` was collected from a model shaped like ``model``.

    Verifies each site's width against the input width of the module that site feeds, the layer
    count (the largest per-layer site index plus one), and -- when the artifact carries a
    ``__meta__`` block -- that the block agrees with the model on hidden size and depth, and with
    ``model_id`` when one is given. An artifact silently mismatched on any of these produces
    anchoring pressure toward the wrong function, so this is a hard failure.

    Raises:
        ArtifactModelMismatch: naming both sides of the first disagreement found.
    """
    embed_tokens, _ = adapter.embed_modules(model)
    hidden_size = embed_tokens.embedding_dim
    expected_layers = adapter.num_layers(model)

    layer_indices = []
    for key, entry in params.items():
        parsed = parse_site_key(key)
        if parsed is None or not isinstance(entry, dict):
            continue
        layer, site = parsed
        mean = entry.get("mean")
        expected_width = _site_input_width(adapter, model, layer, site)
        if torch.is_tensor(mean) and expected_width is not None and mean.shape[0] != expected_width:
            raise ArtifactModelMismatch(
                f"Artifact site {key!r} has width {mean.shape[0]}, but this model's {site} input "
                f"is {expected_width} wide (hidden size {hidden_size})."
            )
        if site != LM_HEAD_SITE:
            layer_indices.append(layer)

    if layer_indices:
        artifact_layers = max(layer_indices) + 1
        if artifact_layers != expected_layers:
            raise ArtifactModelMismatch(
                f"Artifact covers {artifact_layers} layers, but the model has {expected_layers}."
            )

    meta = params.get(META_KEY)
    if meta is None:
        return

    if meta.get("hidden_size") is not None and int(meta["hidden_size"]) != hidden_size:
        raise ArtifactModelMismatch(
            f"Artifact meta declares hidden size {meta['hidden_size']}, but the model's hidden "
            f"size is {hidden_size}."
        )
    if meta.get("num_layers") is not None and int(meta["num_layers"]) != expected_layers:
        raise ArtifactModelMismatch(
            f"Artifact meta declares {meta['num_layers']} layers, but the model has "
            f"{expected_layers}."
        )
    if model_id is not None and meta.get("model_id") != model_id:
        raise ArtifactModelMismatch(
            f"Artifact was collected from model_id {meta.get('model_id')!r}, but it is being used "
            f"with {model_id!r}."
        )
