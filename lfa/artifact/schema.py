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
) -> dict:
    """The ``__meta__`` block: what the statistics describe, and what built them."""
    return {
        "model_id": model_id,
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "sites": list(sites),
        "n_samples_total": None if n_samples_total is None else int(n_samples_total),
        "built_with": "lfa-anchoring",
        "lfa_version": __version__,
    }


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


def validate_against_model(params: dict, model, adapter, model_id: str | None = None) -> None:
    """Check that ``params`` was collected from a model shaped like ``model``.

    Verifies the hidden size (every site's ``mean``), the layer count (the largest per-layer site
    index plus one), and -- when the artifact carries a ``__meta__`` block and ``model_id`` is
    given -- that the two model identifiers agree. An artifact silently mismatched on any of these
    produces anchoring pressure toward the wrong function, so this is a hard failure.

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
        if torch.is_tensor(mean) and mean.shape[0] != hidden_size:
            raise ArtifactModelMismatch(
                f"Artifact site {key!r} has hidden size {mean.shape[0]}, but the model's hidden "
                f"size is {hidden_size}."
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
    if meta is not None and model_id is not None and meta.get("model_id") != model_id:
        raise ArtifactModelMismatch(
            f"Artifact was collected from model_id {meta.get('model_id')!r}, but it is being used "
            f"with {model_id!r}."
        )
