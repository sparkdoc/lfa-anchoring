"""Blockwise int8 (de)quantization for p(h) distribution artifacts — a STORAGE format.

The shipped LFA artifact is dominated by the fp16 PCA basis (`pca_components`, ~208 MB for
qwen3-0.6b-gmm1543k) plus the GMM parameters (~18 MB). Blockwise-int8 halves those to ~113 MB
with ~1% covariance error (measured: int8 blockwise-64 = 1.03%, the best 8-bit option — FP8 is
4× worse because orthonormal basis entries have no dynamic range for exponent bits to spend).

This is quantization for STORAGE ONLY: the artifact is dequantized once, on load
(`dequantize_params`), back to its original dtype in memory. The hot sampling path is untouched,
so a run using the int8 file is identical to the fp16 run up to the ~1% round-trip error — which
is exactly the quantity the validation training arm tests downstream.

Format: a quantized field is a dict `{"__q8__": True, "q8": int8[n_blocks, block],
"scales": fp16[n_blocks], "shape": ..., "numel": ..., "block": ..., "dtype": ...}`.
"""
from __future__ import annotations

import torch

DEFAULT_BLOCK = 64
# fields worth quantizing (the large fp tensors); moments/eigenvalues are negligible and kept exact
QUANTIZABLE = ("pca_components", "gmm_means", "gmm_covariances")


def quantize_blockwise(t: torch.Tensor, block: int = DEFAULT_BLOCK) -> dict:
    """Blockwise absmax int8: split the flattened tensor into `block`-sized groups, each with its
    own fp16 scale = absmax/127. Symmetric (zero-point 0), so orthonormal/zero-mean data stays
    zero-mean. Padding to a multiple of `block` is stripped at dequant via `numel`."""
    orig_dtype = str(t.dtype).replace("torch.", "")
    flat = t.detach().float().reshape(-1)
    numel = flat.numel()
    pad = (-numel) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, block)
    scales = blocks.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q = (blocks / scales[:, None]).round().clamp_(-127, 127).to(torch.int8)
    return {"__q8__": True, "q8": q, "scales": scales.half(),
            "shape": tuple(t.shape), "numel": numel, "block": block, "dtype": orig_dtype}


def dequantize_blockwise(d: dict) -> torch.Tensor:
    """Inverse of `quantize_blockwise` → tensor in the original dtype and shape."""
    flat = (d["q8"].float() * d["scales"].float()[:, None]).reshape(-1)[: d["numel"]]
    out = flat.reshape(d["shape"])
    return out.to(getattr(torch, d["dtype"]))


def is_quantized_field(v) -> bool:
    return isinstance(v, dict) and v.get("__q8__") is True


def dequantize_params(params: dict) -> dict:
    """In place: reconstruct every quantized field in every sub-module entry. No-op on fp artifacts.

    The top-level `"__meta__"` block is a dict of scalars/strings/lists, not a site, and is skipped.
    """
    for key, entry in params.items():
        if key == "__meta__":
            continue
        if not isinstance(entry, dict):
            continue
        for field, v in list(entry.items()):
            if is_quantized_field(v):
                entry[field] = dequantize_blockwise(v)
    return params


def quantize_params(params: dict, fields=QUANTIZABLE, block: int = DEFAULT_BLOCK) -> dict:
    """New params dict with `fields` blockwise-int8 quantized (entries copied shallowly)."""
    out = {}
    for key, entry in params.items():
        if not isinstance(entry, dict):
            out[key] = entry
            continue
        e = dict(entry)
        for field in fields:
            if field in e and torch.is_tensor(e[field]):
                e[field] = quantize_blockwise(e[field], block=block)
        out[key] = e
    return out
