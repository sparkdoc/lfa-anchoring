"""The anchoring losses: what Layerwise Function Anchoring actually optimizes.

Three terms live here, and they are different objects:

* **The function anchor** (:func:`anchor_loss`) -- ``E_{h~p(h)} ||f_s(h) - f_t(h)||^2`` over the
  sub-modules LFA preserves, with ``h`` drawn from the fitted p(h) by a
  :class:`~lfa.sampler.Sampler`. This is the method: it prices a *function* on the distribution
  the model actually visits, so preservation effort lands where the model works, not uniformly
  over weight space. It is scaled by lambda at the call site.
* **The embedding anchor** (:func:`embed_anchor_loss`) -- the same idea for the composed
  ``input_layernorm(embed_tokens(id))`` function, sampled over token IDs by corpus frequency.
* **The weight term** (:func:`weight_loss`) -- plain ``||W_s - W_t||^2_F``, scaled by mu. NOT the
  anchor and not a cheap approximation of it: it is the isotropic-Gaussian degenerate case, kept
  as a global-shrinkage backstop against the non-destructive drift the function anchor is blind
  to. Under LoRA it has an exact factored form; see :func:`lora_factored_weight_loss`.

Anything the artifact has no statistics for is left **unanchored** rather than anchored on noise:
:meth:`~lfa.sampler.Sampler.sample_best` returns ``None`` for an absent site and that site's
contribution is skipped.

All model layout goes through a :class:`~lfa.adapters.ModelAdapter`, and every module output is
compared in float32 on the *student's* device, so the gradient path stays on the student even when
teacher and student sit on different GPUs.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

from .adapters import ModelAdapter, effective_weight
from .artifact.schema import LM_HEAD_SITE
from .sampler import Sampler

logger = logging.getLogger(__name__)

__all__ = [
    "ANCHOR_SCHEDULES",
    "schedule_weight",
    "compute_layer_weights",
    "qkv_anchor_loss",
    "mlp_anchor_loss",
    "lm_head_anchor_loss",
    "embed_anchor_loss",
    "anchor_loss",
    "lora_factored_weight_loss",
    "weight_loss",
]

#: Shapes :func:`compute_layer_weights` can interpolate with.
ANCHOR_SCHEDULES = ("cosine", "linear", "exponential")

_QKV_NAMES = ("q_proj", "k_proj", "v_proj")
_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

#: One warning per process when the artifact carries no ``pre_lm_head`` statistics.
_warned_no_lm_head_site = False


def schedule_weight(t: float, end_ratio: float, schedule: str) -> float:
    """The anchoring weight at normalized depth ``t``.

    Every schedule satisfies ``w(0) = 1.0`` and ``w(1) = end_ratio``.

    Args:
        t: normalized position in ``[0, 1]``; 0 is the first layer, 1 the last.
        end_ratio: weight at ``t = 1`` relative to ``t = 0``.
        schedule: one of :data:`ANCHOR_SCHEDULES`.

    Raises:
        ValueError: for an unknown schedule name.
    """
    if schedule == "linear":
        return 1.0 - (1.0 - end_ratio) * t
    if schedule == "cosine":
        return end_ratio + (1.0 - end_ratio) * (1.0 + math.cos(math.pi * t)) / 2.0
    if schedule == "exponential":
        return end_ratio ** t
    raise ValueError(f"Unknown schedule {schedule!r}. Must be one of {ANCHOR_SCHEDULES}")


def compute_layer_weights(
    num_layers: int,
    end_ratio: float = 0.1,
    schedule: str = "cosine",
    normalize: bool = True,
) -> list[float]:
    """Per-layer anchoring weights, decaying from 1.0 at layer 0 to ``end_ratio`` at the last.

    .. warning::

       **This knob is scale compensation, not a hierarchy** (measured 2026-07-25). The original
       rationale -- anchor early layers hard so new knowledge piggybacks on shared structure --
       is the reverse of what the schedule does. The anchor is an *absolute* per-element MSE and
       teacher activation scale ``E||f_t(h)||^2`` grows enormously with depth (layer 0 -> 27 on
       Qwen3-0.6B, n=32768: pre_qkv 1703x, pre_o 165x, pre_mlp 11578x), so effective *relative*
       pressure is ``w(l) * scale(l, site)`` -- which, even with a 10x decay, still rises 17x-1191x
       with depth. Realized drift confirms it: unanchored relative drift rises with depth
       (2.2-4.2x) while anchored drift falls (0.00-0.05x). Two consequences: a per-*layer* schedule
       structurally cannot equalize relative pressure, because the scale growth differs ~70x across
       *sites*; and the recipe may work partly because of this accidental depth weighting. Four
       reallocation schemes were tried and none beat the hand-set schedule, so do not "fix" it
       without re-tuning lambda.

    Args:
        num_layers: number of transformer layers.
        end_ratio: last layer's weight relative to layer 0 (``1.0`` = uniform, no decay).
        schedule: interpolation shape, one of :data:`ANCHOR_SCHEDULES`.
        normalize: rescale the weights to **sum to 1.0**.

            .. note::

               This is the convention the published lambda is calibrated against, so do not change
               it: with ``normalize=True`` a scheduled anchor is ``num_layers`` times smaller than
               the same anchor under the uniform default (``layer_weights=None``, all ones), and
               lambda absorbs that factor. ``compute_layer_weights(4, 0.1, "cosine")`` is
               ``[0.3774, 0.3276, 0.2075, 0.0875]``.

    Returns:
        ``[w_0, ..., w_{L-1}]``.
    """
    weights = [schedule_weight(l / num_layers, end_ratio, schedule) for l in range(num_layers)]

    if normalize:
        total = sum(weights)
        weights = [w / total for w in weights]

    return weights


def _module_io(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    """The device and dtype a module's parameters live in."""
    param = next(module.parameters())
    return param.device, param.dtype


def _site_mse(
    t_module: nn.Module,
    s_module: nn.Module,
    h: torch.Tensor,
    t_device: torch.device,
    t_dtype: torch.dtype,
    s_device: torch.device,
    s_dtype: torch.dtype,
) -> torch.Tensor:
    """``MSE(f_s(h), f_t(h))`` for one sub-module, in float32 on the student's device.

    The modules are *called* rather than multiplied out, so any adapter delta (LoRA,
    ``modules_to_save``) is part of the anchored function exactly as it is in the forward pass.
    """
    with torch.no_grad():
        t_out = t_module(h.to(device=t_device, dtype=t_dtype))
    s_out = s_module(h.to(device=s_device, dtype=s_dtype))
    return F.mse_loss(s_out.float(), t_out.float().to(s_device))


def qkv_anchor_loss(
    teacher: nn.Module,
    student: nn.Module,
    sampler: Sampler,
    adapter: ModelAdapter,
    n_samples: int = 16,
    layer_weights: list[float] | None = None,
    include_o_proj: bool = True,
) -> torch.Tensor:
    """Attention-projection anchoring: ``sum_l w(l) * E_h ||W_s h - W_t h||^2`` over q, k, v, o.

    ``q/k/v`` read the ``pre_qkv`` site and ``o_proj`` reads ``pre_o`` -- different distributions,
    so they are sampled separately. The result is averaged over ``num_layers * 4`` projections
    (``* 3`` when ``include_o_proj`` is False), which keeps its magnitude independent of depth.

    Returns:
        Scalar float32 tensor on the teacher's device.
    """
    num_layers = adapter.num_layers(teacher)
    device = next(teacher.parameters()).device

    if layer_weights is None:
        layer_weights = [1.0] * num_layers

    total = torch.zeros((), device=device, dtype=torch.float32)
    n_projections = 4 if include_o_proj else 3

    for layer_idx in range(num_layers):
        h = sampler.sample_best(layer_idx, "pre_qkv", n_samples)
        if h is None:  # no statistics for this site -> leave it unanchored
            continue

        t_modules = adapter.qkv_modules(teacher, layer_idx)
        s_modules = adapter.qkv_modules(student, layer_idx)

        t_device, t_dtype = _module_io(t_modules["q_proj"])
        s_device, s_dtype = _module_io(s_modules["q_proj"])

        layer_loss = torch.zeros((), device=s_device, dtype=torch.float32)
        for name in _QKV_NAMES:
            layer_loss = layer_loss + _site_mse(
                t_modules[name], s_modules[name], h, t_device, t_dtype, s_device, s_dtype
            )

        if include_o_proj:
            h_o = sampler.sample_best(layer_idx, "pre_o", n_samples)
            if h_o is not None:
                layer_loss = layer_loss + _site_mse(
                    adapter.o_proj_module(teacher, layer_idx),
                    adapter.o_proj_module(student, layer_idx),
                    h_o, t_device, t_dtype, s_device, s_dtype,
                )

        total = total + layer_weights[layer_idx] * layer_loss.to(device)

    return total / (num_layers * n_projections)


def mlp_anchor_loss(
    teacher: nn.Module,
    student: nn.Module,
    sampler: Sampler,
    adapter: ModelAdapter,
    n_samples: int = 16,
    layer_weights: list[float] | None = None,
) -> torch.Tensor:
    """MLP anchoring: ``sum_l w(l) * E_h ||MLP_s(h) - MLP_t(h)||^2``, averaged over layers.

    The MLP is anchored as a whole function, nonlinearity included -- that is where the importance
    weighting lives, since SiLU amplifies some directions and suppresses others, and a per-matrix
    anchor cannot see it.

    Returns:
        Scalar float32 tensor on the teacher's device.
    """
    num_layers = adapter.num_layers(teacher)
    device = next(teacher.parameters()).device

    if layer_weights is None:
        layer_weights = [1.0] * num_layers

    total = torch.zeros((), device=device, dtype=torch.float32)

    for layer_idx in range(num_layers):
        h = sampler.sample_best(layer_idx, "pre_mlp", n_samples)
        if h is None:  # no statistics for this site -> leave it unanchored
            continue

        t_mlp = adapter.mlp_module(teacher, layer_idx)
        s_mlp = adapter.mlp_module(student, layer_idx)
        t_device, t_dtype = _module_io(t_mlp)
        s_device, s_dtype = _module_io(s_mlp)

        layer_loss = _site_mse(t_mlp, s_mlp, h, t_device, t_dtype, s_device, s_dtype)
        total = total + layer_weights[layer_idx] * layer_loss.to(device)

    return total / num_layers


def lm_head_anchor_loss(
    teacher: nn.Module,
    student: nn.Module,
    sampler: Sampler,
    adapter: ModelAdapter,
    n_samples: int = 16,
) -> torch.Tensor:
    """LM-head anchoring: ``E_h ||lm_head_s(h) - lm_head_t(h)||^2`` on ``pre_lm_head`` samples.

    ``pre_lm_head`` is the output of the final norm, stored under layer index ``num_layers``. An
    artifact fitted without that site (a diagonal-moments-only build, say) simply leaves the head
    unanchored: this returns 0 and warns once, rather than failing a training run.

    Returns:
        Scalar float32 tensor on the teacher's device.
    """
    global _warned_no_lm_head_site

    num_layers = adapter.num_layers(teacher)
    device = next(teacher.parameters()).device

    if not sampler.has_site(num_layers, LM_HEAD_SITE):
        if not _warned_no_lm_head_site:
            _warned_no_lm_head_site = True
            logger.warning(
                "No %r statistics in the p(h) artifact (expected key %d_%s); the LM head will be "
                "left unanchored. Collect that site to anchor it.",
                LM_HEAD_SITE, num_layers, LM_HEAD_SITE,
            )
        return torch.zeros((), device=device, dtype=torch.float32)

    h = sampler.sample_best(num_layers, LM_HEAD_SITE, n_samples)
    if h is None:  # pragma: no cover - has_site() just said otherwise
        return torch.zeros((), device=device, dtype=torch.float32)

    t_lm_head = adapter.lm_head_module(teacher)
    s_lm_head = adapter.lm_head_module(student)
    t_device, t_dtype = _module_io(t_lm_head)
    s_device, s_dtype = _module_io(s_lm_head)

    loss = _site_mse(t_lm_head, s_lm_head, h, t_device, t_dtype, s_device, s_dtype)
    return loss.to(device)


def embed_anchor_loss(
    teacher: nn.Module,
    student: nn.Module,
    sampler: Sampler,
    adapter: ModelAdapter,
    n_samples: int = 16,
) -> torch.Tensor:
    """Embedding anchoring: ``E_{id ~ freq} ||ln_s(emb_s(id)) - ln_t(emb_t(id))||^2``.

    Anchors ``embed_tokens.weight`` (which is ``lm_head.weight`` when tied) together with layer 0's
    ``input_layernorm`` as one composed function. The teacher reference is recomputed at runtime
    rather than read from the artifact's lookup table, so the term is exactly zero at
    initialization even when the shipped table was quantized.

    Token IDs are drawn by corpus frequency when the artifact carries frequencies, else uniformly
    over the vocabulary. That draw uses the **global** RNG (as in the reference implementation),
    not the sampler's private generator, so seeding the :class:`~lfa.sampler.Sampler` alone does
    not make this term reproducible -- seed torch itself.

    Raises:
        ValueError: if the sampler has no embedding lookup. Build one from the teacher with
            :meth:`~lfa.sampler.Sampler.build_embedding_lookup_from_model`.

    Returns:
        Scalar float32 tensor on the teacher's device.
    """
    device = next(teacher.parameters()).device

    if not sampler.has_embedding_lookup():
        raise ValueError(
            "Embedding anchoring needs the layer-0 lookup entry (for the vocabulary size and "
            "token frequencies). Call Sampler.build_embedding_lookup_from_model(model, adapter) "
            "first, or disable the term with include_embed=False."
        )

    lookup = sampler.params["embedding_lookup"]
    if "token_frequencies" in lookup:
        token_freqs = lookup["token_frequencies"].to(device)
        token_ids = torch.multinomial(token_freqs, n_samples, replacement=True)
    else:
        token_ids = torch.randint(0, lookup["vocab_size"], (n_samples,), device=device)

    t_embed, t_ln = adapter.embed_modules(teacher)
    s_embed, s_ln = adapter.embed_modules(student)
    t_device = next(t_embed.parameters()).device
    s_device = next(s_embed.parameters()).device

    with torch.no_grad():
        t_out = t_ln(t_embed(token_ids.to(t_device)))
    s_out = s_ln(s_embed(token_ids.to(s_device)))

    loss = F.mse_loss(s_out.float(), t_out.float().to(s_device))
    return loss.to(device)


def anchor_loss(
    teacher: nn.Module,
    student: nn.Module,
    sampler: Sampler,
    adapter: ModelAdapter,
    *,
    n_samples: int = 16,
    layer_weights: list[float] | None = None,
    include_qkv: bool = True,
    include_mlp: bool = True,
    include_lm_head: bool = True,
    include_embed: bool = True,
    qkv_weight: float = 1.0,
    mlp_weight: float = 1.0,
    lm_head_weight: float = 1.0,
    embed_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """The full function anchor: QKV + MLP + LM head + embedding.

    ``L_anchor = qkv_weight * L_qkv + mlp_weight * L_mlp + lm_head_weight * L_lm_head
    + embed_weight * L_embed``, with lambda applied by the caller.

    Args:
        teacher: the frozen reference model.
        student: the model being trained.
        sampler: draws ``h ~ p(h)`` for each site.
        adapter: resolves the sub-modules to anchor.
        n_samples: samples per site per layer. The loss is mean-reduced, so this is unbiased at
            every value and trades variance for cost only.
        layer_weights: per-layer weights (see :func:`compute_layer_weights`); ``None`` = uniform.
        include_*: switch a block off entirely (its key is then absent from the result).
        *_weight: relative weight of each block in ``"total"``.

    Returns:
        ``{"total": ..., "qkv": ..., "mlp": ..., "lm_head": ..., "embed": ...}`` -- every included
        block plus the combined total, all scalar float32 tensors on the teacher's device.
    """
    device = next(teacher.parameters()).device
    results: dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=device, dtype=torch.float32)

    if include_qkv:
        results["qkv"] = qkv_anchor_loss(
            teacher, student, sampler, adapter, n_samples, layer_weights
        )
        total = total + qkv_weight * results["qkv"]

    if include_mlp:
        results["mlp"] = mlp_anchor_loss(
            teacher, student, sampler, adapter, n_samples, layer_weights
        )
        total = total + mlp_weight * results["mlp"]

    if include_lm_head:
        results["lm_head"] = lm_head_anchor_loss(teacher, student, sampler, adapter, n_samples)
        total = total + lm_head_weight * results["lm_head"]

    if include_embed:
        results["embed"] = embed_anchor_loss(teacher, student, sampler, adapter, n_samples)
        total = total + embed_weight * results["embed"]

    results["total"] = total
    return results


def _target_modules(
    model: nn.Module, adapter: ModelAdapter, layer_idx: int, names: list[str]
) -> list[tuple[str, nn.Module]]:
    """``(name, module)`` for each requested projection in one layer, in the given order."""
    modules: list[tuple[str, nn.Module]] = []
    qkv = None
    mlp = None
    for name in names:
        if name == "o_proj":
            modules.append((name, adapter.o_proj_module(model, layer_idx)))
        elif name in _QKV_NAMES:
            if qkv is None:
                qkv = adapter.qkv_modules(model, layer_idx)
            modules.append((name, qkv[name]))
        else:
            if mlp is None:
                mlp = adapter.mlp_module(model, layer_idx)
            modules.append((name, getattr(mlp, name)))
    return modules


def lora_factored_weight_loss(
    teacher: nn.Module,
    student: nn.Module,
    adapter: ModelAdapter,
    layer_weights: list[float],
) -> torch.Tensor | None:
    """mu's weight loss computed from the LoRA factors, or ``None`` when that is not valid here.

    When the student is a LoRA model whose frozen base weights *are* the teacher's, the difference
    is exactly the adapter delta, so the Frobenius norm needs neither the merged weight nor the
    subtraction::

        ||W_s - W_t||_F^2 = ||s.BA||_F^2 = s^2 . tr((B^T B)(A A^T))

    Two reasons to prefer it, both measured:

    * **Speed.** ``B^T B`` and ``A A^T`` are ``[r, r]`` for *every* module shape, so all modules
      collapse into one batched matmul per shape group -- ~15 kernels instead of ~1200. The direct
      form is launch-bound, not FLOP-bound: on Qwen3-0.6B at the paper's recipe this took mu from
      8.8% of a training step to 1.1%.
    * **Accuracy.** The general path materializes ``W_base + BA`` in the model dtype and subtracts
      a nearly equal teacher weight -- catastrophic cancellation in bf16's 8-bit mantissa. The
      factored form never forms that difference and works from the fp32 factors.

    Returns ``None`` -- so :func:`weight_loss` falls back to the general path -- unless *every*
    targeted module is a LoRA-wrapped linear whose base weight equals the teacher's. That covers
    full-weight training, rescale-style adapters, and any resume from a merged checkpoint, where
    the identity does not hold. The base-equals-teacher check is exact (``torch.equal``) and runs
    once per student, cached on the model as ``_lfa_mu_fastpath_ok``.

    Returns:
        Scalar float32 tensor on the teacher's device, averaged over modules, or ``None``.
    """
    num_layers = adapter.num_layers(teacher)
    device = next(teacher.parameters()).device
    names = [
        n for n in adapter.lora_target_modules()
        if n in _ATTENTION_PROJECTIONS or n in _MLP_PROJECTIONS
    ]

    # (A, B, scaling, layer_weight) per targeted module, plus the (base, teacher) pairs to verify.
    factors: list[tuple[torch.Tensor, torch.Tensor, float, float]] = []
    to_verify: list[tuple[torch.Tensor, torch.Tensor]] = []

    for layer_idx in range(num_layers):
        w = layer_weights[layer_idx]
        t_modules = dict(_target_modules(teacher, adapter, layer_idx, names))
        for name, s_module in _target_modules(student, adapter, layer_idx, names):
            if not (hasattr(s_module, "lora_A") and hasattr(s_module, "lora_B")
                    and hasattr(s_module, "scaling") and len(s_module.lora_A) == 1):
                return None
            base = getattr(s_module, "weight", None)
            if base is None:
                return None
            to_verify.append((base, t_modules[name].weight))
            adapter_name = next(iter(s_module.lora_A))
            factors.append((
                s_module.lora_A[adapter_name].weight,
                s_module.lora_B[adapter_name].weight,
                float(s_module.scaling[adapter_name]),
                w,
            ))

    if not factors:
        return None

    # The identity holds only if the frozen base IS the teacher. Verify exactly, once.
    cached = getattr(student, "_lfa_mu_fastpath_ok", None)
    if cached is None:
        cached = all(b.shape == t.shape and torch.equal(b, t.to(b.device)) for b, t in to_verify)
        try:
            student._lfa_mu_fastpath_ok = cached
        except Exception:  # pragma: no cover - defensive, e.g. a module that forbids attributes
            pass
    if not cached:
        return None

    groups: dict[tuple, list[tuple]] = {}
    for A, B, scaling, w in factors:
        groups.setdefault((tuple(A.shape), tuple(B.shape)), []).append((A, B, scaling, w))

    total = torch.zeros((), device=device, dtype=torch.float32)
    for items in groups.values():
        As = torch.stack([A for A, _, _, _ in items]).float()             # [M, r, d_in]
        Bs = torch.stack([B for _, B, _, _ in items]).float()             # [M, d_out, r]
        coeff = torch.tensor([s * s * w for _, _, s, w in items],
                             device=As.device, dtype=As.dtype)            # s^2 . layer weight
        AAt = torch.bmm(As, As.transpose(1, 2))                           # [M, r, r]
        BtB = torch.bmm(Bs.transpose(1, 2), Bs)                           # [M, r, r]
        # tr((B^T B)(A A^T)) per module, without forming the [r, r] product
        trace = (BtB * AAt.transpose(1, 2)).sum(dim=(1, 2))               # [M]
        total = total + (coeff * trace).sum().to(device)

    return total / len(factors)


def weight_loss(
    teacher: nn.Module,
    student: nn.Module,
    adapter: ModelAdapter,
    layer_weights: list[float] | None = None,
    force_general: bool = False,
) -> torch.Tensor:
    """mu's term: ``sum_l sum_proj w(l) * ||W_s - W_t||^2_F``, averaged over modules.

    Direct weight anchoring -- the isotropic-Gaussian degenerate case of the function anchor, in
    which every weight element is priced equally regardless of what the model does with it. It
    earns its place as a *backstop*: global shrinkage that also catches the non-destructive drift
    the function anchor does not price.

    Tries :func:`lora_factored_weight_loss` first and falls back to the general form (which uses
    :func:`~lfa.adapters.effective_weight`, so an adapter delta counts) whenever the fast path does
    not apply.

    Args:
        layer_weights: per-layer weights; ``None`` = uniform. mu is usually applied uniformly even
            when the anchor is scheduled -- it targets drift the schedule is not aimed at.
        force_general: skip the fast path. For testing the two forms against each other.

    Returns:
        Scalar float32 tensor on the teacher's device.
    """
    num_layers = adapter.num_layers(teacher)
    device = next(teacher.parameters()).device

    if layer_weights is None:
        layer_weights = [1.0] * num_layers

    if not force_general:
        factored = lora_factored_weight_loss(teacher, student, adapter, layer_weights)
        if factored is not None:
            return factored

    names = [
        n for n in adapter.lora_target_modules()
        if n in _ATTENTION_PROJECTIONS or n in _MLP_PROJECTIONS
    ]

    total = torch.zeros((), device=device, dtype=torch.float32)
    n_modules = 0

    for layer_idx in range(num_layers):
        w = layer_weights[layer_idx]
        t_modules = dict(_target_modules(teacher, adapter, layer_idx, names))
        for name, s_module in _target_modules(student, adapter, layer_idx, names):
            W_s = effective_weight(s_module)
            W_t = t_modules[name].weight
            diff = W_s.float() - W_t.float().to(W_s.device)
            total = total + w * (diff ** 2).sum().to(device)
            n_modules += 1

    if n_modules > 0:
        total = total / n_modules

    return total
