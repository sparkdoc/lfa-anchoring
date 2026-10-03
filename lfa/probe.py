"""Check how an artifact prices real update directions: a minutes-long smoke alarm, not a guarantee.

The anchor charges ``E_{h~p(h)} ||Delta f(h)||^2`` for an update ``Delta f`` of a sub-module, with
``h`` drawn from the artifact. So what matters about an artifact is whether it *prices* update
directions the way the model's real activations do. This module measures exactly that, per
anchoring site, for a family of witness directions:

* **Witnesses** -- the LoRA deltas ``(lora_alpha / r) * B @ A`` of one or more trained adapters
  (:func:`lora_deltas`), plus ``n_random`` random rank-32 deltas per real one, each matched to the
  real delta's Frobenius norm (:func:`random_like`). A random witness sees little more than the
  overall scale of the distribution, so SHAPE is set by the real ones, and they must come from an
  **unanchored** run (lambda 0). A run trained anchored against an artifact drifts into the
  directions that artifact underprices, so its deltas are biased witnesses: the alarm is not
  evaluated for them (:func:`adapter_training`), though the numbers are still reported.
* **Prices** (:func:`site_price`) -- ``pre_qkv``: ``mean_h sum_{q,k,v} ||dW h||^2``; ``pre_o``:
  ``mean_h ||dW_o h||^2``; ``pre_mlp``: ``mean_h ||MLP'(h) - MLP(h)||^2`` for the whole SwiGLU
  block with the deltas on gate/up/down. ``pre_lm_head`` is not probed (LoRA does not touch the
  head), and neither is layer 0's ``pre_qkv`` (an exact embedding lookup, not a fitted density).
* **Truth** -- the same prices on real activations: the base model run over WikiText-2 in
  sequences of at most ``seq_len`` tokens, ``n_real`` vectors collected per site at the module
  inputs and split into two disjoint halves. The first half is the truth.
* **Ratios** -- ``rho_j = price under the samples / price under the truth`` per witness ``j``,
  summarised by :func:`level_shape`: **LEVEL** ``exp(mean log rho)`` is a uniform mis-scaling,
  absorbed by lambda and reported only; **SHAPE** ``std(log rho)`` is direction-dependent
  mispricing that no scalar lambda absorbs, and is the number to read.
* **Floor** -- SHAPE with the second real half in place of the artifact: what real-vs-real noise
  alone gives on these witnesses.
* **Diagonal reference** -- the floor half of the real activations with every feature column
  independently permuted across rows (:func:`decorrelate`): every real marginal kept, every
  correlation gone. It is what a perfect diagonal model of the real data would price, fixed per
  model and independent of the artifact: the reference point for a broken artifact. It permutes
  the floor half rather than the truth half so that, like the artifact, it shares no rows with
  the truth it is priced against (see :func:`probe_artifact`).

**The alarm** is threshold-free: a site class (linear = ``pre_qkv`` + ``pre_o``; MLP =
``pre_mlp``) whose median artifact SHAPE is at or above the diagonal reference's median SHAPE
-- the artifact prices update directions no better than a perfect diagonal model of the real
activations. It detects gross failures only -- a collapsed or degenerate artifact -- and does not
grade near-misses. A scale or layer error shows in LEVEL, which is printed beside SHAPE and which
lambda absorbs, not in SHAPE. It says what was measured on these witnesses and this text, and
nothing about the behaviour a model trained against the artifact keeps: pricing fidelity and
preserved behaviour have been seen to come apart. Silence from the alarm is not a mark of
quality.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .adapters import ModelAdapter, effective_weight, get_adapter
from .artifact.schema import SITES, validate_against_model
from .evaluate import DatasetUnavailable, _eval_mode, _model_device, _wikitext2_tokens
from .sampler import Sampler

logger = logging.getLogger(__name__)

__all__ = [
    "SITE_PROJECTIONS",
    "SITE_CLASSES",
    "CLASS_ORDER",
    "RANDOM_RANK",
    "ProbeReport",
    "lora_deltas",
    "adapter_training",
    "random_like",
    "site_price",
    "level_shape",
    "decorrelate",
    "default_layers",
    "collect_activations",
    "alarms_for",
    "probe_artifact",
]

#: The projections whose input each probed site is, and so which deltas a site's price reads.
SITE_PROJECTIONS = {
    "pre_qkv": ("q_proj", "k_proj", "v_proj"),
    "pre_o": ("o_proj",),
    "pre_mlp": ("gate_proj", "up_proj", "down_proj"),
}

#: The class each site's SHAPE is summarised in.
SITE_CLASSES = {"pre_qkv": "linear", "pre_o": "linear", "pre_mlp": "mlp"}

#: The site classes, in report order.
CLASS_ORDER = ("linear", "mlp")

#: Rank of the random witnesses.
RANDOM_RANK = 32

#: Ratios are clamped here before the log, so one witness priced at zero cannot dominate SHAPE.
LOG_FLOOR = 1e-3

#: Rows priced per matmul; the result does not depend on it.
PRICE_CHUNK = 8192

#: Feature columns permuted per pass in :func:`decorrelate`, which bounds its index memory.
_DECORRELATE_COLUMNS = 256


# ==============================================================================================
# Witnesses
# ==============================================================================================

def lora_deltas(adapter_dir, layers=None) -> dict[tuple[int, str], torch.Tensor]:
    """``(lora_alpha / r) * B @ A`` for every (layer, projection) a saved PEFT adapter trains.

    Reads ``adapter_config.json`` and ``adapter_model.safetensors``. The projection name is the
    last component of the module path (``q_proj``, ``gate_proj``, ...), the layer the index after
    ``layers``; LoRA weights outside the transformer layers are not deltas of a probed site and
    are left out. With ``layers``, only those layers' deltas are formed: a full set for a
    1.7B-parameter model is several GB of host memory, and the probe reads a handful of layers.
    Returns float32 tensors on the CPU.

    Raises:
        ValueError: the adapter scales its deltas some other way (rsLoRA, DoRA, per-module rank or
            alpha patterns), or holds no LoRA weights inside the transformer layers.
    """
    from safetensors.torch import load_file

    path = Path(adapter_dir)
    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    other_scaling = [name for name in ("use_rslora", "use_dora", "rank_pattern", "alpha_pattern")
                     if config.get(name)]
    if other_scaling:
        raise ValueError(f"The adapter at {path} sets {', '.join(other_scaling)}; the probe reads "
                         "plain LoRA deltas, (lora_alpha / r) * B @ A, only.")
    scale = config["lora_alpha"] / config["r"]

    wanted = None if layers is None else set(layers)
    tensors = load_file(str(path / "adapter_model.safetensors"))
    deltas, in_layers = {}, False
    for key, a in tensors.items():
        if ".lora_A." not in key:
            continue
        prefix, _, suffix = key.partition(".lora_A.")
        parts = prefix.split(".")
        if "layers" not in parts:
            continue
        in_layers = True
        layer = int(parts[parts.index("layers") + 1])
        if wanted is not None and layer not in wanted:
            continue
        b = tensors[f"{prefix}.lora_B.{suffix}"]
        deltas[(layer, parts[-1])] = scale * (b.float() @ a.float())
    if not in_layers:
        raise ValueError(f"The adapter at {path} holds no LoRA weights inside the transformer "
                         "layers, so it has no update directions to probe with.")
    return deltas


def adapter_training(adapter_dir) -> dict:
    """How the run that wrote ``adapter_dir`` was anchored, from its ``config.json``.

    A training run writes ``config.json`` one level above its ``final_model`` /
    ``latest_model`` / ``checkpoint_epoch_N`` adapter directories (:func:`lfa.train.train`). The
    result is ``{"adapter", "config", "training", ...}`` where ``training`` is ``"unanchored"``
    (``lambda_qkv`` and ``lambda_mlp`` both 0), ``"anchored"`` (either above 0; ``lambda_qkv``,
    ``lambda_mlp``, ``mu`` and ``artifact_path`` are recorded) or ``"unknown"`` (no readable
    config with both lambdas beside the adapter).
    """
    config_path = Path(adapter_dir).parent / "config.json"
    record = {"adapter": str(adapter_dir), "config": None, "training": "unknown"}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return record
    if not isinstance(config, dict) or not all(
            isinstance(config.get(key), (int, float)) for key in ("lambda_qkv", "lambda_mlp")):
        return record
    record["config"] = str(config_path)
    record.update({key: config.get(key)
                   for key in ("lambda_qkv", "lambda_mlp", "mu", "artifact_path")})
    anchored = config["lambda_qkv"] > 0 or config["lambda_mlp"] > 0
    record["training"] = "anchored" if anchored else "unanchored"
    return record


def _witness_notes(histories: list[dict]) -> tuple[bool, list[str]]:
    """Whether the alarm may be evaluated with these adapters, and the lines that say why not."""
    anchored = [h for h in histories if h["training"] == "anchored"]
    if anchored:
        return False, [
            f"alarm not evaluated: adapter {h['adapter']} was trained anchored "
            f"(lambda_qkv={h['lambda_qkv']:g}, lambda_mlp={h['lambda_mlp']:g}); anchored training "
            "moves into directions its artifact underprices, so its deltas are biased witnesses "
            "-- use an adapter from an unanchored run (lambda 0)"
            for h in anchored]
    unknown = [h["adapter"] for h in histories if h["training"] == "unknown"]
    if unknown:
        return True, [
            "caveat: the witnesses must come from an unanchored run (lambda 0), and the training "
            f"history of {', '.join(unknown)} could not be read (no config.json with its lambdas "
            "beside the adapter)"]
    return True, []


def random_like(ref: torch.Tensor, rank: int, generator: torch.Generator) -> torch.Tensor:
    """A random rank-``rank`` delta shaped like ``ref`` and scaled to ``ref``'s Frobenius norm.

    ``B ~ N(0, 1)^{out x rank}``, ``A ~ N(0, 1)^{rank x in}``, drawn on ``generator``'s device and
    returned in float32 on ``ref``'s device.
    """
    out_features, in_features = ref.shape
    b = torch.randn(out_features, rank, generator=generator, device=generator.device)
    a = torch.randn(rank, in_features, generator=generator, device=generator.device)
    delta = (b @ a).to(ref.device)
    return delta * (ref.float().norm() / delta.norm().clamp_min(1e-12))


# ==============================================================================================
# Prices
# ==============================================================================================

def site_price(site: str, modules: dict[str, nn.Module], deltas: dict[str, torch.Tensor],
               h: torch.Tensor, chunk: int = PRICE_CHUNK) -> float:
    """``mean_h ||Delta f(h)||^2`` for ``site``'s sub-module under ``deltas``, in float32.

    ``pre_qkv`` sums ``||dW h||^2`` over the ``q_proj``/``k_proj``/``v_proj`` deltas given;
    ``pre_o`` is ``||dW_o h||^2``. Both are linear, so ``modules`` is not read. ``pre_mlp`` is the
    whole SwiGLU block, ``down(silu(gate(h)) * up(h))``, with each delta given added to its
    projection's weight: ``modules`` maps ``gate_proj``/``up_proj``/``down_proj`` to the model's
    projections, whose effective weights and biases are read. A projection with no delta is left
    as it is. The work runs on ``h``'s device in chunks of ``chunk`` rows.

    Raises:
        ValueError: an unknown site, no deltas, or a delta for a projection the site does not feed.
    """
    names = SITE_PROJECTIONS.get(site)
    if names is None:
        raise ValueError(f"Cannot price site {site!r}; the probe prices {list(SITE_PROJECTIONS)}.")
    stray = sorted(set(deltas) - set(names))
    if stray:
        raise ValueError(f"Deltas {stray} are not projections of {site}, which feeds {list(names)}.")
    if not deltas:
        raise ValueError(f"No deltas to price at {site}.")

    device = h.device
    deltas = {name: d.detach().to(device=device, dtype=torch.float32) for name, d in deltas.items()}
    if site == "pre_mlp":
        weights = {}
        for name in names:
            module = modules[name]
            bias = getattr(module, "bias", None)
            weights[name] = (
                effective_weight(module).detach().to(device=device, dtype=torch.float32),
                None if bias is None else bias.detach().to(device=device, dtype=torch.float32),
            )

    total, count = 0.0, 0
    for start in range(0, h.shape[0], chunk):
        x = h[start:start + chunk].to(torch.float32)
        if site == "pre_mlp":
            squared = _swiglu_difference(x, weights, deltas).pow(2).sum(dim=1)
        else:
            squared = sum((x @ d.T).pow(2).sum(dim=1) for d in deltas.values())
        total += squared.sum(dtype=torch.float64).item()
        count += x.shape[0]
    return total / count


def _swiglu_difference(x: torch.Tensor, weights: dict, deltas: dict) -> torch.Tensor:
    """``MLP'(x) - MLP(x)`` for a SwiGLU block, written to avoid subtracting two full outputs.

    ``down`` is linear, so ``(W_d + dW_d) m' - W_d m = W_d (m' - m) + dW_d m'``; the down bias
    cancels. The gate and up biases do not, and are applied on both sides.
    """
    (w_gate, b_gate), (w_up, b_up), (w_down, _) = (
        weights["gate_proj"], weights["up_proj"], weights["down_proj"])
    gate, up = F.linear(x, w_gate, b_gate), F.linear(x, w_up, b_up)
    before = F.silu(gate) * up
    if "gate_proj" in deltas:
        gate = gate + x @ deltas["gate_proj"].T
    if "up_proj" in deltas:
        up = up + x @ deltas["up_proj"].T
    after = F.silu(gate) * up
    difference = (after - before) @ w_down.T
    if "down_proj" in deltas:
        difference = difference + after @ deltas["down_proj"].T
    return difference


# ==============================================================================================
# LEVEL, SHAPE and the diagonal reference
# ==============================================================================================

def level_shape(ratios: list[float]) -> tuple[float, float]:
    """``(LEVEL, SHAPE)`` of a list of price ratios.

    LEVEL is ``exp(mean log rho)`` -- the geometric-mean mis-scaling, which lambda absorbs. SHAPE
    is the sample standard deviation (``n - 1``) of ``log rho`` -- the spread of mispricing across
    directions, which it does not. Each ratio enters as ``log(max(rho, 1e-3))``; a single ratio
    has SHAPE 0.

    Raises:
        ValueError: no ratios.
    """
    if not ratios:
        raise ValueError("level_shape needs at least one ratio.")
    logs = [math.log(max(ratio, LOG_FLOOR)) for ratio in ratios]
    mean = sum(logs) / len(logs)
    variance = sum((value - mean) ** 2 for value in logs) / max(len(logs) - 1, 1)
    return math.exp(mean), math.sqrt(variance)


def decorrelate(samples: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """``samples`` with each column independently permuted across rows.

    Every column keeps exactly its values (so every marginal is kept) while the pairing between
    columns is destroyed (so every correlation is gone): what a diagonal artifact would sample.
    The permutations are drawn on ``generator``'s device.
    """
    rows, columns = samples.shape
    out = torch.empty_like(samples)
    for start in range(0, columns, _DECORRELATE_COLUMNS):
        stop = min(start + _DECORRELATE_COLUMNS, columns)
        order = torch.rand(rows, stop - start, generator=generator, device=generator.device)
        order = order.argsort(dim=0).to(samples.device)
        out[:, start:stop] = samples[:, start:stop].gather(0, order)
    return out


# ==============================================================================================
# Real activations
# ==============================================================================================

def default_layers(num_layers: int) -> list[int]:
    """Five evenly spaced layers from the first to the last (fewer on a shallower model)."""
    return sorted({round(i * (num_layers - 1) / 4) for i in range(5)})


def collect_activations(model: nn.Module, adapter: ModelAdapter, tokens: torch.Tensor,
                        sites: list[tuple[int, str]], n: int,
                        seq_len: int) -> dict[tuple[int, str], torch.Tensor]:
    """The first ``n`` input vectors of each ``(layer, site)``, from running ``model`` over ``tokens``.

    ``tokens`` (one 1-D stream) is cut into consecutive sequences of ``seq_len`` tokens, each run
    on its own; hooks on :meth:`~lfa.adapters.ModelAdapter.site_module` read every position's
    input. Returns float32 CPU tensors ``[n, width]``, in stream order.

    Raises:
        ValueError: the stream ran out before every site had ``n`` vectors.
    """
    store: dict[tuple[int, str], list[torch.Tensor]] = {key: [] for key in sites}
    counts = dict.fromkeys(sites, 0)

    def hook_for(key):
        def hook(module, args):
            if counts[key] >= n:
                return
            vectors = args[0].detach().reshape(-1, args[0].shape[-1]).float().cpu()
            store[key].append(vectors)
            counts[key] += vectors.shape[0]
        return hook

    handles = [adapter.site_module(model, layer, site).register_forward_pre_hook(hook_for((layer, site)))
               for layer, site in sites]
    device = _model_device(model, "cpu")
    try:
        with torch.no_grad(), _eval_mode(model):
            for window in tokens.split(seq_len):
                if all(count >= n for count in counts.values()):
                    break
                model(window.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    short = min(counts.values(), default=n)
    if short < n:
        raise ValueError(f"The text gave {short} activation vectors per site, fewer than the {n} "
                         "asked for: pass a smaller --n-real (n_real), or more text.")
    return {key: torch.cat(chunks)[:n] for key, chunks in store.items()}


# ==============================================================================================
# The report
# ==============================================================================================

def alarms_for(summary: dict[str, dict]) -> list[str]:
    """One sentence per site class whose median artifact SHAPE is >= the diagonal reference's."""
    alarms = []
    for site_class, row in summary.items():
        artifact, diagonal = row["artifact_shape"], row["diagonal_shape"]
        if artifact >= diagonal:
            alarms.append(
                f"{site_class} sites: the artifact's median SHAPE ({artifact:.3f}) is at or above "
                f"the diagonal reference's ({diagonal:.3f}). On these witnesses it prices update "
                "directions no better than a perfect diagonal model of the real activations.")
    return alarms


def _over(value: float, floor: float) -> float | None:
    return value / floor if floor > 0 else None


@dataclass
class ProbeReport:
    """What :func:`probe_artifact` measured.

    Attributes:
        sites: per ``"{layer}_{site}"``: ``layer``, ``site``, ``class``, ``n_witnesses``, the
            artifact's and the diagonal reference's LEVEL and SHAPE, and the floor's (LEVEL
            close to 1 and SHAPE close to 0 when the two real halves agree).
        summary: per site class (``linear``, ``mlp``): the medians over its sites of every SHAPE
            and LEVEL, and the artifact's and the diagonal reference's median SHAPE as multiples
            of the floor's
            (``None`` when the floor is exactly 0).
        alarms: one sentence per class whose artifact median SHAPE is at or above the diagonal
            reference's; ``None`` when the alarm was not evaluated (an adapter trained
            anchored), so that an unevaluated alarm never reads as a quiet one.
        settings: how the measurement was taken, including each adapter's training history
            (``witness_training``, from :func:`adapter_training`).
        notes: why the alarm was not evaluated, or the caveat it was evaluated under.
    """

    sites: dict[str, dict]
    summary: dict[str, dict]
    alarms: list[str] | None
    settings: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        """The report as plain JSON-serialisable data."""
        return {"sites": self.sites, "summary": self.summary,
                "alarms": None if self.alarms is None else list(self.alarms),
                "notes": list(self.notes), "settings": self.settings}

    def format_table(self) -> str:
        """The per-site table, the class medians against the floor, and the alarms."""

        def number(value, digits=3):
            return f"{value:.{digits}f}" if value is not None else "-"

        def times(value):
            return f"{value:.1f}x" if value is not None else "-"

        lines = [
            "LEVEL = geometric-mean price ratio (absorbed by lambda; reported only). "
            "SHAPE = spread of log price ratios across update directions (not absorbable).",
            "",
            f"{'site':<14}{'floor':>9}{'artifact':>18}{'diagonal reference':>24}",
            f"{'':<14}{'shape':>9}{'level':>9}{'shape':>9}{'level':>12}{'shape':>12}",
        ]
        for key, row in self.sites.items():
            lines.append(f"{row['layer']:<3}{row['site']:<11}{number(row['floor_shape']):>9}"
                         f"{number(row['artifact_level']):>9}{number(row['artifact_shape']):>9}"
                         f"{number(row['diagonal_level']):>12}{number(row['diagonal_shape']):>12}")
        lines += ["", "median SHAPE by site class (multiple of the floor):"]
        for site_class, row in self.summary.items():
            lines.append(
                f"  {site_class:<7} floor {number(row['floor_shape'])}   "
                f"artifact {number(row['artifact_shape'])} "
                f"({times(row['artifact_shape_over_floor'])})   "
                f"diagonal reference {number(row['diagonal_shape'])} "
                f"({times(row['diagonal_shape_over_floor'])})   over {row['n_sites']} sites")
        unprobed = [site_class for site_class in CLASS_ORDER if site_class not in self.summary]
        for site_class in unprobed:
            lines.append(f"  {site_class:<7} not probed: no site of this class was priced")
        skipped = self.settings.get("skipped") or []
        if skipped:
            lines += ["", "skipped sites:"] + [f"  {reason}" for reason in skipped]
        lines.append("")
        lines += [f"NOTE: {note}" for note in self.notes]
        if self.alarms:
            lines += [f"ALARM: {alarm}" for alarm in self.alarms]
        elif self.alarms is not None:                  # None: not evaluated, said in the notes
            probed = " and ".join(self.summary)
            lines.append(f"No alarm at the probed classes ({probed}): the artifact's median SHAPE "
                         "is below the diagonal reference's. This measures pricing on these "
                         "witnesses, not the behaviour a trained model keeps.")
        return "\n".join(lines)


# ==============================================================================================
# The probe
# ==============================================================================================

def _site_modules(adapter: ModelAdapter, model: nn.Module, layer: int, site: str) -> dict:
    """The projections ``site`` feeds in ``layer``, keyed by name."""
    if site == "pre_qkv":
        return adapter.qkv_modules(model, layer)
    if site == "pre_o":
        return {"o_proj": adapter.o_proj_module(model, layer)}
    mlp = adapter.mlp_module(model, layer)
    missing = [name for name in SITE_PROJECTIONS["pre_mlp"] if not hasattr(mlp, name)]
    if missing:
        raise ValueError(f"Layer {layer}'s MLP has no {', '.join(missing)}: the probe prices a "
                         "SwiGLU block (gate_proj, up_proj, down_proj).")
    act = getattr(mlp, "act_fn", None)
    if act is not None:
        x = torch.linspace(-6.0, 6.0, 97, device=next(mlp.parameters()).device)
        if not torch.allclose(act(x).float(), F.silu(x), atol=1e-5):
            raise ValueError(f"Layer {layer}'s MLP activation is not SiLU: the probe prices a "
                             "SwiGLU block.")
    return {name: getattr(mlp, name) for name in SITE_PROJECTIONS["pre_mlp"]}


def _witnesses(adapters: list[tuple[str, dict]], layer: int, modules: dict, n_random: int,
               generator: torch.Generator) -> list[dict[str, torch.Tensor]]:
    """Every adapter's real deltas at one site, each followed by ``n_random`` random ones."""
    witnesses = []
    for adapter_dir, deltas in adapters:
        real = {name: deltas[(layer, name)] for name in modules if (layer, name) in deltas}
        if not real:
            continue
        for name, delta in real.items():
            expected = tuple(effective_weight(modules[name]).shape)
            if tuple(delta.shape) != expected:
                raise ValueError(f"The adapter at {adapter_dir} has a {tuple(delta.shape)} delta "
                                 f"for layer {layer} {name}, but this model's weight is "
                                 f"{expected}: it was trained on another model.")
        witnesses.append(real)
        for _ in range(n_random):
            witnesses.append({name: random_like(delta, RANDOM_RANK, generator)
                              for name, delta in real.items()})
    return witnesses


def _nothing_priced(layers: list[int], skipped: list[str]) -> str:
    reasons = "; ".join(skipped) or "no site to probe"
    return (f"Nothing was priced at layers {layers}: {reasons}. Probe layers the artifact has "
            "statistics for, with an adapter that trains their projections.")


def probe_artifact(model, tokenizer, artifact, adapter_dirs, *, layers=None, n_real=30_000,
                   n_model=30_000, n_random=4, seq_len=512, seed=0, device="cuda:0",
                   tokens: torch.Tensor | None = None) -> ProbeReport:
    """Price real update directions under ``artifact`` against real activations (module docstring).

    Args:
        model: the base model the artifact describes, in float32 and already placed.
        tokenizer: its tokenizer, for the WikiText-2 text.
        artifact: an artifact path, or a loaded artifact dict (:class:`~lfa.sampler.Sampler`).
        adapter_dirs: saved PEFT adapters trained on ``model``; their deltas are the witnesses.
        layers: the layers to probe (default: :func:`default_layers`).
        n_real: real activation vectors per site, split into truth and floor halves.
        n_model: artifact samples per site.
        n_random: random witnesses per real delta.
        seq_len: tokens per sequence of real text.
        seed: seeds the artifact sampler, the random witnesses and the diagonal reference's
            permutations.
        device: where the samples are drawn and the prices computed.
        tokens: one token stream to use instead of WikiText-2's test split.

    Raises:
        ValueError: a model carrying LoRA adapters or not in float32, a layer out of range, too
            little text for ``n_real``, adapters that do not fit the model, or no site priced
            (naming why each was skipped).
        lfa.artifact.schema.ArtifactModelMismatch: the artifact is shaped for another model.
        lfa.evaluate.DatasetUnavailable: WikiText-2 could not be loaded.
    """
    started = time.time()
    if hasattr(model, "peft_config") or any(hasattr(module, "lora_A") for module in model.modules()):
        raise ValueError("The model carries LoRA adapters, so its activations and weights would "
                         "include them: probe the base model, and pass the adapter with "
                         "--adapter (adapter_dirs).")
    adapter = get_adapter(model)
    dtype = next(model.parameters()).dtype
    if dtype != torch.float32:
        raise ValueError(f"The probe reads real activations from a float32 model; this one is "
                         f"{dtype}. Load it with dtype=torch.float32.")

    sampler = Sampler(artifact, device=device, seed=seed)
    validate_against_model(sampler.params, model, adapter)

    num_layers = adapter.num_layers(model)
    layers = default_layers(num_layers) if layers is None else sorted(set(layers))
    out_of_range = [layer for layer in layers if not 0 <= layer < num_layers]
    if out_of_range:
        raise ValueError(f"Layers {out_of_range} are out of range for a {num_layers}-layer model.")

    sites, skipped = [], []
    for layer in layers:
        for site in SITES:
            if layer == 0 and site == "pre_qkv":
                continue
            if sampler.has_site(layer, site):
                sites.append((layer, site))
            else:
                skipped.append(f"{layer}_{site}: no statistics in the artifact")

    if not sites:
        raise ValueError(_nothing_priced(layers, skipped))
    adapters = [(str(path), lora_deltas(path, layers=layers)) for path in adapter_dirs]
    histories = [adapter_training(path) for path in adapter_dirs]

    text = "WikiText-2 test split" if tokens is None else "caller-supplied tokens"
    if tokens is None:
        try:
            tokens = _wikitext2_tokens(tokenizer, math.ceil(n_real / seq_len), seq_len)
        except DatasetUnavailable as error:
            cause = error.__cause__
            raise DatasetUnavailable(
                "The probe's real activations come from the WikiText-2 test split, which could "
                f"not be loaded ({type(cause).__name__}: {cause}). It is read from the Hugging "
                "Face Hub: run once with network access to cache it.") from cause
    real = collect_activations(model, adapter, tokens, sites, n_real, seq_len)
    logger.info("Collected %d real vectors at %d sites (%.0fs)", n_real, len(sites),
                time.time() - started)

    half = n_real // 2
    witness_generator = torch.Generator().manual_seed(seed)
    diagonal_generator = torch.Generator(device=device).manual_seed(seed)
    rows = {}
    for layer, site in sites:
        modules = _site_modules(adapter, model, layer, site)
        witnesses = _witnesses(adapters, layer, modules, n_random, witness_generator)
        if not witnesses:
            skipped.append(f"{layer}_{site}: no adapter trains its projections")
            continue
        truth = real[(layer, site)][:half].to(device)
        floor = real[(layer, site)][half:2 * half].to(device)
        samples = sampler.sample_best(layer, site, n_model).to(device=device, dtype=torch.float32)
        # The diagonal reference permutes the floor half, so that it shares no rows with the
        # truth it is priced against, as the artifact's samples do not.
        diagonal = decorrelate(floor, diagonal_generator)

        ratios = {"artifact": [], "diagonal": [], "floor": []}
        for witness in witnesses:
            reference = site_price(site, modules, witness, truth)
            if not reference > 0:
                continue
            for name, h in (("artifact", samples), ("diagonal", diagonal), ("floor", floor)):
                ratios[name].append(site_price(site, modules, witness, h) / reference)
        if not ratios["artifact"]:
            skipped.append(f"{layer}_{site}: every witness prices to zero on real activations")
            continue

        row = {"layer": layer, "site": site, "class": SITE_CLASSES[site],
               "n_witnesses": len(ratios["artifact"])}
        for name, values in ratios.items():
            row[f"{name}_level"], row[f"{name}_shape"] = level_shape(values)
        rows[f"{layer}_{site}"] = row
        logger.info("Priced %d_%s: floor %.3f, artifact %.3f, diagonal reference %.3f (%.0fs)",
                    layer, site, row["floor_shape"], row["artifact_shape"],
                    row["diagonal_shape"], time.time() - started)

    if not rows:
        raise ValueError(_nothing_priced(layers, skipped))

    summary = {}
    for site_class in CLASS_ORDER:
        members = [row for row in rows.values() if row["class"] == site_class]
        if not members:
            continue
        medians = {f"{name}_{stat}": statistics.median(row[f"{name}_{stat}"] for row in members)
                   for name in ("floor", "artifact", "diagonal") for stat in ("level", "shape")}
        medians["artifact_shape_over_floor"] = _over(medians["artifact_shape"],
                                                     medians["floor_shape"])
        medians["diagonal_shape_over_floor"] = _over(medians["diagonal_shape"],
                                                     medians["floor_shape"])
        medians["n_sites"] = len(members)
        summary[site_class] = medians

    settings = {
        "artifact": str(artifact) if not isinstance(artifact, dict) else "<in-memory artifact>",
        "adapters": [path for path, _ in adapters], "witness_training": histories,
        "layers": layers, "n_real": n_real,
        "n_model": n_model, "n_random": n_random, "random_rank": RANDOM_RANK,
        "seq_len": seq_len, "seed": seed,
        "text": text,
        "skipped": skipped,
    }
    evaluate, notes = _witness_notes(histories)
    return ProbeReport(sites=rows, summary=summary,
                       alarms=alarms_for(summary) if evaluate else None, settings=settings,
                       notes=notes)
