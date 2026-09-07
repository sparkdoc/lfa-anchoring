"""The Layerwise Function Anchoring (LFA) training loop.

A step runs two paths that share one gradient update:

* the **content** path -- an ordinary causal-LM forward over a batch of the new domain's text,
  giving ``L_content``;
* the **anchor** path -- ``n_anchor_samples`` hidden states drawn from the estimated ``p(h)``,
  pushed through the teacher's and the student's sub-modules, giving ``L_anchor`` (see
  :mod:`lfa.losses`). It never touches the batch.

with the weight term ``mu * L_weight`` added as a backstop::

    L = L_content + L_anchor + mu * L_weight

The two paths use completely separate data, which is what makes the method data-free: nothing
from the previously-learned domains is stored or replayed, and the preservation signal comes
from the artifact plus the frozen teacher alone.

Everything else here is bookkeeping around that step -- gradient accumulation, the learning-rate
schedule, per-epoch re-chunking of the corpus, per-epoch validation on the held-out documents,
checkpoints, and resume. Two guards earn their keep: a run whose optimizer steps produce a
gradient norm of exactly zero is announced loudly rather than left to finish as a silent no-op,
and full-weight training says up front that it is outside the validated envelope.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .adapters import ModelAdapter
from .corpus import ChunkedCorpus, make_dataloader
from .losses import anchor_loss, compute_layer_weights, weight_loss
from .models import assert_trainable_params, load_adapter_for_training
from .sampler import Sampler

logger = logging.getLogger("lfa.train")

__all__ = [
    "TrainConfig",
    "LR_SCHEDULES",
    "ResumeSourceHasNoAdapter",
    "StepMetrics",
    "EpochMetrics",
    "TrainingState",
    "FULL_WEIGHT_NOTICE",
    "EMBED_ANCHOR_DISABLED_NOTICE",
    "train_step",
    "train_epoch",
    "validation_loss",
    "train",
]

#: The learning-rate schedules a run may ask for. Both start with ``warmup_steps`` of linear
#: warmup from 0.1x to 1.0x of ``learning_rate``; they differ in what follows.
LR_SCHEDULES = ("cosine", "constant")


class ResumeSourceHasNoAdapter(RuntimeError):
    """Raised when a LoRA run resumes from a checkpoint that saved no adapter."""


#: Said once, at the start of a full-weight run. Every published LFA result is LoRA.
FULL_WEIGHT_NOTICE = (
    "Full-weight anchoring is unvalidated on this model in the LFA paper (LoRA is the validated "
    "path); calibrate λ in 50,000–100,000 and check held-out domain perplexity, not only "
    "general-text perplexity."
)

#: Said once, at the start of a run whose sampler has no layer-0 embedding table. The embedding
#: and the LM head are the two ends of one tied matrix, and L_embed is the only term that anchors
#: the embedding end -- so losing it silently leaves that matrix anchored from one side only. The
#: table is not shipped (it is ~300 MB and exactly reconstructible), so the omission is easy to
#: make and invisible in the metrics: `loss_embed` is simply 0.0 rather than missing.
EMBED_ANCHOR_DISABLED_NOTICE = (
    "embedding anchor disabled: the artifact has no embedding lookup; call "
    "Sampler.build_embedding_lookup_from_model(model, adapter) before training"
)


# ==============================================================================================
# Configuration
# ==============================================================================================

@dataclass
class TrainConfig:
    """Every knob of an LFA run. The defaults are the shipped Qwen3-0.6B operating point.

    The recipe is a *joint* operating point, not a set of independent settings. In particular
    ``lambda_qkv``/``lambda_mlp`` are coupled to ``lora_rank``, to the artifact's sharpness, and
    to the composition of the corpus: lambda constrains motion inside the rank-``r`` update
    subspace, so the same lambda binds far harder at a lower rank, and a lambda carried across
    ranks or corpora is not the same regularizer. Re-tune it rather than porting it.

    Learning rate schedule: ``warmup_steps`` of linear warmup (0.1x -> 1.0x of
    ``learning_rate``), then either cosine decay over every remaining step of ``num_epochs`` to a
    floor of ``lr_floor * learning_rate`` (``lr_schedule="cosine"``, the default) or a flat hold
    at the peak (``lr_schedule="constant"``). The schedule is always laid over the epochs actually
    trained, so a change of ``num_epochs`` is a change of the whole curve, not only of where it
    stops.

    Checkpoint modes: ``"none"`` writes only ``final_model``; ``"rolling"`` overwrites
    ``latest_model`` every ``checkpoint_every`` epochs; ``"all"`` accumulates
    ``checkpoint_epoch_N``. Both periodic modes also write ``training_state.pt``, so only they
    make a mid-run resume possible.
    """

    # -- model
    model_id: str = "Qwen/Qwen3-0.6B"

    # -- anchoring (lambda) and the weight backstop (mu)
    lambda_qkv: float = 100000.0
    lambda_mlp: float = 100000.0
    mu: float = 0.05
    n_anchor_samples: int = 16
    anchor_end_ratio: float = 0.1
    anchor_schedule: str = "cosine"
    artifact_path: str = ""

    # -- optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 50
    lr_schedule: str = "cosine"
    lr_floor: float = 0.0
    batch_size: int = 6
    gradient_accumulation_steps: int = 1
    num_epochs: int = 15
    sequence_length: int = 512

    # -- checkpointing and logging
    checkpoint_mode: str = "none"
    checkpoint_every: int = 10
    logging_steps: int = 10

    # -- LoRA / full weight
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    freeze_embed: bool = True
    full_weight: bool = False

    # -- data
    seed: int = 42
    keep_short_whole: bool = True
    #: Share of DOCUMENTS held out of training, shuffled under ``seed``. :func:`lfa.train.train`
    #: is handed a corpus that is already built, so this field is read by whoever builds it --
    #: :meth:`lfa.workspace.Workspace.train` passes it to :func:`lfa.corpus.load_corpus`, and
    #: hands the trainer the held-out half for the per-epoch :func:`validation_loss` pass -- and
    #: it is recorded in ``config.json`` so a run says which documents it was allowed to see.
    val_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"lr_schedule must be one of {', '.join(LR_SCHEDULES)}, got "
                f"{self.lr_schedule!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        """The config as a JSON-serializable dict (what lands in ``config.json``)."""
        return asdict(self)


# ==============================================================================================
# Metrics and state
# ==============================================================================================

@dataclass
class StepMetrics:
    """Losses from one micro-batch, all as reported values (before accumulation scaling)."""

    loss_total: float
    loss_content: float
    loss_anchor: float
    loss_qkv: float = 0.0
    loss_mlp: float = 0.0
    loss_lm_head: float = 0.0
    loss_embed: float = 0.0
    loss_mu: float = 0.0
    grad_norm: float | None = None


@dataclass
class EpochMetrics:
    """Micro-batch averages over one epoch (``avg_grad_norm`` averages optimizer steps)."""

    avg_loss_total: float
    avg_loss_content: float
    avg_loss_anchor: float
    avg_loss_qkv: float
    avg_loss_mlp: float
    avg_loss_lm_head: float
    avg_loss_embed: float
    avg_loss_mu: float
    avg_grad_norm: float
    num_steps: int
    duration_seconds: float


@dataclass
class TrainingState:
    """What a run carries across epochs, and what a resume restores."""

    epoch: int = 0
    global_step: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    #: Losses of the untrained student over the whole training set, measured before epoch 1.
    baseline: dict[str, float] | None = None
    #: The model that was actually trained. A resume re-attaches a saved adapter, which produces a
    #: NEW object -- the caller's own variable still points at the bare base model -- so the run
    #: hands its model back here rather than leaving that to be discovered.
    model: nn.Module | None = None


def _value(x: Any) -> float:
    """A float from a tensor, a number, or the ``0`` that an excluded loss block leaves behind."""
    return x.item() if torch.is_tensor(x) else float(x)


# ==============================================================================================
# One step
# ==============================================================================================

def train_step(
    student: nn.Module,
    teacher: nn.Module,
    batch: dict[str, torch.Tensor],
    sampler: Sampler | None,
    adapter: ModelAdapter,
    config: TrainConfig,
    layer_weights: list[float] | None = None,
    accumulation_steps: int = 1,
) -> tuple[torch.Tensor, StepMetrics]:
    """One micro-batch: content loss, function anchor, weight backstop.

    Args:
        batch: ``input_ids`` / ``attention_mask`` / ``labels``, as :mod:`lfa.corpus` collates them.
        sampler: draws the anchor's hidden states; ``None`` disables anchoring entirely.
        layer_weights: the anchor's per-layer schedule; ``None`` = uniform.
        accumulation_steps: the returned loss is divided by this, so that accumulating
            ``accumulation_steps`` micro-batches gives the same gradient as one large batch. The
            reported metrics are *not* scaled -- they stay comparable across batch geometries.

    Returns:
        ``(loss_to_backward, metrics)``.
    """
    device = next(student.parameters()).device

    # The LM head and the embedding are two ends of one tied matrix, so both anchor at layer 0's
    # strength -- and both are pointless when freeze_embed has frozen that matrix.
    lm_head_weight = config.lambda_qkv if not config.freeze_embed else 0.0
    include_embed = (
        config.lambda_qkv > 0
        and sampler is not None
        and sampler.has_embedding_lookup()
        and not config.freeze_embed
    )
    embed_weight = config.lambda_qkv if include_embed else 0.0

    # -- L_content: cross-entropy on the new domain's text
    attention_mask = batch.get("attention_mask")
    outputs = student(
        input_ids=batch["input_ids"].to(device),
        attention_mask=None if attention_mask is None else attention_mask.to(device),
        labels=batch["labels"].to(device),
    )
    loss_content = outputs.loss

    # -- L_anchor: distribution-weighted function anchoring
    loss_anchor: Any = 0.0
    loss_qkv = loss_mlp = loss_lm_head = loss_embed = 0.0
    if sampler is not None:
        anchor = anchor_loss(
            teacher=teacher,
            student=student,
            sampler=sampler,
            adapter=adapter,
            n_samples=config.n_anchor_samples,
            layer_weights=layer_weights,
            include_qkv=config.lambda_qkv > 0,
            include_mlp=config.lambda_mlp > 0,
            include_lm_head=lm_head_weight > 0,
            include_embed=include_embed,
            qkv_weight=config.lambda_qkv,
            mlp_weight=config.lambda_mlp,
            lm_head_weight=lm_head_weight,
            embed_weight=embed_weight,
        )
        # Excluded blocks are absent from the result, not zero-valued.
        loss_anchor = anchor["total"]
        loss_qkv = anchor.get("qkv", 0.0)
        loss_mlp = anchor.get("mlp", 0.0)
        loss_lm_head = anchor.get("lm_head", 0.0)
        loss_embed = anchor.get("embed", 0.0)

    # -- mu * L_weight: L2 toward the teacher, UNIFORM over layers. mu catches the drift the
    # function anchor does not price -- global shrinkage, deliberately not aimed anywhere -- so it
    # does not take the anchor's layer schedule.
    loss_mu: Any = 0.0
    if config.mu > 0:
        loss_mu = config.mu * weight_loss(
            teacher=teacher, student=student, adapter=adapter, layer_weights=None,
        )

    loss_total = (loss_content + loss_anchor + loss_mu) / accumulation_steps

    metrics = StepMetrics(
        loss_total=_value(loss_content) + _value(loss_anchor) + _value(loss_mu),
        loss_content=_value(loss_content),
        loss_anchor=_value(loss_anchor),
        loss_qkv=_value(loss_qkv),
        loss_mlp=_value(loss_mlp),
        loss_lm_head=_value(loss_lm_head),
        loss_embed=_value(loss_embed),
        loss_mu=_value(loss_mu),
    )
    return loss_total, metrics


# ==============================================================================================
# One epoch
# ==============================================================================================

#: Said when an optimizer step moved nothing. Once per epoch -- it is a setup fault, not news.
NO_GRADIENT_NOTICE = (
    "Gradient norm was exactly 0.0 at an optimizer step: no parameter received gradient — "
    "adapter frozen? Nothing is being learned and every checkpoint will be identical to the "
    "last. (PeftModel.from_pretrained defaults to is_trainable=False; re-attach a saved adapter "
    "with lfa.models.load_adapter_for_training instead.)"
)


def train_epoch(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader,
    sampler: Sampler | None,
    adapter: ModelAdapter,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    layer_weights: list[float] | None = None,
    global_step: int = 0,
    run_logger: logging.Logger | None = None,
) -> tuple[EpochMetrics, int]:
    """One pass over ``dataloader`` with gradient accumulation.

    An epoch can legitimately be empty: :meth:`~lfa.corpus.ChunkedCorpus.rechunk` cuts every
    document from a per-epoch offset, and with ``keep_short_whole=False`` a corpus of short
    documents yields nothing at some offsets. That is skipped with a warning -- no optimizer
    step, ``global_step`` unchanged -- rather than dividing by zero inside the sampler.

    Returns:
        ``(metrics, global_step)``.
    """
    log = run_logger or logger
    student.train()
    if teacher is not None:
        teacher.eval()

    accumulation_steps = config.gradient_accumulation_steps
    start_time = time.time()

    if len(dataloader.dataset) == 0:
        log.warning(
            "Empty training epoch: the corpus chunked to 0 examples at this epoch's offset. "
            "Skipping the epoch; global_step stays %d.", global_step,
        )
        return EpochMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
                            time.time() - start_time), global_step

    totals = dict(total=0.0, content=0.0, anchor=0.0, qkv=0.0, mlp=0.0, lm_head=0.0,
                  embed=0.0, mu=0.0)
    total_grad_norm = 0.0
    num_steps = 0
    num_micro_batches = 0
    warned_no_gradient = False

    def optimizer_step(last_metrics: StepMetrics | None) -> None:
        nonlocal num_steps, global_step, total_grad_norm, warned_no_gradient

        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.max_grad_norm)
        grad_norm = _value(grad_norm)
        total_grad_norm += grad_norm

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        num_steps += 1
        global_step += 1

        if grad_norm == 0.0 and not warned_no_gradient:
            warned_no_gradient = True
            log.warning(NO_GRADIENT_NOTICE)

        if global_step % config.logging_steps == 0 and last_metrics is not None:
            anchor_str = (f"{last_metrics.loss_anchor:.2e}"
                          if last_metrics.loss_anchor > 0 else "0")
            log.info(
                "Step %d: loss=%.4f content=%.4f anchor=%s lr=%.2e",
                global_step, last_metrics.loss_total, last_metrics.loss_content, anchor_str,
                optimizer.param_groups[0]["lr"],
            )

    optimizer.zero_grad()
    accumulated = 0
    last_metrics = None

    for batch in dataloader:
        loss, last_metrics = train_step(
            student=student, teacher=teacher, batch=batch, sampler=sampler, adapter=adapter,
            config=config, layer_weights=layer_weights, accumulation_steps=accumulation_steps,
        )
        loss.backward()

        totals["total"] += last_metrics.loss_total
        totals["content"] += last_metrics.loss_content
        totals["anchor"] += last_metrics.loss_anchor
        totals["qkv"] += last_metrics.loss_qkv
        totals["mlp"] += last_metrics.loss_mlp
        totals["lm_head"] += last_metrics.loss_lm_head
        totals["embed"] += last_metrics.loss_embed
        totals["mu"] += last_metrics.loss_mu
        num_micro_batches += 1

        accumulated += 1
        if accumulated >= accumulation_steps:
            optimizer_step(last_metrics)
            accumulated = 0

    if accumulated > 0:                      # a trailing partial accumulation window still steps
        optimizer_step(last_metrics)

    n = max(num_micro_batches, 1)
    metrics = EpochMetrics(
        avg_loss_total=totals["total"] / n,
        avg_loss_content=totals["content"] / n,
        avg_loss_anchor=totals["anchor"] / n,
        avg_loss_qkv=totals["qkv"] / n,
        avg_loss_mlp=totals["mlp"] / n,
        avg_loss_lm_head=totals["lm_head"] / n,
        avg_loss_embed=totals["embed"] / n,
        avg_loss_mu=totals["mu"] / n,
        avg_grad_norm=total_grad_norm / max(num_steps, 1),
        num_steps=num_steps,
        duration_seconds=time.time() - start_time,
    )
    return metrics, global_step


# ==============================================================================================
# Checkpoint state
# ==============================================================================================

#: Module names PEFT treats as embedding layers when it decides whether to save them.
_EMBEDDING_MODULE_NAMES = frozenset({"embed_tokens", "lm_head"})


def _save_student(student: nn.Module, output_dir: Path) -> None:
    """Write a checkpoint: the LoRA adapter, or the full weights.

    The only subtlety is one PEFT default. ``PeftModel.save_pretrained`` leaves
    ``save_embedding_layers="auto"``, and resolving "auto" means asking the Hub whether the base
    model's ``config.json`` exists, to find out whether the vocabulary was resized. When the base
    is a Hub id and the Hub cannot be reached -- which is the normal case here, since a long run
    sets ``HF_HUB_OFFLINE=1`` so a flaky network cannot strand it -- that check cannot answer, and
    PEFT warns once per save. Deciding it ourselves removes the network call and the warning:
    "auto" only ever resolves to ``True`` when the embedding is a LoRA *target* or the vocabulary
    was resized, and this package does neither (the embedding is reached through
    ``modules_to_save``, which is a different mechanism and is unaffected by this flag -- verified
    by saving the same adapter both ways under ``freeze_embed`` True and False and comparing the
    tensors, which are identical).

    A ``target_modules`` given as a regex is left to PEFT: guessing wrong there would silently drop
    a resized embedding, and a warning is much cheaper than that.
    """
    if not hasattr(student, "peft_config"):
        student.save_pretrained(output_dir)          # full weight: an ordinary HF checkpoint
        return

    targets: set[str] = set()
    for peft_config in student.peft_config.values():
        configured = getattr(peft_config, "target_modules", None) or ()
        if isinstance(configured, str):              # a regex; let PEFT work it out
            student.save_pretrained(output_dir)
            return
        targets.update(configured)

    student.save_pretrained(output_dir,
                            save_embedding_layers=bool(targets & _EMBEDDING_MODULE_NAMES))


def _save_training_state(
    state: TrainingState,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    model_checkpoint: str,
    run_logger: logging.Logger,
) -> None:
    """Write ``training_state.pt``: progress, history, and the optimizer's moments.

    ``model_checkpoint`` names the directory the state belongs to, so that a resume knows which
    weights the optimizer moments were computed against.
    """
    torch.save(
        {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "history": state.history,
            "baseline": state.baseline,
            "model_checkpoint": model_checkpoint,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        output_dir / "training_state.pt",
    )
    run_logger.info(
        "  Training state saved (epoch=%d, step=%d, checkpoint=%s)",
        state.epoch, state.global_step, model_checkpoint,
    )


def _load_training_state(output_dir: Path) -> dict:
    """Read ``training_state.pt``.

    Loaded onto the CPU: the state may have been written under a different GPU visibility, and
    ``optimizer.load_state_dict`` moves the moments back to wherever the parameters now live.

    Raises:
        FileNotFoundError: when the directory holds no prior run.
    """
    state_path = output_dir / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            f"No training state found at {state_path}. Cannot resume without a prior training "
            "run in this directory (checkpoint_mode='none' writes one only at the end of a run)."
        )
    return torch.load(state_path, weights_only=False, map_location="cpu")


# ==============================================================================================
# The run
# ==============================================================================================

def _build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainConfig, total_steps: int,
    run_logger: logging.Logger,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup, then cosine decay to the floor (or a flat hold at the peak).

    ``total_steps`` is the whole run: ``steps_per_epoch * num_epochs``. A warmup longer than the
    run is clamped to it, which is the only case in which ``"cosine"`` has no cosine phase at all.
    """
    warmup_steps = min(config.warmup_steps, total_steps)
    post_warmup_steps = total_steps - warmup_steps

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)

    if config.lr_schedule == "cosine" and post_warmup_steps > 0:
        eta_min = config.lr_floor * config.learning_rate
        floor_str = f", floor={config.lr_floor}" if config.lr_floor > 0 else ""
        decay = CosineAnnealingLR(optimizer, T_max=post_warmup_steps, eta_min=eta_min)
        run_logger.info("LR schedule: warmup(%d) → cosine(%d%s)",
                        warmup_steps, post_warmup_steps, floor_str)
        return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])

    constant = ConstantLR(optimizer, factor=1.0, total_iters=post_warmup_steps)
    run_logger.info("LR schedule: warmup(%d) → constant(%d)", warmup_steps, post_warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, constant], milestones=[warmup_steps])


@torch.no_grad()
def validation_loss(student: nn.Module, dataloader: DataLoader) -> dict[str, float]:
    """Cross-entropy on the held-out documents: the number that says whether a run over-fits.

    Token-weighted, so a short trailing batch does not count as much as a full one, and reported
    with its perplexity (``exp(loss)``) because that is the unit every other domain measurement in
    this package is in.

    It leaves the training stream untouched, which is what makes it safe to add to a run whose
    numbers are being compared against another implementation: no gradients, no optimizer, and a
    loader built with ``shuffle=False``, which therefore takes no generator. (A ``DataLoader`` with
    ``generator=None`` still draws a ``_base_seed`` from torch's global CPU RNG once per epoch.
    That is harmless whenever nothing in the training path reads the global stream, which holds for
    a **seeded** sampler -- every anchor draw, the embedding term included, then goes through its
    private generator, and LoRA dropout is 0. An unseeded sampler (``Sampler(..., seed=None)``,
    the mode that reproduces the reference implementation call for call) does read the global
    stream, so a run configured that way *with* the embedding term live is the one case where a
    validation pass shifts the anchor's draws.) The student is put in ``eval()`` for the pass and
    restored to whatever mode it was in.

    ``tokens`` counts label positions that are not ``-100``, which is one per sequence more than
    the causal-LM loss averages over, since the labels are shifted inside the model. That is
    deliberate parity with the research code's ``evaluate_holdout`` (``scripts/lra_run_experiment.py``),
    which weights the same way: it is what makes the two implementations' held-out losses and
    token counts the same quantity, and the equivalence harness compares both. Do not "fix" the
    arithmetic without saying so there.

    Returns:
        ``{"loss": ..., "perplexity": ..., "tokens": ...}``; an empty split gives loss ``0.0``.
    """
    was_training = student.training
    student.eval()
    device = next(student.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    try:
        for batch in dataloader:
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            outputs = student(
                input_ids=batch["input_ids"].to(device),
                attention_mask=None if attention_mask is None else attention_mask.to(device),
                labels=labels,
            )
            # Weighted by the tokens that actually carried a target: left padding writes -100.
            batch_tokens = int((labels != -100).sum().item())
            total_loss += float(outputs.loss.item()) * batch_tokens
            total_tokens += batch_tokens
    finally:
        if was_training:
            student.train()

    loss = total_loss / max(total_tokens, 1)
    return {"loss": loss, "perplexity": float(torch.exp(torch.tensor(loss))),
            "tokens": total_tokens}


@torch.no_grad()
def _measure_baseline(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader,
    sampler: Sampler | None,
    adapter: ModelAdapter,
    config: TrainConfig,
    layer_weights: list[float] | None,
    run_logger: logging.Logger,
) -> dict[str, float]:
    """The untrained student's losses over the whole training set, before epoch 1.

    Averaged per micro-batch, so it is directly comparable with an epoch's averages -- which is
    the point: the anchor starts at exactly zero (the student *is* the teacher), so this is the
    reference every later epoch is read against.
    """
    was_training = student.training
    student.eval()
    totals = dict(total=0.0, content=0.0, anchor=0.0, qkv=0.0, mlp=0.0, lm_head=0.0,
                  embed=0.0, mu=0.0)
    n = 0
    try:
        for batch in dataloader:
            _, metrics = train_step(
                student=student, teacher=teacher, batch=batch, sampler=sampler, adapter=adapter,
                config=config, layer_weights=layer_weights, accumulation_steps=1,
            )
            totals["total"] += metrics.loss_total
            totals["content"] += metrics.loss_content
            totals["anchor"] += metrics.loss_anchor
            totals["qkv"] += metrics.loss_qkv
            totals["mlp"] += metrics.loss_mlp
            totals["lm_head"] += metrics.loss_lm_head
            totals["embed"] += metrics.loss_embed
            totals["mu"] += metrics.loss_mu
            n += 1
    finally:
        if was_training:
            student.train()

    baseline = {f"loss_{key}": value / max(n, 1) for key, value in totals.items()}
    run_logger.info("Baseline (epoch 0): content=%.4f anchor=%.4e mu=%.4e over %d micro-batches",
                    baseline["loss_content"], baseline["loss_anchor"], baseline["loss_mu"], n)
    return baseline


def train(
    teacher: nn.Module,
    student: nn.Module,
    dataset: ChunkedCorpus,
    sampler: Sampler | None,
    adapter: ModelAdapter,
    config: TrainConfig,
    output_dir: str | Path,
    logger: logging.Logger | None = None,
    resume: bool = False,
    tokenizer=None,
    val_dataset: ChunkedCorpus | None = None,
) -> TrainingState:
    """Train ``student`` against the frozen ``teacher`` and write the run to ``output_dir``.

    Writes ``config.json``, ``training_history.json``, ``final_model/`` (the LoRA adapter, or
    the full weights) and ``training_state.pt``, plus any periodic checkpoints the config asks
    for. ``final_model`` is what ships: there is no separate "best" checkpoint, because a
    checkpoint chosen by the lowest loss is chosen on one axis of a method whose whole point is
    the trade between two.

    Args:
        dataset: re-chunked at the start of every epoch, from an offset determined by
            ``config.seed`` and the epoch number, for positional diversity.
        sampler: the ``p(h)`` sampler; ``None`` trains with no anchor at all (the unanchored
            control), which is the one case where mu is the only preservation pressure.
        val_dataset: the documents held out of training (``config.val_fraction`` of them, built
            by the caller with the same chunker). When given, every epoch ends with a
            :func:`validation_loss` pass whose loss and perplexity land in the history, so a run
            that has started to over-fit says so at the epoch it happens rather than at the end.
            It is chunked once, at offset 0, and never re-chunked -- the epoch-to-epoch change
            has to come from the model, not from the text moving.
        resume: continue a run in this directory. The saved optimizer moments, epoch counter,
            history and scheduler position are restored, and a saved LoRA adapter is re-attached
            to ``student`` *trainable* -- attaching one for inference instead is the classic
            silent no-op, so this path asserts that something can train. A LoRA resume whose
            checkpoint saved no adapter is refused up front
            (:class:`ResumeSourceHasNoAdapter`). Re-attaching builds a new
            model object, so a resumed run's trained model is the one in
            :attr:`TrainingState.model` (and on disk under ``final_model/``), not the ``student``
            the caller passed in.

    Returns:
        The final :class:`TrainingState`, with ``history``, ``baseline`` and the trained ``model``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_logger = logger or logging.getLogger("lfa.train")

    if config.full_weight:
        run_logger.warning(FULL_WEIGHT_NOTICE)

    # Said here, once, rather than in `train_step`, which would repeat it every micro-batch. The
    # conditions are exactly those under which `include_embed` there comes out False *because the
    # table is missing* -- freeze_embed and lambda_qkv=0 switch the term off deliberately.
    if (config.lambda_qkv > 0 and sampler is not None and not config.freeze_embed
            and not sampler.has_embedding_lookup()):
        run_logger.warning(EMBED_ANCHOR_DISABLED_NOTICE)

    num_layers = adapter.num_layers(teacher)
    layer_weights = None
    if config.anchor_end_ratio < 1.0:
        layer_weights = compute_layer_weights(
            num_layers, end_ratio=config.anchor_end_ratio, schedule=config.anchor_schedule,
            normalize=True,
        )
        run_logger.info("Layer weights (%s, end_ratio=%s): L0=%.4f, L%d=%.4f",
                        config.anchor_schedule, config.anchor_end_ratio, layer_weights[0],
                        num_layers - 1, layer_weights[-1])

    state = TrainingState()
    saved = None
    if resume:
        # Restore before the optimizer is built: re-attaching an adapter creates new parameter
        # objects, and an optimizer built over the pre-resume ones would optimize nothing.
        saved = _load_training_state(output_dir)
        state.epoch = saved["epoch"]
        state.global_step = saved["global_step"]
        state.history = saved["history"]
        state.baseline = saved.get("baseline")

        checkpoint_dir = output_dir / saved.get("model_checkpoint", "final_model")
        if config.use_lora and not (checkpoint_dir / "adapter_config.json").exists():
            # Said here rather than left to a later symptom. A LoRA resume whose checkpoint has
            # no adapter used to continue into `optimizer.load_state_dict`, which raises about a
            # parameter-group size mismatch -- true, but only because the saved moments happen to
            # be there to disagree, and only after the run has been set up. There is nothing to
            # resume: `load_student` hands back a fully trainable model, so the frozen-model
            # guard below would not catch it either.
            raise ResumeSourceHasNoAdapter(
                f"resume from {output_dir}: this is a LoRA run, but {checkpoint_dir} carries no "
                "adapter_config.json, so there is no adapter to continue. Training from here "
                "would start a fresh adapter over the restored epoch counter, history and "
                "optimizer moments, and then overwrite the checkpoint it resumed from. Point at "
                "a run that saved an adapter, or start a new run without resume."
            )
        if config.use_lora and (checkpoint_dir / "adapter_config.json").exists():
            if not hasattr(student, "peft_config"):
                student = load_adapter_for_training(student, checkpoint_dir)
                run_logger.info("  Adapter re-attached (trainable) from %s", checkpoint_dir)
        assert_trainable_params(student, context=f"resume from {output_dir}")

    state.model = student            # the object a resume re-attached, not the caller's base
    optimizer = AdamW(student.parameters(), lr=config.learning_rate,
                      weight_decay=config.weight_decay)

    pad_token_id = 0
    for source in (tokenizer, getattr(dataset, "tokenizer", None)):
        if getattr(source, "pad_token_id", None) is not None:
            pad_token_id = source.pad_token_id
            break
    dataloader = make_dataloader(dataset, config.batch_size, True, config.seed, pad_token_id)
    # shuffle=False: the held-out pass takes no generator. With a seeded sampler nothing in the
    # training path reads the global RNG either, so the training stream is bit-identical to a run
    # without validation -- which `test_the_held_out_pass_leaves_the_training_stream_untouched`
    # asserts by comparing the two runs' adapters byte for byte, with the embedding anchor both
    # off and on. (Unseeded, the embedding draw follows the global stream by design; see
    # `lfa.losses.embed_anchor_loss`.)
    val_dataloader = (None if val_dataset is None else
                      make_dataloader(val_dataset, config.batch_size, False, config.seed,
                                      pad_token_id))

    # Ceiling division: a trailing partial accumulation window takes an optimizer step too.
    steps_per_epoch = ((len(dataloader) + config.gradient_accumulation_steps - 1)
                       // config.gradient_accumulation_steps)
    scheduler = _build_scheduler(optimizer, config, steps_per_epoch * config.num_epochs,
                                 run_logger)

    if resume:
        # torch's LR schedulers are chainable: each step multiplies the group's CURRENT lr rather
        # than recomputing it from the base. `load_state_dict` overwrites that lr with whatever
        # was saved mid-schedule, so fast-forwarding from there compounds the saved value instead
        # of replaying the schedule. Put the peak lr back first, then fast-forward.
        peak_lrs = [group["lr"] for group in optimizer.param_groups]
        if "optimizer_state_dict" in saved:
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            run_logger.info("  Optimizer state restored")
        for group, lr in zip(optimizer.param_groups, peak_lrs):
            group["lr"] = lr
        with warnings.catch_warnings():
            # Fast-forwarding a fresh scheduler is the only way to restore its position, and
            # torch cannot tell that from the real mistake it warns about (stepping the
            # scheduler before the optimizer).
            warnings.filterwarnings("ignore", message=r".*before `optimizer\.step\(\)`.*")
            for _ in range(state.global_step):
                scheduler.step()
        run_logger.info("Resuming from epoch %d, global_step %d", state.epoch, state.global_step)

    # Merge rather than clobber: a caller may already have written run metadata here.
    config_path = output_dir / "config.json"
    saved_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    saved_config.update(config.to_dict())
    config_path.write_text(json.dumps(saved_config, indent=2))

    run_logger.info("%s training for %d epochs", "Resuming" if resume else "Starting",
                    config.num_epochs)
    run_logger.info("  λ_qkv=%s, λ_mlp=%s, μ=%s, n_anchor=%d",
                    config.lambda_qkv, config.lambda_mlp, config.mu, config.n_anchor_samples)
    run_logger.info("  Batch size: %d x %d = %d", config.batch_size,
                    config.gradient_accumulation_steps,
                    config.batch_size * config.gradient_accumulation_steps)
    if config.freeze_embed:
        run_logger.info("  Frozen: embed_tokens + lm_head (their anchor terms are off)")

    # The baseline measures the model at *initialization*; a resumed run keeps the one it saved.
    if not resume:
        state.baseline = _measure_baseline(student, teacher, dataloader, sampler, adapter,
                                           config, layer_weights, run_logger)

    for epoch in range(state.epoch, config.num_epochs):
        dataset.rechunk(epoch, config.seed)

        run_logger.info("Epoch %d/%d (%d chunks, %d tokens)", epoch + 1, config.num_epochs,
                        len(dataset), dataset.total_tokens())

        epoch_metrics, state.global_step = train_epoch(
            student=student, teacher=teacher, dataloader=dataloader, sampler=sampler,
            adapter=adapter, config=config, optimizer=optimizer, scheduler=scheduler,
            layer_weights=layer_weights, global_step=state.global_step, run_logger=run_logger,
        )
        state.epoch = epoch + 1

        run_logger.info(
            "Epoch %d complete: loss=%.4f content=%.4f anchor=%.4e μ=%.4e grad_norm=%.4f (%.1fs)",
            epoch + 1, epoch_metrics.avg_loss_total, epoch_metrics.avg_loss_content,
            epoch_metrics.avg_loss_anchor, epoch_metrics.avg_loss_mu,
            epoch_metrics.avg_grad_norm, epoch_metrics.duration_seconds,
        )

        record = {
            "epoch": epoch + 1,
            "global_step": state.global_step,
            "loss_total": epoch_metrics.avg_loss_total,
            "loss_content": epoch_metrics.avg_loss_content,
            "loss_anchor": epoch_metrics.avg_loss_anchor,
            "loss_qkv": epoch_metrics.avg_loss_qkv,
            "loss_mlp": epoch_metrics.avg_loss_mlp,
            "loss_lm_head": epoch_metrics.avg_loss_lm_head,
            "loss_embed": epoch_metrics.avg_loss_embed,
            "loss_mu": epoch_metrics.avg_loss_mu,
            "grad_norm": epoch_metrics.avg_grad_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "duration": epoch_metrics.duration_seconds,
        }
        if val_dataloader is not None:
            validation = validation_loss(student, val_dataloader)
            record["val_loss"] = validation["loss"]
            record["val_perplexity"] = validation["perplexity"]
            record["val_tokens"] = validation["tokens"]
            run_logger.info("  Held-out: loss=%.4f perplexity=%.3f over %d tokens",
                            validation["loss"], validation["perplexity"], validation["tokens"])
        state.history.append(record)

        if config.checkpoint_mode != "none" and (epoch + 1) % config.checkpoint_every == 0:
            name = ("latest_model" if config.checkpoint_mode == "rolling"
                    else f"checkpoint_epoch_{epoch + 1}")
            _save_student(student, output_dir / name)
            run_logger.info("  Checkpoint saved to %s", output_dir / name)
            _save_training_state(state, optimizer, output_dir, name, run_logger)

    _save_student(student, output_dir / "final_model")
    run_logger.info("Final model saved to %s", output_dir / "final_model")
    _save_training_state(state, optimizer, output_dir, "final_model", run_logger)

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(state.history, indent=2))
    run_logger.info("Training history saved to %s", history_path)

    return state
