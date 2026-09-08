"""Loading the teacher and student, wrapping the student in LoRA, and fusing an adapter.

Layerwise Function Anchoring (LFA) trains a student against a frozen copy of the same model,
so a run holds two models at once. Both are loaded here, in the same dtype, from the same
checkpoint, so that at step 0 the student *is* the teacher and the anchoring signal starts at
exactly zero.

**One GPU by default.** :func:`resolve_device` pins a single device and refuses ``"auto"``
unless the caller passes ``allow_sharding=True``. Sharding a model across cards is model
parallelism -- it buys memory, not speed -- and at the sizes LFA targets it measured ~8% slower
than pinning one card, because every anchoring hidden state then crosses a device boundary. Use
it only when the student plus the frozen teacher genuinely do not fit beside each other.

The trainer moves between three states of the student, all of them handled here: a fresh model
(:func:`load_student`), a LoRA-wrapped one (:func:`apply_lora`), and a saved adapter re-attached
to keep training (:func:`load_adapter_for_training`). :func:`fuse` produces the plain,
adapter-free model that evaluation and release consume.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import sysconfig
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters import ModelAdapter

if TYPE_CHECKING:                       # peft is imported lazily, only where it is used
    from peft import PeftModel

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DEVICE",
    "ShardingRefused",
    "NoTrainableParameters",
    "resolve_device",
    "resolve_model_path",
    "load_tokenizer",
    "load_teacher",
    "load_student",
    "apply_lora",
    "MissingBuildToolchain",
    "check_gpu_toolchain",
    "assert_trainable_params",
    "load_adapter_for_training",
    "fuse",
]

#: The single card LFA pins by default. See :func:`resolve_device`.
DEFAULT_DEVICE = "cuda:0"

#: Device-map values that ask transformers to spread one model over several devices.
_SHARDING_MAPS = ("auto", "balanced", "balanced_low_0", "sequential")


class ShardingRefused(RuntimeError):
    """Raised when a device request would shard one model across devices unasked."""


class NoTrainableParameters(RuntimeError):
    """Raised when a model that is about to be trained has nothing with ``requires_grad``."""


class MissingBuildToolchain(RuntimeError):
    """Raised when a CUDA run would need to compile and this machine cannot.

    torch's triton backend JIT-compiles a small CUDA shim at the first GPU kernel launch, which
    needs ``Python.h`` and a C compiler. Interpreters that ship their own headers (uv- and
    conda-managed ones, most Docker images) have both; a distribution ``python3`` without its
    ``-dev`` package has neither, and the failure otherwise arrives as a gcc error and a
    sixty-line traceback through torch, minutes into a run.
    """


#: Set to skip :func:`check_gpu_toolchain` on a machine where torch never JIT-compiles.
TOOLCHAIN_CHECK_OFF = "LFA_SKIP_TOOLCHAIN_CHECK"


def check_gpu_toolchain(device: str | dict) -> None:
    """Refuse a CUDA run up front when the machine cannot compile triton's shim.

    Called from :func:`resolve_device`, so it runs once per workspace operation, before a model
    is loaded -- the point of it is to say this in one line at second zero rather than as
    somebody else's traceback at minute five. It checks only what is cheap and decisive: the
    presence of ``Python.h`` for the running interpreter, and a C compiler on ``PATH``.

    Raises:
        MissingBuildToolchain: naming what is missing, for this interpreter, with the remedy.
    """
    if os.environ.get(TOOLCHAIN_CHECK_OFF):
        return
    if not _single_device(device).startswith("cuda"):
        return

    header = Path(sysconfig.get_paths()["include"]) / "Python.h"
    compiler = next((found for candidate in (os.environ.get("CC"), "gcc", "cc", "clang")
                     if candidate and (found := shutil.which(candidate))), None)
    missing = []
    if not header.is_file():
        missing.append(f"the Python development headers ({header} does not exist)")
    if compiler is None:
        missing.append("a C compiler on PATH (gcc, cc or clang)")
    if not missing:
        return

    raise MissingBuildToolchain(
        f"This machine cannot compile for the GPU, and a CUDA run needs to: {' and '.join(missing)}"
        f" is missing for {sys.executable}. torch's triton backend compiles a small CUDA shim at "
        "the first GPU kernel launch, so without them the run dies mid-training in gcc rather "
        "than here. Install your distribution's development package for this interpreter "
        "(`python3-dev` / `python3.13-dev`, plus `build-essential`), or use an interpreter that "
        "ships its own headers (uv- or conda-managed). This is an environment prerequisite, not "
        f"a defect in this package. Set {TOOLCHAIN_CHECK_OFF}=1 to skip this check on a machine "
        "where your torch build never compiles."
    )


def _single_device(device: str | dict) -> str:
    """The device string to test, for a plain device or a single-device map."""
    if isinstance(device, dict):
        values = set(device.values())
        return str(next(iter(values))) if len(values) == 1 else "cuda"
    return str(device)


def resolve_device(device: str | dict | None = None, allow_sharding: bool = False) -> str | dict:
    """The device map to load a model with, defaulting to a single pinned card.

    ``None`` gives :data:`DEFAULT_DEVICE`. An explicit single device (``"cpu"``, ``"cuda:1"``,
    or a dict placing every module on one device) passes through. A request that would spread
    one model across devices -- ``"auto"``, ``"balanced"``, or a dict naming more than one
    device -- raises :class:`ShardingRefused` unless ``allow_sharding=True``.

    Sharding is refused by default because it is the wrong tool for the usual case: it exists to
    fit a model that does not fit on one card, and at LFA's scale it costs speed rather than
    buying it (~8% slower in the research code's measurement, since each anchored hidden state then has
    to hop between cards). Pass ``allow_sharding=True`` deliberately, when the student and the
    frozen teacher together do not fit on a single card.

    Every CUDA resolution also runs :func:`check_gpu_toolchain`, since this is the one place
    every workspace operation passes through before it loads anything.

    Raises:
        ShardingRefused: for a sharding request without ``allow_sharding=True``.
        MissingBuildToolchain: a CUDA device on a machine that cannot compile triton's shim.
    """
    if device is None:
        check_gpu_toolchain(DEFAULT_DEVICE)
        return DEFAULT_DEVICE

    if isinstance(device, dict):
        if allow_sharding or len(set(device.values())) <= 1:
            check_gpu_toolchain(device)
            return device
        raise ShardingRefused(
            f"Device map {device} spreads one model over {sorted(set(device.values()))}. "
            "LFA pins a single device by default: sharding buys memory, not speed, and costs "
            "~8% here because anchored hidden states then cross devices. Pass "
            "allow_sharding=True if the student and the frozen teacher do not fit on one card."
        )

    if device in _SHARDING_MAPS and not allow_sharding:
        raise ShardingRefused(
            f"device={device!r} asks transformers to shard the model across devices. LFA pins a "
            f"single device by default (e.g. {DEFAULT_DEVICE!r}): sharding buys memory, not "
            "speed, and costs ~8% here because anchored hidden states then cross devices. Pass "
            "allow_sharding=True if the student and the frozen teacher do not fit on one card."
        )

    check_gpu_toolchain(device)
    return device


def resolve_model_path(model_id: str, local_files_only: bool = False) -> str:
    """The path to load ``model_id`` from, resolved against the local cache when offline.

    Offline, a bare Hub id is turned into its cache snapshot path so that loading never reaches
    for the network. Online, or for something that is already a path, the id is returned as-is.

    Raises:
        ValueError: offline, when the model is not in the cache.
    """
    if not local_files_only or os.path.exists(model_id):
        return model_id

    cache_root = Path(
        os.environ.get("HF_HUB_CACHE")
        or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    )
    cache_dir = cache_root / ("models--" + model_id.replace("/", "--"))
    snapshots = sorted((cache_dir / "snapshots").iterdir()) if (cache_dir / "snapshots").is_dir() else []
    if not snapshots:
        raise ValueError(
            f"Model {model_id!r} not found in cache at {cache_dir}. Download it with network "
            "access first, or pass a local path."
        )
    return str(snapshots[0])


def load_tokenizer(model_id: str, local_files_only: bool = False) -> AutoTokenizer:
    """The tokenizer for ``model_id``, with a pad token guaranteed (falling back to EOS)."""
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_path(model_id, local_files_only), local_files_only=local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_teacher(
    model_id: str,
    device: str | dict | None = DEFAULT_DEVICE,
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = False,
    allow_sharding: bool = False,
) -> AutoModelForCausalLM:
    """The frozen reference model: every parameter ``requires_grad=False``, in eval mode.

    The teacher is loaded unquantized and in the same dtype as the student, so that the two are
    identical at step 0 and the anchoring loss measures drift rather than load-time error.
    """
    teacher = AutoModelForCausalLM.from_pretrained(
        resolve_model_path(model_id, local_files_only),
        device_map=resolve_device(device, allow_sharding),
        dtype=dtype,
        local_files_only=local_files_only,
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def load_student(
    model_id: str,
    device: str | dict | None = DEFAULT_DEVICE,
    dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = True,
    local_files_only: bool = False,
    allow_sharding: bool = False,
) -> AutoModelForCausalLM:
    """The model being adapted, in train mode with every weight trainable.

    This is the full-weight student. For LoRA training, pass the result to :func:`apply_lora`,
    which freezes the base weights; the anchoring loss is identical either way, since it acts on
    sub-module *outputs* rather than on parameters.

    Gradient checkpointing is on by default: an LFA step holds a teacher, a student and the
    anchoring forward passes at once, so activation memory is the binding constraint.
    """
    student = AutoModelForCausalLM.from_pretrained(
        resolve_model_path(model_id, local_files_only),
        device_map=resolve_device(device, allow_sharding),
        dtype=dtype,
        local_files_only=local_files_only,
    )
    student.train()
    for param in student.parameters():
        param.requires_grad = True
    if gradient_checkpointing:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return student


def apply_lora(
    model: AutoModelForCausalLM,
    adapter: ModelAdapter,
    rank: int = 32,
    alpha: int = 64,
    dropout: float = 0.0,
    freeze_embed: bool = True,
) -> PeftModel:
    """Wrap ``model``'s projections in LoRA adapters, in place, and return the PEFT model.

    The targeted modules are ``adapter.lora_target_modules()`` -- the seven attention and MLP
    projections of the layout. Note that this target set does *not* narrow the objective: the
    weight-anchoring term (mu) always spans all seven projections whatever LoRA is attached to,
    as it does in the research code, so a smaller LoRA target set means fewer trainable
    directions, not a weaker anchor.

    ``freeze_embed=True`` (the recipe) leaves the embedding and the tied LM head frozen: no
    ``modules_to_save``, and weight tying is left alone. A fully trainable output head absorbs
    most of the adaptation pressure and degrades generation quietly, so it is off by default.
    ``freeze_embed=False`` restores it, adding ``modules_to_save=["embed_tokens"]`` with
    ``ensure_weight_tying=True`` so PEFT wraps the LM head over a single shared weight tensor.

    Sharded models are not gathered to one device first, as the research code's version had to do: PEFT
    handles a ``device_map="auto"`` model itself in the pinned transformers/PEFT versions, and
    LFA pins one device by default anyway (see :func:`resolve_device`).
    """
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=adapter.lora_target_modules(),
        modules_to_save=None if freeze_embed else ["embed_tokens"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        ensure_weight_tying=not freeze_embed,
    )
    return get_peft_model(model, config)


def assert_trainable_params(model: nn.Module, context: str = "training") -> None:
    """Raise unless ``model`` has at least one parameter with ``requires_grad=True``.

    A model with nothing to optimize trains silently: the forward pass and the loss still run
    (and the loss even wiggles as the data reshuffles between epochs), but ``grad_norm`` is
    exactly 0, no weight ever moves, and every checkpoint written is a copy of the last. Fail at
    setup instead of after the run.

    Raises:
        NoTrainableParameters: when every parameter is frozen.
    """
    if not any(param.requires_grad for param in model.parameters()):
        raise NoTrainableParameters(
            f"{context}: model has no trainable parameters (every requires_grad is False). "
            "Training would be a silent no-op with grad_norm == 0."
        )


def load_adapter_for_training(base_model: nn.Module, adapter_dir: str | Path) -> PeftModel:
    """Re-attach a saved LoRA adapter to ``base_model`` so that it keeps training.

    ``PeftModel.from_pretrained`` defaults to ``is_trainable=False``, which attaches the adapter
    in *inference* mode with ``requires_grad=False`` on every LoRA parameter. Resuming that way
    is a silent no-op -- the loss still moves as the data reshuffles, but ``grad_norm`` is 0.0
    and every post-resume checkpoint is byte-identical to the one it resumed from. That bug
    corrupted several the research code runs before it was caught, so this function both passes
    ``is_trainable=True`` and asserts afterwards that something can actually train.

    Raises:
        NoTrainableParameters: if the re-attached model has nothing to train.
    """
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=True)
    assert_trainable_params(model, context=f"resume from {adapter_dir}")
    return model


def fuse(
    adapter_dir: str | Path,
    base_model_id_or_path: str,
    out_dir: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = False,
) -> Path:
    """Fold a LoRA adapter into its base model and save the plain model plus tokenizer.

    The result carries no PEFT wrapper and no adapter files: it loads with
    ``AutoModelForCausalLM.from_pretrained`` like any other checkpoint, which is what evaluation,
    conversion and release want. Merging happens on CPU, so fusing does not need a free GPU.

    Returns:
        The output directory.
    """
    from peft import PeftModel

    out_dir = Path(out_dir)
    base_path = resolve_model_path(base_model_id_or_path, local_files_only)
    logger.info("Fusing adapter %s into %s", adapter_dir, base_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=dtype, device_map="cpu", local_files_only=local_files_only
    )
    merged = PeftModel.from_pretrained(base_model, str(adapter_dir)).merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    load_tokenizer(base_model_id_or_path, local_files_only).save_pretrained(out_dir)
    logger.info("Fused model and tokenizer written to %s", out_dir)
    return out_dir
