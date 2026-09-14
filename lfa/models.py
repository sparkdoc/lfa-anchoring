"""Loading the teacher and student, wrapping the student in LoRA, and fusing an adapter.

Layerwise Function Anchoring (LFA) trains a student against a frozen reference of the same model.
**Under LoRA that reference is already inside the student**: PEFT freezes the base weight of every
adapted linear, the norms are never adapter targets, and ``freeze_embed`` leaves the embedding and
the tied head alone -- so the teacher's output on an anchor vector ``h`` is what the student's own
sub-module computes with its adapter switched off. :class:`AdapterDisabledTeacher` reads it that
way and no second model is loaded (``teacher_mode="adapter_disabled"``, the default for a LoRA run
since 0.1.1). Full-weight training moves ``W_base``, so there the student holds no teacher and
:func:`load_teacher` loads a real one, in the same dtype from the same checkpoint, so that at step
0 the student *is* the teacher and the anchoring signal starts at exactly zero.

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

import functools
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
    "TEACHER_MODES",
    "ShardingRefused",
    "NoTrainableParameters",
    "TeacherModeRefused",
    "resolve_device",
    "resolve_model_path",
    "load_tokenizer",
    "load_teacher",
    "load_student",
    "apply_lora",
    "resolve_teacher_mode",
    "AdapterDisabledTeacher",
    "make_adapter_disabled_teacher",
    "frozen_reference",
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


class TeacherModeRefused(ValueError):
    """Raised when the frozen teacher cannot be read the way ``teacher_mode`` asks for.

    Every case is one where the student's frozen base is *not* the teacher's weights, or where the
    caller has asked for two incompatible things at once. A ``ValueError``, so the CLI reports it
    as one line rather than as a traceback.
    """


#: What ``teacher_mode`` accepts. ``"auto"`` resolves by training mode -- see
#: :func:`resolve_teacher_mode`.
TEACHER_MODES = ("auto", "separate", "adapter_disabled")


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
    buying it (~8% slower in the research code's measurement, since each anchored hidden state
    then has to hop between cards). Pass ``allow_sharding=True`` deliberately, when the student
    and the frozen teacher together do not fit on a single card.

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

    Sharded models are not gathered to one device first, as the research code's version had to
    do: PEFT handles a ``device_map="auto"`` model itself in the pinned transformers/PEFT
    versions, and LFA pins one device by default anyway (see :func:`resolve_device`).
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


# ==============================================================================================
# The teacher a LoRA student already holds
# ==============================================================================================
#
# PEFT freezes the base weight of every module it adapts and routes the forward through a wrapper
# that adds a trainable delta to the base's output. Disable the wrapper and the student computes
# `W_base . h` -- which is exactly what the separate teacher computes, on the same tensor, in the
# same dtype, on the same device. So under LoRA the second model is pure redundancy.
#
# Two shapes of read, and the difference matters:
#
#   * A module that is CALLED (q/k/v/o, the MLP's three projections, lm_head, embed_tokens) is
#     handed back as its frozen base -- the very `nn.Linear` the separate teacher would have used.
#     No wrapper, no state, nothing to restore.
#   * A CONTAINER whose forward composes adapted children (the MLP, whose `down(act(gate(x)) *
#     up(x))` is anchored as one function, and the whole model) is not re-implemented. The view
#     switches the adapters beneath it off, calls the container's own forward, and switches them
#     back -- so the nonlinearity and the composition cannot drift from the teacher's.
#
# The switch is `_disable_adapters`, set directly. PEFT's `enable_adapters(False)` also calls
# `requires_grad_(False)` on the adapter tensors: a teacher *read* that silently freezes the
# student is precisely the class of bug `load_adapter_for_training` exists to prevent, so it is
# not used here. `_disable_adapters` is what `lora.Linear.forward` branches on and has no other
# effect.
#
# Nothing here knows a model's layout. The view mirrors whatever tree the student has, so a
# `ModelAdapter` walks it exactly as it walks the student -- which is the rule this package holds
# to everywhere else: layout lives in `lfa.adapters` and nowhere else.


@functools.lru_cache(maxsize=1)
def _adapter_wrapper_kinds() -> tuple[type, ...]:
    """The PEFT base classes whose forward can add a trainable delta.

    Cached because it is asked on the hot path: every module under an anchored container is
    tested on every teacher read, and a bare ``from x import y`` per test is a measurable share
    of a training step at 28 layers x 16 anchor draws.

    TWO of them, and missing the second is a real defect rather than a tidiness point:

    * ``BaseTunerLayer`` -- ``lora.Linear`` and its kin, which add ``s.BA`` to the base output.
    * ``AuxiliaryTrainingWrapper`` -- ``ModulesToSaveWrapper``, which :func:`apply_lora` puts
      around ``embed_tokens`` (and, through ``ensure_weight_tying``, the head) whenever
      ``freeze_embed`` is False. It keeps ``original_module`` frozen and routes the forward to a
      *trainable copy* instead, and it is **not** a ``BaseTunerLayer``. A view that disabled only
      the LoRA layers would serve the student's trained embeddings from a full-model forward while
      serving the frozen base from every weight read -- silently, and only in that configuration.

    PEFT's own ``disable_adapter()`` context covers both (``LoraModel._set_adapter_layers``), and
    both expose ``_disable_adapters`` and branch on it in ``forward``.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    try:
        from peft.utils.other import AuxiliaryTrainingWrapper
    except ImportError:                      # pragma: no cover - peft without the auxiliary base
        return (BaseTunerLayer,)
    return (BaseTunerLayer, AuxiliaryTrainingWrapper)


def _is_adapter_wrapper(module: nn.Module) -> bool:
    """Whether ``module`` is a PEFT wrapper that can add a trainable delta."""
    return isinstance(module, _adapter_wrapper_kinds())


def _frozen_base_module(module: nn.Module) -> nn.Module:
    """The frozen base underneath a possibly PEFT-wrapped module.

    ``lora.Linear`` (any ``BaseTunerLayer``) gives its ``base_layer``; ``ModulesToSaveWrapper``
    gives its ``original_module``, the copy PEFT keeps frozen while training another; anything
    else is already the base and is returned as it is.
    """
    original = getattr(module, "original_module", None)
    if original is not None:
        return original
    get_base_layer = getattr(module, "get_base_layer", None)
    if callable(get_base_layer):
        return get_base_layer()
    return module


def _is_adapter_parameter(name: str) -> bool:
    """Whether a parameter name belongs to an adapter rather than to the frozen base."""
    return "lora_" in name or "modules_to_save" in name


def _frozen_parameters(module: nn.Module):
    """Yield only the frozen-base parameters under ``module``, never an adapter tensor.

    Load-bearing for dtype as well as for correctness: :func:`lfa.losses._module_io` reads
    ``next(module.parameters())`` for the dtype it casts anchor vectors to, and an adapter tensor
    is not guaranteed to carry the model's dtype.
    """
    for name, parameter in module.named_parameters():
        if not _is_adapter_parameter(name):
            yield parameter


def _refuse_merged_adapters(module: nn.Module) -> None:
    """Refuse a student whose adapter has been merged into the base weight.

    After a merge the delta *is* in ``base_layer.weight``, so the frozen base has stopped being
    the teacher and every read through this view would be the student's own adapted function.
    """
    for child in module.modules():
        if _is_adapter_wrapper(child) and getattr(child, "merged", False):
            raise TeacherModeRefused(
                "adapter-disabled teacher: this student has a merged adapter, so its base weight "
                "is no longer the teacher's. Train from an unmerged adapter, or pass "
                "teacher_mode='separate' to load a real teacher."
            )


class _FrozenModuleView:
    """One module as the teacher sees it: every PEFT wrapper beneath it disabled.

    Built only for a module that still has an adapter wrapper somewhere under it. Anything
    without one is handed back unwrapped instead (:func:`_frozen_view`), so the objects a
    ``ModelAdapter`` finally resolves -- the projections, the norms, the embedding -- are plain
    modules with no indirection at all.

    The view mirrors the student's own tree by attribute and by index, so a ``ModelAdapter``
    resolves sub-modules on it exactly as it does on the student.
    """

    def __init__(self, module: nn.Module, cache: dict[int, "_FrozenModuleView"] | None = None):
        self._module = module
        # Shared across the whole view, so each sub-module is scanned for wrappers once rather
        # than on every accessor call, and so `id()` of a view is stable (an adapter that walks
        # `.model` in a loop compares identities to detect a cycle).
        self._cache = {} if cache is None else cache
        self._wrappers = [child for child in module.modules() if _is_adapter_wrapper(child)]

    # -- the student's tree, mirrored ----------------------------------------------------------

    def __getattr__(self, name):
        if name.startswith("_"):             # never recurse through this view's own state
            raise AttributeError(name)
        return self._as_teacher(getattr(self._module, name))

    def __getitem__(self, index):
        return self._as_teacher(self._module[index])

    def __len__(self) -> int:
        return len(self._module)

    def __iter__(self):
        return (self._as_teacher(child) for child in self._module)

    def _as_teacher(self, value):
        return _frozen_view(value, self._cache) if isinstance(value, nn.Module) else value

    # -- the surface the losses read -----------------------------------------------------------

    def parameters(self, recurse: bool = True):
        return _frozen_parameters(self._module)

    def named_parameters(self, *args, **kwargs):
        for name, parameter in self._module.named_parameters(*args, **kwargs):
            if not _is_adapter_parameter(name):
                yield name, parameter

    def modules(self):
        """The wrapped module's real sub-modules, wrappers included and unfiltered.

        Deliberately not a view: the callers are shape inspections
        (:func:`lfa.artifact.schema._site_input_width` looks for the first ``nn.Linear`` under a
        composite site to read its input width), and a PEFT wrapper is not an ``nn.Linear`` while
        the ``base_layer`` beneath it is -- so the unfiltered walk gives the base's own width.
        """
        return self._module.modules()

    def __bool__(self) -> bool:
        # Without this, `__len__` would answer truthiness and raise TypeError for a module that
        # has no length. A view is always a real object.
        return True

    @property
    def training(self) -> bool:
        return self._module.training

    def __call__(self, *args, **kwargs):
        """The wrapped module's own forward, with every adapter beneath it switched off.

        The merged-adapter check rides on the list this loop already walks rather than taking a
        traversal of its own: this runs once per anchored container per step.
        """
        previous = []
        for wrapper in self._wrappers:
            if getattr(wrapper, "merged", False):
                raise TeacherModeRefused(
                    "adapter-disabled teacher: an adapter has been merged into the base weight "
                    "mid-run, so the base is no longer the teacher. Use teacher_mode='separate'."
                )
            previous.append(wrapper._disable_adapters)
            wrapper._disable_adapters = True
        try:
            return self._module(*args, **kwargs)
        finally:
            for wrapper, was_disabled in zip(self._wrappers, previous):
                wrapper._disable_adapters = was_disabled

    def __repr__(self) -> str:               # pragma: no cover - diagnostics only
        return f"_FrozenModuleView({type(self._module).__name__})"


def _frozen_view(module: nn.Module, cache: dict[int, object] | None = None):
    """``module`` as the teacher sees it: the frozen base itself, or a view over it.

    The cache is consulted **first**, and it caches the plain-module answer as well as the view.
    Deciding whether a module needs a view means walking everything under it, and the losses
    resolve every anchored sub-module of every layer on every step -- so deciding again on each
    access costs a full walk of the model per accessor call per step. Measured at the shipped
    recipe on Qwen3-0.6B (RTX 3090, rank 32, batch 6 x 512, 16 anchor samples): 0.7228 s/step
    deciding each time against 0.7087 s/step with this lookup, against a ``separate`` arm at
    0.7018 -- 2% of the whole run for a question whose answer cannot change.

    Keying by ``id`` is sound because everything the cache answers for is reachable from the
    student the view holds, so no entry can be collected and no id reused while the view lives.
    """
    if cache is not None and (cached := cache.get(id(module))) is not None:
        return cached
    base = _frozen_base_module(module)
    if not any(_is_adapter_wrapper(child) for child in base.modules()):
        view = base                          # the very object a separate teacher would have used
    else:
        view = _FrozenModuleView(base, cache)
    if cache is not None:
        cache[id(module)] = view
    return view


class AdapterDisabledTeacher(_FrozenModuleView):
    """The frozen teacher, read out of a LoRA student instead of loaded as a second model.

    A :class:`~lfa.adapters.ModelAdapter` resolves every anchored sub-module on this exactly as it
    does on the student, and each one comes back as the student's own frozen base. The anchoring
    loss, mu's weight term and the layer-0 embedding table are then computed against the same
    tensors a separate teacher would have held -- bit for bit, not approximately.

    ``eval()`` and ``train()`` are no-ops that return ``self``. The caller means "put the teacher
    in eval mode", and the teacher here *is* the student, which must stay in whatever mode
    training left it in. (No numerical difference at the shipped recipe, whose dropouts are zero,
    but silently flipping the student out of train mode from a teacher read would be a real trap.)

    Raises:
        TeacherModeRefused: if ``student`` carries no LoRA adapter, or carries a merged one.
    """

    def __init__(self, student: nn.Module):
        from peft.tuners.tuners_utils import BaseTunerLayer

        if not any(isinstance(module, BaseTunerLayer) for module in student.modules()):
            raise TeacherModeRefused(
                "teacher_mode='adapter_disabled' needs a LoRA (PEFT) student: no adapter layer "
                "was found in this model. Full-weight training moves the base weights, so the "
                "student holds no teacher there -- use teacher_mode='separate'."
            )
        _refuse_merged_adapters(student)
        super().__init__(student)

    def eval(self) -> "AdapterDisabledTeacher":
        return self

    def train(self, mode: bool = True) -> "AdapterDisabledTeacher":
        return self

    @property
    def training(self) -> bool:
        return False

    def __call__(self, *args, **kwargs):
        """A full-model forward as the teacher would run it: adapters off, in eval mode."""
        was_training = self._module.training
        self._module.eval()
        try:
            return super().__call__(*args, **kwargs)
        finally:
            if was_training:
                self._module.train()

    def __repr__(self) -> str:               # pragma: no cover - diagnostics only
        return f"AdapterDisabledTeacher(student={type(self._module).__name__})"


def make_adapter_disabled_teacher(student: nn.Module) -> AdapterDisabledTeacher:
    """The teacher view over a LoRA ``student`` (see :class:`AdapterDisabledTeacher`)."""
    return AdapterDisabledTeacher(student)


def frozen_reference(model: nn.Module):
    """``model``'s frozen base, whether or not LoRA has been attached to it yet.

    A LoRA-wrapped student gives an :class:`AdapterDisabledTeacher`. A student that has *not* been
    wrapped -- the resume path, where the trainer re-attaches the saved adapter itself -- is
    returned unchanged: it was loaded from the base checkpoint moments ago and its weights are
    that base.

    For the reads that happen before training starts (checking the artifact against the model, and
    rebuilding the layer-0 embedding table) this is the teacher, in either state.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    if any(isinstance(module, BaseTunerLayer) for module in model.modules()):
        return make_adapter_disabled_teacher(model)
    return model


def resolve_teacher_mode(mode: str, *, use_lora: bool, full_weight: bool) -> str:
    """Turn ``"auto"`` into the mode a run will actually use, and refuse the impossible one.

    ``"auto"`` is ``"adapter_disabled"`` for a LoRA run -- the student holds the teacher, so
    loading a second copy would allocate a whole model for nothing -- and ``"separate"`` for
    full-weight training, where ``W_base`` moves and the student is no longer a copy of anything.

    Args:
        mode: one of :data:`TEACHER_MODES`.
        use_lora: whether the run attaches a LoRA adapter.
        full_weight: whether the run trains the full weights.

    Returns:
        ``"separate"`` or ``"adapter_disabled"``.

    Raises:
        TeacherModeRefused: ``"adapter_disabled"`` asked for a full-weight run.
        ValueError: an unknown mode.
    """
    if mode not in TEACHER_MODES:
        raise ValueError(
            f"teacher_mode must be one of {', '.join(TEACHER_MODES)}, got {mode!r}."
        )
    is_lora = use_lora and not full_weight
    if mode == "adapter_disabled" and not is_lora:
        raise TeacherModeRefused(
            "teacher_mode='adapter_disabled' reads the frozen teacher out of the student's own "
            "LoRA base, and full-weight training moves that base -- there would be no teacher "
            "left to read. Use teacher_mode='separate' for a full-weight run, or drop "
            "--full-weight."
        )
    if mode == "auto":
        return "adapter_disabled" if is_lora else "separate"
    return mode


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
    corrupted several research runs before it was caught, so this function both passes
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
