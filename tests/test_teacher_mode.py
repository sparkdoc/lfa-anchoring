"""``teacher_mode``: reading the frozen teacher out of the LoRA student instead of a second copy.

Under LoRA the student already *holds* the teacher. PEFT freezes the base weight of every adapted
linear, the norms are never adapter targets, and ``freeze_embed`` leaves the embedding and the tied
head alone -- so the teacher's output on an anchor vector ``h`` is what the student's own
sub-module computes with its adapter disabled. The separate ``from_pretrained`` teacher is a
byte-for-byte copy of weights already on the card.

Every assertion here is therefore ``torch.equal`` rather than ``allclose``: the claim is not that
the two teachers agree closely, it is that they are the same weights. And every comparison is
repeated **after the adapter has been moved off zero**, because PEFT initializes ``lora_B`` to 0 --
before that the student *is* the base and every check passes with the view broken. That is what the
``stepped`` parametrization is for; the ``False`` variant is kept only to show it is vacuous.

The other parametrization is ``freeze_embed``. With it off, PEFT wraps ``embed_tokens`` (and, via
``ensure_weight_tying``, the head) in a ``ModulesToSaveWrapper`` -- an ``AuxiliaryTrainingWrapper``
and *not* a ``BaseTunerLayer`` -- which keeps ``original_module`` frozen and routes the forward to a
trainable copy. A view that disabled only the LoRA layers would serve the student's *trained*
embeddings from a full-model forward while serving the frozen base from every weight read. That
configuration is the only one that can fail, so it is parametrized rather than assumed away.

Everything runs on the CPU against the tiny fixture model, in float32, with real PEFT layers.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest
import torch
from peft.tuners.tuners_utils import BaseTunerLayer

from lfa.adapters import effective_weight, get_adapter
from lfa.losses import anchor_loss, lora_factored_weight_loss, weight_loss
from lfa.models import (
    TEACHER_MODES,
    AdapterDisabledTeacher,
    TeacherModeRefused,
    apply_lora,
    frozen_reference,
    make_adapter_disabled_teacher,
    resolve_teacher_mode,
)
from lfa.sampler import Sampler
from lfa.train import TrainConfig

RANK, ALPHA = 2, 4


# ==============================================================================================
# Fixtures: a LoRA student, and a genuinely separate teacher of the same weights
# ==============================================================================================

def _move_adapter_off_zero(student, freeze_embed: bool) -> None:
    """What a few optimizer steps would do, done in one line so the test stays a unit test.

    ``lora_B`` is zero at initialization, so an unstepped student computes exactly the base
    function and every teacher/student comparison is vacuously true. Perturbing ``lora_B`` (and,
    when the embedding is trainable, the ``modules_to_save`` copy) is what makes the checks bite.
    """
    with torch.no_grad():
        for name, parameter in student.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.1)
            elif not freeze_embed and "modules_to_save" in name:
                parameter.add_(0.05)


@pytest.fixture(params=[True, False], ids=["freeze_embed", "trainable_embed"])
def freeze_embed(request) -> bool:
    return request.param


@pytest.fixture(params=[True, False], ids=["stepped", "unstepped"])
def stepped(request) -> bool:
    return request.param


@pytest.fixture
def pair(tiny_model_fresh, freeze_embed, stepped):
    """``(separate_teacher, lora_student, adapter)`` -- two objects, one set of base weights.

    The teacher is a deep copy taken *before* the student is wrapped, frozen and in eval mode, so
    it is exactly what :func:`lfa.models.load_teacher` would have produced from the same
    checkpoint.
    """
    model, _ = tiny_model_fresh
    adapter = get_adapter(model)
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    student = apply_lora(model, adapter, rank=RANK, alpha=ALPHA, freeze_embed=freeze_embed)
    if stepped:
        _move_adapter_off_zero(student, freeze_embed)
    return teacher, student, adapter


@pytest.fixture
def view(pair):
    teacher, student, _ = pair
    return make_adapter_disabled_teacher(student)


def _sampler(path):
    return Sampler(path, device="cpu", seed=0)


# ==============================================================================================
# The accessor contract: what the adapter resolves on the view is the teacher's own module
# ==============================================================================================

def test_module_accessors_match_the_separate_teacher(pair, view):
    """Every module LFA anchors, called on the same input, gives the teacher's output bit for bit."""
    teacher, _, adapter = pair
    torch.manual_seed(0)

    for layer in range(adapter.num_layers(teacher)):
        h = torch.randn(8, teacher.config.hidden_size)
        t_qkv = adapter.qkv_modules(teacher, layer)
        v_qkv = adapter.qkv_modules(view, layer)
        for name in ("q_proj", "k_proj", "v_proj"):
            assert torch.equal(v_qkv[name](h), t_qkv[name](h)), f"layer {layer} {name}"

        o_in = torch.randn(8, adapter.o_proj_module(teacher, layer).in_features)
        assert torch.equal(adapter.o_proj_module(view, layer)(o_in),
                           adapter.o_proj_module(teacher, layer)(o_in)), f"layer {layer} o_proj"

        assert torch.equal(adapter.mlp_module(view, layer)(h),
                           adapter.mlp_module(teacher, layer)(h)), f"layer {layer} mlp"

    head_in = torch.randn(8, teacher.config.hidden_size)
    assert torch.equal(adapter.lm_head_module(view)(head_in),
                       adapter.lm_head_module(teacher)(head_in))

    ids = torch.randint(0, teacher.config.vocab_size, (8,))
    t_embed, t_ln = adapter.embed_modules(teacher)
    v_embed, v_ln = adapter.embed_modules(view)
    assert torch.equal(v_ln(v_embed(ids)), t_ln(t_embed(ids)))
    assert adapter.num_layers(view) == adapter.num_layers(teacher)


def test_weight_reads_on_the_view_are_the_frozen_base(pair, view):
    """``effective_weight`` over the view returns the teacher's weight, adapter delta excluded."""
    teacher, _, adapter = pair
    for layer in range(adapter.num_layers(teacher)):
        for name, module in adapter.qkv_modules(view, layer).items():
            assert torch.equal(effective_weight(module),
                               adapter.qkv_modules(teacher, layer)[name].weight), name
        assert torch.equal(effective_weight(adapter.o_proj_module(view, layer)),
                           adapter.o_proj_module(teacher, layer).weight)
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert torch.equal(effective_weight(getattr(adapter.mlp_module(view, layer), name)),
                               getattr(adapter.mlp_module(teacher, layer), name).weight), name


def test_the_views_parameters_are_base_parameters_only(pair, view):
    """No ``lora_A``/``lora_B`` and no ``modules_to_save`` copy is ever yielded as the teacher's.

    Load-bearing beyond tidiness: ``lfa.losses._module_io`` reads ``next(module.parameters())`` for
    the dtype it casts anchor vectors to, and the caller's device comes from
    ``next(teacher.parameters()).device``.
    """
    _, student, adapter = pair
    names = [name for name, _ in view.named_parameters()]
    assert names, "the view yielded no parameters at all"
    assert not [name for name in names if "lora_" in name or "modules_to_save" in name]
    assert next(view.parameters()) is not None

    for layer in range(adapter.num_layers(student)):
        mlp_names = [name for name, _ in adapter.mlp_module(view, layer).named_parameters()]
        assert not [name for name in mlp_names if "lora_" in name]


# ==============================================================================================
# The losses: the same number, to the last bit, either way
# ==============================================================================================

def test_anchor_loss_is_bit_identical_either_way(pair, view, tiny_artifact):
    """``L_anchor`` from the view equals ``L_anchor`` from the separate teacher, term by term."""
    teacher, student, adapter = pair
    _, path = tiny_artifact

    separate_sampler, view_sampler = _sampler(path), _sampler(path)
    separate_sampler.build_embedding_lookup_from_model(teacher, adapter)
    view_sampler.build_embedding_lookup_from_model(view, adapter)

    from_separate = anchor_loss(teacher, student, separate_sampler, adapter, n_samples=4)
    from_view = anchor_loss(view, student, view_sampler, adapter, n_samples=4)

    assert set(from_separate) == set(from_view)
    for key, value in from_separate.items():
        assert torch.equal(from_view[key], value), f"{key}: {from_view[key]} vs {value}"


def test_the_anchor_is_not_vacuously_zero_when_the_adapter_has_moved(pair, view, tiny_artifact,
                                                                    stepped):
    """The comparison above is worth something only if the number it compares is nonzero."""
    if not stepped:
        pytest.skip("an unstepped adapter anchors at exactly zero, which is the point of `stepped`")
    _, student, adapter = pair
    _, path = tiny_artifact
    sampler = _sampler(path)
    sampler.build_embedding_lookup_from_model(view, adapter)
    assert anchor_loss(view, student, sampler, adapter, n_samples=4)["total"].item() > 0


def test_weight_loss_is_bit_identical_either_way(pair, view):
    """mu's term agrees on both the LoRA fast path and the general path."""
    teacher, student, adapter = pair
    layer_weights = [1.0] * adapter.num_layers(teacher)

    assert torch.equal(weight_loss(view, student, adapter, layer_weights),
                       weight_loss(teacher, student, adapter, layer_weights))
    assert torch.equal(weight_loss(view, student, adapter, layer_weights, force_general=True),
                       weight_loss(teacher, student, adapter, layer_weights, force_general=True))


def test_mus_fast_path_still_verifies_and_engages_under_the_view(pair, view):
    """The fast path's ``base is the teacher`` check is exact, and it passes on the view.

    It is the one place the two teachers are not interchangeable by construction: under the view
    the student's base weight and the teacher's weight are the *same tensor*, so a check written
    as identity rather than as ``torch.equal`` would pass for the wrong reason. It is written as
    ``torch.equal``, and this pins that it still fires.
    """
    teacher, student, adapter = pair
    layer_weights = [1.0] * adapter.num_layers(teacher)

    factored = lora_factored_weight_loss(view, student, adapter, layer_weights)
    assert factored is not None
    assert student._lfa_mu_fastpath_ok is True
    assert torch.equal(factored, lora_factored_weight_loss(teacher, student, adapter,
                                                           layer_weights))


def test_the_layer_zero_table_is_rebuilt_from_the_students_frozen_embedding(pair, view,
                                                                           tiny_artifact):
    """The reconstructed ``input_layernorm(embed_tokens(id))`` table equals the teacher's.

    Layer-0 ``pre_qkv`` is the one artifact entry that is *rebuilt from the model* rather than
    shipped (``Sampler.build_embedding_lookup_from_model``). If the view served the student's
    trainable embedding copy instead of the frozen original, the whole layer-0 anchor would point
    at the student and pull toward nothing.
    """
    teacher, _, adapter = pair
    _, path = tiny_artifact

    from_teacher, from_view = _sampler(path), _sampler(path)
    from_teacher.build_embedding_lookup_from_model(teacher, adapter)
    from_view.build_embedding_lookup_from_model(view, adapter)

    expected = from_teacher.params["embedding_lookup"]
    got = from_view.params["embedding_lookup"]
    assert got["vocab_size"] == expected["vocab_size"]
    assert got["hidden_dim"] == expected["hidden_dim"]
    assert torch.equal(got["pre_qkv_table"], expected["pre_qkv_table"])


def test_full_model_forward_matches_the_separate_teacher(pair, view):
    """The whole-model forward with every adapter wrapper disabled is the teacher's forward.

    Both kinds of PEFT wrapper have to be disabled for this: ``lora.Linear`` (a ``BaseTunerLayer``)
    and ``ModulesToSaveWrapper`` (an ``AuxiliaryTrainingWrapper``, which is not one). The
    ``trainable_embed`` variant is the only one that can catch the second.
    """
    teacher, _, _ = pair
    ids = torch.randint(0, teacher.config.vocab_size, (2, 6))
    with torch.no_grad():
        expected = teacher(input_ids=ids).logits
        got = view(input_ids=ids).logits
    assert torch.equal(got, expected)


def test_every_adapter_wrapper_is_disabled_for_a_teacher_read(pair, view):
    """Observed *during* the forward: every PEFT wrapper that runs, runs disabled.

    The disable set has to be PEFT's own -- ``BaseTunerLayer`` **and**
    ``AuxiliaryTrainingWrapper``, which is what ``LoraModel._set_adapter_layers`` covers. A
    forward pre-hook records each wrapper's ``_disable_adapters`` at the moment it is called, so
    this reads the state the forward actually branched on rather than the state left behind.
    """
    from peft.utils.other import AuxiliaryTrainingWrapper

    _, student, _ = pair
    wrappers = {name: module for name, module in student.named_modules()
                if isinstance(module, (BaseTunerLayer, AuxiliaryTrainingWrapper))}
    assert wrappers, "no adapter wrapper in this student to check"

    observed: dict[str, bool] = {}
    handles = [module.register_forward_pre_hook(
        lambda m, args, name=name: observed.__setitem__(name, m._disable_adapters))
        for name, module in wrappers.items()]
    try:
        ids = torch.randint(0, student.config.vocab_size, (1, 4))
        with torch.no_grad():
            view(input_ids=ids)
    finally:
        for handle in handles:
            handle.remove()

    assert set(observed) == set(wrappers), "some wrapper never ran during the teacher forward"
    assert all(observed.values()), [name for name, off in observed.items() if not off]


# ==============================================================================================
# The view is a READ: it leaves the student exactly as it found it
# ==============================================================================================

def test_the_view_leaves_the_student_untouched(pair, view):
    """requires_grad, train/eval mode and adapter enablement are the same after a teacher read.

    ``enable_adapters(False)`` would have been the obvious way to write the view and is the wrong
    one: it also calls ``requires_grad_(False)`` on the adapter tensors, which silently freezes the
    student mid-run. ``_disable_adapters`` is exactly what ``lora.Linear.forward`` branches on and
    has no other effect.
    """
    _, student, adapter = pair
    student.train()
    before_grad = {name: parameter.requires_grad for name, parameter in student.named_parameters()}
    before_mode = {name: module.training for name, module in student.named_modules()}
    before_disabled = {name: module._disable_adapters for name, module in student.named_modules()
                       if hasattr(module, "_disable_adapters")}
    assert any(before_grad.values()), "the student had nothing trainable to protect"
    assert before_disabled, "no adapter wrapper found to check"

    ids = torch.randint(0, student.config.vocab_size, (1, 4))
    with torch.no_grad():
        view(input_ids=ids)
    h = torch.randn(4, student.config.hidden_size)
    adapter.mlp_module(view, 0)(h)
    view.eval()
    view.train()

    assert {n: p.requires_grad for n, p in student.named_parameters()} == before_grad
    assert {n: m.training for n, m in student.named_modules()} == before_mode
    assert {n: m._disable_adapters for n, m in student.named_modules()
            if hasattr(m, "_disable_adapters")} == before_disabled


def test_the_student_still_computes_its_adapted_function_after_a_teacher_read(pair, view, stepped):
    """The adapters come back on: a teacher read must not turn the student into the teacher."""
    if not stepped:
        pytest.skip("an unstepped adapter is the base function, so there is nothing to tell apart")
    teacher, student, _ = pair
    ids = torch.randint(0, student.config.vocab_size, (1, 5))
    with torch.no_grad():
        view(input_ids=ids)
        after = student(input_ids=ids).logits
        teacher_logits = teacher(input_ids=ids).logits
    assert not torch.equal(after, teacher_logits)


def test_a_merged_adapter_is_refused_rather_than_read_as_the_teacher(pair):
    """Merging folds the delta into the base weight, so the base stops being the teacher."""
    _, student, _ = pair
    student.merge_adapter()
    with pytest.raises(TeacherModeRefused, match="merged"):
        make_adapter_disabled_teacher(student)


# ==============================================================================================
# Resolution, refusals, and what a run records
# ==============================================================================================

def test_the_default_resolution_table():
    """``auto`` is ``adapter_disabled`` under LoRA and ``separate`` for full-weight training."""
    assert resolve_teacher_mode("auto", use_lora=True, full_weight=False) == "adapter_disabled"
    assert resolve_teacher_mode("auto", use_lora=False, full_weight=True) == "separate"
    assert resolve_teacher_mode("auto", use_lora=False, full_weight=False) == "separate"
    assert resolve_teacher_mode("separate", use_lora=True, full_weight=False) == "separate"
    assert (resolve_teacher_mode("adapter_disabled", use_lora=True, full_weight=False)
            == "adapter_disabled")
    assert set(TEACHER_MODES) == {"auto", "separate", "adapter_disabled"}


def test_adapter_disabled_is_refused_for_full_weight_training():
    """Full weight moves ``W_base``, so the student holds no teacher: refused, not approximated."""
    with pytest.raises(TeacherModeRefused, match="full-weight"):
        resolve_teacher_mode("adapter_disabled", use_lora=False, full_weight=True)
    with pytest.raises(TeacherModeRefused, match="full-weight"):
        TrainConfig(teacher_mode="adapter_disabled", use_lora=False, full_weight=True)


def test_an_unknown_teacher_mode_is_refused_by_the_config():
    with pytest.raises(ValueError, match="teacher_mode"):
        TrainConfig(teacher_mode="borrowed")


def test_the_view_refuses_a_student_with_no_adapter(tiny_model_fresh):
    """Nothing to disable means nothing is frozen: a plain model is not its own teacher."""
    model, _ = tiny_model_fresh
    with pytest.raises(TeacherModeRefused, match="LoRA"):
        make_adapter_disabled_teacher(model)


def test_frozen_reference_passes_an_unwrapped_student_through(tiny_model):
    """Before ``apply_lora`` -- a resume -- the student IS the base, so it is its own reference."""
    model, _ = tiny_model
    unwrapped = copy.deepcopy(model)
    assert frozen_reference(unwrapped) is unwrapped


def test_frozen_reference_views_a_wrapped_student(pair):
    _, student, _ = pair
    assert isinstance(frozen_reference(student), AdapterDisabledTeacher)


def test_the_config_carries_the_mode_into_config_json():
    """``teacher_mode`` is a recorded frame field, so a run says which teacher it trained against."""
    assert "teacher_mode" in TrainConfig().to_dict()
    assert TrainConfig().teacher_mode == "auto"
    assert json.loads(json.dumps(TrainConfig().to_dict()))["teacher_mode"] == "auto"


# ==============================================================================================
# Through the trainer: no second model, and a run that says which teacher it trained against
# ==============================================================================================

def _training_setup(tiny_model, tiny_texts, tiny_artifact, tmp_path):
    """(student, dataset, sampler, adapter, config) for a two-epoch CPU run, as `test_train` does."""
    from lfa.corpus import ChunkedCorpus

    model, tokenizer = tiny_model
    _, artifact_path = tiny_artifact
    base_dir = tmp_path / "base"
    model.config.save_pretrained(base_dir)

    adapter = get_adapter(model)
    student = copy.deepcopy(model)
    student.config._name_or_path = str(base_dir)
    student.name_or_path = str(base_dir)
    student = apply_lora(student, adapter, rank=RANK, alpha=ALPHA)

    config = TrainConfig(
        lambda_qkv=10.0, lambda_mlp=10.0, mu=0.05, n_anchor_samples=4, learning_rate=1e-2,
        warmup_steps=1, logging_steps=1, batch_size=2, gradient_accumulation_steps=1,
        num_epochs=1, sequence_length=64, use_lora=True, lora_rank=RANK, lora_alpha=ALPHA,
        teacher_mode="adapter_disabled",
    )
    dataset = ChunkedCorpus(tiny_texts[:8], tokenizer, max_length=64)
    sampler = Sampler(artifact_path, device="cpu", seed=0)
    return student, dataset, sampler, adapter, config


def test_a_run_with_no_teacher_trains_and_records_the_mode(tiny_model, tiny_texts, tiny_artifact,
                                                           tmp_path):
    """``train(teacher=None, ...)`` under ``adapter_disabled``: it trains, and ``config.json`` says so."""
    from lfa.train import train

    student, dataset, sampler, adapter, config = _training_setup(
        tiny_model, tiny_texts, tiny_artifact, tmp_path)
    sampler.build_embedding_lookup_from_model(frozen_reference(student), adapter)

    state = train(None, student, dataset, sampler, adapter, config, tmp_path)

    assert state.history and state.history[-1]["loss_anchor"] >= 0.0
    assert json.loads((tmp_path / "config.json").read_text())["teacher_mode"] == "adapter_disabled"


def test_a_run_that_is_handed_a_teacher_records_separate(tiny_model, tiny_texts, tiny_artifact,
                                                         tmp_path):
    """``auto`` plus a teacher object is ``separate``: the run really did hold a second model."""
    from lfa.train import train

    model, _ = tiny_model
    student, dataset, sampler, adapter, config = _training_setup(
        tiny_model, tiny_texts, tiny_artifact, tmp_path)
    teacher = copy.deepcopy(model).eval()
    sampler.build_embedding_lookup_from_model(teacher, adapter)

    train(teacher, student, dataset, sampler, adapter,
          dataclasses.replace(config, teacher_mode="auto"), tmp_path)

    assert json.loads((tmp_path / "config.json").read_text())["teacher_mode"] == "separate"


def test_separate_without_a_teacher_is_refused(tiny_model, tiny_texts, tiny_artifact, tmp_path):
    from lfa.train import train

    student, dataset, sampler, adapter, config = _training_setup(
        tiny_model, tiny_texts, tiny_artifact, tmp_path)
    with pytest.raises(TeacherModeRefused, match="no teacher"):
        train(None, student, dataset, sampler, adapter,
              dataclasses.replace(config, teacher_mode="separate"), tmp_path)


def test_adapter_disabled_with_a_teacher_handed_in_is_refused(tiny_model, tiny_texts,
                                                              tiny_artifact, tmp_path):
    """Asking for no second model and then handing one over is a contradiction, not a preference."""
    from lfa.train import train

    model, _ = tiny_model
    student, dataset, sampler, adapter, config = _training_setup(
        tiny_model, tiny_texts, tiny_artifact, tmp_path)
    with pytest.raises(TeacherModeRefused, match="adapter_disabled"):
        train(copy.deepcopy(model), student, dataset, sampler, adapter, config, tmp_path)


def test_a_workspace_stage_loads_no_teacher_by_default(tmp_path, base_dir, tiny_artifact,
                                                       corpus_a, monkeypatch):
    """The shipped default: a LoRA stage never calls ``load_teacher``, and its config says so.

    Monkeypatched to fail rather than counted: the claim is that no second model is loaded at
    all, and a call that returns something would leave the claim resting on a count nobody reads.
    """
    import lfa.workspace as workspace_module
    from conftest import tiny_recipe
    from lfa.workspace import Workspace

    _, artifact_path = tiny_artifact

    def refuse(*args, **kwargs):
        raise AssertionError("load_teacher was called under teacher_mode=adapter_disabled")

    monkeypatch.setattr(workspace_module, "load_teacher", refuse)
    workspace = Workspace.init(tmp_path / "ws", str(base_dir), artifact=str(artifact_path))
    entry = workspace.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")

    assert entry["teacher_mode"] == "adapter_disabled"
    run_config = json.loads((Path(entry["output_dir"]) / "config.json").read_text())
    assert run_config["teacher_mode"] == "adapter_disabled"


def test_a_workspace_stage_can_still_ask_for_a_separate_teacher(tmp_path, base_dir, tiny_artifact,
                                                                corpus_a):
    """``teacher_mode="separate"`` is the pre-0.1.1 behaviour, still available and still recorded."""
    from conftest import tiny_recipe
    from lfa.workspace import Workspace

    _, artifact_path = tiny_artifact
    workspace = Workspace.init(tmp_path / "ws2", str(base_dir), artifact=str(artifact_path))
    entry = workspace.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu",
                            teacher_mode="separate")

    assert entry["teacher_mode"] == "separate"
    assert json.loads((Path(entry["output_dir"]) / "config.json").read_text())["teacher_mode"] \
        == "separate"


def test_a_full_weight_stage_resolves_to_separate(tmp_path, base_dir, tiny_artifact, corpus_a):
    """Full weight moves the base, so ``auto`` keeps the second model rather than refusing."""
    from conftest import tiny_recipe
    from lfa.workspace import Workspace

    _, artifact_path = tiny_artifact
    workspace = Workspace.init(tmp_path / "ws3", str(base_dir), artifact=str(artifact_path))
    entry = workspace.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu",
                            full_weight=True)

    assert entry["teacher_mode"] == "separate"


def test_a_full_weight_stage_refuses_adapter_disabled(tmp_path, base_dir, tiny_artifact, corpus_a):
    from conftest import tiny_recipe
    from lfa.workspace import Workspace

    _, artifact_path = tiny_artifact
    workspace = Workspace.init(tmp_path / "ws4", str(base_dir), artifact=str(artifact_path))
    with pytest.raises(TeacherModeRefused, match="full-weight"):
        workspace.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu",
                        full_weight=True, teacher_mode="adapter_disabled")


def test_a_resumed_stage_reads_the_teacher_out_of_the_re_attached_adapter(tmp_path, base_dir,
                                                                         tiny_artifact, corpus_a):
    """The view attaches to whatever the student is *after* the resume re-attaches the adapter.

    A resume is handed the bare student -- ``lfa.train.train`` re-attaches the saved adapter
    itself, producing a new model object -- so a view built any earlier would have collected no
    adapter wrapper to switch off and would have served the student's own adapted function as the
    teacher's. The probe is the one ``test_workspace`` uses: a resume with nothing left to do
    leaves the saved adapter exactly as it was, and a run that had wrapped a fresh adapter or
    anchored against itself would not.
    """
    import safetensors.torch
    from conftest import tiny_recipe
    from lfa.workspace import Workspace

    _, artifact_path = tiny_artifact
    workspace = Workspace.init(tmp_path / "ws5", str(base_dir), artifact=str(artifact_path))
    recipe = tiny_recipe(base_dir)
    entry = workspace.train(corpus_a, recipe=recipe, device="cpu")
    trained = safetensors.torch.load_file(Path(entry["adapter"]) / "adapter_model.safetensors")
    assert any(t.abs().sum() > 0 for name, t in trained.items() if "lora_B" in name)

    resumed = workspace.train(corpus_a, recipe=recipe, resume=True, device="cpu")

    assert resumed["teacher_mode"] == "adapter_disabled"
    after = safetensors.torch.load_file(Path(resumed["adapter"]) / "adapter_model.safetensors")
    assert set(after) == set(trained)
    for name, tensor in trained.items():
        assert torch.equal(after[name], tensor), name
