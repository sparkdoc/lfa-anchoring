"""Model loading, device pinning, LoRA wrapping, the resume guard, and fuse.

Everything here runs on CPU against the tiny fixture model, saved to a temp directory so the
loaders exercise their real `from_pretrained` path. Weights are float32 throughout: bf16 (the
library default) cannot carry the 1e-5 agreement the fuse test checks.
"""

import pytest, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from lfa.adapters import effective_weight, get_adapter
from lfa.models import (
    DEFAULT_DEVICE,
    TOOLCHAIN_CHECK_OFF,
    NoTrainableParameters,
    ShardingRefused,
    apply_lora,
    assert_trainable_params,
    fuse,
    load_adapter_for_training,
    load_student,
    load_teacher,
    check_gpu_toolchain,
    load_tokenizer,
    resolve_device,
    resolve_model_path,
)


@pytest.fixture
def base_dir(tiny_model_fresh, tmp_path):
    """The tiny model and its tokenizer saved as a plain HF checkpoint directory."""
    model, tokenizer = tiny_model_fresh
    path = tmp_path / "base"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture
def lora_setup(tiny_model_fresh, base_dir, tmp_path):
    """A LoRA-wrapped tiny model with a real (nonzero) delta, and its saved adapter directory.

    Returns ``(peft_model, adapter, adapter_dir, expected_q_proj)`` where ``expected_q_proj`` is
    the effective weight of layer 1's query projection -- what fusing the adapter must produce.
    """
    model, _ = tiny_model_fresh
    adapter = get_adapter(model)
    peft_model = apply_lora(model, adapter, rank=2, alpha=4)

    q_proj = adapter.qkv_modules(peft_model, 1)["q_proj"]
    with torch.no_grad():                      # B starts at zero; make the delta real
        q_proj.lora_B["default"].weight.normal_(0, 0.1)
    expected = effective_weight(q_proj).detach().clone()

    # The fixture model was built in memory, so it carries no base path; point the adapter at the
    # checkpoint it belongs to, as a real run's would (PEFT reads it back when saving/loading).
    peft_model.peft_config["default"].base_model_name_or_path = str(base_dir)

    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(adapter_dir)
    return peft_model, adapter, adapter_dir, expected


def trainable_names(model):
    return [name for name, param in model.named_parameters() if param.requires_grad]


# --- device pinning -------------------------------------------------------------------------

def test_auto_device_is_refused_by_default():
    with pytest.raises(ShardingRefused, match="allow_sharding"):
        resolve_device("auto")


def test_auto_device_is_allowed_when_asked_for():
    assert resolve_device("auto", allow_sharding=True) == "auto"


def test_none_resolves_to_the_pinned_default():
    assert resolve_device(None) == "cuda:0" == DEFAULT_DEVICE


def test_explicit_devices_pass_through():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


def test_balanced_is_a_sharding_request_too():
    with pytest.raises(ShardingRefused):
        resolve_device("balanced")


def test_single_device_dict_passes_through_but_a_split_one_is_refused():
    assert resolve_device({"": "cuda:0"}) == {"": "cuda:0"}
    with pytest.raises(ShardingRefused):
        resolve_device({"model.layers.0": "cuda:0", "model.layers.1": "cuda:1"})


# --- path resolution and tokenizer ---------------------------------------------------------

def test_model_path_is_untouched_when_online():
    assert resolve_model_path("org/some-model", local_files_only=False) == "org/some-model"


def test_local_path_resolves_to_itself_offline(base_dir):
    assert resolve_model_path(str(base_dir), local_files_only=True) == str(base_dir)


def test_missing_offline_model_names_the_cache_it_looked_in(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    with pytest.raises(ValueError, match="not found in cache"):
        resolve_model_path("org/never-downloaded", local_files_only=True)


def test_tokenizer_gets_a_pad_token(base_dir):
    tokenizer = load_tokenizer(str(base_dir), local_files_only=True)
    assert tokenizer.pad_token is not None


# --- teacher / student ----------------------------------------------------------------------

def test_teacher_is_frozen_and_in_eval_mode(base_dir):
    teacher = load_teacher(str(base_dir), device="cpu", dtype=torch.float32, local_files_only=True)
    assert not teacher.training
    assert trainable_names(teacher) == []
    assert next(teacher.parameters()).device.type == "cpu"
    assert next(teacher.parameters()).dtype is torch.float32


def test_student_is_trainable_and_checkpointing(base_dir):
    student = load_student(str(base_dir), device="cpu", dtype=torch.float32, local_files_only=True)
    assert student.training
    assert all(p.requires_grad for p in student.parameters())
    assert student.is_gradient_checkpointing


def test_student_without_gradient_checkpointing(base_dir):
    student = load_student(
        str(base_dir), device="cpu", dtype=torch.float32,
        gradient_checkpointing=False, local_files_only=True,
    )
    assert not student.is_gradient_checkpointing


# --- LoRA wrapping --------------------------------------------------------------------------

def test_lora_trains_only_lora_parameters(lora_setup):
    peft_model, adapter, _, _ = lora_setup
    names = trainable_names(peft_model)
    assert names, "LoRA wrapping left nothing to train"
    assert all("lora_" in name for name in names), names
    cfg = peft_model.peft_config["default"]
    assert (cfg.r, cfg.lora_alpha) == (2, 4)


def test_lora_leaves_the_embedding_frozen(lora_setup):
    peft_model, adapter, _, _ = lora_setup
    embed_tokens, _ = adapter.embed_modules(peft_model)
    assert embed_tokens.weight.requires_grad is False


def test_lora_targets_the_adapters_projections(lora_setup):
    peft_model, adapter, _, _ = lora_setup
    targeted = {name.split(".lora_")[0].rsplit(".", 1)[-1] for name in trainable_names(peft_model)}
    assert targeted == set(adapter.lora_target_modules())


def test_freeze_embed_false_trains_the_embedding(tiny_model_fresh):
    model, _ = tiny_model_fresh
    adapter = get_adapter(model)
    peft_model = apply_lora(model, adapter, rank=2, alpha=4, freeze_embed=False)
    embed_tokens, _ = adapter.embed_modules(peft_model)
    assert embed_tokens.weight.requires_grad is True
    assert any("modules_to_save" in name for name in trainable_names(peft_model))


# --- resume guard ---------------------------------------------------------------------------

def test_adapter_loaded_for_training_has_trainable_parameters(base_dir, lora_setup):
    _, _, adapter_dir, _ = lora_setup
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
    resumed = load_adapter_for_training(base, adapter_dir)
    names = trainable_names(resumed)
    assert names and all("lora_" in name for name in names), names


def test_the_guard_rejects_a_frozen_adapter(base_dir, lora_setup):
    """PeftModel.from_pretrained defaults to is_trainable=False -- the silent-no-op resume bug."""
    _, _, adapter_dir, _ = lora_setup
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
    frozen = PeftModel.from_pretrained(base, str(adapter_dir))
    assert trainable_names(frozen) == []

    with pytest.raises(NoTrainableParameters, match="resume"):
        assert_trainable_params(frozen, context="resume")


def test_the_guard_passes_a_model_with_something_to_train(tiny_model_fresh):
    model, _ = tiny_model_fresh
    assert_trainable_params(model)          # all weights trainable: must not raise


# --- fuse -----------------------------------------------------------------------------------

def test_fuse_writes_a_plain_model_carrying_the_lora_delta(base_dir, lora_setup, tmp_path):
    peft_model, adapter, adapter_dir, expected = lora_setup
    out = fuse(adapter_dir, base_dir, tmp_path / "fused", dtype=torch.float32)

    assert out == tmp_path / "fused"
    assert (out / "config.json").exists()

    fused = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    assert not hasattr(fused, "peft_config")
    fused_q_proj = get_adapter(fused).qkv_modules(fused, 1)["q_proj"].weight
    assert torch.allclose(fused_q_proj, expected, atol=1e-5)


def test_fuse_saves_the_tokenizer_beside_the_model(base_dir, lora_setup, tmp_path):
    _, _, adapter_dir, _ = lora_setup
    out = fuse(adapter_dir, base_dir, tmp_path / "fused", dtype=torch.float32)
    assert (out / "tokenizer_config.json").exists()
    assert load_tokenizer(str(out), local_files_only=True) is not None


def test_fuse_leaves_an_untouched_projection_alone(base_dir, lora_setup, tmp_path):
    """Only the module whose B was made nonzero moves; the rest fuse back to the base weight."""
    peft_model, adapter, adapter_dir, _ = lora_setup
    untouched = effective_weight(adapter.qkv_modules(peft_model, 0)["k_proj"]).detach().clone()
    out = fuse(adapter_dir, base_dir, tmp_path / "fused", dtype=torch.float32)

    fused = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    fused_k_proj = get_adapter(fused).qkv_modules(fused, 0)["k_proj"].weight
    assert torch.allclose(fused_k_proj, untouched, atol=1e-5)


# ------------------------------------------------------- the CUDA build-toolchain preflight
#
# A distribution `python3` without its `-dev` package has no `Python.h`, and some torch paths
# (torch.compile, custom triton kernels) JIT-compile a small CUDA shim at the first kernel
# launch. The package's own training and generation paths do not, so the preflight warns once,
# naming the remedy, and lets the run proceed.


@pytest.fixture
def no_headers(tmp_path, monkeypatch):
    """An interpreter whose include directory holds no `Python.h`."""
    import sysconfig

    empty = tmp_path / "include"
    empty.mkdir()
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"include": str(empty)})
    monkeypatch.delenv(TOOLCHAIN_CHECK_OFF, raising=False)
    return empty


def test_a_cuda_run_without_python_headers_warns_once_and_proceeds(no_headers, caplog):
    """A real Qwen3-0.6B LoRA step and a generation ran on an RTX 2070 with no headers
    (2026-09-26), so the missing toolchain is a warning about a path some torch builds take,
    not a refusal."""
    import lfa.models as models_module

    models_module._toolchain_warned = False
    with caplog.at_level("WARNING", logger="lfa.models"):
        check_gpu_toolchain("cuda:0")
        check_gpu_toolchain("cuda:0")

    messages = [r.getMessage() for r in caplog.records if "Python development headers" in r.getMessage()]
    assert len(messages) == 1                            # once per process
    assert "python3-dev" in messages[0]
    assert TOOLCHAIN_CHECK_OFF in messages[0]
    assert resolve_device("cuda:0") == "cuda:0"          # and the resolver proceeds


def test_the_preflight_also_mentions_a_missing_compiler(monkeypatch, caplog):
    import lfa.models as models_module

    monkeypatch.delenv(TOOLCHAIN_CHECK_OFF, raising=False)
    monkeypatch.setattr(models_module.shutil, "which", lambda name: None)
    models_module._toolchain_warned = False
    with caplog.at_level("WARNING", logger="lfa.models"):
        check_gpu_toolchain({"": "cuda:0"})
    assert any("C compiler" in r.getMessage() for r in caplog.records)


def test_a_cpu_run_needs_no_toolchain(no_headers):
    check_gpu_toolchain("cpu")
    assert resolve_device("cpu") == "cpu"                # and the resolver stays out of the way


def test_the_preflight_can_be_switched_off(no_headers, monkeypatch, caplog):
    import lfa.models as models_module

    monkeypatch.setenv(TOOLCHAIN_CHECK_OFF, "1")
    models_module._toolchain_warned = False
    with caplog.at_level("WARNING", logger="lfa.models"):
        check_gpu_toolchain("cuda:0")
    assert not [r for r in caplog.records if "headers" in r.getMessage()]
