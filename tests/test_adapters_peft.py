"""The adapter's PEFT paths: unwrapping a PeftModel, and effective_weight's LoRA branches."""

import pytest, torch
from peft import LoraConfig, get_peft_model

from lfa.adapters import get_adapter, effective_weight


@pytest.fixture
def peft_pair(tiny_model_fresh):
    """A LoRA-wrapped tiny model, its adapter, and the causal LM PEFT wraps."""
    model, _ = tiny_model_fresh
    adapter = get_adapter(model)
    peft_model = get_peft_model(
        model,
        LoraConfig(r=2, lora_alpha=4, target_modules=adapter.lora_target_modules(), task_type="CAUSAL_LM"),
    )
    return peft_model, adapter, peft_model.base_model.model


@pytest.fixture
def q_proj(peft_pair):
    """Layer 0's query projection with a nonzero LoRA B, so its delta is real."""
    peft_model, adapter, _ = peft_pair
    module = adapter.qkv_modules(peft_model, 0)["q_proj"]
    with torch.no_grad():
        module.lora_B["default"].weight.normal_(0, 0.1)
    return module


def lora_delta(module):
    a = module.lora_A["default"].weight
    b = module.lora_B["default"].weight
    return (b @ a) * module.scaling["default"]


def test_base_model_unwraps_peft_model(peft_pair):
    peft_model, adapter, inner = peft_pair
    assert adapter.base_model(peft_model) is inner.model
    assert adapter.num_layers(peft_model) == 2


def test_modules_resolve_through_peft_wrapper(peft_pair):
    peft_model, adapter, inner = peft_pair
    assert adapter.qkv_modules(peft_model, 1)["q_proj"] is inner.model.layers[1].self_attn.q_proj
    assert adapter.o_proj_module(peft_model, 0) is inner.model.layers[0].self_attn.o_proj
    assert adapter.mlp_module(peft_model, 1) is inner.model.layers[1].mlp
    emb, ln0 = adapter.embed_modules(peft_model)
    assert emb is inner.model.embed_tokens and ln0 is inner.model.layers[0].input_layernorm
    assert adapter.final_norm_module(peft_model) is inner.model.norm
    assert adapter.lm_head_module(peft_model) is inner.lm_head


def test_effective_weight_adds_lora_delta(q_proj):
    expected = q_proj.weight + lora_delta(q_proj)
    assert torch.allclose(effective_weight(q_proj), expected, atol=1e-6)


def test_effective_weight_matches_the_forward_pass(q_proj):
    x = torch.randn(5, q_proj.in_features)
    assert torch.allclose(q_proj(x), x @ effective_weight(q_proj).T, atol=1e-5)


def test_effective_weight_does_not_double_count_a_merged_adapter(peft_pair, q_proj):
    peft_model, _, _ = peft_pair
    base_weight = q_proj.weight.detach().clone()
    delta = lora_delta(q_proj).detach().clone()

    peft_model.merge_adapter()

    assert q_proj.merged
    assert torch.allclose(q_proj.weight, base_weight + delta, atol=1e-6)  # the delta is in .weight now
    merged = effective_weight(q_proj)
    assert torch.allclose(merged, q_proj.weight, atol=1e-6)
    assert not torch.allclose(merged, base_weight + 2 * delta, atol=1e-6)


def test_effective_weight_ignores_disabled_adapters(peft_pair, q_proj):
    peft_model, _, _ = peft_pair
    base_weight = q_proj.weight.detach().clone()
    with peft_model.disable_adapter():
        assert torch.allclose(effective_weight(q_proj), base_weight, atol=1e-6)
    assert torch.allclose(effective_weight(q_proj), base_weight + lora_delta(q_proj), atol=1e-6)


def test_effective_weight_is_differentiable(q_proj):
    weight = effective_weight(q_proj)
    assert weight.requires_grad and weight.grad_fn is not None

    weight.pow(2).sum().backward()
    grad = q_proj.lora_B["default"].weight.grad
    assert grad is not None and torch.any(grad != 0)
