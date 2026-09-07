import pytest, torch
from lfa.adapters import get_adapter, ModelAdapter, UnsupportedModelError, effective_weight

def test_llama_layout_selected(tiny_model):
    model, _ = tiny_model
    ad = get_adapter(model)
    assert ad.name == "llama_layout" and ad.num_layers(model) == 2

def test_modules_resolve(tiny_model):
    model, _ = tiny_model
    ad = get_adapter(model)
    q = ad.qkv_modules(model, 1)
    assert set(q) == {"q_proj", "k_proj", "v_proj"}
    assert q["q_proj"] is model.model.layers[1].self_attn.q_proj
    assert ad.o_proj_module(model, 0) is model.model.layers[0].self_attn.o_proj
    assert ad.mlp_module(model, 0) is model.model.layers[0].mlp
    emb, ln0 = ad.embed_modules(model)
    assert emb is model.model.embed_tokens and ln0 is model.model.layers[0].input_layernorm
    assert ad.lm_head_module(model) is model.lm_head
    assert ad.site_module(model, 1, "pre_mlp") is model.model.layers[1].mlp
    assert ad.site_module(model, 2, "pre_lm_head") is model.lm_head

def test_unsupported_model_raises():
    class Weird(torch.nn.Module):
        config = type("C", (), {"model_type": "weird"})()
    with pytest.raises(UnsupportedModelError, match="weird"):
        get_adapter(Weird())

def test_effective_weight_plain_linear():
    lin = torch.nn.Linear(4, 3)
    assert torch.equal(effective_weight(lin), lin.weight)

def test_site_module_resolves_the_module_whose_input_is_the_site(tiny_model):
    model, _ = tiny_model
    ad = get_adapter(model)
    # pre_qkv is the input_layernorm output, i.e. what q_proj reads; pre_o is o_proj's input.
    assert ad.site_module(model, 0, "pre_qkv") is model.model.layers[0].self_attn.q_proj
    assert ad.site_module(model, 1, "pre_o") is model.model.layers[1].self_attn.o_proj

def test_site_module_rejects_an_unknown_site(tiny_model):
    model, _ = tiny_model
    with pytest.raises(ValueError, match="post_mlp"):
        get_adapter(model).site_module(model, 0, "post_mlp")
