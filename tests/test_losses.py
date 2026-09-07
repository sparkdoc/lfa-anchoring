import copy, torch, pytest
from lfa.losses import anchor_loss, weight_loss, lora_factored_weight_loss, compute_layer_weights
from lfa.sampler import Sampler
from lfa.adapters import get_adapter

def test_zero_when_identical(tiny_model, tiny_artifact):
    model, _ = tiny_model; _, path = tiny_artifact; ad = get_adapter(model)
    s = Sampler(path, device="cpu", seed=0); s.build_embedding_lookup_from_model(model, ad)
    out = anchor_loss(model, copy.deepcopy(model), s, ad, n_samples=4)
    assert out["total"].item() == 0.0 and out["embed"].item() == 0.0

def test_positive_after_perturbation(tiny_model, tiny_artifact):
    model, _ = tiny_model; _, path = tiny_artifact; ad = get_adapter(model)
    student = copy.deepcopy(model)
    with torch.no_grad(): student.model.layers[1].mlp.down_proj.weight.add_(0.05)
    s = Sampler(path, device="cpu", seed=0); s.build_embedding_lookup_from_model(model, ad)
    out = anchor_loss(model, student, s, ad, n_samples=8)
    assert out["mlp"].item() > 0 and out["qkv"].item() == 0.0 and out["total"].item() > 0

def test_layer_weights_schedule():
    w = compute_layer_weights(4, end_ratio=0.1, schedule="cosine")
    assert len(w) == 4 and w[0] > w[-1] and abs(sum(w) - 4) < 1e-6

def test_mu_fast_path_matches_general(tiny_model):
    from peft import LoraConfig, get_peft_model
    model, _ = tiny_model; ad = get_adapter(model)
    student = get_peft_model(copy.deepcopy(model), LoraConfig(r=2, lora_alpha=4, target_modules=ad.lora_target_modules()))
    for n, p in student.named_parameters():
        if "lora_B" in n:
            with torch.no_grad(): p.normal_(0, 0.1)
    fast = lora_factored_weight_loss(model, student, ad, [1.0] * 2)
    assert fast is not None
    slow = weight_loss(model, student, ad, [1.0] * 2, force_general=True)
    assert torch.allclose(fast, slow, rtol=1e-3, atol=1e-6)

def test_mu_fast_path_refused_when_base_differs(tiny_model):
    from peft import LoraConfig, get_peft_model
    model, _ = tiny_model; ad = get_adapter(model)
    student = get_peft_model(copy.deepcopy(model), LoraConfig(r=2, lora_alpha=4, target_modules=ad.lora_target_modules()))
    with torch.no_grad(): student.base_model.model.model.layers[0].mlp.up_proj.weight.add_(1e-3)
    assert lora_factored_weight_loss(model, student, ad, [1.0] * 2) is None
