"""The Qwen3 family the package bundles recipes for, proven on CPU without downloading weights."""
import json

import torch
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from lfa.adapters import get_adapter
from lfa.artifact.build import build_artifact, choose_layer_group_size
from lfa.artifact.schema import LM_HEAD_SITE, SITES, _site_input_width, load_artifact
from lfa.sampler import Sampler

from conftest import make_corpus, tiny_recipe

QWEN3_1_7B = dict(hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
                  num_key_value_heads=8, head_dim=128, intermediate_size=6144, vocab_size=151936,
                  tie_word_embeddings=True)


def _site_widths(model):
    ad = get_adapter(model)
    layers = ad.num_layers(model)
    widths = [_site_input_width(ad, model, layer, site) for layer in range(layers) for site in SITES
              if not (layer == 0 and site == "pre_qkv")]
    widths.append(_site_input_width(ad, model, layers, LM_HEAD_SITE))
    return ad, widths


def test_the_real_1_7b_geometry_is_the_llama_layout_with_84_sites_2048_wide():
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(Qwen3Config(**QWEN3_1_7B))
    ad, widths = _site_widths(model)
    assert ad.name == "llama_layout"
    assert len(widths) == 84 and set(widths) == {2048}


def test_the_layer_group_for_the_1_7b_model_on_118_gib():
    # The self-generated build passes SelfGenOptions().reservoir_size (200,000) and fp16's itemsize
    # (2) to choose_layer_group_size (lfa/artifact/build.py, build_artifact_self_generated), so
    # one layer costs 200k x (2*2048 + 2048) x 2 B = 2.4576e9 B; half of 118 GiB is 25.78 layers.
    assert choose_layer_group_size(2048, 2048, 28, 200_000, 2, 118 * 2**30) == 25


def test_the_two_qwen3_models_have_two_store_entries(monkeypatch, tmp_path):
    import lfa.artifact.store as store
    from lfa.selfgen.artifact_corpus import SelfGenOptions
    monkeypatch.setenv(store.STORE_ENV, str(tmp_path))
    entries = {store.entry_dir(m, "a" * 64, SelfGenOptions())
               for m in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")}
    assert len(entries) == 2


def test_a_tiny_qwen3_builds_samples_and_trains_on_cpu(monkeypatch, tmp_path, tiny_model):
    """The real Qwen3 class (attention with q_norm/k_norm) through build -> sample -> init -> train.

    The conftest fixtures are Llama, so this is the one test that runs `Qwen3ForCausalLM` itself.
    """
    from lfa import Workspace
    monkeypatch.setenv("LFA_ARTIFACT_STORE", str(tmp_path / "store"))  # never the real store

    _, tokenizer = tiny_model
    torch.manual_seed(0)
    config = Qwen3Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, intermediate_size=64,
                         vocab_size=len(tokenizer), max_position_embeddings=512,
                         tie_word_embeddings=True)
    model = Qwen3ForCausalLM(config).eval()
    assert hasattr(model.model.layers[0].self_attn, "q_norm")      # the Qwen3 attention, not Llama's
    assert get_adapter(model).name == "llama_layout"
    model_dir = tmp_path / "tiny_qwen3"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    texts = [f"document {i} " + "anchoring hidden states on sampled vectors " * (2 + i % 4)
             for i in range(40)]
    corpus_file = tmp_path / "texts.jsonl"
    corpus_file.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts),
                           encoding="utf-8")
    artifact_path = tmp_path / "distribution_stats.pt"
    build_artifact(str(model_dir), corpus_file, artifact_path, max_samples=1000, seq_len=128,
                   pca_variance=0.9, gmm_k=2, reservoir_size=500, quantize=True, device="cpu",
                   seed=0)

    params = load_artifact(artifact_path)
    assert {k for k in params if k[0].isdigit()} == {
        "0_pre_o", "0_pre_mlp", "1_pre_qkv", "1_pre_o", "1_pre_mlp", "2_pre_lm_head"}
    assert Sampler(artifact_path, device="cpu", seed=0).sample_best(1, "pre_mlp", 4).shape == (4, 32)

    # The recipe names this model directory, so init's other-model warning stays silent.
    recipe_path = tiny_recipe(model_dir).save(tmp_path / "tiny_recipe.yaml")
    ws = Workspace.init(tmp_path / "ws", str(model_dir), artifact=str(artifact_path),
                        recipe=str(recipe_path))
    entry = ws.train(make_corpus(tmp_path / "domain", "consciousness"), device="cpu")

    assert entry["epochs"] == 1
    assert (tmp_path / "ws" / "runs" / "stage1" / "final_model").is_dir()
    assert entry["output_dir"] == str(tmp_path / "ws" / "runs" / "stage1")
