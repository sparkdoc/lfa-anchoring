import copy

import pytest, torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers

@pytest.fixture(scope="session")
def tiny_model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True)
    model = LlamaForCausalLM(cfg).eval()
    tok = Tokenizer(models.WordLevel({chr(i): i for i in range(256)}, unk_token=chr(0)))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token=chr(0), pad_token=chr(1), eos_token=chr(2))
    return model, tokenizer

@pytest.fixture
def tiny_model_fresh(tiny_model):
    """A private deep copy of the session model, for tests that wrap, mutate, train or move it."""
    model, tokenizer = tiny_model
    return copy.deepcopy(model), tokenizer


@pytest.fixture
def tiny_texts():
    return [f"document number {i} about anchoring functions on sampled hidden states " * (1 + i % 4) for i in range(12)]


@pytest.fixture(scope="session")
def tiny_artifact(tmp_path_factory):
    """A minimal p(h) artifact for the 2-layer, D=32 `tiny_model`: dict + a file on disk.

    Sites deliberately omit `0_pre_qkv`: on a real artifact that site is either the embedding
    lookup table or absent, so leaving it out exercises both the `sample_best` fallback chain and
    the "no statistics for this site" path. `2_pre_lm_head` sits at layer index `num_layers`, the
    convention for the model-level head.

    Tests must treat the returned dict as read-only (it is shared for the whole session); a
    `Sampler` is given the PATH so it loads and mutates only its own copy.
    """
    from lfa.artifact.schema import LM_HEAD_SITE, SITES, make_meta

    g = torch.Generator().manual_seed(11)
    hidden, n_comp, n_gmm = 32, 8, 3
    params = {}
    for key in ("0_pre_o", "0_pre_mlp", "1_pre_qkv", "1_pre_o", "1_pre_mlp", "2_pre_lm_head"):
        basis, _ = torch.linalg.qr(torch.randn(hidden, n_comp, generator=g))   # orthonormal [32, 8]
        weights = torch.rand(n_gmm, generator=g) + 0.1
        params[key] = {
            "mean": torch.randn(hidden, generator=g) * 0.1,
            "std": torch.rand(hidden, generator=g) + 0.5,
            "pca_components": basis,
            "pca_eigenvalues": torch.linspace(1.0, 0.1, n_comp),
            "pca_n_components": n_comp,
            "gmm_weights": weights / weights.sum(),
            "gmm_means": torch.randn(n_gmm, n_comp, generator=g),
            "gmm_covariances": torch.rand(n_gmm, n_comp, generator=g) * 0.2 + 0.05,
            "gmm_n_components": n_gmm,
            "gmm_covariance_type": "diag",
            "n_samples": 1000,
        }
    params["__meta__"] = make_meta("tiny", hidden, 2, list(SITES) + [LM_HEAD_SITE], 6000)

    path = tmp_path_factory.mktemp("art") / "distribution_stats.pt"
    torch.save(params, path)
    return params, path
