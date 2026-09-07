import copy
from pathlib import Path

import pytest, torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers

def pytest_runtest_setup(item):
    """A ``gpu``-marked test skips, rather than errors, on a machine with no CUDA.

    The marker's usual job is selection -- the default ``addopts`` deselects it -- but a run that
    asks for a marker explicitly (``-m equivalence``) selects the GPU tests along with the rest,
    and without this they would fail inside torch on "No CUDA GPUs are available" instead of
    saying what they need.
    """
    if item.get_closest_marker("gpu") and not torch.cuda.is_available():
        pytest.skip("needs CUDA (marked gpu)")


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
    # `n_samples_total` is a PER-SITE count (every site sees the same token stream), so it
    # agrees with each entry's own `n_samples` rather than summing them.
    params["__meta__"] = make_meta("tiny", hidden, 2, list(SITES) + [LM_HEAD_SITE], 1000)

    path = tmp_path_factory.mktemp("art") / "distribution_stats.pt"
    torch.save(params, path)
    return params, path


# ============================================================================================
# A workspace-sized setup: the tiny model as a checkpoint, a corpus, a recipe, and a registry
# that serves the fixture artifact. Shared by `test_workspace.py` and `test_cli.py`, which
# exercise the same state machine through two different front doors.
# ============================================================================================

@pytest.fixture(scope="module")
def base_dir(tmp_path_factory, tiny_model):
    """The tiny model as a checkpoint directory -- what a workspace's `model_id` points at."""
    model, tokenizer = tiny_model
    path = tmp_path_factory.mktemp("base_model")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def make_corpus(directory, word, n_docs=8):
    """A handful of short documents about `word`, written into `directory`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n_docs):
        text = f"document {i} about {word}: anchoring the function of a sub-module " * 6
        (directory / f"doc_{i}.txt").write_text(text, encoding="utf-8")
    return directory


@pytest.fixture(scope="module")
def corpus_a(tmp_path_factory):
    return make_corpus(tmp_path_factory.mktemp("domains") / "domain_a", "consciousness")


@pytest.fixture(scope="module")
def corpus_b(tmp_path_factory):
    return make_corpus(tmp_path_factory.mktemp("domains") / "domain_b", "archaeology")


def tiny_recipe(base_dir, **overrides):
    """A one-epoch rank-2 operating point, calibrated (by declaration) at exactly this point."""
    from lfa import Recipe

    kwargs = dict(
        name="tiny", model_id=str(base_dir), artifact="tiny",
        lora_rank=2, lora_alpha=4, lambda_qkv=10.0, lambda_mlp=10.0, mu=0.05,
        n_anchor_samples=4, epochs=1, checkpoint_mode="none",
        learning_rate=1e-2, warmup_steps=1, batch_size=2, sequence_length=64, seed=0,
        calibrated_rank=2, calibrated_artifact="tiny",
        # These workspaces have eight short documents each: holding a tenth of them out
        # (the shipped recipe's default) would change every document count the workspace
        # tests assert on. Tests about the hold-out itself set it explicitly.
        val_fraction=0.0,
    )
    kwargs.update(overrides)
    return Recipe(**kwargs)


@pytest.fixture(scope="module")
def registry(tiny_artifact):
    """The "tiny" artifact, published in the registry and served by a local copy."""
    import lfa.artifact.fetch as fetch_module
    from lfa.artifact.fetch import ARTIFACTS, sha256_file

    _, path = tiny_artifact
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(ARTIFACTS, "tiny", {
            "model_id": "tiny",
            "url": "https://example.invalid/artifacts-v1/tiny.pt",
            "sha256": sha256_file(path),
            "n_samples_total": 1000,
            "kind": "test fixture",
            "size_mb": 1,
        })
        patch.setattr(fetch_module, "_download_with_requests",
                      lambda url, dest: dest.write_bytes(path.read_bytes()))
        yield ARTIFACTS["tiny"]
