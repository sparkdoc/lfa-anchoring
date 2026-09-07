"""Does the companion compute what the paper computed?

Every other test in this suite says the package is self-consistent. These say it is the *same*
package: each one replays a fixture captured from the the research code research code
(``capture_fixtures.py``) against the companion's own implementation, at the tolerance the
quantity deserves.

* **The sampler is checked at zero tolerance.** ``sample_best`` is the input side of the anchor,
  and a drifted sample stream is a different ``p(h)``: every anchor number would move with it and
  nothing would raise. The 85 draws of one anchoring step must be bit-identical.
* **The losses are checked to 1e-4 relative.** They are reductions over bf16 forward passes, so
  the accumulation order is visible in the last digits; what is being claimed is that the same
  four blocks are computed the same way, not that two float32 sums agree bitwise. mu is checked
  on both of its paths -- and the LoRA-factored one is the *more* accurate of the two, since the
  general form subtracts near-equal bf16 weights.
* **The extension is checked by likelihood.** Its two halves separate cleanly: the collection is
  deterministic and is compared activation by activation, while the mixture fitted on those
  activations is not (the two GMM implementations draw different k-means++ initializations), so
  the fitted domain mixture is compared by the density it puts on held-out activations.

Everything here needs the the research code checkout the fixtures were captured from -- for the model
checkpoints, the artifact and the corpus, which are far too large to commit. Each test names the
path it wanted and skips when it is absent.

Run them with the marker, which the default ``addopts`` deselects::

    CUDA_VISIBLE_DEVICES=0 pytest tests/equivalence -m equivalence -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from lfa.adapters import get_adapter
from lfa.artifact.extend import extend_artifact
from lfa.artifact.schema import load_artifact
from lfa.corpus import load_corpus, make_dataloader
from lfa.losses import anchor_loss, weight_loss
from lfa.models import load_adapter_for_training, load_teacher, load_tokenizer
from lfa.quantize import dequantize_blockwise, quantize_blockwise
from lfa.sampler import Sampler

from capture_fixtures import (
    DEFAULT_RESEARCH_ROOT,
    N_ANCHOR_SAMPLES,
    gaussian_mixture_log_likelihood,
    link_txt_corpus,
)

pytestmark = pytest.mark.equivalence

FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: Relative tolerance for a loss: a reduction over bf16 forwards, not a bitwise claim.
LOSS_RTOL = 1e-4
#: Relative tolerance against the research code's GENERAL mu, which is the one number in this file that
#: the companion deliberately does not reproduce exactly: the research code accumulates the LoRA delta
#: into a bfloat16 buffer in place (`get_effective_weight`, src/lra_models.py:616-621) and then
#: subtracts a near-equal teacher weight, which costs ~0.5% to cancellation. The companion's
#: `effective_weight` promotes to the wider of the base and adapter dtypes, so its general form
#: lands on the exact value instead -- and the arbiter is the research code's OWN factored path, which
#: the companion matches bit for bit.
MU_GENERAL_RTOL = 1e-2
#: How far apart two domain mixtures fitted from DIFFERENT k-means++ initializations may sit, as
#: a fraction of the held-out mean log-likelihood. Measured over five initializations of the
#: reference fitter on the captured activations: 3.7% at 1_pre_mlp (-624.2 to -647.2 nats) and
#: 0.6% at 12_pre_mlp. It is the tolerance for the end-to-end call only -- at a MATCHED
#: initialization the two fitters agree exactly, which is what
#: `test_the_domain_mixture_is_the_reference_fit_at_a_matched_initialization` asserts.
LL_BAND = 0.05


# ------------------------------------------------------------------------------------ helpers

def load_reference(name: str) -> dict:
    """One captured fixture, or a skip naming how to regenerate it."""
    path = FIXTURES / name
    if not path.is_file():
        pytest.skip(f"no fixture at {path} -- regenerate with tests/equivalence/"
                    f"capture_fixtures.py inside the research code's venv (see the README there)")
    return torch.load(path, map_location="cpu", weights_only=False)


def require(path: str | Path, what: str) -> Path:
    """A path the fixture recorded, or a skip naming what is missing.

    The fixtures are checked in; the checkpoints, artifacts and corpora they were captured from
    are not (gigabytes), so a checkout without them skips rather than fails.
    """
    resolved = Path(path)
    if not resolved.exists():
        pytest.skip(f"{what} not present at {resolved} (the research code checkout required)")
    return resolved


def import_from_research(module: str):
    """Import a module from the research checkout, or skip.

    Used only by the tests that compare two implementations *in the same process*; the rest work
    from the captured fixtures and never import ``src``.
    """
    root = DEFAULT_RESEARCH_ROOT
    if not (root / "src").is_dir():
        pytest.skip(f"no the research code checkout at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        __import__(module)
    except ImportError as error:                 # e.g. scipy, which src.lra_distribution imports
        pytest.skip(f"cannot import {module} from {root}: {error}")
    return sys.modules[module]


def release_cuda() -> None:
    """Hand back the caching allocator's blocks between two model-loading tests."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def sampler_reference():
    return load_reference("sampler_draws.pt")


@pytest.fixture(scope="module")
def losses_reference():
    return load_reference("losses.pt")


@pytest.fixture(scope="module")
def extend_reference():
    return load_reference("extend_ref.pt")


# ==============================================================================================
# T16.1 -- the sampler, at zero tolerance
# ==============================================================================================

@pytest.mark.gpu
def test_the_sampler_replays_every_reference_draw_bit_for_bit(sampler_reference):
    """85 draws of one anchoring step, from the shipped int8 artifact, under one global seed.

    No tolerance is allowed here. The draws share a single RNG stream, so this checks the
    fidelity ladder, the residual-variance terms, the component draw's device (the multinomial
    stays on the CPU generator on purpose) and the call order all at once: any one of them wrong
    changes the stream from that point on.
    """
    reference = sampler_reference
    artifact = require(reference["artifact"], "the shipped p(h) artifact")
    base_model = require(reference["base_model"], "the base model checkpoint")

    model = load_teacher(str(base_model), device=reference["device"],
                         dtype=getattr(torch, reference["model_dtype"]))
    adapter = get_adapter(model)

    # Seeded after the load, exactly as the capture did: `from_pretrained` touches the RNG.
    torch.manual_seed(reference["seed"])
    sampler = Sampler(str(artifact), device=reference["device"], seed=None)
    sampler.build_embedding_lookup_from_model(model, adapter)

    mismatches = []
    worst = 0.0
    for (site, layer, n), expected in zip(reference["calls"], reference["draws"]):
        drawn = sampler.sample_best(layer, site, n)
        if expected is None:
            assert drawn is None, f"{site} at layer {layer}: reference drew nothing, we drew"
            continue
        assert drawn is not None, f"{site} at layer {layer}: reference drew, we drew nothing"
        drawn = drawn.detach().cpu().float()
        assert drawn.shape == expected.shape, f"{site} at layer {layer}: {drawn.shape} vs {expected.shape}"
        if not torch.equal(drawn, expected):
            difference = (drawn - expected).abs().max().item()
            worst = max(worst, difference)
            mismatches.append((site, layer, difference))

    del model, sampler
    release_cuda()
    assert not mismatches, (
        f"{len(mismatches)} of {len(reference['draws'])} draws differ from the reference. "
        f"First: site {mismatches[0][0]!r} layer {mismatches[0][1]} (max abs diff "
        f"{mismatches[0][2]:.3e}); worst over all draws {worst:.3e}. This is a p(h) change, not "
        "a rounding difference -- diff Sampler.sample_gmm/sample_pca/sample_embedding_lookup "
        "against src/lra_distribution.py before touching the tolerance, which stays at zero."
    )


# ==============================================================================================
# T16.2 -- the anchor and mu
# ==============================================================================================

def test_the_loader_reproduces_the_reference_training_batch(losses_reference, tmp_path):
    """The batch the trainer's first step sees, chunk for chunk.

    A loss fixture is only as good as the stream it was measured on, and the chunking has three
    ways to drift silently (the epoch-0 offset, the short-document frame, the seeded document
    shuffle before the split). CPU-only: this is the loader, not the model.
    """
    reference = losses_reference
    corpus = link_txt_corpus(require(reference["corpus"], "the domain corpus"),
                             tmp_path / "corpus", reference["documents"])
    tokenizer = load_tokenizer(str(require(reference["base_model"], "the base model checkpoint")))

    dataset, holdout = load_corpus(corpus, tokenizer, max_length=reference["sequence_length"],
                                   val_fraction=0.0, seed=reference["corpus_seed"],
                                   keep_short_whole=reference["keep_short_whole"])
    assert holdout is None
    assert len(dataset) == reference["n_chunks"]

    loader = make_dataloader(dataset, batch_size=reference["batch"]["input_ids"].shape[0],
                             shuffle=False, pad_token_id=tokenizer.pad_token_id)
    batch = next(iter(loader))
    for key, expected in reference["batch"].items():
        assert torch.equal(batch[key], expected), f"{key} differs from the reference batch"


@pytest.mark.gpu
def test_the_anchor_components_match_the_reference(losses_reference):
    """L_qkv, L_mlp, L_lm_head, L_embed and their total, for the archived recipe adapter.

    The LM-head and embedding terms are structurally zero here and are asserted to *stay* zero:
    the archived adapter targets only the seven projections and froze the embedding, so those two
    sub-modules are the teacher's own. That is a real property of the run, and a non-zero value
    would mean the student had been perturbed somewhere it should not have been.
    """
    reference = losses_reference
    artifact = require(reference["artifact"], "the shipped p(h) artifact")
    base_model = require(reference["base_model"], "the base model checkpoint")
    adapter_dir = require(reference["adapter"], "the archived recipe adapter")
    device, dtype = reference["device"], getattr(torch, reference["model_dtype"])

    teacher = load_teacher(str(base_model), device=device, dtype=dtype)
    adapter = get_adapter(teacher)
    student = load_adapter_for_training(load_teacher(str(base_model), device=device, dtype=dtype),
                                        adapter_dir)
    student.eval()
    student.requires_grad_(False)

    sampler = Sampler(str(artifact), device=device, seed=None)
    sampler.build_embedding_lookup_from_model(teacher, adapter)

    torch.manual_seed(reference["seed"])
    computed = anchor_loss(teacher, student, sampler, adapter,
                           n_samples=reference["n_samples_per_layer"])
    computed = {name: float(value) for name, value in computed.items()}
    del teacher, student, sampler
    release_cuda()

    assert set(computed) == set(reference["anchor"])
    for name, expected in reference["anchor"].items():
        if expected == 0.0:
            assert computed[name] == 0.0, f"L_{name}: expected exactly 0, got {computed[name]!r}"
        else:
            assert math.isclose(computed[name], expected, rel_tol=LOSS_RTOL), (
                f"L_{name}: {computed[name]!r} vs reference {expected!r} "
                f"({abs(computed[name] - expected) / abs(expected):.2e} relative)"
            )


@pytest.mark.gpu
def test_mu_matches_the_reference_on_both_of_its_paths(losses_reference):
    """mu, computed through the LoRA factors and the general way.

    The factored path is the equivalence claim and is exact on both sides -- ``||s.BA||_F^2``
    from the fp32 factors, with no near-equal subtraction anywhere in it.

    The general path is the one place the companion knowingly departs from the research code, and
    it departs by being *right*: the research code accumulates the LoRA delta into a bfloat16 buffer in
    place and then subtracts a nearly equal teacher weight, which loses ~0.5% to cancellation,
    while the companion's ``effective_weight`` promotes to fp32 and keeps it. So the two general
    forms differ by that 0.5%, and the check that settles which one is a port defect is the third
    one below: the companion's general value sits on the research code's own factored number, not on its
    rounded one.
    """
    reference = losses_reference
    base_model = require(reference["base_model"], "the base model checkpoint")
    adapter_dir = require(reference["adapter"], "the archived recipe adapter")
    device, dtype = reference["device"], getattr(torch, reference["model_dtype"])

    teacher = load_teacher(str(base_model), device=device, dtype=dtype)
    adapter = get_adapter(teacher)
    student = load_adapter_for_training(load_teacher(str(base_model), device=device, dtype=dtype),
                                        adapter_dir)
    student.eval()
    student.requires_grad_(False)

    general = float(weight_loss(teacher, student, adapter, force_general=True))
    fast = float(weight_loss(teacher, student, adapter))
    del teacher, student
    release_cuda()

    assert math.isclose(fast, reference["mu_fast"], rel_tol=LOSS_RTOL), (
        f"factored mu {fast!r} vs reference {reference['mu_fast']!r}")
    assert math.isclose(general, fast, rel_tol=LOSS_RTOL), (
        f"our own two mu paths disagree: general {general!r}, factored {fast!r}. They agree only "
        "because the general form keeps the delta in fp32; a bf16 materialization here would "
        "reintroduce the cancellation the factored path exists to avoid."
    )
    assert math.isclose(general, reference["mu_general"], rel_tol=MU_GENERAL_RTOL), (
        f"general mu {general!r} vs reference {reference['mu_general']!r} "
        f"({abs(general - reference['mu_general']) / reference['mu_general']:.2e} relative, "
        f"tolerance {MU_GENERAL_RTOL:.0e})")

    # Which side of the research code's own disagreement we land on. The factored value is exact, so
    # being nearer to it than the research code's general form is what makes the 0.5% gap above their
    # rounding rather than our error.
    ours_to_exact = abs(general - reference["mu_fast"])
    theirs_to_exact = abs(reference["mu_general"] - reference["mu_fast"])
    assert ours_to_exact < theirs_to_exact / 100, (
        f"the companion's general mu ({general!r}) should sit on the research code's exact factored value "
        f"({reference['mu_fast']!r}) rather than on its bf16-rounded general one "
        f"({reference['mu_general']!r}): {ours_to_exact:.3e} away versus {theirs_to_exact:.3e}."
    )


# ==============================================================================================
# ==============================================================================================
# T16.3 -- the continual extension
# ==============================================================================================

def _extension_inputs(reference, tmp_path):
    artifact = require(reference["artifact"], "the shipped p(h) artifact")
    fused = require(reference["fused_model"], "the fused base+A model")
    corpus = link_txt_corpus(require(reference["corpus"], "the domain corpus"),
                             tmp_path / "domain", reference["documents"])
    return artifact, fused, corpus


def _collect(reference, fused, corpus, need):
    """The extension's collection half, run through the companion's own collector.

    Reaches for the private function on purpose: ``extend_artifact`` collects, fits and merges,
    and the three fail for different reasons. Separating them is what lets the tests below say
    *which* one moved.
    """
    from lfa.artifact.extend import _collect_domain_activations

    model = load_teacher(str(fused), device=reference["device"], dtype=torch.float32)
    try:
        return _collect_domain_activations(
            model, load_tokenizer(str(fused)), get_adapter(model), corpus,
            list(reference["heldout"]), need=need, seq_len=reference["seq_len"],
            seed=reference["seed"], device=reference["device"],
            keep_short_whole=reference["keep_short_whole"],
        )
    finally:
        del model
        release_cuda()


def _held_out_coordinates(reference, base: dict, key: str) -> torch.Tensor:
    """The held-out activations in un-whitened base coordinates.

    ``(h - mean) @ V`` is the convention both extensions emit their domain components in: the fit
    itself runs whitened, for conditioning, and un-whitens by ``sqrt(eig)`` on the way out.
    """
    return ((reference["heldout"][key].float() - base[key]["mean"].float())
            @ base[key]["pca_components"].float())


@pytest.mark.gpu
def test_the_extension_collects_the_same_activations(extend_reference, tmp_path):
    """The collection half, activation by activation.

    Deterministic on both sides -- same documents, same chunking, same seeded chunk order, same
    fused model -- so it is compared directly rather than by summary. The slice compared is the
    one *past* the fit's ``need``, which no mixture on either side was fitted on, so the same
    stream also underwrites the likelihood comparisons below.
    """
    reference = extend_reference
    _, fused, corpus = _extension_inputs(reference, tmp_path)
    offset = reference["heldout_offset"]
    rows = next(iter(reference["heldout"].values())).shape[0]

    collected = _collect(reference, fused, corpus, offset + rows)

    for key, expected in reference["heldout"].items():
        ours = collected[key][offset:]
        assert ours.shape == expected.shape, f"{key}: {tuple(ours.shape)} vs {tuple(expected.shape)}"
        difference = (ours - expected).abs().max().item()
        assert torch.allclose(ours, expected, rtol=1e-4, atol=1e-3), (
            f"{key}: collected activations differ from the reference stream (max abs diff "
            f"{difference:.3e}). The extension fits whatever this collects, so a difference here "
            "is a different domain, not a different fit."
        )


@pytest.mark.gpu
def test_the_domain_mixture_is_the_reference_fit_at_a_matched_initialization(extend_reference,
                                                                             tmp_path):
    """The fitting half, at the initialization the reference used.

    ``fit_domain_gmm`` is EM from a k-means++ seeding, so a mixture is only comparable with
    another mixture started from the same draw. Started from the same one it is not merely close:
    the fitted means and variances come out identical, which is the strongest statement available
    about the port of the fitter, the whitening and the un-whitening back into the base's frame.

    The reference's initialization is ``0``, not the ``42`` the capture passed: the research code's
    extension script forwards its ``--seed`` to the chunk permutation only and leaves every
    site's fit at ``fit_domain_gmm``'s default. The companion deliberately does not do that --
    one seed drives the whole extension there -- so the end-to-end test below sees the two fits
    started differently, and this test is where the fitters are compared like with like.
    """
    from lfa.artifact.extend import fit_domain_gmm

    reference = extend_reference
    artifact, fused, corpus = _extension_inputs(reference, tmp_path)
    base = load_artifact(artifact)
    collected = _collect(reference, fused, corpus, reference["need"])

    for key, expected in reference["sites"].items():
        block = fit_domain_gmm(collected[key], base[key], reference["k_domain"], seed=0,
                               device=reference["device"])
        assert int(block["gmm_n_components"]) == reference["k_domain"]
        assert int(block["n_samples"]) == reference["need"]

        weights = block["gmm_weights"] / block["gmm_weights"].sum()
        assert torch.allclose(weights, expected["domain_weights"], rtol=1e-5, atol=1e-7)
        assert torch.equal(block["gmm_means"], expected["domain_means"])
        assert torch.equal(block["gmm_covariances"], expected["domain_covariances"])

        z = _held_out_coordinates(reference, base, key)
        ours = gaussian_mixture_log_likelihood(z, weights, block["gmm_means"],
                                               block["gmm_covariances"])
        theirs = gaussian_mixture_log_likelihood(z, expected["domain_weights"],
                                                 expected["domain_means"],
                                                 expected["domain_covariances"])
        assert math.isclose(ours, theirs, rel_tol=1e-9), (
            f"{key}: held-out mean log-likelihood {ours:.6f} vs reference {theirs:.6f}")


@pytest.mark.gpu
def test_the_extension_merges_the_domain_into_the_artifact(extend_reference, tmp_path):
    """The whole call, end to end: collect, fit, merge, write.

    The merge is exact arithmetic and is checked term by term -- component count, accumulated
    sample count, weights summing to one, and the new components carrying exactly the domain's
    sample share. The mixture itself is checked only to the width of the fitter's initialization
    band, because this call's fits start from a different k-means++ draw than the reference's
    (see the test above); the tight comparison of the fits lives there, not here.
    """
    reference = extend_reference
    artifact, fused, corpus = _extension_inputs(reference, tmp_path)
    k = reference["k_domain"]

    out = extend_artifact(
        str(fused), str(artifact), corpus, tmp_path / "extended.pt",
        base_n=reference["base_n"], k_domain=k, need=reference["need"],
        seq_len=reference["seq_len"], seed=reference["seed"], device=reference["device"],
        quantize=False, keep_short_whole=reference["keep_short_whole"],
    )
    merged = load_artifact(out)
    base = load_artifact(artifact)
    share = reference["need"] / (reference["base_n"] + reference["need"])

    for key, expected in reference["sites"].items():
        entry = merged[key]
        assert int(entry["gmm_n_components"]) == expected["gmm_n_components"]
        assert int(entry["n_samples"]) == expected["n_samples"] == \
            reference["base_n"] + reference["need"]
        assert math.isclose(float(entry["gmm_weights"].double().sum()),
                            expected["gmm_weights_sum"], rel_tol=1e-6)
        assert math.isclose(float(entry["gmm_weights"][-k:].double().sum()), share, rel_tol=1e-5), (
            f"{key}: the new components carry {float(entry['gmm_weights'][-k:].sum()):.6f} of the "
            f"mixture, but the domain's sample share is {share:.6f}")

        z = _held_out_coordinates(reference, base, key)
        weights = entry["gmm_weights"][-k:].double()
        ours = gaussian_mixture_log_likelihood(
            z, weights / weights.sum(), entry["gmm_means"][-k:], entry["gmm_covariances"][-k:])
        theirs = gaussian_mixture_log_likelihood(
            z, expected["domain_weights"], expected["domain_means"],
            expected["domain_covariances"])
        assert abs(ours - theirs) <= LL_BAND * abs(theirs), (
            f"{key}: held-out mean log-likelihood {ours:.4f} vs reference {theirs:.4f} "
            f"({abs(ours - theirs) / abs(theirs):.3%} apart, initialization band {LL_BAND:.0%})"
        )


# Two implementations in one process: the storage format and the synthetic-artifact ladder
# ==============================================================================================

def test_blockwise_quantization_is_bit_identical_on_a_real_basis_block():
    """``quantize_blockwise`` against the research implementation, on real artifact data.

    Run on a shipped ``pca_components`` basis rather than on random data: the format is absmax
    per 64 values, so what a synthetic tensor would not exercise is the block boundary of a real
    orthonormal basis, whose scales vary over orders of magnitude down the spectrum.
    """
    reference_quantize = import_from_research("src.lra_quantize")
    reference = load_reference("losses.pt")          # the smallest fixture naming the artifact
    params = load_artifact(require(reference["artifact"], "the shipped p(h) artifact"))

    basis = params["12_pre_mlp"]["pca_components"]                  # dequantized on load
    ours = quantize_blockwise(basis)
    theirs = reference_quantize.quantize_blockwise(basis)

    assert ours.keys() == theirs.keys()
    assert torch.equal(ours["q8"], theirs["q8"])
    assert torch.equal(ours["scales"], theirs["scales"])
    for field in ("shape", "numel", "block", "dtype", "__q8__"):
        assert ours[field] == theirs[field], field
    assert torch.equal(dequantize_blockwise(ours),
                       reference_quantize.dequantize_blockwise(theirs))


def _synthetic_artifact(hidden: int, path: Path) -> Path:
    """An artifact exercising the rungs the shipped one does not: full covariance, whitened top-m.

    The shipped artifact is diagonal, full-width and unwhitened at every site, so replaying it
    leaves three branches of ``sample_gmm`` untested. This one carries a full-covariance head, a
    whitened top-m head with a Gaussian tail, a PCA-only site and a moments-only site.
    """
    generator = torch.Generator().manual_seed(7)
    n_comp, top_m, k = 8, 3, 4

    def basis():
        q, _ = torch.linalg.qr(torch.randn(hidden, n_comp, generator=generator))
        return q

    def block(**extra):
        entry = {
            "mean": torch.randn(hidden, generator=generator) * 0.1,
            "std": torch.rand(hidden, generator=generator) + 0.5,
        }
        entry.update(extra)
        return entry

    eigenvalues = torch.linspace(2.0, 0.2, n_comp)
    weights = torch.rand(k, generator=generator) + 0.1
    full_cov = torch.randn(k, n_comp, n_comp, generator=generator) * 0.1
    full_cov = full_cov @ full_cov.transpose(1, 2) + torch.eye(n_comp) * 0.5

    params = {
        # full-covariance head, full width, unwhitened
        "0_pre_o": block(pca_components=basis(), pca_eigenvalues=eigenvalues,
                         pca_n_components=n_comp, gmm_weights=weights / weights.sum(),
                         gmm_means=torch.randn(k, n_comp, generator=generator),
                         gmm_covariances=full_cov, gmm_n_components=k,
                         gmm_covariance_type="full"),
        # whitened top-m head: the remaining coordinates are sampled as a fitted Gaussian tail
        "1_pre_mlp": block(pca_components=basis(), pca_eigenvalues=eigenvalues,
                           pca_n_components=n_comp, gmm_weights=weights / weights.sum(),
                           gmm_means=torch.randn(k, top_m, generator=generator),
                           gmm_covariances=torch.rand(k, top_m, generator=generator) + 0.1,
                           gmm_n_components=k, gmm_covariance_type="diag", gmm_whitened=True),
        # PCA only, and moments only: the two lower rungs of the ladder
        "1_pre_qkv": block(pca_components=basis(), pca_eigenvalues=eigenvalues,
                           pca_n_components=n_comp),
        "0_pre_mlp": block(),
        "2_pre_lm_head": block(pca_components=basis(), pca_eigenvalues=eigenvalues,
                               pca_n_components=n_comp),
    }
    torch.save(params, path)
    return path


def test_the_whole_fidelity_ladder_replays_bit_for_bit_on_a_synthetic_artifact(
        tiny_model, tmp_path):
    """The same zero-tolerance claim as T16.1, over the branches the shipped artifact never takes.

    Both samplers read one file and draw the same call sequence from the global RNG, in the same
    process on the CPU -- so this needs no captured fixture and cannot go stale.
    """
    reference_distribution = import_from_research("src.lra_distribution")

    model, _ = tiny_model
    adapter = get_adapter(model)
    hidden = model.config.hidden_size
    path = _synthetic_artifact(hidden, tmp_path / "synthetic.pt")
    frequencies = torch.rand(model.config.vocab_size, generator=torch.Generator().manual_seed(3))

    calls = [("pre_qkv", 0), ("pre_o", 0), ("pre_mlp", 0), ("pre_qkv", 1), ("pre_mlp", 1),
             ("pre_lm_head", 2)]

    torch.manual_seed(4242)
    theirs = reference_distribution.LayerDistributionSampler(str(path), device="cpu")
    theirs.build_embedding_lookup_from_model(model, token_frequencies=frequencies)
    reference_draws = [theirs.sample_best(layer, site, N_ANCHOR_SAMPLES) for site, layer in calls]

    torch.manual_seed(4242)
    ours = Sampler(str(path), device="cpu", seed=None)
    ours.build_embedding_lookup_from_model(model, adapter, token_frequencies=frequencies)
    our_draws = [ours.sample_best(layer, site, N_ANCHOR_SAMPLES) for site, layer in calls]

    for (site, layer), mine, reference in zip(calls, our_draws, reference_draws):
        assert mine is not None and reference is not None, f"{site} at layer {layer}"
        assert torch.equal(mine, reference), (
            f"{site} at layer {layer}: draws differ (max abs diff "
            f"{(mine - reference).abs().max().item():.3e})"
        )
    # The layer-0 draw came off the exact lookup table, so it must be a table row verbatim.
    table = ours.params["embedding_lookup"]["pre_qkv_table"]
    assert ((our_draws[0].unsqueeze(1) == table.unsqueeze(0)).all(-1).any(-1)).all()


def test_the_artifact_build_fits_the_same_pca_basis(tiny_model):
    """``fit_site``'s PCA against the research pipeline's, on a real activation covariance.

    The build's other half is already pinned elsewhere -- the collection by the extension's
    stream comparison, the mixture by the fitter comparison -- which leaves the basis: the
    eigendecomposition, the descending sort, the clamp, and the rule that turns a 95% variance
    target into a component count. That count is what the artifact's whole size and the sampler's
    residual term are computed from, and it is an integer with a threshold in it, so it can move
    by one without anything else looking different.

    The covariance is reconstructed from the shipped artifact's own basis plus its off-basis
    residual, so the spectrum is a real one (~1024 dimensions of real decay) rather than a
    synthetic matrix whose threshold would fall in an easy place. Passing no reservoir stops the
    reference method after its PCA block, which is the part under comparison.
    """
    reference_distribution = import_from_research("src.lra_distribution")

    from lfa.artifact.collect import SiteStats
    from lfa.artifact.fit import fit_site

    reference = load_reference("losses.pt")          # the smallest fixture naming the artifact
    entry = load_artifact(require(reference["artifact"], "the shipped p(h) artifact"))["12_pre_mlp"]
    basis = entry["pca_components"].float()
    eigenvalues, mean, std = (entry["pca_eigenvalues"].float(), entry["mean"].float(),
                              entry["std"].float())
    explained = (basis ** 2 * eigenvalues).sum(dim=1)
    covariance = basis @ torch.diag(eigenvalues) @ basis.T + torch.diag(
        (std ** 2 - explained).clamp_min(0))
    covariance = (covariance + covariance.T) / 2                    # exactly symmetric for eigh

    n = 4_000
    stats = SiteStats()
    stats.n, stats.hidden_dim = n, covariance.shape[0]
    stats._sum_x = mean.double() * n
    stats._sum_xx = (covariance.double() + torch.outer(mean.double(), mean.double())) * n
    stats._m2 = std.double() ** 2 * n
    ours = fit_site(stats, pca_variance=0.95, device="cpu")

    model, _ = tiny_model
    estimator = reference_distribution.LayerDistributionEstimator(
        model, None, device="cpu", max_samples=8, use_covariance_accumulation=True)
    site = reference_distribution.LayerStatistics(layer_idx=12, level="pre_mlp")
    site.mean, site.std = mean, std
    estimator.statistics = {(12, "pre_mlp"): site}
    estimator.compute_micro_structure_from_covariance(
        {(12, "pre_mlp"): covariance}, {}, show_progress=False, store_pca_components=True,
        pca_variance_threshold=0.95)

    assert ours["pca_n_components"] == site.pca_n_components
    assert torch.equal(ours["pca_eigenvalues"], site.pca_eigenvalues)
    # The companion stores the basis fp16 (as the shipped artifact does) and the reference keeps
    # fp32, so the bases agree to the storage rounding rather than bitwise.
    assert torch.allclose(ours["pca_components"].float(), site.pca_components, atol=1e-3)
