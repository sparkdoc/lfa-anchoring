# Equivalence against `the research code`

Every other test in this suite says the package is self-consistent. These say it is the **same**
package: that `lfa` computes what the research code behind the Layerwise Function Anchoring paper
computed, at the tolerance each quantity deserves.

The comparison is fixture-based. `capture_fixtures.py` runs **inside the research code's virtualenv**,
records what the research code produces, and writes three `.pt` files into `fixtures/`;
`test_equivalence.py` runs in the companion's own venv and replays each of them. Three tests
compare the two implementations directly in one process instead, where that is possible without
a GPU.

The fixtures are committed (~12 MB). The checkpoints, artifact and corpus they were captured
from are not — they are gigabytes — so each test names the path it wanted and **skips** when it
is absent.

## Running them

The marker is deselected by the default `addopts`, so ask for it explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 pytest tests/equivalence -m equivalence -q      # all thirteen
pytest tests/equivalence -m "equivalence and not gpu" -q               # the seven CPU ones
```

Five of them import `src.*` from the the research code checkout in-process (appended to `sys.path`, never
prepended — the research root holds packages with common names). `src.lra_distribution` imports
`scipy` at module level, which is not a dependency of this package; without it the tests that
import it skip with a message naming it (`uv pip install scipy -p .venv` to run them).

Fixture paths are stored **relative to the the research code root** and re-rooted at test time, so a
checkout elsewhere works: set `LFA_RESEARCH_ROOT=/path/to/the research code`. Each fixture also records the
research checkout's git revision it was captured against.

## What each test proves

| Test | Claim | Tolerance |
|---|---|---|
| `test_the_sampler_replays_every_reference_draw_bit_for_bit` | The 85 draws of one anchoring step — 28 layers × (`pre_qkv`, `pre_o`, `pre_mlp`) then the LM-head site — from the shipped int8 artifact under one global seed | **zero** |
| `test_the_loader_reproduces_the_reference_training_batch` | The chunk count and the first training batch, collated | exact |
| `test_the_anchor_components_match_the_reference` | `L_qkv`, `L_mlp`, `L_lm_head`, `L_embed`, total | 1e-4 relative |
| `test_mu_matches_the_reference_on_both_of_its_paths` | mu through the LoRA factors and the general way | 1e-4 / see below |
| `test_the_extension_collects_the_same_activations` | The activations a continual extension collects through the fused model | `allclose(rtol=1e-4, atol=1e-3)` (measured 0.0) |
| `test_the_domain_mixture_is_the_reference_fit_at_a_matched_initialization` | The fitted domain mixture's means and variances | exact |
| `test_the_extension_merges_the_domain_into_the_artifact` | Component count, accumulated sample count, the domain's weight share | exact |
| `test_blockwise_quantization_is_bit_identical_on_a_real_basis_block` | `quantize_blockwise` on a shipped `pca_components` basis | **zero** |
| `test_the_whole_fidelity_ladder_replays_bit_for_bit_on_a_synthetic_artifact` | Full-covariance heads, whitened top-m heads with a Gaussian tail, PCA-only and moments-only sites, and the frequency-weighted layer-0 lookup | **zero** |
| `test_the_artifact_build_fits_the_same_pca_basis` | The build's `pca_n_components` and eigen-spectrum on a real activation covariance | exact / fp16 storage |
| `test_all_four_anchor_blocks_match_in_process` | `L_qkv`, `L_mlp`, `L_lm_head`, `L_embed` with **all four non-zero**, against a student perturbed everywhere | **zero** |
| `test_the_layer_schedule_matches_the_reference` | `compute_layer_weights` over {cosine, linear, exponential} × `end_ratio` {0.1, 0.5, 1.0} × normalize {True, False} | **zero** |
| `test_a_bumped_seed_changes_every_draw` | The negative control: one off in the global seed moves every draw | — |

**The sampler is the one checked at zero tolerance**, because it is the input side of the anchor:
a sample stream that has drifted is a different `p(h)`, every anchor number moves with it, and
nothing raises. The fixture-backed losses are reductions over bf16 forward passes, so their last
digits carry the accumulation order; what is claimed there is that the same blocks are computed
the same way (they come out bit-identical in fact). Where both implementations can run in one
process on the CPU — the four anchor blocks, the layer schedule, the fidelity ladder, the storage
format, the basis fit — the bar is bit-identity, and `test_a_bumped_seed_changes_every_draw` is
the control saying those replays can fail.

The extension's collection is compared with `allclose(rtol=1e-4, atol=1e-3)` rather than
bitwise — it runs a 0.6 B model forward on both sides — though it measures 0.0 in practice. Its
held-out rows come from the capture script's own mirror of the research collector
(`_collect_heldout`: the same loader, the same seeded chunk order, a hook on `gate_proj`, whose
input *is* the `pre_mlp` site), run past the fit's `need` so no mixture on either side saw them.
They are the coordinates both mixtures are scored on, not a claim in themselves.

### Two places the companion deliberately differs

Both are real, both are documented in the code, and neither is a port defect.

* **mu's general path.** the research code accumulates the LoRA delta into a bfloat16 buffer in place
  (`get_effective_weight`, `src/lra_models.py:616`) and then subtracts a nearly equal teacher
  weight, losing ~0.5% to cancellation; the companion's `effective_weight` promotes to the wider
  of the base and adapter dtypes and keeps it. So the two general forms sit 4.9e-3 apart — and
  the arbiter is the research code's *own* LoRA-factored path, which the companion reproduces bit for
  bit and which the companion's general form lands on to 7e-7. The test asserts that direction
  rather than papering over the gap.

* **The extension's fit seed.** the research code's `lra_extend_distribution_gmm.py` forwards `--seed` to
  the chunk permutation only and leaves every site's mixture at `fit_domain_gmm`'s default
  `seed=0`; the companion's `extend_artifact` uses one seed for the whole extension, fits
  included. Started from the same k-means++ draw the two fitters produce **identical** means and
  variances — that is what the matched-initialization test asserts. The end-to-end test therefore
  does not compare the mixture at all: it could only compare mixtures started from different
  draws, against a band 3.7% wide at `1_pre_mlp` (five initializations of the reference fitter
  span −624.2 to −647.2 nats) and 0.6% at `12_pre_mlp`, and every failure such a check could
  catch is caught exactly one test earlier. It asserts the merge terms, which are exact.

## Regenerating the fixtures

Needs the the research code checkout, its venv, and one GPU. Nothing under the research code is written: the
checkout is read, and the script's scratch (corpus symlinks, a dequantized copy of the base
artifact, the full merged artifact) goes to `--work-dir`, a temporary directory by default.

```bash
cd /path/to/the research code && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python lfa-anchoring/tests/equivalence/capture_fixtures.py
# or one at a time: --only sampler | losses | extend
```

Paths it reads (all under the the research code root; `capture_fixtures.py` fails fast naming any that
are missing):

| What | Path |
|---|---|
| p(h) artifact | `data/distributions/qwen3-0.6b-gmm1543k-int8/distribution_stats.pt` |
| base model | `outputs/lra/qwen3-0.6b/original` |
| recipe adapter (e15) | `outputs/lra/qwen3-0.6b/chalmers/judge_search/gmm_r32_lam100000_f0.13/checkpoint_epoch_15` |
| domain corpus | `data/domain/chalmers` (the 77 `.txt` documents only) |
| fused base+A model | `outputs/lra/qwen3-0.6b/continual_ab/lra_gmm_r4_s42/base_plus_A` |
| extension script | `scripts/lra_extend_distribution_gmm.py` |

Two details of the capture worth knowing before changing it:

* The corpus is presented to both loaders as a directory of **symlinks to the `.txt` documents
  only**. The research corpus directory also holds `.jsonl` evaluation dumps and a README, and
  the two loaders disagree about those (the research code's domain path drops a `prompt`/`response`
  record, the companion's renders it) — a disagreement that is real but is not what these tests
  are about, so it is kept out of the comparison.
* the research code's extension script loads its base artifact with a plain `torch.load` and never
  dequantizes, so it cannot read the shipped int8 file. The capture hands it a dequantized copy —
  the same tensors the companion's loader reconstructs — so that both sides fit in one basis.

## The fixtures

| File | Size | Contents |
|---|---|---|
| `sampler_draws.pt` | 7.4 MB | seed, device, the 85 `(site, layer, n)` calls and every returned tensor |
| `losses.pt` | 0.08 MB | the anchor's components, mu on both paths, the training batch, the document list |
| `extend_ref.pt` | 4.2 MB | the merged artifact's two compared sites, the fitted domain components, and 500 held-out activations per site |

Every payload also carries `research_revision` — the research checkout's `git rev-parse HEAD`
(with a `-dirty` marker) at capture time — and stores its paths relative to the the research code root.
The committed fixtures were captured at `6895aee4dc4c50f8f9d0a32916e2918da5c5c695-dirty` (the
dirty files were three `docs/*.csv`, which the capture does not read).

Draws are stored as float32. The layer-0 draw comes off a bfloat16 lookup table, and
bfloat16 → float32 is exact and injective, so equality in float32 is equality in bfloat16 — the
zero-tolerance claim survives the conversion.
