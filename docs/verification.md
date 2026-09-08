# Verification: what was checked, and what came out

This package is a port. The method, the recipe and the published `p(h)` artifacts come out of a
private research repository — the record behind the Layerwise Function Anchoring paper — and the
question that matters is whether this code still computes what that code computes.

It was checked, and this page reports the result. **It reports; it does not prove.** The harness
that produced these numbers needs *both* implementations at once, plus checkpoints, artifacts and
corpora that are gigabytes and are not public, so it lives with the research code and cannot run
here. Nothing on this page is reproducible from this repository alone. A reviewer who wants the
harness, the fixtures or the raw run records can ask for them.

Two things this page is therefore not. It is not a claim that the package is correct — agreement
with another implementation is not correctness, and both could be wrong in the same way. And it is
not a reproduction of the paper's headline numbers, which were measured on the paper's corpora and
instruments and are quoted in [concepts.md](concepts.md) as the paper's, not as this package's.

Measured 2026-09-07, re-checked 2026-09-08, on one RTX 3090.

## One deliberate divergence, after the fact

Everything below was measured against the loader as it stood on 2026-09-08. **Later that day the
corpus loader was deliberately changed, and the training stream is no longer identical to the
research code's for a corpus that contains documents longer than one chunk.** The claim is not
withdrawn — it is dated, and it is conditioned.

What changed: the per-epoch chunk offset used to be where each document *started*, so a document
longer than `sequence_length` lost its first `offset` tokens in every epoch after the first — 42.6 %
of a 600-token document in an average epoch, 25.0 % of a 1,024-token one, 5.1 % of a 5,000-token
one. The offset now moves the chunk **boundaries** instead: the leading segment is emitted as a
chunk of its own, and nothing is discarded. Corpora of book-length documents — the paper's, and
this comparison's — sit in the harmless tail of that table, which is why the defect survived the
port and this page; corpora of articles, documentation pages or chapters do not, and that is a
public package's problem rather than the research record's.

What that does to the numbers on this page:

* **Unchanged.** Everything measured at offset 0, which both loaders share: the corpus counts
  (training chunks, held-out chunks, held-out tokens — all taken when the corpus is built), the
  first collated batch, and the run's **first** epoch.
* **Diverged.** Every later epoch. Each document longer than one chunk now yields one additional
  chunk per epoch, so the per-epoch optimizer-step counts and the per-epoch content and held-out
  losses in the Tier-2 table would no longer land where they did — not by drifting, but because
  the two loaders are now feeding different text.
* **Not recoverable from this package.** The truncating stream was briefly kept reachable as
  `rotate_offset=False`, and it has since been **removed**. Its only use was reproducing the
  per-epoch series on this page, and a switch that silently discards up to 85 % of a document's
  tokens in an epoch is not something to leave in a loader that strangers point at their own
  corpora — the package exists to be used on new corpora, not to reproduce these numbers. Re-doing
  the per-epoch half of this comparison means matching the two loaders deliberately: cut
  `ChunkedCorpus._chunk_bounds` back to `range(offset, n_tokens, stride)` on a branch, or compare
  at offset 0, where they still agree exactly.

The divergence is deliberate and it is in the package's favour: a user's corpus is not silently
trained on a fraction of itself. It is written down here because "bit-identical to the research
code" is the kind of claim that has to say when it stopped being true.

## Tier 1 — the pieces, against captured reference values

Each check below replays a value recorded from the research implementation, or runs both
implementations in one process where they are small enough to. Thirteen tests.

### The sampler: 85 draws, zero tolerance

One full anchoring step draws hidden states 85 times — 28 layers × (`pre_qkv`, `pre_o`, `pre_mlp`),
then the LM-head site — from the shipped int8 artifact under one seed. **All 85 replay
bit-for-bit.** Not to a tolerance: `torch.equal`.

This is the one quantity that gets no tolerance at all, because it is the *input* side of the
anchor. A sample stream that has drifted is a different `p(h)`; every anchor number downstream
moves with it and nothing raises. The draws also share one RNG stream, so a single wrong step —
the fidelity ladder picking the wrong rung, a residual-variance term, the component draw landing on
the wrong device's generator, the call order — changes every draw after it.

**The negative control fails, as it must.** A companion test bumps the seed by one and asserts that
*every* draw moves. Without it, "85 draws matched" would be consistent with a replay that compares
nothing.

### The anchor, mu, the schedule, the loader

| what | result |
|---|---|
| All four anchor blocks — `L_qkv`, `L_mlp`, `L_lm_head`, `L_embed` — run in one process against a student perturbed in all four | **bit-identical** |
| The same four, plus the total, replayed from a fixture captured on the real recipe adapter | agree to 1e-4 relative (reductions over bf16 forwards, so the last digits carry accumulation order) |
| The weight-regularization term (`mu`) on its **fast path** — the LoRA-factored form, `‖s·BA‖²_F = s²·tr((BᵀB)(AAᵀ))` | agrees to 1e-4 relative with the research implementation's own factored path (both are exact arithmetic over the fp32 factors — no near-equal subtraction on either side) |
| The layer schedule, `compute_layer_weights`, over **36 configurations** — {cosine, linear, exponential} × end-ratio {0.1, 0.5, 1.0} × normalize {on, off} × {2, 28} layers | **bit-identical** at every one |
| The training stream: the corpus's chunk count and the first collated batch | **exact** (2,453 chunks on the paper's domain-A corpus at 512 tokens) — both are read at offset 0, which the loader change above leaves alone |
| Blockwise int8 quantization of a real `pca_components` basis block | **bit-identical** |
| The whole fidelity ladder on a synthetic artifact — full-covariance heads, whitened top-*m* heads with a Gaussian tail, PCA-only and moments-only sites, the frequency-weighted layer-0 lookup | **bit-identical** |
| The artifact build's PCA basis on a real activation covariance | same component count and eigen-spectrum, to fp16 storage |

The layer schedule earns its own row because a drift there would be silent. The schedule multiplies
every layer's contribution, and λ is calibrated against the normalized convention, so a changed
schedule does not fail anywhere — it quietly re-prices depth and makes the published λ mean
something else.

**One deliberate difference, and it is in this package's favour.** `mu`'s *general* path (not the
fast one) sits 4.9e-3 from the research implementation's general path, because that one accumulates
the LoRA delta into a bfloat16 buffer and then subtracts a nearly equal teacher weight, losing about
0.5 % to cancellation. This package promotes to the wider dtype and keeps it. The arbiter is the
research code's *own* factored path — exact arithmetic, no cancellation — which this package
reproduces to 1e-4 and which its own general form lands on to 7e-7, two orders inside the 4.9e-3
gap. The test asserts that direction rather than papering over the gap.

### The artifact extension

Extending a `p(h)` artifact with a new domain is two stages, and they behave differently.

* **Collection** — the activations gathered by running the fused model over the new corpus — is
  deterministic, and matches activation by activation (`rtol=1e-4, atol=1e-3` on a 0.6 B forward;
  it measures 0.0).
* **The mixture fitted on them** is not: the two GMM implementations draw different k-means++
  initializations. **At a matched initialization the fitted means and variances are identical.**
  That is what is asserted; the end-to-end extension then asserts the merge terms — component
  count, accumulated sample count, the new domain's weight share — which are exact.

## Tier 2 — one full run, against a matched run of the research code

The bundled `qwen3-0.6b` recipe was trained end to end, and compared against a research-code run of
the **identical configuration**: same corpus, same int8 artifact, same seed, same fifteen epochs of
cosine, same loader frame. Before anything trained, the two configurations were compared field by
field — rank, α, both λ, μ, the anchor schedule, epochs, learning rate, batch geometry, warmup,
sequence length, seed, loader frame, held-out fraction — and matched on all of them. A run that
compares two different experiments produces numbers about nothing.

What is compared is the **deterministic** part of the run:

| quantity | result | tolerance |
|---|---|---|
| optimizer steps, every one of the 15 epochs | **exact** (603 … 8,969) | integer equality |
| training chunks | **exact**, 3,614 on both sides | integer equality |
| held-out chunks | **exact**, 435 on both sides | integer equality |
| held-out tokens | **exact**, 157,366 on both sides | integer equality |
| per-epoch content loss, all 15 epochs | worst epoch **0.191 %** | 0.5 % |
| per-epoch held-out loss, all 15 epochs | worst epoch **0.0104 nats** | 0.03 nats |

The step count is the tight one — and it is the row the loader change above moves, from the second
epoch on. It is a function of the document set, the split, the chunker, each epoch's chunk offset,
the batch size and the accumulation window, so a difference in any of those lands there — as an integer, which no amount of floating-point drift can blur; and because the
learning-rate schedule is a function of the step, matching steps also mean matching learning rates.
None of these is a *proof* of identity: equal chunk and token counts are arithmetically consistent
with different text. What rules that out is at the source level — both loaders enumerate the corpus
as the same sorted walk, shuffle it with the same Mersenne Twister under the same seed, and split it
at the same index — and the counts are what would catch it if that ever stopped being true.

### The two perplexities are coarse sanity checks, not criteria

Beside the table above the run also reports two end-of-run numbers, measured with the research
code's own scorer on both checkpoints:

* domain direct-QA perplexity **10.6996**, against the reference run's **10.9122** (−1.95 %);
* WikiText-2 drift **−8.202 %**, against the reference run's **−7.898 %** (0.304 points), both read
  against the same base model measured in the same run.

**Nothing asserts these, and they carry no band.** The reason is structural, not a matter of being
careful. The anchor is a Monte-Carlo term: sixteen hidden states are drawn from `p(h)` per site per
step. The research code's sampler draws them from torch's *global* RNG; this package's sampler owns
a *private* seeded generator. So the two runs are not two evaluations of one deterministic
objective — they are two **draws** of a stochastic one, and bit-equivalence between full runs is
impossible by construction. The gap is seed-scale, orders above bf16 kernel noise, and it is a wash
rather than a bias: same distribution, same *n*, same λ, same schedule.

The honest consequence is that these two numbers have no calibrated tolerance, because the spread of
that draw has never been measured on either side. A band asserted on an unmeasured spread is a
guess. They are here as sanity checks — they say the run produced a domain-adapted model in the
right place on the same instrument — and a large gap in them would mean *investigate*, starting with
a second seed on one side, which is also what would earn them a band. A miss on the deterministic
rows above is the regression.

## Where the harness is

With the research code, as two opt-in test tiers that need both sides on one machine. This
repository keeps the report; it does not keep a suite it could never run. What it does keep is
everything the *package* needs to be checked on its own — `pytest -q` here exercises the loop, the
artifact, the recipe, the workspace and the CLI without a GPU, a corpus or a network.
