# Concepts: what Layerwise Function Anchoring does

Fine-tuning a language model on a new domain moves every weight it is allowed to move, and the
things the model used to do get moved along with it. The usual answers either keep old data around
to rehearse (replay, which needs the old corpus and grows with the number of domains) or hold the
*weights* near where they were (L2-SP, EWC, which price a parameter without asking whether the
model ever uses it).

Layerwise Function Anchoring (LFA) prices something else: the **function each sub-module computes,
on the hidden states that sub-module actually sees**. Preservation effort then lands where the
model works rather than uniformly over weight space, and — because "the states it actually sees"
is captured once, in advance, as a distribution — adaptation itself needs none of the old text.

## The distribution p(h)

Run a general-purpose seed corpus through the frozen model once and record, at each anchoring
site, what the vectors arriving there look like: a mean, a covariance (kept as a PCA basis with its
eigenvalues), and a small mixture fitted in that basis. That is the **artifact**. It is a
*per-site statistic* over hidden-state vectors — no text, no token ids, no ordering, nothing
sequence-shaped — and the same file serves every adaptation of that model afterwards.

The sites, per layer, are the *inputs* of the sub-modules being preserved
(`lfa/adapters/__init__.py::ANCHOR_SITES`):

| site | feeds | width on Qwen3-0.6B |
|---|---|---|
| `pre_qkv` | `q_proj`, `k_proj`, `v_proj` | 1024 |
| `pre_o` | `o_proj` | 2048 (heads × head dim, not the hidden size) |
| `pre_mlp` | the whole MLP, nonlinearity included | 1024 |
| `pre_lm_head` | `lm_head` | 1024 (stored under layer index = layer count) |

Layer 0's `pre_qkv` is not fitted at all: it is `input_layernorm(embed_tokens(id))`, exactly
reconstructible from the model's own weights, so the shipped artifact stores no table and
`Sampler.build_embedding_lookup_from_model` rebuilds it at load time (~300 MB not shipped).

**Data-free at adaptation time, not at artifact-build time.** The artifact is built from a corpus;
what is data-free is every adaptation afterwards, and — in a chain — every *earlier domain*, none
of which is ever stored or replayed. Say it that way; the unqualified claim is not the one this
method supports.

## The objective

One training step runs two paths that share a gradient update (`lfa/train.py::train_step`):

```
L = L_content  +  L_anchor(λ)  +  μ · L_weight
```

* **`L_content`** — ordinary causal-LM cross-entropy on the new domain's text. The only place the
  domain enters.
* **`L_anchor`** — `E_{h ~ p(h)} ‖f_student(h) − f_teacher(h)‖²`, summed over the anchored
  sub-modules with a per-layer weight, scaled by `λ`. It never touches the batch: the hidden
  states come from the artifact, the two forward passes are of the sub-modules alone. Blocks:
  attention projections and the whole MLP (`λ_qkv`, `λ_mlp`), and — only when the tied
  embedding/LM-head matrix is *not* frozen — the LM head and the composed embedding function
  `input_layernorm(embed_tokens(id))`, both at `λ_qkv`. The shipped recipe sets
  `freeze_embed: true`, so those last two terms are off: there is nothing there to preserve.
* **`μ · L_weight`** — plain `‖W_s − W_t‖²_F`, uniform over layers. Not a cheap approximation of
  the anchor but its isotropic degenerate case, kept as a **backstop** for drift the function
  anchor does not price. It is global shrinkage and is deliberately not aimed anywhere, which is
  why it does not take the anchor's layer schedule.

Anything the artifact has no statistics for is left **unanchored** rather than anchored on noise:
`Sampler.sample_best` returns `None` and that sub-module is skipped.

## λ is coupled; port it and it is a different regularizer

`λ` constrains motion inside the update subspace, so the same number binds far harder in a smaller
one. Three couplings, all enforced as warnings by `Recipe.warnings`:

* **LoRA rank.** Lower rank ⇒ lower λ. The shipped point is rank 32 at λ = 100,000; at rank 16 the
  measured frontier on a corpus of this kind sits nearer 2·10⁴–5·10⁴.
* **The artifact.** A sharper or flatter p(h) changes the anchor's scale, so an artifact swap is a
  re-tune. The recipe records `calibrated_artifact` for exactly this reason.
* **The corpus.** A different mix of document lengths or formats is a different regularization
  problem.

And diagnose against **held-out domain** perplexity, never against general-text perplexity alone:
over-anchoring makes the general number look its best while the domain collapses.

This is not an LFA quirk. Every method's strength knob is coupled to the protocol it was tuned
under; a knob ported across ranks or schedules measures the port, not the method.

## The layer schedule is scale compensation

`anchor_end_ratio: 0.1` decays the per-layer weight from 1.0 at layer 0 to 0.1 at the last layer,
and the weights are normalized to sum to 1 — the convention the published λ is calibrated against,
so changing `normalize` changes what λ means by a factor of the layer count.

It is tempting to read the decay as "anchor early layers hard so the new domain builds on shared
structure". Measured, it does the opposite (`lfa/losses.py::compute_layer_weights` carries the
numbers): the anchor is an *absolute* per-element MSE and teacher activation scale grows enormously
with depth, so even after a 10× decay the effective relative pressure still rises with depth. The
knob is scale compensation, not a hierarchy. It is kept because it is what the operating point was
tuned with and four reallocation schemes failed to beat it — not because the original story was
right.

## Why the correlated + mixture artifact, and not the diagonal one

A diagonal artifact — per-dimension mean and standard deviation — is about 1 MB and is a *known
inferior* option rather than a cheap equivalent: the linear sites' inputs are strongly correlated,
and a diagonal model misprices them by a factor of several, which shows up directly in the anchor
because the loss scales with the second moment of p(h). The shipped artifact keeps a correlated
basis and a K = 32 mixture per site instead, quantized blockwise to int8 (~108 MB against ~226 MB
in fp16, dequantized on load). Both paths add back the **off-basis residual variance** the ~95 %
basis truncates, so the sampled marginals are right and λ means what it was calibrated to mean.

The registry ships both (`lfa list-artifacts`); the diagonal one is a budget floor and needs its
own λ.

## Reading a run: two axes, never one

`lfa evaluate` prints two rows because either alone is meaningless — a run that reports only the
domain has not said what it gave up, and one that reports only WikiText-2 has not said whether it
learned anything:

| metric | what it is |
|---|---|
| general (WikiText-2) | sliding-window test perplexity, window 2048 / stride 512 |
| domain | held-out perplexity on the new domain's own documents |

`--compare-unanchored` adds a third column: the same run with λ = μ = 0. That control is what says
what the anchor bought, and it costs a second training run.

A general perplexity *below* the base model's is not a win — see [faq.md](faq.md).

## Chains: adding a domain without revisiting the last one

Between two domains the workspace does two things (`lfa extend`): merge the stage's adapter into
the model, and add the domain it learned to p(h) as a small mixture in the base's own basis,
weighted by sample share:

```
p(h) = (1 − a) · p_base(h)  +  a · p_domain(h),      a = n_domain / (n_base + n_domain)
```

That union is exact — it is the distribution of "draw from the base pool with probability 1 − a,
else from the domain pool" — so nothing is refitted and the merged artifact is a sufficient
statistic for the next round too. Each round reads only the *new* domain and adds a fixed number of
components per site: the cost per round does not grow with the number of rounds. See
[multi-domain-chains.md](multi-domain-chains.md).

## What the numbers are

Two different things, kept apart on purpose.

**The paper's result.** On Qwen3-0.6B at the shipped operating point (rank 32, λ = 100,000,
μ = 0.05, 15 epochs) the LFA paper reports domain perplexity **8.76** at a seed ΔPPL of
**−10.0 %** — seed-corpus perplexity 10 % *below* the base model's, which is the favourable end of
that axis and the contrast the paper draws with methods that pay a positive drift. (A general
perplexity below base is still not by itself evidence that anything was preserved; see
[faq.md](faq.md).) That is the paper's measurement, on the paper's corpus and instruments; this
repository does not reproduce it and does not claim to.

**This package's own measured point.** One full run of the bundled recipe was compared against a
research-code run of the *identical* configuration. (Measured before the loader change of
2026-09-08, which made the per-epoch chunk offset rotate the chunk boundaries rather than discard
each document's leading tokens. The corpus counts below are read at offset 0 and are unaffected;
the per-epoch series are, from the second epoch on, and the stream they were measured under is no
longer reachable from this package. [verification.md](verification.md) says what that changes and
why.) The anchor is a Monte-Carlo term — 16 hidden
states drawn per site per step — and the two implementations draw from independent RNG streams, so
two full runs are two draws of a stochastic objective. What is compared is therefore the
deterministic part of the run, which is not resampled. Measured 2026-09-07:

| quantity | result | tolerance |
|---|---|---|
| optimizer steps, every epoch | exact (603 … 8,969) | integer equality |
| corpus: training / held-out chunks, held-out tokens | exact (3,614 / 435 / 157,366) | integer equality |
| per-epoch content loss, all 15 epochs | worst 0.191 % | 0.5 % |
| per-epoch held-out loss, all 15 epochs | worst 0.0104 nats | 0.03 nats |

Two end-of-run perplexities sit beside those as *reported* numbers rather than as checks — domain
direct-QA 10.6996 against 10.9122, WikiText-2 drift −8.202 % against −7.898 % — each being one draw
of a sampled objective, and nothing asserts them: the spread of that draw has never been measured,
so any band on it would be a guess. The full report, including what was checked piece by piece, is
[verification.md](verification.md); the harness that produced it lives with the research code,
because it needs both implementations at once.

**After a third domain.** The paper reports that in a three-domain chain perplexity on the earlier
domains keeps accumulating — it does not degrade — while *judged answering* on those earlier
domains falls after the third stage, and that the drop is carried by the question-and-answer
*pairs* in the stage's generated supplement rather than by the anchor. The paper records that
finding as double-edged, and half of it is easy to lose: removing the supplement altogether makes
judged retention on the earlier domain **worse**, not better. This companion has no QA supplement: its
loader mixes nothing into the corpus you give it, and it computes no judged score at all. Every
number it reports is a perplexity computed locally.

## Where this sits

The research code behind the paper is a private record of every arm, every retraction and every
ladder. It is not distributed. This package is the method itself: the training loop, the artifact,
the recipe and the chain, ported and checked against that record
([verification.md](verification.md)), with the research-only scaffolding left behind.

Citation: *Layerwise Function Anchoring: Preserving Sub-Module Functions on Sampled Hidden States
for Continual Domain Adaptation* — the LFA paper (2026), authors withheld for review.
