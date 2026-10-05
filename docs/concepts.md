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

## The building blocks

| Block | What it is | Where it lives |
|---|---|---|
| **Model** | Any causal LM the package has an adapter for (Qwen3 today). Referenced by Hub id or path; never copied. | `lfa.adapters` |
| **Artifact** | The `p(h)` statistic for that model: per-site mean, covariance basis and K = 32 mixture, int8. Built from the model's own text, kept in the local store and reused by every workspace over that model ([the-artifact.md](the-artifact.md)). | `lfa.artifact` |
| **Recipe** | The tuned operating point (rank, λ, μ, epochs, schedule) *and what it was tuned against*, so a run that changes rank or artifact is told λ no longer means what it meant. | `lfa.Recipe` |
| **Corpus** | A flat directory of `.txt` files. `lfa prepare-domain` makes one from text, Markdown, HTML or PDF ([preparing-your-data.md](preparing-your-data.md)). | `lfa.corpus` |
| **Supplement** | Question-and-answer pairs the model writes over the corpus's training side, mixed in at the recipe's token fraction. They make the domain's knowledge answerable when the model is asked about it; they do not protect skills ([below](#what-the-supplement-does-and-does-not-do)). | `lfa.supplements`, `lfa.selfgen` |
| **Workspace** | The state machine that holds the others together across domains: which model the next stage adapts, which artifact version it anchors against, and a history entry per stage. | `lfa.Workspace` |
| **Train / Evaluate / Fuse / Extend** | The four operations on a workspace: adapt one domain; read the stage on both axes; export a plain checkpoint; fold the stage into the model *and* into `p(h)` for the next domain (or `regenerate-artifact`: fold it into the model and refit `p(h)` on that model's own text). | `lfa.train`, `lfa.evaluate` |

## The distribution p(h)

Run text through the frozen model once — this package has the model write it
([the-artifact.md](the-artifact.md#the-self-generated-artifact)) — and record, at each anchoring
site, what the vectors arriving there look like: a mean, a
covariance (kept as a PCA basis with its eigenvalues), and a small mixture fitted in that basis.
That is the **artifact**. It is a *per-site statistic* over hidden-state vectors — no text, no
token ids, no ordering, nothing sequence-shaped — and the same file serves every adaptation of that
model afterwards.

The sites, per layer, are the *inputs* of the sub-modules being preserved
(`lfa/adapters/__init__.py::ANCHOR_SITES`):

| site | feeds | width on Qwen3-0.6B |
|---|---|---|
| `pre_qkv` | `q_proj`, `k_proj`, `v_proj` | 1024 |
| `pre_o` | `o_proj` | 2048 (heads × head dim, not the hidden size) |
| `pre_mlp` | the whole MLP, nonlinearity included | 1024 |
| `pre_lm_head` | `lm_head` | 1024 (stored under layer index = layer count) |

Layer 0's `pre_qkv` is not fitted at all: it is `input_layernorm(embed_tokens(id))`, exactly
reconstructible from the model's own weights, so an artifact stores only the token frequencies
and `Sampler.build_embedding_lookup_from_model` rebuilds the table at load time (~300 MB not
stored).

**Data-free at adaptation time, not at artifact-build time.** The artifact is built from a corpus
(one the model wrote, so no dataset is downloaded — but it is still text); what is data-free is every adaptation afterwards, and — in a chain — every *earlier
domain*, none of which is ever stored or replayed. Say it that way; the unqualified claim is not
the one this method supports.

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

* **LoRA rank.** Lower rank ⇒ lower λ. Both bundled recipes are rank 32 at λ = 1,000,000 and were
  measured at no other rank; on the research corpus, at the paper's point, rank 16's frontier sat
  at roughly a fifth to a half of rank 32's λ.
* **The artifact.** A sharper or flatter p(h) changes the anchor's scale, so an artifact swap is a
  re-tune. The recipe records `calibrated_artifact` for exactly this reason, and for the
  self-generated artifact it is calibrated against, the frame it was built at
  (`self_generated_frame`).
* **The corpus.** A different mix of document lengths or formats is a different regularization
  problem.

And diagnose against **held-out domain** perplexity, never against general-text perplexity alone:
over-anchoring makes the general number look its best while the domain collapses.

This is not an LFA quirk. Every method's strength knob is coupled to the protocol it was tuned
under; a knob ported across ranks or schedules measures the port, not the method.

## The layer schedule is scale compensation

`anchor_end_ratio: 0.1` decays the per-layer weight from 1.0 at layer 0 to 0.1 at the last layer,
and the weights are normalized to sum to 1 — the convention the recipe's λ is calibrated against,
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
because the loss scales with the second moment of p(h). The artifact this package builds keeps a
correlated basis and a K = 32 mixture per site instead, quantized blockwise to int8 and dequantized
on load (the real-text Qwen3-0.6B artifact the paper's operating point was tuned against is ~108 MB int8
against ~226 MB in fp16). Both kinds add back the **off-basis residual variance** the ~95 % basis
truncates, so the sampled marginals are right and λ means what it was calibrated to mean. A
diagonal artifact is not what this package builds, and would need its own λ.

## Reading a run: two axes, never one

`lfa evaluate` prints two rows because either alone is meaningless — a run that reports only the
domain has not said what it gave up, and one that reports only WikiText-2 has not said whether it
learned anything:

| metric | what it is |
|---|---|
| general (WikiText-2) | sliding-window test perplexity, window 2048 / stride 512 |
| domain | held-out perplexity on the new domain's own documents |

`--compare-unanchored` adds a third column: the same run with λ = μ = 0, trained on the same mix
(the stage's own supplement at the stage's fraction). That control is what says what the anchor
bought, and it costs a second training run.

A general perplexity *below* the base model's is not a win — see [faq.md](faq.md).

## What the supplement does, and does not do

`train` mixes a question-and-answer supplement into the domain: the stage's entry model reads each
training-side passage and writes six question-and-answer pairs about it, and those pairs make up
`supplement_fraction` (0.13) of the training tokens. The domain content comes from the passage;
only the question-forming, the answer construction and the assistant's voice come from the model.

**What it does: reachability.** Its measured effect is on whether the domain's knowledge can be
reached when the model is asked about it: the new knowledge becomes answerable in
question-and-answer form. The recipe's λ was tuned with a supplement at 0.13 in the mix, which is
why `train` writes one by default.

**What it does not do: protect skills.** A supplement written in a skill's mode left that skill no
better, measured on instruction following (IFEval) and reasoning (GSM8K), under any method.
Keeping skills is the anchor's job: with the anchor on, instruction following stayed at the base
model's level with or without the supplement. LFA's own loss of about 10 points on GSM8K is not
repaired by self-generated inputs either; rehearsal on model-written text, a replay method, is
outside this package. So read the supplement as what makes the new domain reachable in
question-and-answer form, and nothing more.

Scope: one model (Qwen3-0.6B), one seed, one domain, judged by `gpt-5.6-luna@medium`. These are the
research runs' measurements, not this package's: the package computes no judged score.

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

**The paper's result.** On Qwen3-0.6B at the paper's operating point (rank 32, λ = 100,000,
μ = 0.05, 15 epochs) the LFA paper reports domain perplexity **8.76** at a seed ΔPPL of
**−10.0 %** — seed-corpus perplexity 10 % *below* the base model's, which is the favourable end of
that axis and the contrast the paper draws with methods that pay a positive drift. (A general
perplexity below base is still not by itself evidence that anything was preserved; see
[faq.md](faq.md).) That is the paper's measurement, on the paper's corpus and instruments; this
repository does not reproduce it and does not claim to.

**This package's own measured point.** One full run of the bundled recipe (the 0.1.x recipe,
before the supplement; it trained on the raw corpus alone) was compared against a research-code
run of the *identical* configuration. (Measured before the loader change of
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
judged answering on the earlier domain **worse**, not better. This companion writes a
supplement too — [above](#what-the-supplement-does-and-does-not-do) — but it computes no judged
score at all, so it cannot show either half of that finding: every number it reports is a
perplexity computed locally.

## Where this sits

The research code behind the paper holds every arm, every retraction and every ladder. It is
private and not distributed. This package is the method itself: the training loop, the artifact,
the recipe and the chain, ported and checked against that code
([verification.md](verification.md)), with the research-only scaffolding left behind.

Citation: *Layerwise Function Anchoring: Preserving Sub-Module Functions on Sampled Hidden States
for Continual Domain Adaptation* — the LFA paper, submitted to ICLR 2027; authors anonymous during
review. On OpenReview: <https://openreview.net/forum?id=68rQ2UBOOC>.
