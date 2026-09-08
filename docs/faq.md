# FAQ

## How much GPU memory does a run need?

About **9 GB** for Qwen3-0.6B at the shipped recipe. Measured on an RTX 3090 (2026-09-07) at rank
32, batch 6 × 512 tokens, 16 anchor samples: **8.63 GiB allocated at peak, 10.8 GiB reserved** by
the caching allocator. A run holds two models — the student and the frozen teacher — plus the
optimizer state and the activations; the anchor itself is small, since it evaluates sub-modules on
16 vectors rather than on the batch.

LFA **pins one card by default** and refuses a device map that would spread the model across
several (`ShardingRefused`). Sharding is model parallelism: it exists to fit a model that does not
fit, it buys memory rather than speed, and here it costs about 8 % because every anchored hidden
state then crosses a device boundary. Pass `--allow-sharding` deliberately, when student and
teacher together genuinely do not fit.

For sweeps, run one configuration per card as two independent lanes (`--device cuda:0` and
`--device cuda:1`). That is a true 2× on the queue, which no form of parallelism inside one run
gets you here.

The two things that *do* cost real memory are the artifact build (host RAM, tens of GB) and the
continual extension (below).

## Why does a chain's `extend` need so much RAM?

It holds `--need` activations per site in float32 on the host before fitting. For the shipped
Qwen3-0.6B artifact — 84 sites, of which 28 are 2048 wide and 56 are 1024 wide — at the default
`--need 40000`:

```
56 × 40,000 × 1024 × 4 B  +  28 × 40,000 × 2048 × 4 B  ≈  18 GB
```

**Measured, on that exact configuration: 19.4 GiB peak resident** (`/usr/bin/time -v`, one RTX
3090, 2026-09-08) — the arithmetic above plus about 1.4 GiB of interpreter, torch and model. Halve
`--need` to halve the dominant term. The fit's quality floor is `--k-domain` components at one per
200 activations, so 40,000 is far above what 8 components need; it is chosen for coverage of the
domain, not for the fit's arithmetic.

That agreement is recent: until 2026-09-08 the collection kept every chunk in a list and
concatenated at the end, so a user measured **35.9 GiB** against this same paragraph — the chunks
and the concatenations were resident at once, and freeing the chunks did not return their pages.
Each site now fills one `--need × width` buffer in place, which is what makes the budget above the
real bill. If you are on an older version, budget twice the number.

**How long it takes.** About **five minutes** at the defaults on one RTX 3090 (measured 5 min 08 s:
roughly one minute collecting activations through the fused model, then four minutes fitting 84
mixtures). The fitting half logs its progress every ten sites, so a quiet minute is normal and a
quiet five is not.

## Why does the chunk count change from epoch to epoch?

Because the chunker starts each epoch at a different random offset into every document (seeded by
the run's `seed` and the epoch number), so the last partial chunk of a document falls differently
each time and short documents can land in one chunk or two. `Epoch 1/15 (199 chunks)` followed by
`Epoch 3/15 (160 chunks)` is that, not data being dropped: every epoch sees the whole corpus, cut
in different places, which is the positional diversity the offset exists for. What does not move
is the held-out split — it is chunked once, at offset 0, so the per-epoch held-out numbers compare
like with like.

## The anchor loss spikes by orders of magnitude on some steps. Is that a divergence?

No. The anchor is a Monte-Carlo estimate: 16 hidden states are drawn per site per step from p(h),
and an occasional draw lands far out in the distribution's tail, where the teacher and the student
disagree most. A step reading `anchor=2.1e+02` against a steady `anchor≈4e-01` is one such draw,
and the gradient it produces is real rather than spurious — it is exactly the case the anchor is
there to price. Read the **per-epoch averages** the epoch summary prints (`loss_anchor` in
`training_history.json`), not the per-step line: those should fall or hold, and a rising per-epoch
anchor average across many epochs is the thing that would be worth investigating.

## Why is full-weight training flagged?

Because every published LFA result is LoRA. The loss is method-agnostic by construction — it
compares sub-module outputs, not adapters — so full-weight training runs, and
`--full-weight` will do it. But the operating point in the bundled recipe was tuned at rank 32,
and λ's meaning goes with the size of the space it constrains: a full-weight run is not
"rank ∞ at the same λ". The warning says what to do instead — calibrate λ in the 50,000–100,000
region and read **held-out domain** perplexity, not only general-text perplexity.

## WikiText-2 perplexity came out *below* the base model's. Is that a win?

No, and it is worth being clear about why. General-text perplexity can improve for reasons that
have nothing to do with preserving what the model could do: a domain corpus that happens to look
like well-edited prose sharpens next-token statistics generally, and an over-anchored model that
has barely moved will always look good on the axis that measures not moving.

Read the two axes together and always in that order:

1. Did the domain number actually move? If not, the run is over-anchored, whatever the general
   axis says.
2. What did it cost on the general axis? That is a *cost*, and the interesting comparison is
   against the unanchored control (`--compare-unanchored`), which shows how much of that cost the
   anchor removed.

A general number below base is at best incidental and at worst a symptom. It is not evidence that
anything was preserved.

## How do I use a model that is not Qwen3?

Three steps, and only the third is real work: an **adapter** so LFA can find the sub-modules
(usually free — `LlamaLayoutAdapter` covers Llama, Qwen2/3, Mistral and their kin), an
**artifact** built for that model, and a **λ calibration** at your rank and corpus. Never port λ
across models. [adding-a-model.md](adding-a-model.md) has the interface, the fallback orders, and
the calibration procedure.

The natural next models to verify are **Gemma 3 1B** and **Llama 3.2 1B**: both should be placed by
the existing adapter, and both are small enough that the artifact build and a λ sweep fit on one
24 GB card.

## Where is the corpus the paper used?

Not here. The paper's first domain is a collection of philosophy-of-mind papers assembled from the
web by hand; the collection is not ours to redistribute, and this package ships the *recipe* rather
than the texts. The other two domains are a public dataset and a set of open-access papers, and are
likewise not bundled.

Nothing about the method depends on those particular corpora. `lfa prepare-domain` turns whatever
documents you have into the shape the loader reads, and the couplings in
[recipes.md](recipes.md) are what to re-check when your corpus differs in kind from theirs.

## What is the "loader frame" the logs mention?

`keep_short_whole` — whether a document that fits in a single chunk is present in **every** epoch
(the default, `true`) or drops out of every epoch whose random chunk offset is past its end
(`false`).

It is a **frame** field, not a tuning knob: realized exposure to the short documents differs by
about threefold between the two settings, and λ is coupled to corpus composition. A perplexity
produced under one setting is not comparable with one produced under the other. It is invisible in
every metric, which is why a run says out loud which frame it used, and why the setting is recorded
in each stage's history entry rather than only in the recipe.

The default here is `true`. This package does not offer a switch for anything else on the command
line; the field exists in `TrainConfig` and `Recipe` for a caller who must match an external
frame exactly.

## Is the learning-rate schedule exactly restored when I `--resume`?

Nearly, and the caveat is inherited from the research code. The schedule's total is
`steps_per_epoch × num_epochs`, and `steps_per_epoch` is read from the **chunk count the corpus
has when the run starts** — so the whole cosine curve depends on that count. A resume through
`Workspace.train` rebuilds the corpus the same way the first run did (same documents, same seed,
same split, chunked at offset 0), so the two agree and the scheduler is then fast-forwarded to the
saved `global_step`.

Where it can drift is a direct `lfa.train.train` caller who hands in a `ChunkedCorpus` already
re-chunked to some other epoch's offset: the chunk count differs, so the schedule's total differs,
and the resumed run follows a slightly different curve from the one it is continuing. Hand the
trainer a freshly built corpus, or call `dataset.rechunk(0, seed)` first.

Two other resume facts worth knowing. First, only `checkpoint_mode` `rolling` or `all` writes the
`training_state.pt` a resume needs *during* a run; under `none` one is written when the run
finishes, so an interrupted `none` run has nothing to resume from. Second, a resumed LoRA run gets
its adapter re-attached **trainable** — attaching one for inference instead is the classic silent
no-op, where the loss still wiggles and nothing learns. If it ever happens, the run says so: a
gradient norm of exactly 0.0 at an optimizer step is announced loudly.

## Is there anything unchecked in the artifact build?

One thing, and it is worth knowing about: **the fp16 reservoir has no range guard.** The raw
vectors retained for the mixture fit are stored in float16 by default, whose maximum is 65,504. A
model with activations above that would store `inf` and poison that site's fit, and nothing checks
for it.

Qwen3-0.6B does not come near it, but a model with large activation outliers might. Check the
per-site statistics the build logs, and if in doubt build in float32 (`build_artifact(...,
dtype=torch.float32)`), which doubles the reservoir memory —
[rebuilding-the-artifact.md](rebuilding-the-artifact.md) has the arithmetic.

## Does anything here call an external model API?

No. Every score this package computes is a perplexity, computed locally from model logits:
held-out domain perplexity and WikiText-2 by sliding window. There is no judge, no API key, and
nothing to configure. The paper's judged results are in the paper.
