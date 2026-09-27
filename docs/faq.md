# FAQ

## How much GPU memory does a run need?

About **8 GB** for Qwen3-0.6B at the shipped recipe, since 0.1.1. Measured on an RTX 3090
(2026-09-07) at rank 32, batch 6 × 512 tokens, 16 anchor samples: **8.63 GiB allocated at peak,
10.8 GiB reserved** by the caching allocator — with a second, separately loaded teacher, which is
what every run did before 0.1.1 and what `--teacher-mode separate` still does. A LoRA run now
holds **one** model: PEFT freezes the base weight of every module it adapts, so the student *is*
the teacher and it is read there with the adapters switched off. That is the whole of the teacher's
resident weights returned — **1.11 GiB at 0.6B** (596 M parameters in bfloat16), measured on all
three of allocated, reserved and `nvidia-smi`, at no cost in step time (−0.2 %, inside a
run-to-run spread of 0.8 %). What is left is the student, the optimizer state and the activations;
the anchor itself is small, since it evaluates sub-modules on 16 vectors rather than on the batch.

Full-weight training moves the base weights, so there the student is not a copy of anything and a
real teacher is loaded: a full-weight run still holds two models, and `--teacher-mode
adapter_disabled` is refused rather than approximated.

LFA **pins one card by default** and refuses a device map that would spread the model across
several (`ShardingRefused`). Sharding is model parallelism: it exists to fit a model that does not
fit, it buys memory rather than speed, and here it costs about 8 % because every anchored hidden
state then crosses a device boundary. Pass `--allow-sharding` deliberately, when the model — plus
the separate teacher, if the run holds one — genuinely does not fit.

**On an 8 GB card.** On an RTX 2070 (8 GB, 2026-09-26) a Qwen3-0.6B rank-32 step at 512 tokens
peaks at 2.45 / 3.52 / 4.59 GiB for micro-batch 1 / 2 / 3 (1.08 / 1.98 / 2.94 s per step) and
overflows at 6 on the fp32 logits (6 × 512 × 151,936 floats). Set `batch_size: 3` and
`gradient_accumulation_steps: 2` to keep the recipe's 6 × 512 geometry; about 6 s per optimizer
step. bfloat16 is sound on Turing: perplexity 9.67 against fp32's 9.70 (and fp16's 9.68) on one
sentence, greedy outputs equal, at about half fp16 speed. The package does not probe memory and
pick a batch for you, on purpose: the training frame must not depend on the card, so the geometry
is a recipe edit you make and the history records.

For sweeps, run one configuration per card as two independent lanes (`--device cuda:0` and
`--device cuda:1`). That is a true 2× on the queue, which no form of parallelism inside one run
gets you here.

The two things that *do* cost real memory are the artifact build (host RAM, tens of GB; on the
self-generated route, and so under `regenerate-artifact`, the build sizes itself to the RAM it finds
— below) and the continual extension (below).

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

**On the `regenerate` route** (`lfa regenerate-artifact`, or a chain with `artifact: regenerate`)
there is no `--need`: each boundary runs a whole self-generated artifact build, whose bill is the
build's reservoirs, collected `layer_group_size` layers at a time. With no group size given, the
self-generated build chooses it from the model's config and the host RAM available when it starts,
and logs the choice: on the machine this was written on, 7 for Qwen3-0.6B at the 200,000-vector
reservoir (about 10.7 GiB of reservoirs per group, against about 24 GiB available). The choice is
not neutral — one torch generator is shared across a group's sites, so grouping changes the
reservoir draws and the fitted mixtures, though not the exact moments — so the artifact's meta
records `layer_group_size`, and a rebuild reproduces a file by passing `--layer-group-size` with
the recorded value ([rebuilding-the-artifact.md](rebuilding-the-artifact.md)).

**How long it takes.** About **five minutes** at the defaults on one RTX 3090 (measured 5 min 08 s:
roughly one minute collecting activations through the fused model, then four minutes fitting 84
mixtures). The fitting half logs its progress every ten sites, so a quiet minute is normal and a
quiet five is not.

## The run warned that this machine cannot compile for the GPU

That is the toolchain check, and since 0.2.0 it is a warning, printed once, rather than a refusal:

```
This machine cannot compile for the GPU: the Python development headers (…/Python.h does not
exist) is missing for …/python3. This package's own training and generation paths ran without
it, but a torch path that JIT-compiles (torch.compile, custom triton kernels) would fail in gcc
mid-run. …
```

Some torch paths compile a small CUDA shim on the first kernel launch and need `Python.h` and a C
compiler. This package's own paths do not: a Qwen3-0.6B LoRA stage and an unconditional generation
ran on an RTX 2070 under a Python with no development headers (2026-09-26). So the run proceeds.
If a torch path of your own does compile and fails in gcc, install your distribution's development
package for the interpreter (`python3-dev` / `python3.13-dev`, plus `build-essential`); a uv- or
conda-managed interpreter ships its own headers. `LFA_SKIP_TOOLCHAIN_CHECK=1` silences the warning.

## How long does self-generation take?

Timed on an RTX 2070 (8 GB, 2026-09-26), Qwen3-0.6B:

| what | size | time |
|---|---|---|
| generation | 16 documents × 512 tokens | 54 s |
| an artifact corpus | 16 raw + 4 chat-format documents | 77 s |
| a small artifact build | 20k samples per site, K = 4, model loads included | about 3 min |
| a supplement | about six passages at 6 pairs each, batch 4 | 47 s |

None of those is the recorded frame. The full frame — 2,750 documents of up to 2,048 tokens, then
the fit at 600k samples per site — takes several hours on an 8 GB card; not timed. A supplement
costs one generation per 4,000-character passage of the training side (the
default batch is 16 passages), once per corpus and writer: it is cached under
`<workspace>/supplements/<corpus sha256[:12]>/` and reused while the training side's hash, the
writer checkpoint's hash and the template's hash all match. `lfa prepare-supplement --force`
rewrites it.

## Why does the chunk count change from epoch to epoch?

Because the chunker starts each epoch at a different random offset into every document (seeded by
the run's `seed` and the epoch number), so the last partial chunk of a document falls differently
each time and short documents can land in one chunk or two. `Epoch 1/15 (199 chunks)` followed by
`Epoch 3/15 (160 chunks)` is that, not data being dropped: every epoch sees the whole corpus, cut
in different places, which is the positional diversity the offset exists for. What does not move
is the held-out split — it is chunked once, at offset 0, so the per-epoch held-out numbers compare
like with like.

The offset moves the chunk **boundaries** rather than each document's starting point: the leading
segment `[0, offset)` is emitted as a chunk of its own beside the offset-aligned ones. That is why
the chunk count rises with the offset while the token count stays flat — one extra chunk per
multi-chunk document, not one extra pass over the text.

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
**artifact** built for that model — `lfa build-artifact --model <id> --self-generated` writes its
seed corpus with the model itself, no download — and a **λ calibration** at your rank and corpus,
against that artifact. Never port λ across models. [adding-a-model.md](adding-a-model.md) has
the interface, the fallback orders, and
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

## What is the "loader" line the logs print at the start of a run?

One setting, `keep_short_whole`, and it is about short documents rather than about matching anyone
else's stream. Under the default, `true`, a document that fits in a single chunk is trained
**whole** in every epoch: it ignores the epoch's chunk offset. Under `false` it is cut at the
offset like a longer document, into a chunk and a fragment that begins mid-sentence with none of
its own text in front of it.

Neither setting loses any text — the offset rotates the chunk boundaries, so the leading segment is
emitted either way. What changes is whether a short document arrives whole or in two pieces, and
that changes the training stream, which is why a run says which it used and records it in its
history entry rather than only in the recipe.

**Prefer `false` only when the "documents" are themselves arbitrary slices of something longer** —
a scrape cut every 3,000 characters, an export chunked by paragraph. Then keeping them whole
preserves nothing that means anything, and the positional variety between epochs is worth having.
When each document is a unit somebody wrote (an article, a page, a recipe), keep the default.

There is no setting for the chunk offset itself. It rotates the boundaries; that is what the loader
does. A switch that reproduced an older, truncating stream was briefly offered and has been removed —
[verification.md](verification.md) says why, and what it means for the numbers on that page.

## How large a corpus can I train on?

**Budget about 35 bytes of host RAM per token, or roughly eight times the corpus's size on disk**,
resident for the whole run. Tokenization is eager: the entire corpus is held as token tensors from
the moment the run starts. That is on the host and is unrelated to the ~9 GB on the GPU.

Measured end to end through `load_corpus` (2026-09-08, one RTX 3090 host, the Qwen3 tokenizer,
512-token chunks):

| corpus | on disk | tokens | resident |
|---|---|---|---|
| 500 documents | 37.6 MB | 10.1 M | **282 MiB** (29.3 B/token, 7.5× the disk size) |
| 2,000 documents | 150.4 MB | 40.4 M | **1,311 MiB** (34.1 B/token, 8.7× the disk size) |

Twenty-four of those bytes a token are the tensors themselves — `input_ids` and `attention_mask` at
eight bytes each, plus the eight-byte `labels` copy — and the rest is per-chunk object overhead and
allocator retention. So a **1 GB corpus of text needs roughly 8 GB of RAM**, and on a 32 GB machine
the practical ceiling is somewhere near **2 GB of text** (~500 M tokens) before the corpus alone is
half the machine.

A corpus estimated to exceed the memory actually available is **refused** (`CorpusTooLarge`) before
a single document is tokenized, rather than becoming an out-of-memory kill part-way through with
nothing written. The estimate is sized from characters at four characters a token, which
under-counts for scripts that tokenize denser than English — so the guard sooner misses a corpus it
should have caught than refuses one that would have fitted. It reads `MemAvailable` from
`/proc/meminfo`, so on a platform without it (macOS, some containers) **the guard does not fire at
all** and the budget above is all you have; the numbers are documented here for that reason as well
as for planning. Override it with `LFA_CORPUS_MEMORY_LIMIT_GB=<GiB>`, or `LFA_CORPUS_MEMORY_LIMIT_GB=off`.

The supplement adds to the training side on top of that: at the recipe's `supplement_fraction` of
0.13, the written pairs come to about 0.15 of the raw training tokens (0.13 / 0.87).

If the corpus is too big, the answer this package prefers is not more RAM: **split it and train the
parts as successive domains** ([multi-domain-chains.md](multi-domain-chains.md)). That is what LFA
is for, and each stage then holds only its own share.

## The run warned about my corpus's "shape". What do those mean?

Three shapes train badly without failing — the losses fall, the counts look ordinary, and the model
that comes out is quietly worse than the corpus could have made it. The trainer reads them off the
corpus **before the first step** and names the numbers it read them from. None is a refusal.

* **Too few chunks for the batch.** Under eight optimizer steps an epoch, each step's gradient
  comes from more than an eighth of the corpus, so consecutive steps see nearly the same examples
  and the shuffle buys almost nothing; at the recipe's 50-step warmup such a run also spends its
  first six epochs or more below the learning rate the operating point was tuned at. Add documents,
  lower `batch_size`, or lower `sequence_length` so each document yields more chunks.
* **One document dominating.** A single document past half the chunks contributes more gradient
  than the whole of the rest of the corpus, so the run is at least as much a fine-tune on that one
  document. The warning names the document and its share. Split it at its own section boundaries.
* **More epochs than the text can carry.** Under 500,000 training tokens (~2 MB of English) at more
  than five epochs. The shipped 15 were tuned on about 6.6 MB; the two-domain walkthrough, at 164 k
  tokens a stage, reached its held-out minimum at epoch 4 and was worse by epoch 8. Start nearer
  five, keep `val_fraction` above 0, and let the held-out curve pick the dose — the trainer names
  the epoch it bottomed at when the run ends.

A sound corpus produces none of them. To read them without starting a run,
`ChunkedCorpus.shape_warnings(batch_size=…, epochs=…)` returns them as a list of strings.

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
nothing to configure. The text self-generation needs — the artifact corpus and the domain
supplement — is written locally by the model being adapted, never by an external one. The
paper's judged results are in the paper.
