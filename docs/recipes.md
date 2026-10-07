# Recipes

A recipe is one **tuned operating point**, in one file: what to train, how hard to anchor, and —
this is the part that makes it a recipe rather than a bag of defaults — *what it was tuned at*, so
that a run departing from that point can be told λ no longer means what it meant.

```bash
lfa train --workspace runs/world_history --corpus data/world_history --recipe qwen3-0.6b     # a bundled name
lfa train --workspace runs/world_history --corpus data/world_history --recipe my_point.yaml  # or a path
```

```python
from lfa import Recipe
recipe = Recipe.load("qwen3-0.6b")           # bundled, so it works from an installed wheel
config = recipe.to_train_config(stage=1, artifact_path="artifacts/v1.pt")
```

`Recipe.load` resolves a bare name against the recipes inside the package and anything
path-shaped as a path. Values are validated on construction, so a hand-edited YAML fails at load
rather than several GPU-hours into a run.

## The shipped point: `lfa/recipes/qwen3-0.6b.yaml`

Rank-32 LoRA over the model's own self-generated artifact at `self_generated_frame`, λ = 1,000,000
on both site families, μ = 0.05,
16 anchor samples per step, fifteen epochs of cosine, with the model's own question-and-answer
supplement mixed in at 0.13 of training tokens.

### Adapter

| field | value | what it does |
|---|---|---|
| `lora_rank` | 32 | the update subspace λ is calibrated against |
| `lora_alpha` | 64 | α/r = 2 |
| `freeze_embed` | `true` | freezes the tied embedding/LM-head matrix — and with it the LM-head and embedding anchor terms, which have nothing left to preserve |
| `full_weight` | `false` | LoRA is the validated path; see [faq.md](faq.md) |

### Anchoring

| field | value | what it does |
|---|---|---|
| `lambda_qkv` / `lambda_mlp` | 1000000.0 | the function anchor's weight on the attention projections and on the MLPs. Coupled to rank, artifact and corpus — see below |
| `mu` | 0.05 | the uniform weight backstop, `μ‖W_s − W_t‖²_F`, for drift the function anchor does not price |
| `anchor_end_ratio` | 0.1 | the last layer's anchor weight relative to layer 0 (`1.0` = uniform) |
| `anchor_schedule` | `cosine` | how the per-layer weight interpolates (`cosine`, `linear`, `exponential`) |
| `n_anchor_samples` | 16 | hidden states drawn per site per step. The anchor loss is mean-reduced, so this is unbiased at every value and moves variance only |

The per-layer weights are **normalized to sum to 1**, which is the convention the recipe's λ is
calibrated against: under the uniform default a scheduled anchor would be larger by a factor of the
layer count, and λ absorbs it. The schedule is scale compensation rather than a hierarchy —
[concepts.md](concepts.md) has the measurement.

### Run length and checkpointing

| field | value | what it does |
|---|---|---|
| `epochs` | 15 | the run, and the whole learning-rate curve |
| `checkpoint_mode` | `rolling` | overwrite `latest_model` every `checkpoint_every` epochs (`none` writes only `final_model`; `all` accumulates `checkpoint_epoch_N`) |
| `checkpoint_every` | 5 | both periodic modes also write `training_state.pt`, which is what makes `--resume` possible |

`final_model` is what ships. There is no separately-kept "best" checkpoint, and a re-run at the
lowest epoch takes its place: [tuning.md](tuning.md#2-run-the-recipe-once-and-read-its-last-line)
gives the reasons (the schedule, the one axis, the held-out number) and the measured re-runs.

> ⚠ **`epochs: 15` is a dose, and the corpus and λ together set how much of it a run can
> carry.** At the recipe's λ the walkthrough's Darwin text (~189 k training tokens an epoch)
> turned late and shallowly: its held-out perplexity was lowest at epoch 10 (17.62) and ended at
> 18.09, 2.6 % above that ([below](#how-λ-was-chosen)); on Qwen3-1.7B, at the same λ, lowest at
> epoch 11 (12.69) and ended at 12.71, 0.2 % above. The trainer reads both curves as turned, and
> by the rule below advises a re-run at epoch 10 for the first and calls one optional for the
> second. No run at the turn's epoch count exists for Darwin, so whether 10 or 11 epochs would ship
> a better model on this text is unmeasured. A weaker anchor, or a smaller corpus, reaches its
> held-out minimum
> much earlier: at λ = 100,000 the same Darwin text bottomed at epoch 4 and ended worse than the
> base model on both axes, and on a measured 45-document, ~106 k-token corpus, also at
> λ = 100,000, the held-out perplexity bottomed at **epoch 2** (6.67) and ended the fifteenth
> epoch at **16.88** — and since `final_model` is the last epoch, that run shipped a model far
> worse on its own domain than one it had passed through. The anchor was doing its job
> throughout (the unanchored control was worse still on both axes); the *dose* was wrong.
>
> **The rule.** The trainer prints `Held-out: loss=… perplexity=…` after every epoch, and at the
> end of every run one line that reads that column for you — `Held-out perplexity: lowest X at
> epoch k of n; final Y (+z % over the lowest).` — followed by what the curve says:
>
> * **It turned** (the lowest epoch is before the last): the run names that epoch, and what it
>   says depends on how far the end rose above it.
>   * **10 % or more:** a warning ("this run trained past its own optimum"), advising a re-run
>     with `--epochs <that epoch>`.
>   * **1 % to 10 %:** INFO, saying a re-run at that epoch is *likely* to ship a better model on
>     this domain. Likely, not certain: there is one measured case. On a single 757 KB book
>     (H. G. Wells, *A Short History of the World*, split into 67 chapters, 7 held out;
>     Qwen3-0.6B at this recipe; one seed) the held-out perplexity was lowest at epoch 8 (27.02)
>     and ended 6.1 % above it (28.67), and the re-run at `--epochs 8` was better on both axes:
>     WikiText-2 15.94 against 16.59, held-out domain 26.91 against 28.66.
>   * **Under 1 %:** INFO, saying the re-run is optional. The 1 % cut is a judgement, not a
>     measurement. For scale, the run-to-run spread at a fixed recipe and seed (Qwen3-1.7B,
>     λ = 1,000,000, three runs; the Qwen3-1.7B section below) is about 0.15 % on
>     held-out domain perplexity and about 0.5 % on WikiText-2.
> * **It had not turned** (the lowest epoch is the last, which a one-epoch run always is): more
>   epochs may lower it further, and the run says the dose can be raised with `--epochs`. The
>   exception is a re-run at an earlier run's turn, which usually ends at its own lowest: when the
>   workspace holds that earlier run (same stage, same frame but the epochs), the run names it,
>   says to stop there, and compares the two finals — keep the lower, and two less than about
>   0.15 % apart are the same ([tuning.md](tuning.md#when-the-re-run-does-not-turn)).
> * **It diverged** (the last held-out value is not finite): a warning that the run diverged and
>   its `final_model` should not be shipped, with the lowest finite epoch named for a re-run at
>   `--epochs <that epoch>`, or a stronger anchor or a lower learning rate. The summary is not
>   read off the last finite epoch, which would hide the divergence.
> * **There was no curve** (`val_fraction` 0, or a corpus with too few documents to hold any
>   out): nothing chose the dose, and the run says so. Keep `val_fraction` above 0, and split a
>   single long file into documents with `lfa prepare-domain --split-chars 3500`
>   ([preparing-your-data.md](preparing-your-data.md)).
>
> Re-run at the epoch rather than truncating the run you have, because the learning-rate schedule
> is laid over the epoch count (`--epochs 3` is a complete three-epoch run, not the first three
> epochs of fifteen). The same reading — verdict, lowest epoch and value, final value, gap — is
> kept in the stage's entry in the workspace's `history.json`, under `held_out`, and a re-run at
> an earlier run's turn names that run, its turn epoch and its final under `rerun_of`. Re-tuning
> the dose does not re-tune λ: they are separate knobs, and λ's couplings are below.
> [tuning.md](tuning.md) puts the dose, the control and λ together as one procedure for your
> corpus.
>
> **Where the re-run goes.** In the same workspace, before `lfa extend`, `lfa train --corpus <the
> same corpus> --epochs 8` is a second run of the stage, not an overwrite (after `extend`, the same
> corpus would be the next stage): it writes `runs/stage1_run2` (a third,
> `runs/stage1_run3`), and the first run stays in `runs/stage1`. The latest run of a stage is the
> one `lfa evaluate`, `lfa fuse`, `lfa extend` and `lfa regenerate-artifact` read from then on;
> `train` says so when it starts and when it ends, `evaluate` logs the directory it read, and every
> run keeps its own entry in `history.json`. `--resume` is the exception: it continues the latest
> run in its own directory, at the λ and μ that run was started with: without `--lambda`/`--mu`
> it takes them from the run's `config.json` and says so, and an explicit `--lambda` or `--mu`
> that differs is refused, naming the run's value exactly and the flag to give or drop.
>
> **The other end of the same axis.** A corpus so small that the whole run takes fewer optimizer
> steps than `warmup_steps` (50 here) never reaches the learning rate this operating point was
> tuned at — a handful of documents can be two or three steps in total. The trainer says so at the
> start of such a run. More epochs will not fix it, because the schedule is laid over the epochs:
> more documents will.

### Optimization

| field | value |
|---|---|
| `learning_rate` | 3e-4 |
| `lr_schedule` | `cosine` — 50 steps of linear warmup (0.1× → 1×), then cosine decay over every remaining step to zero (the floor is the trainer's `lr_floor`, which recipes do not expose and which defaults to 0); `constant` holds the peak instead |
| `batch_size` × `gradient_accumulation_steps` | 6 × 1, at `sequence_length` 512 |
| `warmup_steps` | 50 |
| `weight_decay` | 0.01 |
| `seed` | 42 — the chunking, the document shuffle and the sampler's private generator |

The schedule is always laid over the epochs *actually* trained, so `--epochs 5` is a different
curve, not a truncation of this one.

### Data

| field | value | what it does |
|---|---|---|
| `keep_short_whole` | `true` | a document that fits in one chunk is trained **whole**, in every epoch, rather than being cut at the epoch's chunk offset into a chunk and a fragment that starts mid-sentence. It changes the training stream, so a run records which setting it used. Prefer `false` only when the documents are themselves arbitrary slices of something longer, so that keeping them whole preserves nothing and the extra positional variety is worth having |
| `val_fraction` | 0.1 | share of *documents* (shuffled under `seed`) held out of training and scored after every epoch. Set it to `0.0` to train on everything — and then read the domain number as a fit |
| `supplement_fraction` | 0.13 | share of training *tokens* made up of the question-and-answer pairs the stage's entry model writes from the training-side documents ([quickstart.md](quickstart.md)); `0.0` trains on the raw corpus alone. The shipped λ was tuned with a question-and-answer supplement at 0.13 of training tokens; every companion run before 0.2.0 trained at 0, off that frame. The held-out documents are split off before anything is mixed, so the domain number stays a raw-text measurement comparable across fractions. The pairs are taken as a prefix of the written file, the one whose achieved share is closest to the target; a pool too small to reach it trains at what it has and warns with the achieved share |

### Calibration record

| field | value |
|---|---|
| `calibrated_rank` | 32 |
| `calibrated_artifact` | `self-generated` |
| `self_generated_frame` | `{n_raw: 2500, n_chat: 0, max_new_tokens: 2048, max_samples: 600000, gmm_k: 32, pca_variance: 0.95}` |
| `stage2_lambda_multiplier` | 3.0 |

`calibrated_rank` and `calibrated_artifact` are what `Recipe.warnings(rank, artifact_id,
artifact_meta)` reads a run against; `stage2_lambda_multiplier` is what
[a chain](multi-domain-chains.md) multiplies λ by from stage 2 on.

`calibrated_artifact: self-generated` means the recipe is calibrated against an artifact fitted
on *this recipe's model's* own text ([the-artifact.md](the-artifact.md#the-self-generated-artifact)),
at the frame `self_generated_frame` records — the frame `lfa init --artifact self-generated`
builds at by default. An artifact carries the frame it was built at in its meta (`selfgen_frame`),
and the recipe compares the two field by field. The recipe's λ was calibrated against this model's
self-generated artifact at this frame (below). Any other value of `calibrated_artifact` is an
artifact id or path, compared as a string with the one the workspace records.

### How λ was chosen

Scope: Qwen3-0.6B; its self-generated artifact at the frame above; the Darwin text of [the
two-domain walkthrough](../examples/two_domain_walkthrough.ipynb) (*On the Origin of Species*)
with the model's own question-and-answer supplement (1,549 pairs) at 0.13 of training tokens,
188,840 training tokens an epoch; rank 32; batch 6; one seed; perplexity; RTX 3090; 2026-10-05.
Every rung trained the recipe's 15 epochs and was scored on WikiText-2 over 100 windows (51,200
tokens) and on held-out Darwin. Base: WikiText-2 17.93, held-out Darwin 30.12. The last column is
the per-epoch validation perplexity (the training split's own 10 %), its lowest and its last.

| λ (μ = 0.05 when λ > 0) | WikiText-2 (100 windows) | held-out Darwin | validation, lowest → last |
|---|---|---|---|
| 0 (μ = 0, unanchored) | 281.36 (+1469 %) | 182.89 (+507 %) | 15.27 at epoch 2 → 182.70 |
| 50,000 | 22.62 (+26.1 %) | 40.49 (+34.4 %) | 16.25 at epoch 4 → 41.56 |
| 100,000 | 19.58 (+9.2 %) | 32.28 (+7.2 %) | 16.63 at epoch 4 → 31.68 |
| 200,000 | 17.86 (−0.4 %) | 24.72 (−17.9 %) | 17.02 at epoch 6 → 24.60 |
| 500,000 | 17.00 (−5.2 %) | 19.75 (−34.4 %) | 17.42 at epoch 6 → 19.57 |
| **1,000,000** | **16.60 (−7.5 %)** | **18.43 (−38.8 %)** | 17.62 at epoch 10 → 18.09 |
| 2,500,000 | 16.53 (−7.8 %) | 18.68 (−38.0 %) | 18.29 at epoch 12 → 18.30 |
| 5,000,000 | 16.66 (−7.1 %) | 19.57 (−35.0 %) | 19.02 at epoch 15 → 19.02 |

Percentages are against the base model. The unanchored control over-trains at 15 epochs, so the
selection is [the cookbook's §5](model-integration-cookbook.md#5-calibrate-λ) over-training
branch, the rule the Qwen3-1.7B recipe was chosen by: the rung best on both axes, extending the
ladder until it turns over. No rung is best on both — 2,500,000's WikiText-2 is 0.4 % below
1,000,000's (16.53 against 16.60) while its held-out Darwin is higher — so the rule's fallback
applies: the lowest held-out Darwin among the rungs within 1 % of the lowest WikiText-2
(≤ 16.70: 1,000,000, 2,500,000 and 5,000,000). That is **1,000,000**, the ladder's turnover
(held-out Darwin 18.43 < 18.68 < 19.57) and the value the same procedure gave Qwen3-1.7B on the
same Darwin text. No repeat was run for this model; the 0.4 % WikiText-2 difference is within the
run-to-run spread measured for Qwen3-1.7B at this λ (about 0.5 %, below).

**The dose at the chosen λ.** The validation perplexity fell from 20.45 after epoch 1 to its
lowest, 17.62, at epoch 10, and ended at 18.09. At λ = 100,000 it bottomed at epoch 4 (16.63) and
ended at 31.68, and that run ended worse than the base model on both axes. An earlier unanchored
4-epoch run on the same corpus reached held-out Darwin 18.73 with WikiText-2 26.80 (+49.4 %); the
1,000,000 rung ends below it on both. One seed, one corpus: on your own corpus, read the per-epoch
column as the dose note above says.

**The stage-2 multiplier.** The same two-stage chain as the 1.7B's below — Darwin for 15 epochs,
the artifact extended, then the walkthrough's cookery text for 15 epochs — at multipliers 1 and 3,
read the same way. One seed, one pair of domains, perplexity, RTX 3090, 2026-10-05. The peak is
nvidia-smi `memory.used` sampled every 10 s over the whole chain, an upper bound (see the memory
note below).

| stage-1 λ | multiplier (stage-2 λ) | cookery held-out | Darwin held-out over stage 2 | WikiText-2 over stage 2 (100 windows) | chain peak MiB |
|---|---|---|---|---|---|
| 1,000,000 | 1× (1,000,000) | 24.07 → 16.79 (−30.3 %) | 18.44 → 21.70 (+17.6 %) | 16.62 → 16.95 (+1.9 %) | 14,588 |
| 1,000,000 | **3× (3,000,000)** | 24.30 → 16.04 (−34.0 %) | 18.49 → 19.83 (+7.2 %) | 16.63 → 16.55 (−0.5 %) | 14,565 |

3× was ahead on all three. Stage 1 of the two chains replicates the ladder's 1,000,000 rung:
held-out Darwin 18.41 and 18.46 against 18.43, WikiText-2 16.61 and 16.63 against 16.60.

**The paper's operating point** is a different one: λ = 100,000 at rank 32, calibrated in the
research code on a corpus about ten times larger (about 1,700 documents, ~1.8 M training
tokens), first against an artifact fitted on real text (the 10 : 1 pretraining-to-instruction
seed corpus, 1.54 M hidden vectors per site;
[the-artifact.md](the-artifact.md#advanced-an-artifact-fitted-on-real-text)), and judged there
([concepts.md](concepts.md#what-the-numbers-are)). It is the paper's point, not this recipe's.

## Qwen3-1.7B: `lfa/recipes/qwen3-1.7b.yaml`

The same operating point for `Qwen/Qwen3-1.7B`: every field above is the Qwen3-0.6B recipe's
except `name` and `model_id`. Its λ was calibrated on this model by the same procedure and came
out at the same `lambda_qkv` = `lambda_mlp` = **1,000,000**.
`lfa init --model Qwen/Qwen3-1.7B` adopts it.

### Calibration record

| field | value |
|---|---|
| `calibrated_rank` | 32 (`lora_rank` 32, `lora_alpha` 64) |
| `lambda_qkv` / `lambda_mlp` | 1000000.0 |
| `stage2_lambda_multiplier` | 3.0 |
| `calibrated_artifact` | `self-generated` — Qwen3-1.7B's own artifact |
| `self_generated_frame` | `{n_raw: 2500, n_chat: 0, max_new_tokens: 2048, max_samples: 600000, gmm_k: 32, pca_variance: 0.95}` |
| `epochs` | 15 |
| `batch_size` × `gradient_accumulation_steps` | 6 × 1, at `sequence_length` 512 |

**How λ was chosen.** Scope: Qwen3-1.7B; its self-generated artifact at the frame above; the
Darwin text of [the two-domain walkthrough](../examples/two_domain_walkthrough.ipynb) (*On the
Origin of Species*) prepared with its question-and-answer supplement; rank 32; one seed;
perplexity; RTX 3090; 2026-10-04/05. Every rung trained the recipe's 15 epochs and was scored by
`lfa evaluate` at its defaults: WikiText-2 over 100 windows (51,200 tokens) and 27 held-out Darwin
documents (26,147 tokens). Base: WikiText-2 15.09, held-out Darwin 21.45.
The selection rule set beforehand assumed that the unanchored control improves the domain; at 15
epochs it does not — on this small corpus (about 189 k tokens an epoch) it over-trains, to
held-out Darwin 162.7 and WikiText-2 +778 %. So λ was chosen on the 15-epoch frontier as the rung
best on **both** axes, extending the ladder until it turned over. 1,000,000 is the interior
optimum. Every WikiText-2 figure in this table and the next paragraph is the 100-window measure.

| λ (μ = 0.05 when λ > 0) | epochs | WikiText-2 (100 windows) | held-out Darwin |
|---|---|---|---|
| 0 (μ = 0, unanchored) | 4 | 15.01 (−0.5 %) | 14.34 (−33.2 %) |
| 0 (μ = 0, unanchored) | 8 | 40.30 (+167 %) | 49.08 (+129 %) |
| 0 (μ = 0, unanchored) | 15 | 132.47 (+778 %) | 162.70 (+658 %) |
| 20,000 | 15 | 15.67 (+3.8 %) | 42.00 (+95.8 %) |
| 50,000 | 15 | 15.03 (−0.4 %) | 27.04 (+26.1 %) |
| 100,000 | 15 | 14.48 (−4.0 %) | 19.89 (−7.3 %) |
| 200,000 | 15 | 14.10 (−6.6 %) | 16.47 (−23.2 %) |
| 500,000 | 15 | 13.43 (−11.0 %) | 14.00 (−34.7 %) |
| **1,000,000** | 15 | **13.15 (−12.8 %)** | **13.63 (−36.5 %)** |
| 2,500,000 | 15 | 13.31 (−11.8 %) | 13.82 (−35.6 %) |
| 5,000,000 | 15 | 13.91 (−7.8 %) | 14.24 (−33.6 %) |

Percentages are against the base model. The stage-1 model was trained three times at each of λ =
100,000 and 1,000,000 (the ladder's run and the first stage of each chain below), which gives the
run-to-run spread on the same measures: at λ = 1,000,000 WikiText-2 13.15, 13.09 and 13.13 (100
windows; about 0.5 %) and held-out Darwin 13.63, 13.64 and 13.65 (about 0.15 %); at λ = 100,000
WikiText-2 14.48, 14.36 and 14.41 (100 windows) and held-out Darwin 19.89, 19.43 and 19.88. The
1,000,000 rung's lead over 2,500,000 — 1.2 % on WikiText-2 (13.15 against 13.31) and 1.4 % on
held-out Darwin (13.63 against 13.82) — is larger than that spread on both axes. WikiText-2 falls
below base at strong λ; that is what was measured, and no mechanism is claimed for it. A general
number below base is not evidence that anything was kept: [the
FAQ](faq.md#wikitext-2-perplexity-came-out-below-the-base-models-is-that-a-win) says how to read it.

**The dose at the chosen λ.** On this corpus (about 189 k training tokens an epoch), the per-epoch
validation perplexity — the training split's own 10 % — fell at λ = 1,000,000 from 14.53 after
epoch 1 to its lowest, 12.69, at epoch 11, and ended at 12.71, 0.2 % above it: a shallow turn,
which the trainer reports with the re-run optional. At
λ = 100,000 it bottomed at epoch 5 (12.25) and ended at 18.45, the turn the dose note above
describes. The unanchored 4-epoch run reaches held-out Darwin 14.34 with WikiText-2 (100 windows)
flat (15.09 → 15.01). One seed, one corpus: on your own corpus, read the per-epoch column as the
dose note above says.

**The stage-2 multiplier.** A two-stage chain — Darwin for 15 epochs, the artifact extended, then
the walkthrough's cookery text (*Domestic Cookery*) for 15 epochs — at multipliers 1 and 3. Cookery
is scored on 15 held-out documents. In every column the first number is the fused stage-1 model
and the second the model after stage 2. WikiText-2 is the 100-window measure (51,200 tokens), as
in the ladder. One seed, one pair of domains, perplexity, RTX 3090, 2026-10-04/05.

| stage-1 λ | multiplier (stage-2 λ) | cookery held-out | Darwin held-out over stage 2 | WikiText-2 over stage 2 (100 windows) | stage-2 training peak MiB |
|---|---|---|---|---|---|
| 1,000,000 | 1× (1,000,000) | 16.16 → 11.90 (−26.4 %) | 13.64 → 16.24 (+19.0 %) | 13.09 → 13.16 (+0.5 %) | 16,206 |
| 1,000,000 | **3× (3,000,000)** | 16.20 → 11.84 (−27.0 %) | 13.65 → 15.40 (+12.8 %) | 13.13 → 12.79 (−2.6 %) | 19,291 |
| 100,000 | 1× (100,000) | 22.35 → 20.38 (−8.8 %) | 19.43 → 29.76 (+53.1 %) | 14.36 → 14.53 (+1.2 %) | 17,511 |
| 100,000 | 3× (300,000) | 22.75 → 13.92 (−38.8 %) | 19.88 → 19.90 (+0.1 %) | 14.41 → 13.87 (−3.8 %) | 16,842 |

At λ = 1,000,000, 3× matched 1× on cookery (−27.0 % against −26.4 %, one seed) and beat it on
Darwin retention and on WikiText-2. At λ = 100,000, 3× was ahead on all three.

**Training memory**, one RTX 3090, batch 6 × 512: an anchored 15-epoch run trained at about
12.7 GiB (maximum 13,033–13,056 MiB across every rung, 20,000 to 5,000,000); the unanchored control
at a maximum of 11,236 MiB; the second stage of a chain, training on the fused stage-1 model,
16,206–19,291 MiB (per arm in the table above; 19,291 MiB is about 18.8 GiB, more than a 16 GB
card holds). The measure is nvidia-smi `memory.used` sampled
every 10 s, which includes PyTorch's caching allocator, so it is an upper bound on what a run
needs. It is not comparable with Qwen3-0.6B's "8.63 GiB allocated" in [faq.md](faq.md), which
is the allocator's own peak, a different measure.

**λ does not port** across artifacts, domains or protocols. A separate research run on another
domain chose 50,000 for this model, against an artifact fitted on real text and at a different
protocol; that is a different point, not a check on this one. The two bundled recipes were
calibrated the same way on the same Darwin text and landed on the same λ: two models, one corpus, one
seed, which is not evidence that λ is independent of model size.

## The couplings, and what the warnings mean

`Workspace.train` calls `Recipe.warnings` before anything is loaded and logs what comes back. None
of them is a refusal — an off-calibration run is allowed, it just is not the measured operating
point:

* **rank ≠ `calibrated_rank`** — λ constrains motion inside the rank-`r` update subspace, so the
  same value binds harder at a lower rank. Lower rank ⇒ lower λ (on the research corpus, at the
  paper's point, Qwen3-0.6B's rank 16 sat at roughly a fifth to a half of rank 32's λ; neither
  bundled recipe was measured at another rank). Re-tune rather than port
  ([model-integration-cookbook.md](model-integration-cookbook.md) §5).
* **artifact ≠ `calibrated_artifact`** — a different p(h) prices the same function differently.
  Re-tune, and read the *frontier* of (preservation, adaptation) points rather than one point.
* **`full_weight: true`** — outside the validated envelope: each recipe's λ was calibrated for
  LoRA. Re-calibrate λ for full weight (for Qwen3-0.6B, the research code's starting range on the
  research corpus, at the paper's point, was 50,000–100,000) and check held-out domain
  perplexity.

Against a `self-generated` calibration, the artifact line above is read off the artifact's meta
rather than its id, and is one of these:

* nothing, when the artifact was fitted on this recipe's model's own text at
  `self_generated_frame` — the default `lfa init --artifact self-generated` on the recipe's model;
* *"this self-generated artifact was built at a different frame from the one this recipe's lambda
  was calibrated at: n_raw 60 (calibrated at 2500), …"* — every field that differs is named. A
  trial build (`--n-raw 60 --max-new-tokens 128`) is the common case, and fine for trying the
  pipeline; for a real run, build at the recorded frame or calibrate λ;
* *"this self-generated artifact describes '…', not this recipe's '…': lambda is coupled to the
  p(h) artifact, so calibrate it against held-out domain perplexity for this model."* — the text
  came from a different model. In a chain on the `regenerate` route it fires from stage 2 on and
  names the fused stage model's path, because the regenerated artifact was written by that model;
  there it carries one more sentence, that regenerating the artifact from each stage's model was
  measured on one configuration (rank 4, one seed) and the stage multiplier is a starting point
  there, not a calibrated constant
  ([multi-domain-chains.md](multi-domain-chains.md#the-regenerate-route));
* *"this recipe's lambda (…) is calibrated against an artifact fitted on the model's own text at
  …, and you are anchoring against '…', which is not one."* — an artifact fitted on real text, or
  one whose provenance is unknown: calibrate λ against held-out domain perplexity
  ([model-integration-cookbook.md](model-integration-cookbook.md), §5).

One more is logged by `train` itself rather than by the recipe: *"Training on the raw corpus alone:
this recipe's lambda was calibrated at supplement_fraction 0.13 and this run mixes none."* It fires
on `--no-supplement` (`supplement=False`) when the recipe's `supplement_fraction` is above 0; a
recipe that sets `0.0` has opted out at the recipe level and is not warned. And a `--lambda` or
`--mu` that differs from what the recipe would have used at that stage is named beside the
recipe's value (`Recipe.override_notes`; [below](#trying-another-λ)).

**Diagnose against held-out domain perplexity.** Over-anchoring makes general-text perplexity look
its best while domain quality collapses, so the general axis alone cannot tell you λ is too high.

## Trying another λ

λ is coupled to the corpus as well as to the rank and the artifact, so the recipe's value is where
to start on your text, not a promise about it. [tuning.md](tuning.md) says when another is worth
trying, which values, and how to pick between them. To try one, set it for one run:

```bash
lfa train --workspace runs/world_history --corpus data/world_history --lambda 2500000
```

`--lambda X` is the stage's λ **exactly**: both `lambda_qkv` and `lambda_mlp`, with no stage
multiplier on top (at stage 2, `--lambda 3000000` is what the shipped recipes would have used).
`--mu Y` does the same for μ. From Python: `ws.train(corpus, lambda_=2.5e6, mu=0.05)`. Every
other field is the recipe's.

The run starts with a note naming both values — *"lambda 2.5e+06 set by --lambda; the recipe
calibrated 1e+06. It applies as given, with no stage multiplier on top. …"* (from Python, *"set by
lambda_="*; at stage 2, *"… the recipe calibrated 3e+06 for stage 2 (1e+06 x its stage-2 multiplier
3)"*) — and ends with its held-out verdict, as every run does. What ran is recorded: the stage's
`history.json` entry carries `lambda_applied`, `lambda_mlp_applied` and `mu_applied` (the values the
trainer used, which the run's `config.json` also holds as `lambda_qkv`, `lambda_mlp` and `mu`),
`lambda_override` and `mu_override` (the values given, `null` where the recipe's were used), and
`recipe`, the recipe as written. `lfa evaluate` logs the λ and μ of the run it reads and where each
came from.

Read the result by its held-out curve and on both axes (`lfa evaluate`), as for the recipe's own
λ. One measured case (H. G. Wells, *A Short History of the World*, Qwen3-0.6B at this recipe, one
seed): λ 2,500,000 at 15 epochs scored WikiText-2 15.91 and held-out 27.12, the recipe's
1,000,000 re-run at its best epoch, 8, scored 15.94 and 26.91, and the recipe's own 15 epochs
16.59 and 28.66. There, a stronger anchor and a shorter run reached about the same point.

In one workspace, each value is a run of the same stage (`runs/stage1_run2`, …;
[above](#run-length-and-checkpointing)). Run `lfa evaluate` after each: the numbers land in that
run's entry. The last run trained is the one `fuse` and `extend` take, so finish on the value you
choose, or give each value its own workspace (`lfa init <new> --model <id> --artifact <first
workspace>/artifacts/v1.pt` copies the very file the first workspace anchored on, where
`--artifact self-generated` would look it up in the store again).

**`--lambda 0 --mu 0` is the unanchored control**, at whatever `--epochs` says. At λ 0 the
function anchor has no term to compute, so the artifact is not sampled (a workspace still needs one
at `init`). `lfa evaluate --compare-unanchored` trains the control at the stage's own epochs; when
that control's held-out curve turned earlier, `evaluate` prints the three commands that train it
at its own best epoch with these flags, in a fresh workspace.

To keep a λ — or to change any other field — write it into a recipe of your own (next section), so
that it is what `train` uses without a flag and its calibration record says where it was tuned.

## Writing your own

Copy the bundled file, change what you mean to change, and — the part that is easy to skip — move
`calibrated_rank` and `calibrated_artifact` to the point you actually tuned at (and
`self_generated_frame` to the frame of the self-generated artifact you tuned against, or
`calibrated_artifact` to the artifact's id or path when it was not self-generated), so that the
warnings stay true for the next person:

```python
import dataclasses

from lfa import Recipe

point = Recipe.load("qwen3-0.6b")
tuned = dataclasses.replace(point, name="qwen3-r16", lora_rank=16, lora_alpha=32,
                            lambda_qkv=3e5, lambda_mlp=3e5,
                            calibrated_rank=16)
tuned.save("recipes/qwen3-r16.yaml")
```

A recipe file must be a YAML mapping, may not carry a field `Recipe` does not have, and must carry
`name`, `model_id` and `artifact`. Unknown or missing fields are named at load.

## Per-run overrides

`Workspace.train` takes `epochs`, `lambda_`, `mu`, `full_weight`, `teacher_mode`,
`keep_short_whole`, `supplement`, `domain_description` and `output_name` per call (`--epochs`,
`--lambda`, `--mu`, `--full-weight`, `--teacher-mode`, `--no-supplement` / `--supplement <file>`,
`--domain-description` on the CLI); the recipe is otherwise used as written. `teacher_mode` is
not a recipe field on purpose: it decides where the frozen teacher is read from, not what is
optimized, and the two modes train the same model to the last bit — so it is not part of a tuned
operating point. `auto`, the default, is `adapter_disabled` for a LoRA run (no second model is
loaded) and `separate` for full weight. Whatever actually ran — both λ values after the stage
multiplier (or as `--lambda` set them), μ, the short-document setting, the held-out fraction, the
domain documents trained on and held out
(`n_train_docs`, `n_val_docs`) and, apart from them, the supplement pairs mixed in
(`n_train_supplement_pairs`), the device and the dtype, and under `supplement` the pairs file, how
many pairs were available and used, the target and achieved fractions and the writer's checkpoint
hash — is recorded in that stage's `history.json` entry, so a run says what it did rather than what
it was asked for.

The chunk offset itself is not a setting: it rotates the chunk boundaries, so every token of every
document is trained on in every epoch.
