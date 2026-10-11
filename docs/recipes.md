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

Rank-32 LoRA over the model's own self-generated artifact at `self_generated_frame`, λ = 160,000
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
| `lambda_qkv` / `lambda_mlp` | 160000.0 | the function anchor's weight on the attention projections and on the MLPs. Coupled to rank, artifact and corpus — see below |
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
> turned early: its held-out perplexity was lowest at epoch 4 (16.97) and ended the fifteenth
> epoch at 30.54, 80.0 % above that, worse than the base model on both axes
> ([below](#how-λ-was-chosen)); on Qwen3-1.7B, at the same λ, lowest at epoch 5 (12.40) and
> ended at 18.29, 47.4 % above. The trainer warns at the end of both runs and advises a re-run at
> that epoch, and the re-runs at `--epochs 4` and `--epochs 5` scored below the base model on both
> axes. That is the recipe's design: 15 epochs is the run that finds the dose, and the re-run at
> its turn is the run to keep. A stronger anchor reaches its held-out minimum later (at λ =
> 1,000,000 the same text turned at epoch 10 on Qwen3-0.6B and 13 on Qwen3-1.7B), and a smaller
> corpus earlier: on a measured 45-document, ~106 k-token corpus at λ = 100,000 (measured under
> 0.1.x, with no supplement and a schedule sized from the first epoch) the held-out
> perplexity bottomed at **epoch 2** (6.67) and ended the fifteenth epoch at **16.88** — and
> since `final_model` is the last epoch, that run shipped a model far worse on its own domain
> than one it had passed through. The anchor was doing its job throughout (the unanchored
> control was worse still on both axes); the *dose* was wrong.
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
>     Qwen3-0.6B at λ = 1,000,000 on an earlier frame — the one-pass supplement, its pairs
>     unshuffled, a schedule sized from the first epoch; one seed) the held-out perplexity was
>     lowest at epoch 8 (27.02)
>     and ended 6.1 % above it (28.67), and the re-run at `--epochs 8` was better on both axes:
>     WikiText-2 15.94 against 16.59, held-out domain 26.91 against 28.66.
>   * **Under 1 %:** INFO, saying the re-run is optional. The 1 % cut is a judgement, not a
>     measurement. For scale, the run-to-run spread at a fixed recipe and seed (Qwen3-1.7B,
>     λ = 1,000,000 on the earlier frame, three runs;
>     [the cookbook's §9](model-integration-cookbook.md#9-worked-example-qwen3-17b)) is about
>     0.15 % on held-out domain perplexity and about 0.5 % on WikiText-2; three Qwen3-0.6B runs at
>     this recipe's λ, 4 epochs, were about 0.2 % apart on each axis ([below](#how-λ-was-chosen)).
> * **It had not turned** (the lowest epoch is the last, which a one-epoch run always is): more
>   epochs may lower it further, and the run says the dose can be raised with `--epochs`. The
>   exception is a re-run at an earlier run's turn, which usually ends at its own lowest: when the
>   workspace holds that earlier run (same stage, same frame but the epochs), the run names it,
>   says to stop there, reads the two held-out finals, and asks for the two runs' `evaluate`
>   tables: keep the run not worse on either axis beyond the run-to-run spread
>   ([tuning.md](tuning.md#when-the-re-run-does-not-turn)). An earlier run that diverged after its
>   lowest epoch is matched the same way, and the re-run is kept.
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
> an earlier run's turn names that run, its turn epoch and its final (`null` for a run that
> diverged) under `rerun_of`. Re-tuning the dose does not re-tune λ: they are separate knobs, and
> λ's couplings are below.
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
> `evaluate` and `fuse` read an earlier run of the stage when `--run` names it (`--run stage1`);
> `extend` and `regenerate-artifact` never do.
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
curve, not a truncation of this one. Its length is the run's real optimizer-step count, summed
epoch by epoch: each epoch is cut at its own chunk offset, and a later epoch can hold more chunks
than the first, so the cosine reaches its floor on the run's last step and never rises after it.
The run's log states that count (`LR schedule: N optimizer steps: warmup(50) → cosine(N − 50)`).
Earlier versions — and the research code — sized it from the first epoch's count alone: a
four-epoch Darwin run (Qwen3-0.6B) took 566 steps against 516 planned, and past its end
PyTorch's cosine climbs back up (a fifteen-epoch run ended at about 5 % of the peak, rising).

### Data

| field | value | what it does |
|---|---|---|
| `keep_short_whole` | `true` | a document that fits in one chunk is trained **whole**, in every epoch, rather than being cut at the epoch's chunk offset into a chunk and a fragment that starts mid-sentence. It changes the training stream, so a run records which setting it used. Prefer `false` only when the documents are themselves arbitrary slices of something longer, so that keeping them whole preserves nothing and the extra positional variety is worth having |
| `val_fraction` | 0.1 | share of *documents* (shuffled under `seed`) held out of training and scored after every epoch. Set it to `0.0` to train on everything — and then read the domain number as a fit |
| `supplement_fraction` | 0.13 | share of training *tokens* made up of the question-and-answer pairs the stage's entry model writes from the training-side documents ([quickstart.md](quickstart.md)); `0.0` trains on the raw corpus alone. The shipped λ was tuned with this release's supplement at 0.13 of training tokens, its pairs shuffled; every companion run before 0.2.0 trained at 0, off that frame. The held-out documents are split off before anything is mixed, so the domain number stays a raw-text measurement comparable across fractions. The pairs are shuffled under the run's seed and taken as a prefix of that order, the one whose achieved share is closest to the target (as the research mixer does; builds before this release took the file's own prefix, its first passages); a pool too small to reach it trains at what it has and warns with the achieved share |

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
two-domain walkthrough](../examples/two_domain_walkthrough.ipynb) (*On the Origin of Species*), 173
training documents and about 189 k training tokens an epoch, with the model's own
question-and-answer supplement (this release's two-pass writer) at 0.13 of training tokens, its
pairs shuffled under seed 42; the cosine laid over the run's real optimizer-step count; rank 32;
batch 6; one seed; perplexity; RTX 3090; 2026-10-08. That is the frame this release trains with.

Each λ was kept at its own dose, as [tuning.md §4](tuning.md#4-λ-when-to-try-another) reads a rung:
a 15-epoch run, then, when its curve turned and ended 1 % or more above its lowest, a re-run at
that epoch. Every run was scored by `lfa evaluate` at its defaults: WikiText-2 over 100 windows
(51,200 tokens) and the 26,147 tokens of held-out Darwin. Base: WikiText-2 17.93, held-out Darwin
30.12. The last column is the 15-epoch run's per-epoch validation perplexity (the training split's
own 10 %): the epoch it was lowest at, and how far the fifteenth ended above it.

| λ (μ = 0.05 when λ > 0) | epochs kept | WikiText-2 (100 windows) | held-out Darwin | 15-epoch run's validation curve |
|---|---|---|---|---|
| 0 (μ = 0, unanchored) | 4 | 25.63 (+42.9 %) | 18.39 (−38.9 %) | — |
| 0 (μ = 0, unanchored) | 15 | 302.49 (+1587 %) | 194.07 (+544 %) | lowest at epoch 2, ended +1175 % |
| 64,000 | 4 | 16.40 (−8.6 %) | 16.81 (−44.2 %) | lowest at epoch 4, ended +184 % |
| **160,000** | 4 | **16.15 (−9.9 %)** | **17.29 (−42.6 %)** | lowest at epoch 4, ended +80.0 % |
| 400,000 | 7 | 16.28 (−9.2 %) | 17.94 (−40.5 %) | lowest at epoch 7, ended +25.6 % |
| 1,000,000 | 10 | 16.40 (−8.5 %) | 18.39 (−39.0 %) | lowest at epoch 10, ended +2.8 % |
| 2,500,000 | 15 | 16.49 (−8.0 %) | 18.87 (−37.3 %) | still falling at epoch 15 |

Percentages are against the base model. The unanchored control at the chosen rung's dose (4 epochs)
improves the domain, so the selection is [the cookbook's
§5](model-integration-cookbook.md#5-calibrate-λ) first rule: among the rungs whose held-out Darwin
gain is at least 90 % of the control's (held-out Darwin at most 19.57: every rung), the lowest
WikiText-2, ties within 1 % going to the smaller λ. That is **160,000** (16.15): 400,000's 16.28 is
within 1 % of it and the larger λ, and 64,000's 16.40 is 1.5 % above. A rung lies on either side, so
the pick is interior. A behaviour reading ties 160,000 with 400,000: the shift in the base model's
own answers to 40 general prompts (teacher-forced NLL of its greedy answers, against the base
model's, in nats; one greedy decode; [tuning.md §4](tuning.md#4-λ-when-to-try-another) describes it)
was +0.121 at 160,000, +0.120 at 400,000, +0.154 at 64,000, +0.140 at 1,000,000 and +0.126 at
2,500,000, against +0.565 for the 4-epoch control.

**The dose at the chosen λ.** The 15-epoch run's validation perplexity fell from 18.78 after epoch
1 to its lowest, 16.97, at epoch 4, and ended at 30.54; that run scored WikiText-2 18.76 and
held-out Darwin 30.93, worse than the base model on both axes, and its behaviour shift was +0.342.
The re-run at `--epochs 4` ended at its own lowest (16.96) and is the row above. The recipe keeps
`epochs: 15`: the run's last line names the turn, and the re-run there is the model to ship
([tuning.md](tuning.md#2-run-the-recipe-once-and-read-its-last-line)). One seed, one corpus: on
your own corpus, read the run's last line as the dose note above says.

**The stage-2 multiplier.** A two-stage chain — Darwin at the chosen dose (4 epochs), the artifact
extended, then the walkthrough's cookery text (*Domestic Cookery*) for the recipe's 15 epochs — at
multipliers 1 and 3. Cookery is scored on its 15 held-out documents; in every column the first
number is the fused stage-1 model and the second the model after stage 2. One seed, one pair of
domains, perplexity, RTX 3090, 2026-10-08. The peak is nvidia-smi `memory.used` sampled every 10 s
over the whole chain, an upper bound (see the memory note below).

| stage-1 λ | multiplier (stage-2 λ) | cookery held-out | Darwin held-out over stage 2 | WikiText-2 over stage 2 (100 windows) | chain peak MiB |
|---|---|---|---|---|---|
| 160,000 | 1× (160,000) | 23.16 → 30.16 (+30.2 %) | 17.30 → 33.70 (+94.8 %) | 16.13 → 18.94 (+17.5 %) | 12,796 |
| 160,000 | **3× (480,000)** | 23.09 → **19.73 (−14.6 %)** | 17.33 → **23.54 (+35.8 %)** | 16.12 → **17.33 (+7.5 %)** | 12,773 |

3× was ahead on all three, which is why the recipe keeps `stage2_lambda_multiplier: 3.0`. Both
arms over-trained the ~104 k-token cookery text at 15 epochs (1× lowest at epoch 5 and ended
102 % above it, 3× lowest at epoch 8 and ended 33 % above). Re-run at their own turns, the two came
out close:

| multiplier | stage-2 epochs | cookery held-out | Darwin held-out over stage 2 | WikiText-2 over stage 2 |
|---|---|---|---|---|
| 1× | 5 | 23.16 → 14.18 (−38.8 %) | 17.30 → 19.79 (+14.4 %) | 16.13 → 16.46 (+2.0 %) |
| 3× | 8 | 23.09 → 15.11 (−34.6 %) | 17.33 → 19.70 (+13.7 %) | 16.12 → 16.44 (+2.0 %) |

By the cookbook's multiplier rule (keep more Darwin while cookery's improvement stays within 90 % of
the better arm's) 1× edges it there: 3×'s cookery gain is 88.9 % of 1×'s, and its Darwin advantage
is 0.5 %. That reading is marginal, one seed, and 3×'s re-run itself turned at epoch 6. Stage 1 of
the two chains replicates the ladder's 160,000 rung: WikiText-2 16.13 and 16.12 against 16.15,
held-out Darwin 17.30 and 17.32 against 17.29, about 0.2 % apart on each axis.

**An earlier calibration** (2026-10-05) read this model on an earlier frame — the one-pass
supplement writer (short third-person answers), its pairs taken unshuffled, and a schedule sized
from the first epoch's chunks — with every rung at a common 15 epochs, and by the cookbook's §5
over-training branch picked λ = 1,000,000. Its numbers are not read against the ladder above. Read
the same common-15-epoch way, this frame's own 15-epoch runs put no rung best on both axes:
2,500,000 has the lowest WikiText-2 (16.49) and 1,000,000 the lowest held-out Darwin (18.81), 1.02 %
higher on WikiText-2, just outside the 1 % band, so the fallback lands on 2,500,000, an end rung,
and the ladder was not extended past it. [The cookbook's
§9](model-integration-cookbook.md#9-worked-example-qwen3-17b) keeps the common-dose procedure on
Qwen3-1.7B as its worked example.

**The paper's operating point** is a different one: λ = 100,000 at rank 32, calibrated in the
research code on a corpus about ten times larger (about 1,700 documents, ~1.8 M training
tokens), first against an artifact fitted on real text (the 10 : 1 pretraining-to-instruction
seed corpus, 1.54 M hidden vectors per site;
[the-artifact.md](the-artifact.md#advanced-an-artifact-fitted-on-real-text)), and judged there
([concepts.md](concepts.md#what-the-numbers-are)). It is the paper's point, not this recipe's.

## Qwen3-1.7B: `lfa/recipes/qwen3-1.7b.yaml`

The same operating point for `Qwen/Qwen3-1.7B`: every field above is the Qwen3-0.6B recipe's
except `name` and `model_id`. Its λ was read on this model and came out at the same
`lambda_qkv` = `lambda_mlp` = **160,000**, by a different reading from the Qwen3-0.6B recipe's
(below). `lfa init --model Qwen/Qwen3-1.7B` adopts it.

### Calibration record

| field | value |
|---|---|
| `calibrated_rank` | 32 (`lora_rank` 32, `lora_alpha` 64) |
| `lambda_qkv` / `lambda_mlp` | 160000.0 |
| `stage2_lambda_multiplier` | 3.0 |
| `calibrated_artifact` | `self-generated` — Qwen3-1.7B's own artifact |
| `self_generated_frame` | `{n_raw: 2500, n_chat: 0, max_new_tokens: 2048, max_samples: 600000, gmm_k: 32, pca_variance: 0.95}` |
| `epochs` | 15 |
| `batch_size` × `gradient_accumulation_steps` | 6 × 1, at `sequence_length` 512 |

**How λ was chosen.** Scope: Qwen3-1.7B; its self-generated artifact at the frame above; the
Darwin text of [the two-domain walkthrough](../examples/two_domain_walkthrough.ipynb) with this
model's own supplement, the same frame as the Qwen3-0.6B ladder's (the two-pass writer, pairs
shuffled, the cosine over the real step count); rank 32; one seed; perplexity; RTX 3090;
2026-10-08/09. Each λ was kept at its own dose, as for Qwen3-0.6B, and scored by `lfa evaluate` at
its defaults: WikiText-2 over 100 windows (51,200 tokens) and 27 held-out Darwin documents (26,147
tokens). Base: WikiText-2 15.09, held-out Darwin 21.45. The last column is the behaviour shift
described for Qwen3-0.6B above (the base model's own greedy answers to 40 general prompts,
teacher-forced; nats against the base model; one greedy decode).

| λ (μ = 0.05 when λ > 0) | epochs kept | WikiText-2 (100 windows) | held-out Darwin | 15-epoch run's validation curve | behaviour shift |
|---|---|---|---|---|---|
| 0 (μ = 0, unanchored) | 3 | 12.62 (−16.4 %) | 12.21 (−43.1 %) | — | not probed |
| 0 (μ = 0, unanchored) | 4 | 14.68 (−2.7 %) | 14.33 (−33.2 %) | — | +0.346 |
| 0 (μ = 0, unanchored) | 5 | 18.41 (+22.0 %) | 19.02 (−11.3 %) | — | not probed |
| 0 (μ = 0, unanchored) | 15 | 142.30 (+843 %) | 163.16 (+661 %) | lowest at epoch 2 | not probed |
| 10,240 | 3 | 11.60 (−23.1 %) | 12.25 (−42.9 %) | lowest at epoch 3 | not probed |
| 25,600 | 4 | 11.81 (−21.7 %) | 12.65 (−41.0 %) | lowest at epoch 4 | +0.117 |
| 64,000 | 5 | 12.10 (−19.8 %) | 12.98 (−39.5 %) | lowest at epoch 5 | +0.118 |
| **160,000** | 5 | **12.19 (−19.2 %)** | **13.15 (−38.7 %)** | lowest at epoch 5, ended +47.4 % | **+0.089** |
| 400,000 | 9 | 13.09 (−13.3 %) | 13.51 (−37.0 %) | lowest at epoch 9 | +0.108 |
| 1,000,000 | 15 | 13.43 (−11.0 %) | 13.76 (−35.9 %) | lowest at epoch 13, ended +0.7 % | +0.134 |
| 2,500,000 | 15 | 13.52 (−10.4 %) | 13.87 (−35.3 %) | still falling at epoch 15 | +0.131 |

**This λ departs from the selection rule, and says so.** Each control (3, 4 and 5 epochs) improves
the domain, so the rule is the cookbook's §5 first rule: the lowest WikiText-2 among the rungs whose
held-out Darwin gain is at least 90 % of the control's. It picked the lowest rung at every extension
of the ladder — 64,000, then 25,600, then 10,240 — because WikiText-2 kept falling as the anchor
weakened. Where it would stop is unmeasured: at 3 epochs the unanchored control is behind 10,240 on
WikiText-2 (12.62 against 11.60), yet itself 16 % below base on it and lower than every rung on
held-out Darwin. On a chat-tuned model a WikiText-2 number below base is not a sign that nothing was
forgotten ([tuning.md §4](tuning.md#4-λ-when-to-try-another)), so on this model it cannot choose λ.
The behaviour shift does separate anchor from no anchor: +0.346 for the 4-epoch control against
+0.089 to +0.134 for every rung probed (10,240 was not), lowest at **160,000**. That is the pick.
Against 64,000 the comparison is dose-matched (both 5 epochs); the kept doses rise with λ above it
(9 and 15 epochs), so against 400,000, 1,000,000 and 2,500,000 the shift is not separated from dose.
Against the rule's domain bar 160,000 passes at the 4- and 5-epoch controls and misses the 3-epoch
control's by 0.02 (13.155 against at most 13.132), and its WikiText-2 is within 1 % of 64,000's. The
shift is not monotone in λ, and the minimum at 160,000 is one probe set, one greedy decode and one
seed; on the Qwen3-0.6B ladder 160,000 and 400,000 tie (+0.121 and +0.120). The frame alone does not
explain the move from the earlier calibration's 1,000,000: at a common 15 epochs this frame's rungs
also put 1,000,000 best on both axes (WikiText-2 13.43, held-out Darwin 13.76, against 13.52 and
13.87 at 2,500,000), so the move is the own-dose reading and this departure.

**The dose at the chosen λ.** The 15-epoch run's validation perplexity — the training split's own
10 % — fell from 13.31 after epoch 1 to its lowest, 12.40, at epoch 5, and ended at 18.29, 47.4 %
above it; that run scored WikiText-2 14.54 and held-out Darwin 19.51. The re-run at `--epochs 5`
is the row above (its own curve lowest at epoch 4, 0.2 % below its end: a re-run the trainer calls
optional). One seed, one corpus: on your own corpus, read the run's last line as the dose note
above says.

**The stage-2 multiplier**, from a two-stage chain at 160,000 (Darwin at the kept 5 epochs, then the
walkthrough's cookery text), one seed, perplexity:

| reading | multiplier | cookery (before → after) | held-out Darwin after stage 2 | WikiText-2 after stage 2 |
|---|---|---|---|---|
| recipe dose, 15 epochs | 1× | 15.88 → 18.47 (+16.3 %) | 20.81 | 13.20 |
| recipe dose, 15 epochs | **3×** | 15.96 → 13.48 (−15.6 %) | **16.42** | **12.75** |
| own turn (1× at 4, 3× at 8 epochs) | **1×** | 15.88 → 11.31 (−28.8 %) | **14.22** | **11.80** |
| own turn (1× at 4, 3× at 8 epochs) | 3× | 15.96 → 11.53 (−27.7 %) | 14.57 | 12.10 |

At the recipe dose 3× wins on all three axes (1× over-trains the small text and ends cookery above
its start). Re-run at their own turns, 1× edges it by the cookbook's rule: 3×'s cookery gain is
96.9 % of 1×'s, so both qualify, and 1× keeps more Darwin and is ahead on all three axes (one seed;
3×'s 8-epoch re-run turned at epoch 6 and ended 1.3 % above its lowest, so 3× is not at its own
lowest epoch). Qwen3-0.6B gives the same verdict by a different margin: there 3× kept marginally more
Darwin but its cookery gain missed the 90 % bar (88.9 %). The recipe keeps 3.0: it is the reading
at the recipe dose, and it protects a chain whose second stage runs long. "Before" is each chain's
own fused stage-1 model. On the earlier frame (the one-pass supplement, its pairs
unshuffled, a schedule sized from the first epoch; 2026-10-04/05), at λ = 1,000,000 with 15 epochs
in each stage, this model's 3× matched 1× on cookery (−27.0 % against −26.4 %) and beat it on
Darwin retention and on WikiText-2; the table is in [the cookbook's
§9](model-integration-cookbook.md#9-worked-example-qwen3-17b), with the run-to-run spread measured
on that frame (three stage-1 runs at λ = 1,000,000: about 0.5 % on WikiText-2 and 0.15 % on
held-out Darwin).

**Training memory**, one RTX 3090, batch 6 × 512: an anchored 15-epoch run trained at about
12.7 GiB (13,033 MiB at λ = 160,000; maximum 13,033–13,056 MiB across every rung of the earlier
frame's ladder, 20,000 to 5,000,000); the unanchored control at a maximum of 11,236 MiB (earlier
frame); the second stage of a chain, training on the fused stage-1 model, 16,206–19,291 MiB (the
earlier frame's chains at λ = 1,000,000 and 100,000; 19,291 MiB, at 1,000,000 × 3, is about
18.8 GiB, more than a 16 GB card holds). A chain at this recipe's λ peaked at 15,478 MiB (1×) and
15,455 MiB (3×), sampled over the whole chain (the peak fell in stage 2's training).
The measure is nvidia-smi `memory.used` sampled every 10 s, which includes PyTorch's caching
allocator, so it is an upper bound on what a run needs. It is not comparable with Qwen3-0.6B's
"8.63 GiB allocated" in [faq.md](faq.md), which is the allocator's own peak, a different measure.

**λ does not port** across artifacts, domains or protocols. A separate research run on another
domain chose 50,000 for this model, against an artifact fitted on real text and at a different
protocol; that is a different point, not a check on this one. The two bundled recipes were read on
the same Darwin text and landed on the same λ by different readings (Qwen3-0.6B by the rule, this
model by behaviour drift): two models, one corpus, one seed, which is not evidence that λ is
independent of model size. On the earlier frame, at a common 15 epochs, the cookbook's §5 picked
1,000,000 for this model ([§9](model-integration-cookbook.md#9-worked-example-qwen3-17b)).

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
multiplier on top (at stage 2, `--lambda 480000` is what the shipped recipes would have used).
`--mu Y` does the same for μ. From Python: `ws.train(corpus, lambda_=2.5e6, mu=0.05)`. Every
other field is the recipe's.

The run starts with a note naming both values — *"lambda 2.5e+06 set by --lambda; the recipe
calibrated 160000. It applies as given, with no stage multiplier on top. …"* (from Python, *"set by
lambda_="*; at stage 2, *"… the recipe calibrated 480000 for stage 2 (160000 x its stage-2
multiplier 3)"*) — and ends with its held-out verdict, as every run does. What ran is recorded: the
stage's `history.json` entry carries `lambda_applied`, `lambda_mlp_applied` and `mu_applied` (the
values the trainer used, which the run's `config.json` also holds as `lambda_qkv`, `lambda_mlp` and
`mu`), `lambda_override` and `mu_override` (the values given, `null` where the recipe's were used),
and `recipe`, the recipe as written. `lfa evaluate` logs the λ and μ of the run it reads and where
each came from.

Read the result by its held-out curve and on both axes (`lfa evaluate`), as for the recipe's own
λ. One measured case (H. G. Wells, *A Short History of the World*, Qwen3-0.6B, one seed, on the
earlier frame of [the dose note](#run-length-and-checkpointing), where the recipe's λ was
1,000,000): λ 2,500,000 at 15 epochs scored WikiText-2 15.91 and held-out 27.12, λ 1,000,000
re-run at its best epoch, 8, scored 15.94 and 26.91, and λ 1,000,000 at 15 epochs 16.59 and
28.66. There, a stronger anchor and a shorter run reached about the same point.

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
