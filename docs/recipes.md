# Recipes

A recipe is one **tuned operating point**, in one file: what to train, how hard to anchor, and —
this is the part that makes it a recipe rather than a bag of defaults — *what it was tuned at*, so
that a run departing from that point can be told λ no longer means what it meant.

```bash
lfa train --workspace runs/my_domain --recipe qwen3-0.6b       # a bundled name
lfa train --workspace runs/my_domain --recipe my_point.yaml    # or a path
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

Rank-32 LoRA over the model's own self-generated artifact at `self_generated_frame`, λ = 100,000
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
| `lambda_qkv` / `lambda_mlp` | 100000.0 | the function anchor's weight on the attention projections and on the MLPs. Coupled to rank, artifact and corpus — see below |
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

`final_model` is what ships. There is no separately-kept "best" checkpoint: a checkpoint chosen by
the lowest training loss is chosen on one axis of a method whose whole point is the trade between
two.

> ⚠ **`epochs: 15` is a dose, and it was tuned on a corpus of about 1,700 documents (~1.8 M
> training tokens). It is too long for a small one.** A first corpus is usually much smaller, and
> a smaller corpus reaches its held-out minimum much earlier: on a measured 45-document,
> ~106 k-token corpus the held-out perplexity bottomed at **epoch 2** (6.67) and ended the
> fifteenth epoch at **16.88** — and since `final_model` is the last epoch, that run shipped a
> model far worse on its own domain than one it had passed through. The anchor was doing its job
> throughout (the unanchored control was worse still on both axes); the *dose* was wrong.
>
> **The rule.** The trainer prints `Held-out: loss=… perplexity=…` after every epoch, and warns at
> the end if the curve turned around. Read that column: the epoch where the perplexity stops
> falling is your dose, and you re-run at it — `lfa train … --epochs <that epoch>` — rather than
> truncating the run you have, because the learning-rate schedule is laid over the epoch count
> (`--epochs 3` is a complete three-epoch run, not the first three epochs of fifteen). Re-tuning
> the dose does not re-tune λ: they are separate knobs, and λ's couplings are below.
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
and the recipe compares the two field by field. Where the λ came from, with its scope: it was
tuned against an artifact fitted on real text (the 10 : 1 pretraining-to-instruction seed corpus,
1.54 M hidden vectors per site; [the-artifact.md](the-artifact.md#advanced-an-artifact-fitted-on-real-text)),
and an artifact fitted on the model's own text at this frame matched it at every λ tried and was
at least as good at the recipe's λ — one model, one seed, one domain. Any other value of
`calibrated_artifact` is an artifact id or path, compared as a string with the one the workspace
records.

A recipe file written before 0.2.0 may carry `calibrated_self_generated`; it is refused at load as
an unknown field, and deleting the line is the whole fix.

## The couplings, and what the warnings mean

`Workspace.train` calls `Recipe.warnings` before anything is loaded and logs what comes back. None
of them is a refusal — an off-calibration run is allowed, it just is not the measured operating
point:

* **rank ≠ `calibrated_rank`** — λ constrains motion inside the rank-`r` update subspace, so the
  same value binds harder at a lower rank. Lower rank ⇒ lower λ (rank 16 measures at roughly
  2·10⁴–5·10⁴ on a corpus of this kind). Re-tune rather than port.
* **artifact ≠ `calibrated_artifact`** — a different p(h) prices the same function differently.
  Re-tune, and read the *frontier* of (preservation, adaptation) points rather than one point.
* **`full_weight: true`** — outside the paper's validated envelope; calibrate λ in 50,000–100,000
  and check held-out domain perplexity.

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
  ([adding-a-model.md](adding-a-model.md), §3).

A self-generated artifact built by an earlier version records no frame, and is told so, with the
rebuild that records one (`lfa init … --artifact self-generated --rebuild`).

One more is logged by `train` itself rather than by the recipe: *"Training on the raw corpus alone:
this recipe's lambda was calibrated at supplement_fraction 0.13 and this run mixes none."* It fires
on `--no-supplement` (`supplement=False`) when the recipe's `supplement_fraction` is above 0; a
recipe that sets `0.0` has opted out at the recipe level and is not warned.

**Diagnose against held-out domain perplexity.** Over-anchoring makes general-text perplexity look
its best while domain quality collapses, so the general axis alone cannot tell you λ is too high.

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
                            lambda_qkv=3e4, lambda_mlp=3e4,
                            calibrated_rank=16)
tuned.save("recipes/qwen3-r16.yaml")
```

A recipe file must be a YAML mapping, may not carry a field `Recipe` does not have, and must carry
`name`, `model_id` and `artifact`. Unknown or missing fields are named at load.

## Per-run overrides

`Workspace.train` takes `epochs`, `full_weight`, `teacher_mode`, `keep_short_whole`,
`supplement`, `domain_description` and `output_name` per call (`--epochs`, `--full-weight`,
`--teacher-mode`, `--no-supplement` / `--supplement <file>`, `--domain-description` on the CLI);
the recipe is otherwise used as written. `teacher_mode` is not a recipe field on purpose: it
decides where the frozen teacher is read from, not what is optimized, and the two modes train the
same model to the last bit — so it is not part of a tuned operating point. `auto`, the default, is
`adapter_disabled` for a LoRA run (no second model is loaded) and `separate` for full weight. Whatever
actually ran — both λ values after the stage multiplier, the short-document setting, the held-out
fraction, the device and the dtype, and under `supplement` the pairs file, how many pairs were
available and used, the target and achieved fractions and the writer's checkpoint hash — is
recorded in that stage's `history.json` entry, so a run says what it did rather than what it was
asked for.

The chunk offset itself is not a setting: it rotates the chunk boundaries, so every token of every
document is trained on in every epoch. A switch that reproduced the older, truncating stream
existed briefly and was removed — see [verification.md](verification.md).
