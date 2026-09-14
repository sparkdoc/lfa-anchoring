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

Rank-32 LoRA over the int8 `gmm1543k` artifact, λ = 100,000 on both site families, μ = 0.05,
16 anchor samples per step, fifteen epochs of cosine.

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

The per-layer weights are **normalized to sum to 1**, which is the convention the published λ is
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

### Calibration record

| field | value |
|---|---|
| `calibrated_rank` | 32 |
| `calibrated_artifact` | `qwen3-0.6b-gmm1543k-int8` |
| `stage2_lambda_multiplier` | 3.0 |

The first two are what `Recipe.warnings(rank, artifact_id)` reads a run against; the third is what
[a chain](multi-domain-chains.md) multiplies λ by from stage 2 on.

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

When you pass a local copy of a published artifact by path, tell `init` which one it is
(`--artifact-id qwen3-0.6b-gmm1543k-int8`), or the recipe will warn that λ was calibrated against
a different artifact when it was calibrated against exactly that one.

**Diagnose against held-out domain perplexity.** Over-anchoring makes general-text perplexity look
its best while domain quality collapses, so the general axis alone cannot tell you λ is too high.

## Writing your own

Copy the bundled file, change what you mean to change, and — the part that is easy to skip — move
`calibrated_rank` and `calibrated_artifact` to the point you actually tuned at, so that the
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

`Workspace.train` takes `epochs`, `full_weight`, `teacher_mode`, `keep_short_whole` and
`output_name` per call (`--epochs`, `--full-weight`, `--teacher-mode` on the CLI); the recipe is
otherwise used as written. `teacher_mode` is not a recipe field on purpose: it decides where the
frozen teacher is read from, not what is optimized, and the two modes train the same model to the
last bit — so it is not part of a tuned operating point. `auto`, the default, is
`adapter_disabled` for a LoRA run (no second model is loaded) and `separate` for full weight. Whatever
actually ran — both λ values after the stage multiplier, the short-document setting, the held-out
fraction, the device and the dtype — is recorded in that stage's `history.json` entry, so a run
says what it did rather than what it was asked for.

The chunk offset itself is not a setting: it rotates the chunk boundaries, so every token of every
document is trained on in every epoch. A switch that reproduced the older, truncating stream
existed briefly and was removed — see [verification.md](verification.md).
