# The equivalence run

One run of the bundled Layerwise Function Anchoring (LFA) recipe, end to end, compared against an
**the research code run of the identical configuration**. `run_recipe.py` is the whole procedure;
`test_acceptance.py` runs that script and asserts the tolerances in `expected.json`.

```bash
CUDA_VISIBLE_DEVICES=0 python tests/acceptance/run_recipe.py --out tests/acceptance/_runs/<date>
# or, as the test:
pytest -m acceptance tests/acceptance/ -s
```

The GPU test is opt-in (`-m acceptance`) because it is over an hour of GPU time and it reads
gigabytes the companion does not distribute — the corpus, the p(h) artifact, the held-out Q&A set
and the reference run all live in the research checkout. Without them it skips, naming what is
missing. The rest of `test_acceptance.py` is cheap and runs in the default suite: it checks the
comparison logic, not the model. Everything the run writes goes under `--out` (gitignored);
the research code is read-only.

## What is compared, and against what

The reference is a the research code run trained from `scripts/_lfa_companion_reference.sh` into
`outputs/lra/qwen3-0.6b/chalmers/judge_search/gmm_r32_lam100000_e15cos_keepshort/` — the same
corpus, the same int8 artifact, the same seed, fifteen epochs of cosine, short documents kept
whole. It is **re-measured beside the companion**, on the same instrument and the same card, so
what is asserted is that two implementations of one objective land in the same place — not that
either lands on a number recorded elsewhere. The reference's measured values are written out to
`reference.json` by the harness; they are not written down in this repository.

### The frame comes first, and it is fatal

Before anything is trained, the reference run's own `config.json` is compared field by field
against the `TrainConfig` the recipe produces (`run_recipe.FRAME_FIELDS`: rank, α, both lambdas, μ,
the anchor schedule, epochs, learning rate, batch geometry, warmup, sequence length, seed, loader
frame, held-out fraction). A difference **refuses the run** (`FrameMismatch`) rather than warning:
two hours of GPU time spent comparing two different experiments produces numbers about nothing.
`--allow-frame-mismatch` trains anyway and reports the difference as a failing check row, for when
you deliberately want to see how far apart two configurations land.

### The criterion (deterministic)

These four are what say the two implementations compute the same thing. None of them is a
resampled quantity, and all four come out of the two runs' own `training_history.json`.

1. **Optimizer steps per epoch** — `global_step` on both sides, **exact integer equality on every
   epoch**. The step count is a function of the document set, the split, the chunker, that epoch's
   offset, the batch size and the accumulation window, so any difference in those lands here; and
   because the learning-rate schedule is a function of the step, matching steps also mean matching
   learning rates.
2. **Corpus** — training chunks, held-out chunks and held-out tokens, exact. the research code logs its
   chunk counts, so they are read back off `training.log` (a required input); the companion's are
   recorded in its history, or rebuilt deterministically from the frame the history records for a
   run made before that field existed.
3. **Content loss per epoch**, every epoch of the run, each within a relative tolerance.
4. **Held-out loss per epoch**, every epoch — the research code's `eval.loss` against the companion's
   `val_loss`, the same token-weighted cross-entropy over the same text.

### The sanity checks (one draw each, *not* the criterion)

5. **Domain perplexity**, `direct_perplexity.overall.mean_perplexity` over the held-out
   chat-formatted Q&A set, from the research code's `scripts/eval_domain_perplexity.py` run in the research code's
   own virtualenv. Held-out raw-prose perplexity is a different quantity and is never comparable
   with it.
6. **WikiText-2 drift**, the full test split at window 2048 / stride 512, each run's perplexity
   read as drift against the base model's — measured in the same run.

These two are coarse end-to-end checks that the run produced a domain-adapted model at all. They
cannot be the criterion: the anchor is a Monte-Carlo term, and the two implementations draw it from
independent RNG streams (the research code's sampler takes torch's global generator, the companion's a
private one), so two full runs are two *draws* of a stochastic objective and their end perplexities
differ by a seed-scale amount. A miss here means **investigate** — run a second seed on each side —
not **regression**. A miss on rows 1–4 is the regression.

The tolerances, and that reasoning, are described in one place: the `_comment` in
[`expected.json`](expected.json). They are not to be widened to accommodate a run.

The companion's own metrics are reported beside the research instrument's, un-banded:
`lfa.evaluate.domain_perplexity` on the stage's held-out *documents* and
`lfa.evaluate.wikitext2_perplexity(n_windows=0)`. The WikiText-2 pair should agree closely (it is
the same computation); the two domain numbers should not, because they measure different text.

## Reuse

Both halves of the run are idempotent, and both refuse to reuse work that was not the work being
asked for:

- a workspace that already carries a trained stage is reused only if its recorded corpus, loader
  frame, held-out fraction and recipe digest match the request;
- a scoring directory is reused only if the result in it names this same checkpoint and this same
  Q&A file — the two fields the research code's scorer records itself.

Otherwise the run stops and says so (`ReusedRunDiffers`). Delete `--out` to start over.

## Output

- `results.json` — both instruments, the checks with their `kind` (`frame` / `criterion` /
  `sanity`) and verdicts, the companion's per-epoch series (`content_curve`, `held_out_curve`,
  `optimizer_steps`, `held_out_tokens`, `val_perplexity_curve`, `learning_rate_curve`), its
  `chunk_counts`, and the provenance of both repositories.
- `reference.json` — where the reference run is, the digest of its config, its two perplexities as
  measured here, and its own `content_curve`, `held_out_curve`, `optimizer_steps`,
  `held_out_tokens` and `chunk_counts`.
