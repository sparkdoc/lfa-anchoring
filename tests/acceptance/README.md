# The equivalence run

One run of the bundled Layerwise Function Anchoring (LFA) recipe, end to end, compared against an
**the research code run of the identical configuration**. `run_recipe.py` is the whole procedure;
`test_acceptance.py` runs that script and asserts the tolerances in `expected.json`.

```bash
CUDA_VISIBLE_DEVICES=0 python tests/acceptance/run_recipe.py --strict --out tests/acceptance/_runs/<date>
# or, as the test:
pytest -m acceptance tests/acceptance/ -s
```

`--strict` is what makes the exit status mean something: without it the script prints its FAIL
rows and still exits 0. `-m acceptance` is the only selector that picks this run up — it is not
marked `gpu`, so `-m gpu` stays the fast GPU tests.

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

### The instrument rows (one draw each, reported and *not* asserted)

5. **Domain perplexity**, `direct_perplexity.overall.mean_perplexity` over the held-out
   chat-formatted Q&A set, from the research code's `scripts/eval_domain_perplexity.py` run in the research code's
   own virtualenv. Held-out raw-prose perplexity is a different quantity and is never comparable
   with it.
6. **WikiText-2 drift**, the full test split at window 2048 / stride 512, each run's perplexity
   read as drift against the base model's — measured in the same run.

These two say the run produced a domain-adapted model on the same instrument, and they carry **no
verdict**: `kind` is `report`, `ok` is `null`, they print as `[----]` with their deviation, and they
cannot fail the run. They cannot be the criterion — the anchor is a Monte-Carlo term and the two
implementations draw it from independent RNG streams (the research code's sampler takes torch's global
generator, the companion's a private one), so two full runs are two *draws* of a stochastic
objective — and, as of 2026-09-07, they carry no band either.

Why no band. They were asserted within 2 % and 1.0 pp, and the 2026-09-07 run consumed 1.95 % of
the 2 %. But **the spread of these quantities across seeds has never been measured on either
side**, so that band was never calibrated: it was a number carried over from an earlier design.
Asserting an unmeasured spread is the same defect the criterion move fixed one level up. The remedy
is not a wider band — it is a **second seed** on one side, which would measure the spread; then
either derive a band from it or leave these rows reported. Until then a large gap here means
**investigate**. A miss on rows 1–4 is the regression.

The tolerances that remain, and that reasoning, are described in one place: the `_comment` in
[`expected.json`](expected.json). They are not to be widened to accommodate a run.

The companion's own metrics are reported beside the research instrument's, un-banded:
`lfa.evaluate.domain_perplexity` on the stage's held-out *documents* and
`lfa.evaluate.wikitext2_perplexity(n_windows=0)`. The WikiText-2 pair should agree closely (it is
the same computation); the two domain numbers should not, because they measure different text.

## Reuse

Both halves of the run are idempotent, and both refuse to reuse work that was not the work being
asked for:

- a workspace that already carries a trained stage is reused only if its recorded corpus, loader
  frame, held-out fraction, recipe digest **and implementation digest** match the request — the
  last of those (`implementation.code_digest`, written by `Workspace.train` from
  `lfa.workspace.code_identity`) is what stops a kept run from certifying code it never ran;
  `--allow-code-change` downgrades that one difference to a printed warning, recorded in
  `results.json`;
- a scoring directory is reused only if the result in it names this same checkpoint and this same
  Q&A file — the two fields the research code's scorer records itself.

Otherwise the run stops and says so (`ReusedRunDiffers`). Delete `--out` to start over.

## Output

- `results.json` — both instruments, the checks with their `kind` (`frame` / `criterion` /
  `report`) and verdicts (a `report` row has none), the provenance of the training itself
  (`companion_commit` and `companion_code_digest` come from the run's own history entry, not from
  the process that scored it), the companion's per-epoch series (`content_curve`, `held_out_curve`,
  `optimizer_steps`, `held_out_tokens`, `val_perplexity_curve`, `learning_rate_curve`), its
  `chunk_counts`, and the provenance of both repositories.
- `reference.json` — where the reference run is, the digest of its config, its two perplexities as
  measured here, and its own `content_curve`, `held_out_curve`, `optimizer_steps`,
  `held_out_tokens` and `chunk_counts`.

Everything under `_runs/` is gitignored **except** those two files for the 2026-09-07 run:
[`_runs/2026-09-07-equiv/results.json`](_runs/2026-09-07-equiv/results.json) and
[`reference.json`](_runs/2026-09-07-equiv/reference.json) are committed, because this run skips on
every machine but the one that has the research checkout, and they are the only source a reader
elsewhere has for the numbers quoted in `README.md`, `docs/concepts.md` and this file. They are
committed **verbatim, as the harness wrote them** — absolute paths and all: they are a record of
one run on one machine, and tidying a record is how a record stops being one. They are pruned from
the sdist (`MANIFEST.in`), so they travel in git only.

Two things to know when reading that particular file, both of which are it being older than the
text around it. Its two instrument rows are recorded under the **old** contract — `"kind":
"sanity"`, a 2 % / 1.0 pp band, `"ok": true` (`results.json:522,531`) — because the run predates
the change to reported rows described above, and it carries none of the provenance fields
(`companion_code_digest`, `scoring_process_commit`) for the same reason. The numbers in it are
unaffected: what changed is what the harness asserts about them, not what it measured. And its
`companion_commit` (`8e7ef3ce…`) is the HEAD of the **scoring** process, not of the training — the
stage was trained at 16:13 and scored at 17:57, and the harness stamped the commit at write time. That is the defect the implementation
digest above now closes; the run's numbers were checked by hand afterwards and stand (the two
training-path commits in that window are inert under `freeze_embed: true`), but the harness did not
establish it and could not have. A record made from this commit on carries `companion_commit` and
`companion_code_digest` from the run's own history entry instead.
