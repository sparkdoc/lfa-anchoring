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

Four things are checked, in this order:

1. **The two runs are the same configuration.** Before anything is trained, the reference run's
   own `config.json` is compared field by field against the `TrainConfig` the recipe produces
   (`run_recipe.FRAME_FIELDS`: rank, α, both lambdas, μ, the anchor schedule, epochs, learning
   rate, batch geometry, warmup, sequence length, seed, loader frame, held-out fraction). A
   difference fails on its own — two runs of different configurations are not evidence about
   either.
2. **Domain perplexity**, `direct_perplexity.overall.mean_perplexity` over the held-out
   chat-formatted Q&A set, from the research code's `scripts/eval_domain_perplexity.py` run in
   the research code's own virtualenv. Held-out raw-prose perplexity is a different quantity and is never
   comparable with it.
3. **WikiText-2 drift**, the full test split at window 2048 / stride 512, each run's perplexity
   read as drift against the base model's — measured in the same run, by the companion's
   `wikitext2_perplexity(n_windows=0)`.
4. **The content-loss curve** over the first epochs, per epoch.

The tolerances for 2–4, and what each one covers, are described in one place: the `_comment` in
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

- `results.json` — both instruments, the checks and their verdicts, the companion's content /
  validation / learning-rate curves, and the provenance of both repositories.
- `reference.json` — where the reference run is, the digest of its config, its two perplexities as
  measured here, and its per-epoch content curve.
