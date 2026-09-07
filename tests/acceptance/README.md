# The acceptance run

One run of the shipped Layerwise Function Anchoring (LFA) recipe, end to end, scored on the axes
the paper's operating point was published on. `run_recipe.py` is the whole procedure;
`test_acceptance.py` runs that script and asserts the band in `expected.json`.

```bash
CUDA_VISIBLE_DEVICES=0 python tests/acceptance/run_recipe.py --out tests/acceptance/_runs/<date>
# or, as the test:
pytest -m acceptance tests/acceptance/ -s
```

It is opt-in (`-m acceptance`) because it is roughly an hour and a half of GPU time and it reads
gigabytes the companion does not distribute — the Chalmers corpus, the p(h) artifact and the
held-out Q&A set all live in the research checkout. Without them the test skips, naming what is
missing. Everything the run writes goes under `--out` (gitignored); the research code is read-only.

## Two instruments, and which one decides

The **research instrument decides the band**: the research code's `scripts/eval_domain_perplexity.py`,
invoked in the research code's own virtualenv, exactly as `scripts/judge_search.sh:34` invoked it when the
published numbers were produced.

- **Domain** = `direct_perplexity.overall.mean_perplexity` over
  `data/domain_perplexity_questions/chalmers/domain_perplexity_qa_direct_20260310_123856.jsonl`
  — held-out *chat-formatted* Q&A (77 queries, 8,760 tokens). Held-out raw-prose perplexity is a
  different quantity and is never comparable with it.
- **Seed** = `wikitext2_perplexity`, the full WikiText-2 test split at window 2048 / stride 512
  (299,078 tokens), read as drift against the base model's 18.18.

The **companion's own metrics are reported beside them**, un-banded: `lfa.evaluate.domain_perplexity`
on the stage's 168 held-out *documents* and `lfa.evaluate.wikitext2_perplexity(n_windows=0)`.

## The run of 2026-09-07

| | |
|---|---|
| Date | 2026-09-07, 10:56–12:20 local |
| Companion commit | `0a068afe37bb453ab57e6468a52d869d0b88b31a` **plus the schedule-horizon fix** described below, which is the commit this README lands in |
| the research code revision | `de810c93772e388b98d8b28b3fd64ddf58250d82` |
| GPU | one NVIDIA GeForce RTX 3090 (`CUDA_VISIBLE_DEVICES=0`), driver 580.173.02; the second card stayed idle |
| Torch | 2.10.0+cu128 |
| Wall clock, training | **4,815.7 s = 80.3 min** for 15 epochs / 6,732 optimizer steps (the whole script, scoring included, took 84 min) |
| Corpus | `data/domain/chalmers_qa0.15` — 1,673 documents: 77 raw Chalmers `.txt` plus 1,596 generated `qa_*.txt` (0.1296 of tokens per its manifest). The Q&A supplement is part of the shipped point. |
| Split | `val_fraction=0.1`, seed 42 → **1,505 train / 168 held-out documents** |
| Loader frame | `keep_short_whole=False` — the research loader, under which a document shorter than the epoch's random chunk offset drops out of that epoch. The bundled recipe ships `True`; this is a *frame* field and the published points were measured under `False`. |
| Artifact | `data/distributions/qwen3-0.6b-gmm1543k-int8/distribution_stats.pt` (int8, 113 MB), passed as a local path — the registry digests are placeholders |
| Recipe | `qwen3-0.6b`: r32/α64, λ=100,000, μ=0.05, n=16, `anchor_end_ratio` 0.1 cosine, lr 3e-4, batch 6, ga 1, **15 epochs of a 20-epoch cosine**, checkpoints e5/e10/e15. The run logs `LR schedule: warmup(50) → cosine(12010)` and `Layer weights (cosine, end_ratio=0.1): L0=0.0631, L27=0.0065` — both byte-identical to the research run's own log lines. |

### Measured

| axis | measured | band | verdict |
|---|---|---|---|
| domain, direct Q&A (research instrument) | **8.8969** | 8.5848 – 8.9352 (8.76 ±2 %) | **PASS**, +1.5 % over the seed-42 point |
| WikiText-2 drift vs base 18.18 (research instrument) | **16.6134 = −8.617 %** | −11.5 % – −8.5 % (−10.0 ±1.5 pp) | **PASS** |
| WikiText-2, companion instrument, `n_windows=0` | 16.613441 | — | agrees with the research instrument to 7 significant figures (16.613441) |
| base WikiText-2, companion instrument | 18.1772 | — | the record's 18.18 |
| domain on the 168 held-out documents, companion instrument | 13.411 (base 23.299, −42.4 %) | — | a different quantity from the Q&A number; not comparable with it |

The published reference points, for scale: fp16 seed 42 **8.7625 / −10.02 %**, fp16 seed 1337
8.8293 / −10.68 %, and — the like-for-like artifact — int8 seed 42 **8.7228 / −8.90 %**. This run
lands inside that spread on both axes, and within 0.28 pp of the int8 arm on preservation.

### What the band's tolerances cover

- the two-seed spread of the published point: 8.7625 (seed 42) / 8.8293 (seed 1337);
- the research code's μ fast path — mathematically exact but not bit-reproducing, "expect ≈8.77";
- the **int8** artifact. The paper's e15 run anchored on the fp16 `gmm1543k`; the research code's own int8
  arm lands at 8.723 / −8.90 %, so int8 ties on domain fit and costs ~1.1 pp of seed preservation.
  The seed band is centred on the fp16 point while this run uses int8, so a passing run is expected
  to sit toward the shallow edge of it, as this one does.

## The first attempt, and what it found

The first run of this script (2026-09-07, 08:59–10:23, 79.9 min, same inputs) measured domain
**9.6858** — outside the band — with seed −11.467 %. The band was not widened; the miss was a real
defect in this package, and finding it is what the acceptance run is for.

**The recipe's `schedule_horizon_epochs` was 100. It should be 20.** The value had been read off
`outputs/lra/qwen3-0.6b/chalmers/judge_search/gmm_r32_lam100000_f0.13/config.json`, which says
`num_epochs: 100` — but that file was rewritten by a resume on 2026-07-20
(`training.log:1307-1312`: *"Resuming from epoch 20, global_step 8757 … Resuming training from
epoch 21 to 100"*), long after the e15 checkpoint was written on 2026-07-18. The original run's own
first log line (`training.log:45`, 2026-07-12) reads `LR schedule: warmup(50) → cosine(12010)` — a
twenty-epoch cosine — and the research code's never-resumed int8 arm, which reproduces that run's per-epoch
content loss to four digits, carries `num_epochs: 20` and the same `cosine(12010)`.

At a 100-epoch horizon the learning rate at e15 is 2.909e-4 where the correct schedule puts it at
1.236e-4 — 2.35× too high — so the run stayed in its exploratory phase instead of converging:

| epoch | step | content loss: corrected run / first attempt / research int8 arm |
|---|---|---|
| 5 | 2,496 | 2.5194 / 2.5251 / 2.5283 |
| 9 | 4,153 | 2.3426 / 2.3750 / 2.3417 |
| 12 | 5,224 | 2.2678 / 2.3241 / 2.2665 |
| 15 | 6,732 | **2.2785 / 2.3818 / 2.2769** |

The first attempt tracked the research arm while the two schedules still agreed (their LR ratio is
0.91 at e5) and separated as the gap opened, ending 0.105 nats high; `exp(0.105) × 8.723 = 9.69`,
which is the domain miss exactly. The corrected run ends 0.0016 nats from the research arm.

Everything else in the port was exonerated in the process, and the checks are worth keeping:

- **Corpus and split, byte for byte.** Both loaders return the same 1,673 documents in the same
  pre-shuffle order, the same shuffle under seed 42, and the same 1,505/168 split.
- **Per-epoch chunking, chunk for chunk**, for all 15 epochs — including the Q&A supplement's
  presence schedule under `keep_short_whole=False` (epochs 0, 1, 4, 7, 13, 14). Epoch 0 holds out
  435 validation chunks totalling **157,366 tokens**, the figure the research code's own run recorded. Every
  `global_step` matches the research run exactly, e1 through e15, in both attempts.
- **The λ/μ wiring.** In both repos the logged `loss_anchor` is the λ-weighted total and
  `loss_total = content + loss_anchor + loss_mu`; λ is applied once, `lm_head` and `embed` anchoring
  are off under `freeze_embed`, and `gradient_accumulation_steps=1` divides nothing.
- **The anchor instrument.** the research code's `scripts/lra_compute_anchor_loss.py` and the companion's
  `lfa.losses.anchor_loss`, on the *same* checkpoints with the fp16 artifact, agree: research e15
  qkv 6.2e-7 / mlp 1.30e-6 against 6.28e-7 / 1.35e-6.
- **The sampler.** The two samplers drawing 200,000 vectors from the same int8 artifact agree on
  `E‖h‖²` to within 0.2 % at every site tested, shallow to deep.

The first attempt's artifacts are kept at `_runs/2026-09-07/` (gitignored), including e5/e10/e15
scores and a dose sweep to e18 that ruled dose out as the axis before the schedule was found.
