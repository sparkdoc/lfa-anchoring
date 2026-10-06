# Tuning on your own corpus

For a bundled model (`Qwen/Qwen3-0.6B` or `Qwen/Qwen3-1.7B`) and documents of your own. The recipe
— λ = 1,000,000, 15 epochs — was calibrated on one text, the walkthrough's Darwin text (~189 k
training tokens an epoch), with one seed ([recipes.md](recipes.md#how-λ-was-chosen)). On your
text, set the number of epochs first, which is cheap, and touch λ second, only when the runs say
so. A model with no bundled recipe needs its λ calibrated from scratch:
[model-integration-cookbook.md §5](model-integration-cookbook.md#5-calibrate-λ) is that procedure.
This page is for your corpus, not a new model.

**What it costs.** Measured for Qwen3-0.6B on one RTX 3090: a 15-epoch run takes about 18 minutes
on a 757 KB book (H. G. Wells, *A Short History of the World*, below) and about 25 minutes on the
Darwin text, and time scales with the epochs (the same book at 8 epochs took 10 minutes). `lfa
evaluate` takes about 30 seconds once WikiText-2 is cached, and `--compare-unanchored` adds a
second training run. A Qwen3-1.7B run takes longer.

The worked numbers below come from that Wells book: one file, its boilerplate stripped, split by
hand into its 67 chapters, 7 of them held out; Qwen3-0.6B at the bundled recipe; one seed;
perplexity.

## 1. Get a held-out curve

The trainer holds out a tenth of the **documents** and scores them after every epoch. That curve
is what chooses the dose, so the corpus needs at least two documents, and more is better. One file
is one document, so a book or a long report has to be split:

```bash
lfa prepare-domain ~/history/world_history.txt --out data/world_history --split-chars 3500 \
    --supplement --model Qwen/Qwen3-0.6B
```

`--split-chars 3500` cuts the file into documents of about 3,500 characters at paragraph
boundaries. Cut a Project Gutenberg licence, a table of contents or an index out of the file
first: cleaning removes markup, not content, and the supplement writes questions about whatever is
there. A corpus with nothing to hold out is refused before the supplement is written, and the
message names the fix. Cost: minutes (under 4 for the Wells book's supplement).
[preparing-your-data.md](preparing-your-data.md) has the rest.

## 2. Run the recipe once and read its last line

```bash
lfa init  runs/world_history --model Qwen/Qwen3-0.6B --artifact self-generated
lfa train --workspace runs/world_history --corpus data/world_history
```

Every run ends with one line that reads the held-out curve and one that says what it means. On
the Wells book's curve:

```
Held-out perplexity: lowest 27.021 at epoch 8 of 15; final 28.668 (+6.1 % over the lowest).
The held-out curve turned at epoch 8: a re-run with --epochs 8 is likely to ship a better model on this domain than this one. …
```

| what the run says | what to do |
|---|---|
| it turned, and ended 10 % or more above its lowest (a WARNING) | re-run with `--epochs <its lowest epoch>` |
| it turned, 1 % to 10 % above: a re-run is "likely to ship a better model" | re-run with `--epochs <its lowest epoch>` |
| it turned, under 1 % above: the re-run is "optional" | keep the run, or re-run; the 1 % cut is a judgement ([§5](#5-when-to-stop) has the measured spread) |
| it had not turned: "more epochs may lower it further" | re-run with a larger `--epochs` |
| it diverged (the last value is not finite) | do not ship it; re-run at the lowest epoch, or with a stronger anchor, or with a lower learning rate (a recipe of your own: [recipes.md](recipes.md#writing-your-own)) |
| no curve | go back to step 1 |

A re-run is a complete run with the learning-rate schedule laid over the new epoch count, not a
truncation of the first, so it costs that many epochs again. On the Wells book the re-run at
`--epochs 8` was better on both axes: WikiText-2 15.94 against 16.59, held-out domain 26.91 against
28.66 (one seed).

```bash
lfa train --workspace runs/world_history --corpus data/world_history --epochs 8
```

**Same workspace or a fresh one.** In the same workspace, before `lfa extend`, the re-run is a
second run of the stage: it writes `runs/stage1_run2`, the first stays in `runs/stage1`, and from
then on `lfa evaluate`, `lfa fuse` and `lfa extend` read the latest run (`train` says so when it
starts and ends). A fresh workspace (`lfa init runs/world_history_e8 …` reuses the stored artifact)
keeps each run on its own. Either way each run keeps its entry in `history.json`
([recipes.md](recipes.md#run-length-and-checkpointing)).

## 3. Read both axes, and the control at its own dose

```bash
lfa evaluate --workspace runs/world_history
```

```
| metric               | before | after |     Δ% |
| -------------------- | -----: | ----: | -----: |
| general (WikiText-2) |  17.93 | 15.94 | -11.1% |
| domain               |  36.32 | 26.91 | -25.9% |
```

That is the Wells book's 8-epoch run. The domain number should fall and WikiText-2 should hold. A
WikiText-2 number *below* the base model's, as here, is not by itself evidence that anything was
kept ([faq.md](faq.md#wikitext-2-perplexity-came-out-below-the-base-models-is-that-a-win)).

**The control.** `lfa evaluate --compare-unanchored` trains the same stage again with λ = μ = 0 and
adds it as a third column. It trains at the anchored run's epochs, and without the anchor a corpus
turns much sooner, so at that dose much of the gap can be dose: on both corpora where it was
measured (the Wells book below and the walkthrough's Darwin text) most of it was. On the Wells book,
at 15 epochs, the control ended at WikiText-2 334.27 and held-out 435.46, but its own curve was
lowest at epoch 1 (25.59). When the control's curve turned (ending 1 % or more above its lowest),
`evaluate` says so beneath the table and prints three commands that train it at its own best epoch
in a fresh workspace; the training one has the shape

```bash
lfa train --workspace <fresh> --corpus data/world_history --lambda 0 --mu 0 --epochs <its best>
```

Run the printed commands rather than this one: they carry the stage's exact recipe and paths.

Read the control there, and expect it to be hard to beat on the domain. On the Wells book, at its
own best epoch the control's held-out perplexity (25.59, epoch 1 of its 15-epoch run) was below
every anchored run's lowest (26.88, the 8-epoch re-run at epoch 7). What the anchor bought there
can only be on the general axis, and that was not measured: neither the control's WikiText-2 at
epoch 1 nor a 1-epoch control run. Where it was measured, in the walkthrough's companion notebook
(Darwin, a 4-epoch demo), the control at its own dose fit the domain more closely (15.50 against
19.25) and kept less of the general axis (WikiText-2 18.93 against 16.06, base 17.80).

## 4. λ: when to try another

λ is coupled to the corpus as well as to the rank and the artifact
([concepts.md](concepts.md#λ-is-coupled-port-it-and-it-is-a-different-regularizer)), so the
recipe's value is a starting point on your text. One run cannot say λ is wrong; a short ladder can.
Two signs make one worth running. They come from the two Darwin ladders in
[recipes.md](recipes.md#how-λ-was-chosen) (15 epochs, one seed), where the held-out curve turned
earlier the weaker the anchor: on Qwen3-0.6B at epoch 4 for λ = 50,000 and 100,000, 6 for 200,000
and 500,000, 10 for 1,000,000, 12 for 2,500,000, and not at all for 5,000,000.

* **The curve turns within a few epochs and WikiText-2 ends above the base model's.** The anchor
  may be too weak for this corpus. On the ladders the weakest rungs did this (Qwen3-0.6B at
  λ = 50,000 and 100,000), and ended worse than the base model on the domain as well. Try a larger
  λ, or fewer epochs. An early turn alone can also be a small corpus (recipes.md's 45-document
  case turned at epoch 2); WikiText-2 above base is what points at λ.
* **The curve is still falling at the last epoch and the domain moved less than you expected.**
  The anchor may be too strong. On the Qwen3-0.6B ladder λ = 5,000,000 was still falling at epoch
  15 and ended with less of a domain gain than 1,000,000 (held-out 19.57 against 18.43); on
  Qwen3-1.7B, 5,000,000 was worse than 1,000,000 on both axes. Try a smaller λ, or more epochs.

**How.** At the dose step 2 chose (8 epochs on the Wells book), run a rung either side of the
recipe's value, a factor of 2.5 apart, each in a workspace of its own (`init` reuses the stored
artifact, and the supplement prepared beside the corpus is reused):

```bash
for lam in 400000 2500000; do
  lfa init     runs/world_history_lam$lam --model Qwen/Qwen3-0.6B --artifact self-generated
  lfa train    --workspace runs/world_history_lam$lam --corpus data/world_history --epochs 8 --lambda $lam
  lfa evaluate --workspace runs/world_history_lam$lam
done
```

`--lambda X` sets both site families to X exactly, logs the recipe's value beside it, and is
recorded in the stage's `history.json` entry ([recipes.md](recipes.md#trying-another-λ)). Each rung
costs one run at that dose plus its evaluation. Read each rung's own end-of-run line as well: a
stronger anchor turns later, so a rung's curve can say the dose no longer fits it.

**How to pick.** With the recipe's λ at that dose as the middle rung, read the unanchored control at
the same dose (`--compare-unanchored` on that run). If its held-out domain perplexity ends above the
base model's, use the cookbook's rule for a control that over-trains ([§5, step
5](model-integration-cookbook.md#5-calibrate-λ), second paragraph): take the rung best on both axes
(lowest held-out domain and lowest WikiText-2); if no rung is, take the lowest held-out domain among
the rungs within 1 % of the lowest WikiText-2. That is the usual case, because the unanchored curve
turns much sooner than the anchored one: every recorded control at the recipe's 15 epochs ended
above base (Darwin on both models, the Wells book). If the control instead improved on the base
model, use step 5's first rule. Either way, if the pick is an end rung, add one a further factor of
2.5 out and read again, until the pick is in the middle. Compare WikiText-2 at one window count
(`evaluate`'s default, 100, throughout).

**What this evidence is.** One seed, perplexity, and one measured case of the trade between dose and
λ. On the Wells book a stronger anchor at the full dose came close to the recipe's anchor at its
best dose, but not to within noise: λ = 2,500,000 at 15 epochs scored WikiText-2 15.91 and held-out
27.12, λ = 1,000,000 at 8 epochs 15.94 and 26.91. The held-out gap, 0.8 % in the 8-epoch run's
favour, is above the 0.15 % spread of [§5](#5-when-to-stop); the WikiText-2 gap, 0.2 %, is below its
0.5 %. To keep a λ, pass `--lambda` on every `train`, or write it into a recipe of your own
([recipes.md](recipes.md#writing-your-own)) so that it is the default.

## 5. When to stop

At a fixed recipe and seed, a repeat of the same run moves held-out domain perplexity by about
0.15 % and WikiText-2 by about 0.5 % (Qwen3-1.7B, λ = 1,000,000, three runs;
[recipes.md](recipes.md#qwen3-17b-lfarecipesqwen3-17byaml)). A difference smaller than that between
two runs is noise: when a re-run or a new rung moves both numbers by less than that, stop. With one
seed, a larger difference is still one reading, not a result.

Then fuse the run you chose:

```bash
lfa fuse --workspace runs/world_history
```

`fuse` reads the workspace's latest run of the stage, so in a workspace that holds several runs,
make the last one trained the one you mean to ship.
