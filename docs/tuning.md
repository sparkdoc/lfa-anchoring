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

`--split-chars 3500` cuts the file into documents of about 3,500 characters at paragraph boundaries.
Cut a Project Gutenberg licence, a table of contents or an index out of the file first: cleaning
removes markup, not content, and the supplement writes questions about whatever is there
([preparing-your-data.md](preparing-your-data.md#a-project-gutenberg-book) lists what a Gutenberg
book carries and how to cut it). A corpus with nothing to hold out is refused before the supplement
is written, and the message names the fix. Cost: minutes (under 4 for the Wells book's supplement).
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
| it had not turned, and it is the re-run at an earlier run's turn: "stop here" | stop; compare the two runs' finals and keep the lower ([below](#when-the-re-run-does-not-turn)) |
| it diverged (the last value is not finite) | do not ship it; re-run at the lowest epoch, or with a stronger anchor, or with a lower learning rate (a recipe of your own: [recipes.md](recipes.md#writing-your-own)) |
| no curve | go back to step 1 |

```bash
lfa train --workspace runs/world_history --corpus data/world_history --epochs 8
```

**Same workspace or a fresh one.** In the same workspace, before `lfa extend`, the re-run is a
second run of the stage: it writes `runs/stage1_run2`, the first stays in `runs/stage1`, and from
then on `lfa evaluate`, `lfa fuse` and `lfa extend` read the latest run (`train` says so when it
starts and ends). A fresh workspace keeps each run on its own: `lfa init runs/world_history_e8
--model Qwen/Qwen3-0.6B --artifact runs/world_history/artifacts/v1.pt`, naming the first
workspace's artifact file, which is the very file its run anchored on (`--artifact
self-generated` would look the artifact up in the store again). Either way each run keeps its
entry in `history.json` ([recipes.md](recipes.md#run-length-and-checkpointing)).

**Why a re-run, not a best checkpoint.** `final_model` is the last epoch, and no checkpoint is
picked off the curve, for three reasons. The learning-rate schedule anneals over the whole run, so
epoch 8 of a 15-epoch run is a model in mid-schedule, and a re-run at `--epochs 8` is a different
model: a complete run whose schedule ends at epoch 8. A checkpoint picked on held-out perplexity is
picked on one axis, the domain, of a method whose point is the trade between two. And the held-out
split is the domain number `evaluate` reports, so a model picked on it would bias that number
down. A re-run costs that many epochs again. Measured, one seed each: on the Wells book the re-run
at `--epochs 8` was better on both axes than the 15-epoch run (WikiText-2 15.94 against 16.59,
held-out domain 26.91 against 28.66) and ended below that run's own epoch-8 value (26.91 against
27.02). On *A Short History of Astronomy* (Arthur Berry, 1898; Qwen3-0.6B at the recipe, 210
documents, ~198 k training tokens an epoch) the 15-epoch run was lowest at epoch 10 (13.773) and
ended at 14.021; the re-run at `--epochs 10` ended at 13.839, 0.5 % above that epoch-10 value, and
still beat the 15-epoch run on both axes (held-out 13.839 against 14.021, WikiText-2 15.97 against
16.24). On the same book at λ = 400,000 the same thing happened: lowest at epoch 6 of 10 (13.435),
the re-run at `--epochs 6` ended at 13.526, 0.7 % above it, and beat the 10-epoch run's final
(13.906) and its WikiText-2 (16.06 against 16.57).

#### When the re-run does not turn

A re-run at the turn usually ends at its own lowest, so its line says the curve had not turned.
That is what such a re-run shows, not a call for more epochs: stop there, compare the two runs'
final held-out perplexity, and keep the lower. In the same workspace the run finds the earlier run
in the stage's history and says so itself, and `evaluate` repeats it. With the astronomy book's
figures (that trial ran before the line existed), the end of the re-run reads:

```
Held-out perplexity: lowest 13.839 at epoch 10 of 10; final 13.839 (+0.0 % over the lowest).
This run is the re-run at stage1's turn (epoch 10), and not turning is what such a re-run shows, not a sign that more epochs would help: stop here. Compare the two finals and keep the lower: this run's 13.839 against stage1's 14.021: this run is 1.3 % lower; keep it.
```

Two finals less than about 0.15 % apart are the same run as far as one seed can tell
([§5](#5-when-to-stop)), and the line says so. A run trained elsewhere cannot be seen from here:
a re-run in a fresh workspace, or the control trained at its own best epoch by the commands
`evaluate` prints ([§3](#3-read-both-axes-and-the-control-at-its-own-dose)), ends with the
generic "had not turned" line. Read it the same way: it is a re-run at a turn, so stop, and compare
its final with the run whose turn it re-ran.

`lfa fuse` and `lfa extend` read the stage's latest run, which after a re-run is the re-run. When
the earlier run came out lower, export it from Python instead, with the same call `fuse` makes:

```python
from lfa.models import fuse

fuse("runs/world_history/runs/stage1/final_model", "Qwen/Qwen3-0.6B",
     "runs/world_history/models/stage1_export")
```

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
Beneath the table, `evaluate` repeats the run's own held-out verdict from step 2
(`This run's held-out curve: …`, with the same advice), read from the run's `history.json` entry.

**The control.** `lfa evaluate --compare-unanchored` trains the same stage again with λ = μ = 0 and
adds it as a third column. It trains at the anchored run's epochs, and without the anchor a corpus
turns much sooner, so at that dose part of the gap can be dose: on the Wells book, at 15 epochs,
the control ended at WikiText-2 334.27 and held-out 435.46, though its own curve was lowest at
epoch 1 (25.59). When the control's curve turned (ending 1 % or more above its lowest), `evaluate`
says so beneath the table and prints three commands that train it at its own best epoch in a fresh
workspace; the training one has the shape

```bash
lfa train --workspace <fresh> --corpus data/world_history --lambda 0 --mu 0 --epochs <its best>
```

Run the printed commands rather than this one: they carry the stage's exact recipe and paths. That
run is a re-run at the control's turn, so its line saying the curve had not turned means stop
([step 2](#when-the-re-run-does-not-turn)).

**Read the control there, and expect it to beat every anchored run on the domain.** At its own
dose the control fits the domain more closely and pays on the general axis; what the anchor buys
is the general axis. One seed each: on *A Short History of Astronomy* the control at its own 2
epochs scored held-out 12.37, below every anchored run (the lowest, 13.53, was λ = 400,000 at 6
epochs), and WikiText-2 19.68, 9.8 % above the base model's 17.93, where the recipe's λ re-run at
10 epochs scored 13.85 and 15.97. In the walkthrough's companion notebook (Darwin, a 4-epoch demo)
the control at its own dose fit the domain more closely (15.55 against 19.30) and kept less of the
general axis (WikiText-2 18.95 against 16.08, base 17.80). On the Wells book its lowest held-out
(25.59, epoch 1 of its 15-epoch run) was below every anchored run's lowest (26.88, the 8-epoch
re-run at epoch 7); its WikiText-2 near that epoch was not measured.

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
recipe's value, a factor of 2.5 apart, each in a workspace of its own on the first workspace's
artifact file (the supplement prepared beside the corpus is reused):

```bash
for lam in 400000 2500000; do
  lfa init     runs/world_history_lam$lam --model Qwen/Qwen3-0.6B --artifact runs/world_history/artifacts/v1.pt
  lfa train    --workspace runs/world_history_lam$lam --corpus data/world_history --epochs 8 --lambda $lam
  lfa evaluate --workspace runs/world_history_lam$lam
done
```

`--lambda X` sets both site families to X exactly, logs the recipe's value beside it, and is
recorded in the stage's `history.json` entry ([recipes.md](recipes.md#trying-another-λ)). Then read
each rung's own end-of-run line: a weaker anchor turns sooner and a stronger one later, so the dose
step 2 chose need not fit a rung. A rung whose line advises a re-run — it turned and ended 1 % or
more above its lowest, the WARNING or the "likely" line — is re-run at its lowest epoch, in its own
workspace, before the comparison; a rung whose re-run is "optional" (under 1 %) is kept as run:

```bash
lfa train    --workspace runs/world_history_lam400000 --corpus data/world_history --epochs <its lowest epoch> --lambda 400000
lfa evaluate --workspace runs/world_history_lam400000
```

Each rung costs one run at that dose plus its evaluation, and one more of each when its line
advises a re-run.

**How to pick.** Compare each λ at its own dose: the recipe's λ at the dose step 2 chose, and each
rung as it was kept (its re-run, when its line advised one). This is not how the shipped recipes
were calibrated: there every rung ran at a common 15 epochs
([recipes.md](recipes.md#how-λ-was-chosen)). Read the unanchored control at the recipe λ's dose
(`--compare-unanchored` on that run). If its held-out domain perplexity ends above the base model's,
use the cookbook's rule for a control that over-trains ([§5, step
5](model-integration-cookbook.md#5-calibrate-λ), second paragraph): take the rung best on both axes
(lowest held-out domain and lowest WikiText-2); if no rung is, take the lowest held-out domain among
the rungs within 1 % of the lowest WikiText-2. That is the usual case, because the unanchored curve
turns much sooner than the anchored one: every recorded control at the recipe's 15 epochs ended
above base (Darwin on both models, the Wells book). If the control instead improved on the base
model, use step 5's first rule. Either way, if the pick is an end rung, add one a further factor of
2.5 out and read again, until the pick is in the middle. Compare WikiText-2 at one window count
(`evaluate`'s default, 100, throughout).

**What this evidence is.** One seed, perplexity, and two measured cases of the trade between dose
and λ. On the Wells book a stronger anchor at the full dose came close to the recipe's anchor at its
best dose, but not to within noise: λ = 2,500,000 at 15 epochs scored WikiText-2 15.91 and held-out
27.12, λ = 1,000,000 at 8 epochs 15.94 and 26.91. The held-out gap, 0.8 % in the 8-epoch run's
favour, is above the 0.15 % spread of [§5](#5-when-to-stop); the WikiText-2 gap, 0.2 %, is below its
0.5 %. On *A Short History of Astronomy* (step 2) the dose changed the pick. At a common 10 epochs λ
= 400,000 scored WikiText-2 16.57 and held-out 13.91 (it turned at epoch 6 and ended 3.5 % above its
lowest), λ = 1,000,000 15.97 and 13.85, λ = 2,500,000 15.92 and 14.44, and the rule picks 1,000,000.
λ = 400,000 re-run at its own 6 epochs scored 16.06 and 13.53: within 1 % of the lowest WikiText-2
and the lowest held-out domain, so by the rule it is the pick, and as an end rung it calls for a
rung at 160,000, which was not run. To keep a λ, pass `--lambda` on every `train`, or write it into
a recipe of your own ([recipes.md](recipes.md#writing-your-own)) so that it is the default.

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
make the last one trained the one you mean to ship, or export an earlier one as
[step 2](#when-the-re-run-does-not-turn) shows. Its default directory, `models/stage1_fused_export`,
belongs to the stage, not the run: a `fuse` after a re-run replaces the export already there and
says so, naming the run the directory now holds (`--out DIR` writes elsewhere).
