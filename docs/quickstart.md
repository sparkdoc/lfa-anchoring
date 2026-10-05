# The full pipeline

From nothing to an adapted model, one step at a time: build the model's p(h) artifact, prepare
your documents, train, evaluate, fuse, and go on to the next domain. One GPU, no API keys — every
number below is a perplexity computed on your own machine. Each step says what it does, what it
costs, and what to do when it goes wrong.

```bash
lfa init runs/my_domain --model Qwen/Qwen3-0.6B --artifact self-generated   # once per model
lfa prepare-domain ~/papers ~/notes.md --out data/my_domain \
    --supplement --model Qwen/Qwen3-0.6B
lfa train    --workspace runs/my_domain --corpus data/my_domain
lfa evaluate --workspace runs/my_domain
lfa fuse     --workspace runs/my_domain
```

## 0. Install and check the card

```bash
pip install lfa-anchoring                       # once it is published; see ../RELEASING.md
pip install -e '.[dev]'                         # from a checkout, with the test tools
```

Add `[html]` or `[pdf]` if your documents are HTML or PDF; `[dev]` is the maintainer's set
(pytest, coverage, build) and an end user does not need it.

**Prerequisites: Python ≥ 3.11 and a CUDA card.** Development headers and a C compiler are
needed only by torch paths that compile a small CUDA shim on the first kernel launch; the
package's own training and generation paths do not, so `lfa` warns once if they are missing and
proceeds. A distribution `python3`
installed without `python3-dev` / `python3.13-dev` (and `build-essential`) has no `Python.h`,
while a uv- or conda-managed interpreter ships its own headers. `LFA_SKIP_TOOLCHAIN_CHECK=1`
silences the warning.

The versions this was built and tested against are pinned in
[`constraints-tested.txt`](../constraints-tested.txt) (torch 2.10.0+cu128, transformers 4.57.6,
accelerate 1.14.0, peft 0.18.1) — nothing here has been run below them, and nothing above them
either, so `pip install -c constraints-tested.txt lfa-anchoring` is the way to get exactly the
tested stack. The package itself accepts `transformers>=4.56,<5` and `peft>=0.18,<1`; the
transformers floor is where `from_pretrained` learned the `dtype=` spelling this package loads
with, and below it a model would load in the checkpoint's own dtype without saying so.

`lfa --version` says which version you have, which is the first thing a bug report needs.

An installed wheel carries the examples as well, so they can be run without a checkout:

```bash
python -m lfa.examples.quickstart --help
python -m lfa.examples.chain_three_domains --help
```

The two notebooks ship in the same place. They are data rather than modules, so open them by
path instead of with `-m`:

```bash
python -c "import lfa.examples, pathlib; print(pathlib.Path(lfa.examples.__file__).parent / 'two_domain_walkthrough.ipynb')"
python -c "import lfa.examples, pathlib; print(pathlib.Path(lfa.examples.__file__).parent / 'what_the_anchor_does.ipynb')"
```

On an 8 GB card, set the recipe's batch geometry before the first `train`:
[faq.md](faq.md#how-much-gpu-memory-does-a-run-need) has the numbers.

## 1. Choose a model

Two models have a bundled recipe — the operating point (λ, μ, rank, batch, epochs) calibrated for
that model — and the `--model` value picks it:

| `--model` | bundled recipe |
|---|---|
| `Qwen/Qwen3-0.6B` | `qwen3-0.6b` |
| `Qwen/Qwen3-1.7B` | `qwen3-1.7b` |

The match is on the recipe's own `model_id`, so it takes the Hub id exactly as written; a local
checkpoint path is another id, gets no default recipe, and needs `--recipe`. What each costs to
build an artifact for and to train is in the README's [Models](../README.md#models) table. The
commands on this page use Qwen3-0.6B; every one of them takes the other id the same way.

`--recipe` names a recipe yourself — a bundled name or a YAML path. When the recipe names a model
other than the workspace's, `init` warns, once:

```
Recipe 'qwen3-0.6b' is calibrated for Qwen/Qwen3-0.6B, and this workspace's model is Qwen/Qwen3-1.7B: lambda does not port between models, so use the recipe for this model or calibrate one (docs/model-integration-cookbook.md).
```

It is a warning, not a refusal: a local path to the same weights is a different id, and for it
the bundled recipe of those weights is the right one. For a model with no bundled recipe,
[model-integration-cookbook.md](model-integration-cookbook.md) is the procedure.

## 2. Build (or reuse) the artifact

The anchor samples hidden states from a fitted p(h) artifact, so a workspace needs one before its
first stage. The model writes it, or, when one has been published for it, `init` downloads it:

```bash
lfa init runs/my_domain --model Qwen/Qwen3-0.6B --artifact self-generated
```

**What it does.** `init` creates the workspace. It *records* the model (a Hub id or a path: the
checkpoint is not copied in) and the recipe — the bundled one that names your model, here
`qwen3-0.6b`, unless you pass `--recipe` — and then has the model write 2,500 documents of its own,
each started from its bare document-start token, and fits p(h) on them at 600k samples per site with
a K = 32 mixture. The build happens in the local store, `~/.cache/lfa/artifacts` (or
`$LFA_ARTIFACT_STORE`), and the finished file is copied into the workspace as `artifacts/v1.pt`,
with the corpus beside it as `artifacts/v1.corpus.jsonl` and its manifest. The workspace records the
artifact as `self-generated:<corpus sha256[:12]>`. Move or delete the checkpoint (or a `--recipe`
file you passed by path) and the workspace will not find it again.

**What it costs.** From a cold store on one RTX 3090 (24 GB) in a host with 125 GiB of RAM,
3 h 51 min for Qwen3-0.6B (83 minutes for the model to write its 2,500 documents, then the fit)
and 5 h 56 min for Qwen3-1.7B (87 minutes of writing), measured 2026-10-03/04
([the-artifact.md](the-artifact.md#what-it-costs)). An 8 GB card has not been measured at this
frame; [faq.md](faq.md) has the smaller pieces that were timed on one. It is paid once per model: a
second `init` over the same model at the same frame finds the finished artifact in the store and
copies it in, logging `Reused the self-generated artifact built <date> from <store path>`.
`lfa list-artifacts` shows what the store holds: the model, the frame (documents × tokens, K), the date
it was built, its size and its path, or how far an unfinished build got.

**When one has been published.** On a store miss, `init` first looks the model up in the list of
published artifacts pinned in the package. If there is one for exactly this model id, checkpoint
and frame, it is downloaded into the store with its corpus and manifest instead of built,
verified (each file's size and sha256; that the artifact is one this release reads, for this model
and frame; that the corpus is the one it was fitted on) and then used exactly as a built one. A
download
that fails or does not verify is refused in one line naming the URL; nothing is kept, and
`--rebuild` builds the artifact here instead (hours on one GPU). Anything with no pin — another
model, another snapshot of the weights, another frame — is built
([the-artifact.md](the-artifact.md#published-artifacts)).

**What it is worth.** On Qwen3-0.6B an artifact fitted on the model's own text at this frame
matched an artifact fitted on real text at every λ tried, and was at least as good at the recipe's
λ — one model, one seed, one domain. Each bundled recipe is calibrated against its own model's
self-generated artifact at this frame and says nothing. On a model with no bundled recipe it is the
way to a first artifact, and λ is then calibrated against it
([model-integration-cookbook.md](model-integration-cookbook.md)). `lfa probe-artifact` checks that
an artifact describes its model ([the-artifact.md](the-artifact.md#checking-an-artifact)).

**While it runs** it logs one line per batch of documents, `self-generated corpus: n/2500
documents (e empty)`. Each finished batch is on disk before the next starts, so:

* **Ctrl-C, a crash, a reboot**: run the same command again. The build resumes at the next batch,
  with the same per-batch seed, and the empties already counted still count.
* **The fit fails after the corpus is written**: the corpus stays in the store, and the next `init`
  fits it without generating again (`lfa list-artifacts` shows the entry as `corpus complete, not
  fitted`).
* **The same build in two terminals**: the second refuses, naming the lock file and the process
  that holds it. A lock left by a process that is no longer running is taken over with a warning.
* **`--rebuild`** builds afresh, here, even when the store has a match, and never downloads. The
  old entry is moved aside to `<entry>.replaced-<timestamp>/`, never deleted.

**Reusing a file.** `--artifact path/to/v1.pt` copies in an artifact this package built instead —
another workspace's `artifacts/v1.pt`, or a file `lfa build-artifact` wrote (`--self-generated`, or
`--corpus` for real text). A file built anywhere else is refused. A file that was fitted on the
model's own text keeps that provenance, so the recipe judges it exactly as if it had been built
here.

**A trial build** for trying the pipeline end to end before paying for the real one:

```bash
lfa init runs/trial --model Qwen/Qwen3-0.6B --artifact self-generated --n-raw 60 --max-new-tokens 128
```

It is a different frame, so it is a different store entry and does not stand in for the real one.
`train` then warns that the artifact was built at a different frame from the one the recipe's λ
was calibrated at, naming each field that differs. That is expected for a trial; for a real run,
build at the default frame. [the-artifact.md](the-artifact.md) has the frame in full, the host-RAM
arithmetic and the store's layout.

## 3. Prepare your data

```bash
lfa prepare-domain ~/papers ~/notes.md --out data/my_domain --supplement --model Qwen/Qwen3-0.6B
lfa prepare-domain ~/papers ~/notes.md --out data/my_domain                 # corpus only
```

The first turns text, Markdown, HTML or PDF into a flat directory of `.txt` files, one document
each, and then has the model write the question-and-answer supplement over it, beside the corpus in
`data/my_domain.supplement/`, where `train` finds it. The second writes the corpus alone, and
`train` writes the supplement itself before the first epoch.
[preparing-your-data.md](preparing-your-data.md) has the formats, the cleaning, the corpus shapes
that train badly and everything about the supplement.

## 4. Train

```bash
lfa train --workspace runs/my_domain --corpus data/my_domain
```

`train` adapts the workspace's current model to the corpus with the anchor on. A tenth of the
documents are held out and scored after every epoch, so the domain number is a measurement rather
than a fit. **Watch that number.** The recipe's 15 epochs were tuned on ~1,700 documents; on a
smaller corpus the held-out perplexity bottoms out early and then climbs, and `final_model` is the
last epoch by design (no best checkpoint is kept — [recipes.md](recipes.md) says why). The trainer
warns at the end if the curve turned around; the fix is to re-run with `--epochs <the epoch it
bottomed at>` — the learning-rate schedule is laid over whatever you say, so that is a complete
shorter run rather than a truncated long one. `--resume` continues an interrupted run.

The supplement is mixed into the training side at the recipe's `supplement_fraction` (0.13 of
training tokens, the frame the recipe's λ was tuned at); the held-out tenth is split off first and
stays raw text. `train` uses a supplement prepared with the data when the same model wrote it, and
writes one into the workspace otherwise. `--no-supplement` trains on the raw corpus alone and warns
that the run is off that frame; `--supplement FILE` mixes in a prompt/response JSONL of your own
instead ([preparing-your-data.md](preparing-your-data.md#bringing-your-own)).

Before the first step the trainer also reads the corpus's shape and warns about the three shapes
that train badly ([preparing-your-data.md](preparing-your-data.md#three-shapes-that-train-badly)).

## 5. Evaluate

```bash
lfa evaluate --workspace runs/my_domain
```

`evaluate` scores the stage on both axes against the model it started from. Add
`--compare-unanchored` for the λ = μ = 0 control (trained on the same mix: the stage's own
supplement at the stage's fraction), and `--n-windows none` on a machine with no network (the
general axis reads WikiText-2 from the Hub).

### What it prints

```
| metric               | before | after |     Δ% |
| -------------------- | -----: | ----: | -----: |
| general (WikiText-2) |  18.18 | 16.69 |  -8.2% |
| domain               |  23.30 | 12.78 | -45.1% |
```

That table is from this package's own verification run on Qwen3-0.6B (2026-09-07,
[verification.md](verification.md)), which predates 0.2.0 and so trained on the raw corpus alone,
with no supplement: the domain moved a long way, the general axis moved a little.
Read both. A general number *below* the base model's is not a win — [faq.md](faq.md) says why.

## 6. Fuse

```bash
lfa fuse --workspace runs/my_domain
```

`fuse` writes a plain checkpoint with the adapter merged in — no PEFT wrapper, loads with
`AutoModelForCausalLM.from_pretrained` like any other model.

The same flow as Python is [`examples/quickstart.py`](../examples/quickstart.py); nothing in the
CLI is decided differently from the way the library decides it for a caller who imports it.

## 7. A next domain

```bash
lfa extend --workspace runs/my_domain
lfa train  --workspace runs/my_domain --corpus data/second_domain
```

`extend` merges the finished stage into the model and folds the domain into p(h), so the next
stage adapts the right model and anchors against a p(h) that describes it; `lfa
regenerate-artifact` is the alternative that fits a fresh p(h) on the merged model's own text. A
whole sequence is one YAML file and `lfa chain domains.yaml --workspace runs/chain`.
[multi-domain-chains.md](multi-domain-chains.md) has both routes and the chain spec.

For the whole thing worked through on real text — two domains one after the other, on two
public-domain books the notebook downloads itself, with **each stage run a second time with the
anchor off** so the control sits beside every number — open
[`examples/two_domain_walkthrough.ipynb`](../examples/two_domain_walkthrough.ipynb). It ran end to
end in 35.4 minutes on one RTX 3090 (four training runs, two of them controls, and a supplement
written for each stage), plus whatever the
model and WikiText-2 cost you on a cold cache and the artifact build when the store has none for
the model yet. Its corpora and epoch count are demo scale — a
quarter of the text the recipe was tuned on, a quarter of its epochs — and the notebook says so
beside every table, so do not read its settings as the recommended ones. Its recorded outputs were
made on 2026-09-30 on that card, with the self-generated artifact at the recorded frame (2,500
documents × 2,048 tokens, K = 32) and the supplement on.
[`examples/what_the_anchor_does.ipynb`](../examples/what_the_anchor_does.ipynb) is optional and
picks up the workspace it leaves behind: each control re-run at its own best number of epochs,
and what the three models say when asked.

## What it costs

The verification run — the `qwen3-0.6b` recipe, 1,673 documents at 512 tokens, 15 epochs, rank 32 —
took **about 1 h 48 m on one RTX 3090** (432 s per epoch) and about 9 GB of GPU memory — that run
held a separate teacher, as every run before 0.1.1 did; the same run today loads no second model and
peaks 1.11 GB lower ([faq.md](faq.md)). LFA pins a single card by default and refuses a sharded
device map unless you pass `--allow-sharding`: sharding buys memory, not speed, and costs about 8 %
here because every anchored hidden state then crosses a device boundary. Qwen3-1.7B needs more:
[faq.md](faq.md#how-much-gpu-memory-does-a-run-need) has the two models side by side.

The artifact build is the expensive part, and you do it once per model, not per domain:
[the-artifact.md](the-artifact.md). The supplement is written once per corpus and writer before
the first epoch; [faq.md](faq.md) has what generation cost on an 8 GB card.

## When something is refused

The library raises rather than guesses, and the CLI prints those refusals as one line and exits 2:
a chain out of order, a workspace that is not there or already is or has not trained anything yet
(what `fuse` and `evaluate` say), a device map that would shard,
an artifact build already running for the same model and frame, a partial build under a different
frame, a dataset that cannot be reached, and a recipe or chain spec
that does not parse or does not validate, a document that needs an optional extra
(`[html]`, `[pdf]`) you have not installed, a self-generated corpus too small or too empty to fit
p(h) on, and a supplement writer that returned no usable pair. The message ends with what to do
instead.

A traceback whose last frames are in `lfa/` is a bug in this package — please report it. A
traceback that ends inside somebody else's code is your environment rather than this package: a
CUDA out-of-memory from torch, a compiler error from triton on a machine without Python headers
(which `lfa` warns about once, up front), a Hub timeout inside `datasets`. The last few frames say
which of the two you have. Ctrl-C is neither: an interrupted command says how to pick it up —
`--resume` for a training run, the same command again for an artifact build (an `init` also
names the store entry it was building) — and exits 130.

One failure is deliberately *not* a refusal: if WikiText-2 cannot be fetched, `evaluate` says so
and reports the general axis as unmeasured rather than losing the domain number you came for.
`--n-windows none` asks for that on purpose.

## Next

* [preparing-your-data.md](preparing-your-data.md) — formats, cleaning, corpus shapes, the supplement.
* [the-artifact.md](the-artifact.md) — the self-generated build in depth, the store, and a real-text artifact.
* [concepts.md](concepts.md) — what the anchor actually does, and what λ and μ are.
* [recipes.md](recipes.md) — the shipped operating point, field by field, and how to depart from it.
* [multi-domain-chains.md](multi-domain-chains.md) — a second and third domain.
* [`examples/two_domain_walkthrough.ipynb`](../examples/two_domain_walkthrough.ipynb) — all of it end to end, with the unanchored control.
* [`examples/what_the_anchor_does.ipynb`](../examples/what_the_anchor_does.ipynb) — optional: the controls at their own best dose, and what the models say.
* [model-integration-cookbook.md](model-integration-cookbook.md) — a model with no bundled recipe.
* [faq.md](faq.md) — memory, full weights, reading the general axis, what is not shipped.
