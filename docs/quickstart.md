# Quickstart

Adapt a model to one domain and read what it cost. Four commands, one GPU, no API keys — every
number below is a perplexity computed on your own machine.

## Install

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

## Get a p(h) artifact

The anchor samples hidden states from a fitted artifact, so a run needs one before it can start.

```bash
lfa list-artifacts
```

`lfa init --artifact <id>` (below) fetches the artifact it names into the workspace and verifies
its checksum, so there is normally nothing to do here. Fetch one by hand — `lfa fetch-artifact <id>
--dest artifacts/` — when you want it in advance, or want one copy shared by several workspaces;
then point `init` at the file with `--artifact artifacts/<id>.pt --artifact-id <id>`, which tells
the recipe that this file *is* the published artifact its λ was calibrated against.

Two are published for Qwen3-0.6B: the recipe artifact (`qwen3-0.6b-gmm1543k-int8`, ~108 MB,
correlated basis plus a K = 32 mixture per site) and a ~1 MB diagonal one kept as a budget floor.
The recipe's λ is calibrated against the first; the second is a known-inferior option that needs
its own λ, not a cheaper equivalent.

> **If the repository is private**, an anonymous download 404s and `fetch-artifact` refuses rather
> than writing something it cannot verify. The assets are published and the registry carries their
> real checksums, so this is a visibility problem rather than a missing release. Either build one — [rebuilding-the-artifact.md](rebuilding-the-artifact.md) — or pass a
> local file *with the id it is a copy of*:
> `lfa init --artifact /path/to/distribution_stats.pt --artifact-id qwen3-0.6b-gmm1543k-int8`.
> Without the id the workspace knows the file only by its path: every stage warns that λ was
> calibrated against a different artifact (it was not), and `lfa extend` refuses — after the stage
> has trained — because the shipped file carries no sample count of its own and the registry entry
> is where that number lives.

### Or let the model write one

```bash
lfa init runs/my_domain --model Qwen/Qwen3-0.6B --artifact self-generated
```

No download: the model writes 2,500 documents of its own, each started from its bare
document-start token, and p(h) is fitted on them at 600k samples per site. The corpus stays
beside the artifact as `artifacts/v1.corpus.jsonl`, with a manifest, and the workspace records
the artifact as `self-generated:<corpus sha256[:12]>`.
The cost: several hours on an 8 GB card; not timed ([faq.md](faq.md) has the pieces that were
timed). On Qwen3-0.6B this artifact matched the real-corpus artifact at every λ tried and was at
least as good as the published one at the recipe's λ — the LFA record's C12, one model, one seed,
one domain — so the bundled recipe treats it as calibrated and says nothing; on
any other model it is the way to a first artifact, and λ is then calibrated against it
([adding-a-model.md](adding-a-model.md)). [rebuilding-the-artifact.md](rebuilding-the-artifact.md)
has the recorded frame.

## Prepare the domain

Whatever the documents arrive as, this turns them into the flat directory of `.txt` files the
loader reads:

```bash
lfa prepare-domain ~/papers ~/notes.md --out data/my_domain
```

`.txt`/`.md` pass through, `.html` needs the `[html]` extra, `.pdf` needs `[pdf]`. A document that
cleans down to less than `--min-length` characters (default 1000) is dropped — a page that
extracted to a nav bar is not training data.

Two things about the corpus that the run will tell you and it is cheaper to know now. The whole
corpus is tokenized eagerly and held in host RAM at **about eight times its size on disk**, and a
corpus too large for the machine is refused before it is tokenized rather than killed part-way
through. And three corpus *shapes* train badly without failing — too few chunks for the batch, one
document contributing most of the gradient, and more epochs than the amount of text can carry; the
trainer names each one, with its numbers, before the first step. [faq.md](faq.md) has the
thresholds and what to do about each.

## The four commands

```bash
lfa init runs/my_domain --model Qwen/Qwen3-0.6B --artifact qwen3-0.6b-gmm1543k-int8
lfa train    --workspace runs/my_domain --corpus data/my_domain
lfa evaluate --workspace runs/my_domain
lfa fuse     --workspace runs/my_domain
```

1. **`init`** creates a workspace. It *records* the model (a Hub id or a path: the checkpoint is
   not copied in, and `init` does not load it — the artifact is checked against it when a stage
   starts) and the recipe, and it *copies in* the p(h) artifact as `artifacts/v1.pt`, so the
   workspace owns its own p(h) and later versions sit beside it. Runs, extended artifacts and
   fused models land there too, with a history entry per stage. `init` picks up the bundled recipe
   that names your model — `qwen3-0.6b` — automatically; pass `--recipe` for another model or your
   own YAML. Move or delete the checkpoint (or a `--recipe` file you passed by path) and the
   workspace will not find it again.
2. **`train`** adapts the workspace's current model to the corpus with the anchor on. A tenth of
   the documents are held out and scored after every epoch, so the domain number is a measurement
   rather than a fit. **Watch that number.** The recipe's 15 epochs were tuned on ~1,700
   documents; on a smaller corpus the held-out perplexity bottoms out early and then climbs, and
   `final_model` is the last epoch by design (no best checkpoint is kept —
   [recipes.md](recipes.md) says why). The trainer warns at the end if the curve turned around;
   the fix is to re-run with `--epochs <the epoch it bottomed at>` — the learning-rate schedule is
   laid over whatever you say, so that is a complete shorter run rather than a truncated long one.
   `--resume` continues an interrupted run.

   `train` first writes the supplement: the entry model reads each training-side passage and
   writes six question-and-answer pairs from a fixed template, cached under `supplements/` and
   reused while the corpus, the writer, the template and the domain description are unchanged.
   `--no-supplement` opts out; `lfa prepare-supplement` writes it ahead of time to inspect. The
   pairs are mixed into the training side at the recipe's `supplement_fraction` (0.13 of
   training tokens, the frame the shipped λ was tuned at); the held-out tenth is split off first
   and stays raw text.
   `--supplement <file.jsonl>` mixes a prompt/response file of your own instead, and
   `--domain-description "<text>"` says what the template calls the text (default: the corpus
   directory's name). With `--no-supplement` the run trains on the raw corpus alone, off the
   frame λ was tuned at, and warns so. What the supplement is for — reachability, not skill
   protection — is in [concepts.md](concepts.md).
3. **`evaluate`** scores the stage on both axes against the model it started from. Add
   `--compare-unanchored` for the λ = μ = 0 control (trained on the same mix: the stage's own
   supplement at the stage's fraction), and `--n-windows none` on a machine with no
   network (the general axis reads WikiText-2 from the Hub).
4. **`fuse`** writes a plain checkpoint with the adapter merged in — no PEFT wrapper, loads with
   `AutoModelForCausalLM.from_pretrained` like any other model.

The same flow as Python is [`examples/quickstart.py`](../examples/quickstart.py); nothing in the
CLI is decided differently from the way the library decides it for a caller who imports it.

For the whole thing worked through on real text — two domains one after the other, on two
public-domain books the notebook downloads itself, with **each stage run a second time with the
anchor off** so the control sits beside every number — open
[`examples/two_domain_walkthrough.ipynb`](../examples/two_domain_walkthrough.ipynb). It is the
runnable version of this page plus [multi-domain-chains.md](multi-domain-chains.md), it needs no
API key, and it ran end to end in 19 minutes on one RTX 3090 (four training runs, two of them
controls), plus whatever the model, the artifact and WikiText-2 cost you on a cold cache. Its corpora and epoch count are demo
scale — a quarter of the text the recipe was tuned on, a quarter of its epochs — and the notebook
says so beside every table, so do not read its settings as the recommended ones. Its recorded
outputs predate 0.2.0 and so trained on the raw text alone; re-run today, each stage first writes
and mixes in its supplement, so the recorded numbers are not a 0.2.0 run's.
[`examples/what_the_anchor_does.ipynb`](../examples/what_the_anchor_does.ipynb) is optional and
picks up the workspace it leaves behind: each control re-run at its own best number of epochs,
and what the three models say when asked.

## What it prints

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

## What it costs

That run — the bundled recipe, 1,673 documents at 512 tokens, 15 epochs, rank 32 — took
**about 1 h 48 m on one RTX 3090** (432 s per epoch) and about 9 GB of GPU memory — that run held
a separate teacher, as every run before 0.1.1 did; the same run today loads no second model and
peaks 1.11 GB lower ([faq.md](faq.md)). LFA pins a
single card by default and refuses a sharded device map unless you pass `--allow-sharding`:
sharding buys memory, not speed, and costs about 8 % here because every anchored hidden state then
crosses a device boundary.

The artifact build is the expensive part, and you do it once per model, not per domain:
[rebuilding-the-artifact.md](rebuilding-the-artifact.md). The supplement is written once per
corpus and writer before the first epoch; [faq.md](faq.md) has what generation cost on an 8 GB
card.

## When something is refused

The library raises rather than guesses, and the CLI prints those refusals as one line and exits 2:
a chain out of order, a workspace that is not there or already is or has not trained anything yet
(what `fuse` and `evaluate` say), a workspace with no p(h) artifact, a device map that would shard,
an artifact that is not published, a dataset that cannot be reached, and a recipe or chain spec
that does not parse or does not validate, a document that needs an optional extra
(`[html]`, `[pdf]`) you have not installed, a self-generated corpus too small or too empty to fit
p(h) on, and a supplement writer that returned no usable pair. The message ends with what to do
instead.

A traceback whose last frames are in `lfa/` is a bug in this package — please report it. A
traceback that ends inside somebody else's code is your environment rather than this package: a
CUDA out-of-memory from torch, a compiler error from triton on a machine without Python headers
(which `lfa` warns about once, up front), a Hub timeout inside `datasets`. The last few frames say
which of the two you have. Ctrl-C is neither: an interrupted command prints one line naming
`--resume` and exits 130.

One failure is deliberately *not* a refusal: if WikiText-2 cannot be fetched, `evaluate` says so
and reports the general axis as unmeasured rather than losing the domain number you came for.
`--n-windows none` asks for that on purpose.

## Next

* [concepts.md](concepts.md) — what the anchor actually does, and what λ and μ are.
* [recipes.md](recipes.md) — the shipped operating point, field by field, and how to depart from it.
* [multi-domain-chains.md](multi-domain-chains.md) — a second and third domain.
* [`examples/two_domain_walkthrough.ipynb`](../examples/two_domain_walkthrough.ipynb) — all of it end to end, with the unanchored control.
* [`examples/what_the_anchor_does.ipynb`](../examples/what_the_anchor_does.ipynb) — optional: the controls at their own best dose, and what the models say.
* [adding-a-model.md](adding-a-model.md) — a model that is not Qwen3.
* [faq.md](faq.md) — memory, full weights, reading the general axis, what is not shipped.
