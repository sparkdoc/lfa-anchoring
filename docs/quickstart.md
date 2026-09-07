# Quickstart

Adapt a model to one domain and read what it cost. Four commands, one GPU, no API keys — every
number below is a perplexity computed on your own machine.

## Install

```bash
pip install -e '.[dev]'          # add ,html or ,pdf if your documents are HTML or PDF
```

Python ≥ 3.11 and a CUDA card. The versions this was built and tested against are pinned in
[`constraints-tested.txt`](../constraints-tested.txt) (torch 2.10.0+cu128, transformers 4.57.6,
accelerate 1.14.0, peft 0.18.1) — nothing here has been run below them. The package itself accepts
`transformers>=4.56,<5` and `peft>=0.18,<1`; the transformers floor is where `from_pretrained`
learned the `dtype=` spelling this package loads with, and below it a model would load in the
checkpoint's own dtype without saying so.

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

> **Before the release assets exist**, `fetch-artifact` refuses rather than downloading something
> it cannot verify (`ArtifactNotPublished`: the registry's checksums are still placeholders). Until
> then, either build one — [rebuilding-the-artifact.md](rebuilding-the-artifact.md) — or pass a
> local file *with the id it is a copy of*:
> `lfa init --artifact /path/to/distribution_stats.pt --artifact-id qwen3-0.6b-gmm1543k-int8`.
> Without the id the workspace knows the file only by its path: every stage warns that λ was
> calibrated against a different artifact (it was not), and `lfa extend` refuses — after the stage
> has trained — because the shipped file carries no sample count of its own and the registry entry
> is where that number lives.

## Prepare the domain

Whatever the documents arrive as, this turns them into the flat directory of `.txt` files the
loader reads:

```bash
lfa prepare-domain ~/papers ~/notes.md --out data/my_domain
```

`.txt`/`.md` pass through, `.html` needs the `[html]` extra, `.pdf` needs `[pdf]`. A document that
cleans down to less than `--min-length` characters (default 1000) is dropped — a page that
extracted to a nav bar is not training data.

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
   rather than a fit. Use `--epochs` to shorten a run (the learning-rate schedule is laid over
   whatever you say, so it changes the whole curve, not only where it stops) and `--resume` to
   continue an interrupted one.
3. **`evaluate`** scores the stage on both axes against the model it started from. Add
   `--compare-unanchored` for the λ = μ = 0 control, and `--n-windows none` on a machine with no
   network (the general axis reads WikiText-2 from the Hub).
4. **`fuse`** writes a plain checkpoint with the adapter merged in — no PEFT wrapper, loads with
   `AutoModelForCausalLM.from_pretrained` like any other model.

The same flow as Python is [`examples/quickstart.py`](../examples/quickstart.py); nothing in the
CLI is decided differently from the way the library decides it for a caller who imports it.

## What it prints

```
| metric               | before | after |     Δ% |
| -------------------- | -----: | ----: | -----: |
| general (WikiText-2) |  18.18 | 16.69 |  -8.2% |
| domain               |  23.30 | 12.78 | -45.1% |
```

That table is from this package's own acceptance run on Qwen3-0.6B (2026-09-07): the domain moved
a long way, the general axis moved a little. Read both. A general number *below* the base model's
is not a win — [faq.md](faq.md) says why.

## What it costs

The acceptance run — the bundled recipe, 1,673 documents at 512 tokens, 15 epochs, rank 32 — took
**about 1 h 48 m on one RTX 3090** (432 s per epoch) and about 9 GB of GPU memory. LFA pins a
single card by default and refuses a sharded device map unless you pass `--allow-sharding`:
sharding buys memory, not speed, and costs about 8 % here because every anchored hidden state then
crosses a device boundary.

The artifact build is the expensive part, and you do it once per model, not per domain:
[rebuilding-the-artifact.md](rebuilding-the-artifact.md).

## When something is refused

The library raises rather than guesses, and the CLI prints those refusals as one line and exits 2:
a chain out of order, a workspace that is not there or already is or has not trained anything yet
(what `fuse` and `evaluate` say), a workspace with no p(h) artifact, a device map that would shard,
an artifact that is not published, a dataset that cannot be reached, and a recipe or chain spec
that does not parse or does not validate. The message ends with what to do instead. Anything that
comes back as a traceback is a bug in this package.

One failure is deliberately *not* a refusal: if WikiText-2 cannot be fetched, `evaluate` says so
and reports the general axis as unmeasured rather than losing the domain number you came for.
`--n-windows none` asks for that on purpose.

## Next

* [concepts.md](concepts.md) — what the anchor actually does, and what λ and μ are.
* [recipes.md](recipes.md) — the shipped operating point, field by field, and how to depart from it.
* [multi-domain-chains.md](multi-domain-chains.md) — a second and third domain.
* [adding-a-model.md](adding-a-model.md) — a model that is not Qwen3.
* [faq.md](faq.md) — memory, full weights, reading the general axis, what is not shipped.
