# lfa-anchoring

**Layerwise Function Anchoring (LFA)** — adapt a language model to a new domain while preserving
what its sub-modules compute on the hidden states it actually sees.

The usual ways to keep a fine-tune from forgetting either rehearse the old data or hold the weights
near where they were. LFA does neither. It estimates, once per model, the distribution `p(h)` of
the vectors arriving at each anchored sub-module, and then — during every later adaptation —
penalizes how far that sub-module's *output on states drawn from `p(h)`* moves:

```
L = L_content  +  λ · E_{h ~ p(h)} ‖f_student(h) − f_teacher(h)‖²  +  μ · ‖W_s − W_t‖²
```

The preservation signal therefore comes from a statistic plus the frozen teacher, never from
stored text: LFA is **data-free at adaptation time**, and in a chain of domains no earlier domain
is ever revisited. The statistic is a per-site mean, covariance and mixture — no text, no token
ids, nothing sequence-shaped.

This is the method as a package: the training loop, the artifact, the recipe, and the multi-domain
chain, ported from the research code behind the LFA paper and checked against it.

## Install

```bash
pip install lfa-anchoring                       # once it is published; see RELEASING.md
pip install -e '.[dev]'                         # from a checkout, with the test tools
```

Add `[html]` or `[pdf]` if your documents are HTML or PDF. To hold to the versions this was
tested at, install against the pin file:
`pip install -c constraints-tested.txt lfa-anchoring`.

**Prerequisites: Python ≥ 3.11 with its development headers, a C compiler, and a CUDA card.**
The headers are the one that surprises people — torch's triton backend compiles a small CUDA
shim the first time a kernel launches, so a distribution `python3` without its `-dev` package
(`python3-dev` / `python3.13-dev`, plus `build-essential`) cannot train, while a uv- or
conda-managed interpreter ships what it needs. `lfa` checks for both before it loads anything
and says so in one line rather than failing inside gcc minutes into a run.

Tested with the versions in
[`constraints-tested.txt`](https://github.com/sparkdoc/lfa-anchoring/blob/main/constraints-tested.txt) — torch 2.10.0+cu128, transformers 4.57.6,
accelerate 1.14.0, peft 0.18.1. Nothing here has been run below those, and nothing above them.

## Four commands

```bash
lfa init runs/my_domain --model Qwen/Qwen3-0.6B --artifact qwen3-0.6b-gmm1543k-int8
lfa train    --workspace runs/my_domain --corpus data/my_domain
lfa evaluate --workspace runs/my_domain
lfa fuse     --workspace runs/my_domain
```

`init` makes a workspace: a directory that *records* which model it adapts (a Hub id or a path —
the model itself is not copied in, and is not even loaded until a stage starts) and which recipe it
uses, and that *carries* the p(h) artifact, fetched by id and checksum-verified or copied from a
path you pass, as `artifacts/v1.pt`. Every run, every extended artifact and every fused model then
lands beside it, with a history entry per stage. `train` adapts the model with the anchor on,
holding a tenth of the documents out so the domain number is a measurement rather than a fit.
`evaluate` reads the stage on both axes, against the model it started from:

```
| metric               | before | after |     Δ% |
| -------------------- | -----: | ----: | -----: |
| general (WikiText-2) |  18.18 | 16.69 |  -8.2% |
| domain               |  23.30 | 12.78 | -45.1% |
```

`fuse` writes a plain checkpoint that loads with `AutoModelForCausalLM.from_pretrained`. A second
domain adds one step — `lfa extend` — which folds the finished stage into both the model and
`p(h)`; `lfa chain domains.yaml` runs a whole sequence.

> **If the repository is private**, an anonymous download of the release asset 404s and the fetch
> refuses rather than writing something it cannot verify. The assets are published and their
> checksums are in the registry; this is a visibility problem, not a missing release. Pass a local
> artifact file instead —
> `--artifact /path/to/distribution_stats.pt --artifact-id qwen3-0.6b-gmm1543k-int8` — or build
> one: [docs/rebuilding-the-artifact.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/rebuilding-the-artifact.md). Pass **both**: the id
> says which published artifact that file is, which is what the recipe's λ is read against (a
> bare path warns on every stage that λ was calibrated elsewhere) and what supplies the base
> sample count `lfa extend` needs, since the shipped file carries none of its own.

Full walkthrough: [docs/quickstart.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/quickstart.md). Same flow as Python:
[`examples/quickstart.py`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/quickstart.py).

**To watch the method work rather than read about it**, run
[`examples/two_domain_walkthrough.ipynb`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/two_domain_walkthrough.ipynb): two domains in sequence on real public-domain
text, with the same two runs repeated with the anchor switched off, so the control is beside
every number. Measured at 21 minutes on one RTX 3090, no API keys, and everything it needs is
downloaded by the notebook itself.

Everything is computed locally. There is no judge, no API key, and nothing to configure.

## Documentation

| | |
|---|---|
| [docs/quickstart.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/quickstart.md) | install, the four commands, what a run costs |
| [`examples/two_domain_walkthrough.ipynb`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/two_domain_walkthrough.ipynb) | the runnable demonstration: two domains, and the same runs unanchored |
| [docs/concepts.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/concepts.md) | what the anchor does, what λ and μ are, how a run is read |
| [docs/recipes.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/recipes.md) | the shipped operating point field by field, and its couplings |
| [docs/multi-domain-chains.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/multi-domain-chains.md) | second and third domains; what `extend` does |
| [docs/adding-a-model.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/adding-a-model.md) | a model that is not Qwen3: adapter, artifact, λ |
| [docs/rebuilding-the-artifact.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/rebuilding-the-artifact.md) | the seed corpus and the build |
| [docs/faq.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/faq.md) | GPU memory, full weights, reading the general axis, what is not shipped |
| [docs/verification.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/verification.md) | what was checked against the research code, how, and what came out |
| [RELEASING.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/RELEASING.md) | how the artifacts and a tag are cut |

## What is in the box

* `lfa.train` — the loop: content loss, function anchor, weight backstop, warmup + cosine, per-epoch
  held-out validation, checkpoints and resume.
* `lfa.artifact` — building a `p(h)` artifact from a seed corpus, quantizing it, fetching a
  published one by id with its checksum, and extending one with a new domain.
* `lfa.Recipe` — a tuned operating point *and what it was tuned at*, so a run that departs from
  either the rank or the artifact is told that λ no longer means what it meant.
* `lfa.Workspace` — the state machine across domains: which model the next stage adapts, which
  artifact version it anchors against, and the order the two may be done in.
* `lfa.evaluate` — the two perplexities, and the table that pairs them.
* `lfa.adapters` — the one place a model's layout is known. A new architecture is one class.

Two published artifacts for Qwen3-0.6B (`lfa list-artifacts`): the recipe artifact
(`qwen3-0.6b-gmm1543k-int8`, ~108 MB) and a ~1 MB diagonal one kept as a budget floor — a
known-inferior option, not a cheaper equivalent.

## Tests

From a checkout (the wheel ships the package and the examples, not the tests):

```bash
pytest -q
pytest tests/test_gpu_smoke.py -m gpu -q     # bf16 placement, one stage on the card, TorchGMM
```

That suite is about this package on its own: the loop, the artifact, the recipe, the workspace and
the CLI. The default run needs no GPU, no corpus and no network; the second line is the `gpu`
marker, which the default deselects.

## Was the port checked against the research code?

Yes, and [docs/verification.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/verification.md)
is the report: the sampler's 85 draws per anchoring step replayed **bit-for-bit** (with a negative
control that fails), all four anchor blocks and the layer schedule bit-identical, and one full run
of the bundled recipe against a research-code run of the identical configuration agreeing on every
deterministic series — optimizer steps and corpus counts exact, per-epoch content loss within
0.191 %, per-epoch held-out loss within 0.0104 nats.

That page **reports; it does not prove.** The harness needs both implementations plus gigabytes of
checkpoints, artifacts and corpora that are not public, so it lives with the research code and is
not in this repository — it is available to a reviewer who asks. And agreement with another
implementation is not correctness.

It is also dated. Since it was measured the corpus loader has **deliberately diverged**: the
per-epoch chunk offset now rotates the chunk boundaries instead of discarding each document's first
`offset` tokens, which the old behaviour did in every epoch after the first — costing a 600-token
document 42.6 % of its tokens in an average epoch, a 5,000-token one 5.1 %. Long documents are the
harmless tail, which is why the paper's corpora never showed it. The epoch-0 stream, and therefore
every corpus count on that page, is unchanged; the per-epoch series are not. The old stream is
not reachable from this package — it was briefly a setting and was removed, because it served
reproducing those numbers and nothing else.

None of it is a reproduction of a published number either. The paper's own headline — domain
perplexity 8.76 on Qwen3-0.6B at a seed ΔPPL of −10.0 %, i.e. seed-corpus perplexity 10 % *below*
the base model's — is the paper's measurement on the paper's corpus and instruments, and is quoted
here only as such.

## Relationship to the research record

The research code behind the paper is a private record of every arm, every ladder and every
retraction. It is not distributed. This package is the method itself: what survived, ported,
tested and documented, with the research-only scaffolding left behind.

## Citing

*Layerwise Function Anchoring: Preserving Sub-Module Functions on Sampled Hidden States for
Continual Domain Adaptation* — the LFA paper (2026); authors withheld for review.

Licensed under Apache-2.0 (see [LICENSE](https://github.com/sparkdoc/lfa-anchoring/blob/main/LICENSE)).
