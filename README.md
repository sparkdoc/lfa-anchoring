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
[`constraints-tested.txt`](https://github.com/<org>/lfa-anchoring/blob/main/constraints-tested.txt) — torch 2.10.0+cu128, transformers 4.57.6,
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

> **Before the release assets exist**, fetching an artifact by id refuses rather than downloading
> something it cannot verify (`ArtifactNotPublished` — the registry's checksums are still
> placeholders), so that first command fails as written. Pass a local artifact file instead —
> `--artifact /path/to/distribution_stats.pt --artifact-id qwen3-0.6b-gmm1543k-int8` — or build
> one: [docs/rebuilding-the-artifact.md](https://github.com/<org>/lfa-anchoring/blob/main/docs/rebuilding-the-artifact.md). Pass **both**: the id
> says which published artifact that file is, which is what the recipe's λ is read against (a
> bare path warns on every stage that λ was calibrated elsewhere) and what supplies the base
> sample count `lfa extend` needs, since the shipped file carries none of its own.

Full walkthrough: [docs/quickstart.md](https://github.com/<org>/lfa-anchoring/blob/main/docs/quickstart.md). Same flow as Python:
[`examples/quickstart.py`](https://github.com/<org>/lfa-anchoring/blob/main/examples/quickstart.py).

Everything is computed locally. There is no judge, no API key, and nothing to configure.

## Documentation

| | |
|---|---|
| [docs/quickstart.md](https://github.com/<org>/lfa-anchoring/blob/main/docs/quickstart.md) | install, the four commands, what a run costs |
| [docs/concepts.md](https://github.com/<org>/lfa-anchoring/blob/main/docs/concepts.md) | what the anchor does, what λ and μ are, how a run is read |
| [docs/recipes.md](https://github.com/<org>/lfa-anchoring/blob/main/docs/recipes.md) | the shipped operating point field by field, and its couplings |
| [docs/multi-domain-chains.md](https://github.com/<org>/lfa-anchoring/blob/main/docs/multi-domain-chains.md) | second and third domains; what `extend` does |
| [docs/adding-a-model.md](https://github.com/<org>/lfa-anchoring/blob/main/docs/adding-a-model.md) | a model that is not Qwen3: adapter, artifact, λ |
| [docs/rebuilding-the-artifact.md](https://github.com/<org>/lfa-anchoring/blob/main/docs/rebuilding-the-artifact.md) | the seed corpus and the build |
| [docs/faq.md](https://github.com/<org>/lfa-anchoring/blob/main/docs/faq.md) | GPU memory, full weights, reading the general axis, what is not shipped |
| [RELEASING.md](https://github.com/<org>/lfa-anchoring/blob/main/RELEASING.md) | how the artifacts and a tag are cut |

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
```

Two suites are opt-in, because they need a GPU and the research checkout this package was ported
from:

```bash
pytest tests/equivalence -m equivalence -q   # sampler, losses and loader, against the research code
pytest tests/acceptance -m acceptance -q -s  # one full run, against a matched the research code run
pytest tests/test_gpu_smoke.py -m gpu -q     # bf16 placement, one stage on the card, TorchGMM
```

The equivalence suite replays captured fixtures: the sampler's draws are asserted **bit-identical**,
the anchor's blocks and the loader's batch to tolerance. See
[`tests/equivalence/README.md`](https://github.com/<org>/lfa-anchoring/blob/main/tests/equivalence/README.md).

The acceptance suite is an **equivalence run**: it trains the bundled recipe end to end and
compares the result against a research-code run of the identical configuration. What it compares is
the *deterministic* part of the run, because the objective itself is not deterministic: both
implementations estimate the anchor from 16 hidden states drawn per site per step, out of
independent RNG streams, so two full runs are two draws of a stochastic objective and
bit-equivalence between them is impossible by construction.

The criterion, measured 2026-09-07 (the run's own records are committed at
[`tests/acceptance/_runs/2026-09-07-equiv/`](https://github.com/<org>/lfa-anchoring/tree/main/tests/acceptance/_runs/2026-09-07-equiv/)):

| quantity | result | tolerance |
|---|---|---|
| optimizer steps, every epoch | **exact** (603 … 8,969) | integer equality |
| corpus: training chunks / held-out chunks / held-out tokens | **exact** (3,614 / 435 / 157,366) | integer equality |
| per-epoch content loss, all 15 epochs | worst 0.191 % | 0.5 % |
| per-epoch held-out loss, all 15 epochs | worst 0.0104 nats | 0.03 nats |

Two end-of-run perplexities are **reported** beside those and asserted by nothing. They are the
*research* instrument's numbers, on different text from the table above -- that domain row is this
package's own `evaluate` on the stage's held-out documents, while these come from the research code's
scorer on a held-out chat-formatted Q&A set, which is why one run has two domain perplexities and
neither is wrong. Domain direct-QA perplexity 10.6996 against the reference's 10.9122 (−1.95 %) and WikiText-2 drift −8.202 % against
−7.898 % (0.304 points). Each is a single draw of a sampled objective whose spread across seeds has
never been measured, so there is no calibrated band to hold them to — a gap there is something to
investigate with a second seed, which is also what would earn them a band back.

None of this is a reproduction of a published number. The paper's own headline — domain perplexity
8.76 on Qwen3-0.6B at a seed ΔPPL of −10.0 %, i.e. seed-corpus perplexity 10 % *below* the base
model's — is the paper's measurement on the paper's corpus and instruments, and is quoted here only
as such. See
[`tests/acceptance/README.md`](https://github.com/<org>/lfa-anchoring/blob/main/tests/acceptance/README.md).

## Relationship to the research record

`the research code` is the research record — every arm, every ladder, every retraction. This package is the
method: what survived, ported, tested, and documented, with the research-only scaffolding left
behind.

## Citing

*Layerwise Function Anchoring: Preserving Sub-Module Functions on Sampled Hidden States for
Continual Domain Adaptation* — the LFA paper (2026); authors withheld for review.

Licensed under Apache-2.0 (see [LICENSE](https://github.com/<org>/lfa-anchoring/blob/main/LICENSE)).
