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

The preservation signal comes from a statistic plus the frozen teacher, never from stored text:
LFA is **data-free at adaptation time**, and in a chain of domains no earlier domain is ever
revisited. The statistic is a per-site mean, covariance and mixture — no text, no token ids,
nothing sequence-shaped. What comes out is a plain Hugging Face checkpoint adapted to your
documents. Everything runs locally on one CUDA card: no judge, no API key.

## Install

```bash
git clone https://github.com/sparkdoc/lfa-anchoring
cd lfa-anchoring
pip install -c constraints-tested.txt -e .     # '.[html]' / '.[pdf]' for those formats, '.[dev]' for the tests
```

Needs Python ≥ 3.11 and a CUDA card; the install takes a few minutes and a few GB. The constraints
file holds the tested stack (torch 2.10.0+cu128, transformers 4.57.6, accelerate 1.14.0, peft
0.18.1). The package is not on PyPI yet; publication is pending.

> Some torch paths compile a small CUDA shim on the first kernel launch and need your Python's
> development headers (`python3.X-dev` for Python 3.X, plus `build-essential`, on Debian and
> Ubuntu; uv- and conda-managed interpreters ship them). The package's own paths do not, so `lfa`
> warns once if the headers are missing and proceeds.

## The pipeline

```bash
lfa init runs/world_history --model Qwen/Qwen3-0.6B --artifact self-generated
lfa prepare-domain ~/history --out data/world_history \
    --supplement --model Qwen/Qwen3-0.6B             # one long file? add --split-chars 3500
lfa train    --workspace runs/world_history --corpus data/world_history
lfa evaluate --workspace runs/world_history
lfa fuse     --workspace runs/world_history
```

**`init`** creates the workspace and puts its p(h) artifact in place. For `Qwen/Qwen3-0.6B` and
`Qwen/Qwen3-1.7B` that is a download of the published artifact, verified by hash: 132 MB / 265 MB
with its corpus, seconds to a minute. Any other model writes 2,500 documents of its own and p(h)
is fitted on them, which takes hours on one GPU, and needs an adapter check and a λ of its own
([the cookbook](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/model-integration-cookbook.md)).
Either way the artifact is kept in a local store and reused by every later workspace over the same
model. `init` ends by printing the next two commands.

**`prepare-domain`** turns text, Markdown, HTML or PDF into the corpus, one `.txt` file per
document; an input file that cleans to under 1,000 characters is dropped (`--min-length`). Training
holds out whole documents, so **one long file — a book, a report — needs `--split-chars 3500`**,
which cuts it into documents at paragraph boundaries. Cut boilerplate such as a Project Gutenberg
licence, a table of contents or an index out of the file first. With `--supplement`, the model then
writes question-and-answer pairs over the corpus, which make the domain's knowledge answerable when
the model is asked about it; they do not protect skills, which is the anchor's job. Without
`--supplement`, `train` writes the same pairs itself before the first epoch. A corpus with nothing
to hold out is refused before any pairs are written, by either command, with the fix in the
message (`train --no-supplement` trains it as it is, with no held-out curve).

**`train`** adapts the model with the anchor on. It holds a tenth of the documents out, scores
them after every epoch, and ends with one line that reads that curve: `Held-out perplexity: lowest
X at epoch k of n; final Y (+z % over the lowest).` `final_model` is the last epoch, so when the
curve turned the run advises a re-run with `--epochs k`: with a warning when it ended 10 % or more
above its lowest, as "likely to ship a better model" from 1 % to 10 %, and as optional under 1 %.
When the curve was still falling at the last epoch, more epochs may lower it. A re-run in the same
workspace writes `runs/stage1_run2`, and `evaluate` and `fuse` then read that run.

**`evaluate`** reads the stage on both axes against the model it started from. The before and
after columns of the walkthrough's stage 1 (Qwen3-0.6B on Darwin, a 4-epoch demo where the recipe
trains 15; 200 WikiText-2 windows; recorded 2026-10-06):

```
| metric               | before | after |     Δ% |
| -------------------- | -----: | ----: | -----: |
| general (WikiText-2) |  17.80 | 16.08 |  -9.7% |
| domain               |  30.12 | 19.30 | -35.9% |
```

Lower is better. **The domain number should fall and WikiText-2 should hold.** A WikiText-2
number below the base model's, as here, is not by itself evidence that anything was kept
([concepts.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/concepts.md#reading-a-run-two-axes-never-one)).
Beneath the table, `evaluate` repeats the run's own held-out verdict in the words `train` ended
with (`This run's held-out curve: lowest X at epoch k of n; …` and its advice).
`--compare-unanchored` adds the λ = μ = 0 control as a third column, at the cost of a second
training run. It trains at the anchored run's epochs; when its own curve turned earlier, part of
its gap is dose, and `evaluate` prints the commands that train it at its own best epoch.
`--n-windows none` skips the general axis offline (it reads WikiText-2 from the Hub).

**`fuse`** writes a plain checkpoint with the adapter merged in; it loads with
`AutoModelForCausalLM.from_pretrained` like any other model, and
[quickstart.md §6](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/quickstart.md#6-fuse)
asks it a question (the chat template, Qwen3's thinking switched off, greedy decoding).

The same thing from Python, which the CLI calls into and decides nothing differently from:

```python
from lfa import Workspace
from lfa.prepare_domain import prepare_domain
from lfa.supplements import prepare_supplement

ws = Workspace.init("runs/world_history", "Qwen/Qwen3-0.6B", artifact="self-generated")
prepare_domain(["history"], "data/world_history")           # split_chars=3500 for one long file
prepare_supplement("data/world_history", "Qwen/Qwen3-0.6B")  # optional: train writes it otherwise
ws.train("data/world_history")        # ends with the held-out curve's verdict
print(ws.evaluate()["table"])         # both axes, against the model the stage started from
ws.fuse()                             # a plain checkpoint: AutoModelForCausalLM.from_pretrained
```

Every step, with what it costs and what can go wrong:
[docs/quickstart.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/quickstart.md); the
Python flow as a script:
[`examples/quickstart.py`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/quickstart.py).

## Tuning on your own corpus

The recipe's 15 epochs and λ = 1,000,000 were calibrated on one text.
[docs/tuning.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/tuning.md) is how to set
the epochs from the held-out curve, read the control at its own dose, try another λ with `lfa
train --lambda`, and tell a real difference from run-to-run noise, each step with its command and
its cost on one card.

## Models

| model (`--model`) | recipe | licence | artifact download | GPU memory to train |
|---|---|---|---:|---:|
| `Qwen/Qwen3-0.6B` | `qwen3-0.6b` | Apache-2.0 | 132 MB | 14.2 GiB |
| `Qwen/Qwen3-1.7B` | `qwen3-1.7b` | Apache-2.0 | 265 MB | 18.8 GiB |

`--model` picks the recipe: `init` uses the bundled recipe whose own model id is the one you pass.
Both are λ = 1,000,000, 15 epochs, rank 32. The download is the artifact with the corpus it was
fitted on. GPU memory is the highest `nvidia-smi` reading over a two-stage chain on an RTX 3090, an
upper bound on what a run needs; a single stage read 13.6 GiB (Qwen3-0.6B, an unanchored 4-epoch
run) and 12.7 GiB (Qwen3-1.7B, anchored), a caching effect of the measure that does not order the
two models by size, and a Qwen3-1.7B chain has only been run on 24 GB cards. Per run, and the
settings for an 8 GB card: [the
FAQ](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/faq.md#how-much-gpu-memory-does-a-run-need).
The licence is the model's own, from its Hub card. Any other model:
[docs/model-integration-cookbook.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/model-integration-cookbook.md),
with Qwen3-1.7B worked through.

## A second domain, and a chain

One extra step between domains. `extend` merges the finished stage into the model and folds the
domain's activations into `p(h)` as a sample-weighted mixture union — exact, no refit, and it reads
only the *new* domain — so the next stage adapts the right model and anchors against a `p(h)`
that describes it.

```bash
lfa extend   --workspace runs/world_history
lfa train    --workspace runs/world_history --corpus data/second_domain
lfa evaluate --workspace runs/world_history
lfa fuse     --workspace runs/world_history
```

```python
ws = Workspace.open("runs/world_history")
ws.extend()
ws.train("data/second_domain"); print(ws.evaluate()["table"]); ws.fuse()
```

A whole sequence is one YAML file and one command; every domain is folded in before the next
starts, and every entry is checked before the first one trains:

```bash
lfa chain domains.yaml --workspace runs/chain
```
```yaml
artifact: extend                                              # or regenerate: refit p(h) per stage
domains:
  - {name: philosophy,   corpus: data/domain_a, epochs: 15}   # paths resolve against this file
  - {name: archaeology,  corpus: data/domain_c}
```

λ from stage 2 on is the recipe's value times its `stage2_lambda_multiplier`. A chain spec takes no
per-domain λ; a single stage takes one with `lfa train --lambda`
([recipes.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/recipes.md#trying-another-λ)).
Details:
[docs/multi-domain-chains.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/multi-domain-chains.md).

## The walkthrough notebooks

**To watch it work rather than read about it**, open
[`examples/two_domain_walkthrough.ipynb`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/two_domain_walkthrough.ipynb):
Qwen3-0.6B adapted to Darwin, then to an 1853 American cookbook, with every stage repeated with the
anchor off so the control sits beside each number. Across stage 2 the anchored model's Darwin
perplexity moves 19.30 → 19.97 while the unanchored one's goes to 37.67, having read no Darwin
either way. Recorded 2026-10-06 on one RTX 3090 with the self-generated artifact at the recorded
frame (2,500 documents × 2,048 tokens, K = 32) and the supplement on; one seed, one run per cell.
35.5 minutes of training and tables on that card once the artifact is in the store, and it
downloads what it needs and needs no API key.

A second notebook is optional and continues from the workspace the first leaves behind:
[`examples/what_the_anchor_does.ipynb`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/what_the_anchor_does.ipynb)
re-runs each control at its own best epoch count, so the gaps above can be split into what is
dose and what is anchor — at its own dose each control fits its new domain more closely than the
anchored run and keeps less of the rest — and then puts the three models to fixed probes, whose
answers, at this scale, split: on the first ten the control's answer is the better one five times
(twice on a general question) and the anchored model's once, and on the eight transfer probes the
anchored model's is the nearer to the question on all three adjacent ones and the nearer to the base
model's on two of the three inward ones; the other three do not separate the arms.

## Documentation

| | |
|---|---|
| [docs/quickstart.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/quickstart.md) | the full pipeline, step by step |
| [docs/preparing-your-data.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/preparing-your-data.md) | formats, cleaning, splitting a long file, corpus shapes, the supplement |
| [docs/tuning.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/tuning.md) | tuning the epochs and λ on your own corpus |
| [docs/the-artifact.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/the-artifact.md) | the self-generated build, the store, published artifacts, and a real-text artifact |
| [docs/concepts.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/concepts.md) | the building blocks, what the anchor does, what λ and μ are, how a run is read |
| [docs/recipes.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/recipes.md) | the shipped operating point field by field, its couplings, and trying another λ (`--lambda`) |
| [docs/multi-domain-chains.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/multi-domain-chains.md) | second and third domains; what `extend` does, and the `regenerate` route |
| [docs/model-integration-cookbook.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/model-integration-cookbook.md) | a model with no bundled recipe: adapter, artifact, probe, λ |
| [docs/faq.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/faq.md) | GPU memory (8 GB cards included), self-generation cost, the teacher, full weights, reading the general axis, what is not shipped |
| [`examples/two_domain_walkthrough.ipynb`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/two_domain_walkthrough.ipynb) | the runnable how-to: two domains one after the other, each stage repeated with the anchor off |
| [`examples/what_the_anchor_does.ipynb`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/what_the_anchor_does.ipynb) | optional, continues from it: the controls at their own best dose, and what the models say |

## Tests

From the checkout, with the test tools installed (`pip install -c constraints-tested.txt -e
'.[dev,html]'`):

```bash
pytest -q                                    # no GPU, no corpus, no network
pytest tests/test_gpu_smoke.py -m gpu -q     # one stage on the card
```

CI runs the first on Python 3.11, 3.12 and 3.13 on every push to `main` and every pull request.
[CONTRIBUTING.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/CONTRIBUTING.md) lists every
tier and what a change needs to pass.

## Provenance

This package is the method behind the [LFA paper](https://openreview.net/forum?id=68rQ2UBOOC), ported from the research code and checked
against it: the sampler's draws replayed bit-for-bit, every anchor block bit-identical, and one
full run at the paper's operating point agreeing with the research run on every deterministic series
(per-epoch content loss within 0.191 %). [docs/verification.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/verification.md)
is the report, with the one deliberate divergence since (the corpus loader's per-epoch chunking)
and what it changes. Agreement with another implementation is not correctness, and nothing here
reproduces a published number: the paper's headline (domain perplexity 8.76 on Qwen3-0.6B at a
seed ΔPPL of −10.0 %) is the paper's measurement on the paper's corpus and instruments.

The paper's Qwen3-0.6B point was tuned on the research corpus against an artifact fitted on real
text; an artifact fitted on the model's own text, which this package builds instead, matched it
there at every lambda tried — one model, one seed, one domain. Both bundled recipes' lambda was
calibrated in this package, each against its own model's self-generated artifact, on the
walkthrough's Darwin text
([docs/recipes.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/recipes.md)).

The research code — every arm, ladder and retraction — is private and is not distributed. This is
what survived, ported, tested and documented.

## Citing

*Layerwise Function Anchoring: Preserving Sub-Module Functions on Sampled Hidden States for
Continual Domain Adaptation* — the LFA paper, submitted to ICLR 2027; authors anonymous during
review. On OpenReview: <https://openreview.net/forum?id=68rQ2UBOOC>.

Licensed under Apache-2.0 (see [LICENSE](https://github.com/sparkdoc/lfa-anchoring/blob/main/LICENSE)).
