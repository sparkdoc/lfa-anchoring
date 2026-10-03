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
revisited. Under LoRA that frozen teacher is the student's own base — PEFT keeps it frozen, so a
run loads no second copy of the model and reads the teacher out of the student with its adapters
switched off, which is bit-identical to holding a separate one and 1.11 GB cheaper on Qwen3-0.6B
(`--teacher-mode`; full-weight training moves the base, so there a real teacher is loaded). The
statistic is a per-site mean, covariance and mixture — no text, no token ids, nothing
sequence-shaped. Everything runs locally: no judge, no API key.

## Install

```bash
pip install lfa-anchoring                       # once published; see RELEASING.md
pip install -e '.[dev]'                         # from a checkout, with the test tools
```

Add `[html]` or `[pdf]` if your documents arrive in those formats. Needs Python ≥ 3.11 and a CUDA
card. Tested at torch 2.10.0+cu128, transformers 4.57.6, accelerate 1.14.0, peft 0.18.1
(`pip install -c constraints-tested.txt lfa-anchoring` holds to those exactly).

> Some torch paths compile a small CUDA shim on the first kernel launch and need your `python3`'s
> development headers (`python3-dev` + `build-essential` on Debian; uv- and conda-managed
> interpreters ship them). The package's own paths do not, so `lfa` warns once if the headers are
> missing and proceeds.

## The pipeline

```bash
lfa init runs/my_domain --model Qwen/Qwen3-0.6B --artifact self-generated   # once per model
lfa prepare-domain ~/papers ~/notes.md --out data/my_domain \
    --supplement --model Qwen/Qwen3-0.6B                                     # your documents
lfa train    --workspace runs/my_domain --corpus data/my_domain
lfa evaluate --workspace runs/my_domain
lfa fuse     --workspace runs/my_domain
```

**`init`** creates the workspace and puts its p(h) artifact in place: the model writes 2,500
documents of its own and p(h) is fitted on them — about 3 h 40 min on one RTX 3090 (24 GB); an
8 GB card has not been measured. The result
is kept in a local store, so every later workspace over the same model reuses it
(`lfa list-artifacts` shows what is there), and a build that was interrupted resumes when the
same command is run again. When the package pins a published artifact for exactly this model and
frame, `init` downloads and verifies that instead of building, and `--rebuild` builds here anyway
([the artifact](docs/the-artifact.md#published-artifacts)). `--artifact` also takes a path, but
only to an artifact this package built: another workspace's `artifacts/v1.pt`, or what
`lfa build-artifact` wrote.

**`prepare-domain`** turns text, Markdown, HTML or PDF into the corpus, a flat directory of `.txt`
files. With `--supplement`, the model then writes question-and-answer pairs over the corpus, which
make the domain's knowledge answerable when the model is asked about it; they do not protect
skills, which is the anchor's job. Without `--supplement`, `train` writes the same pairs itself
before the first epoch.

**`train`** adapts the model with the anchor on. It holds a tenth of the documents out, scores them
after every epoch, and warns at the end if that curve turned around — on a small corpus re-run with
`--epochs <the epoch it bottomed at>`.

**`evaluate`** reads the stage on both axes against the model it started from:

```
| metric               | before | after |     Δ% |
| -------------------- | -----: | ----: | -----: |
| general (WikiText-2) |  18.18 | 16.69 |  -8.2% |
| domain               |  23.30 | 12.78 | -45.1% |
```

That table is this package's own verification run on Qwen3-0.6B (2026-09-07), which predates 0.2.0
and so trained on the raw corpus alone. Add `--compare-unanchored` for the λ = μ = 0 control as a
third column, and `--n-windows none` offline (the general axis reads WikiText-2 from the Hub).

**`fuse`** writes a plain checkpoint with the adapter merged in; it loads with
`AutoModelForCausalLM.from_pretrained` like any other model.

The same thing from Python, which the CLI calls into and decides nothing differently from:

```python
from lfa import Workspace
from lfa.prepare_domain import prepare_domain
from lfa.supplements import prepare_supplement

ws = Workspace.init("runs/my_domain", "Qwen/Qwen3-0.6B", artifact="self-generated")
prepare_domain(["papers", "notes.md"], "data/my_domain")      # .txt/.md/.html/.pdf -> .txt files
prepare_supplement("data/my_domain", "Qwen/Qwen3-0.6B")       # optional: train writes it otherwise
ws.train("data/my_domain")            # holds a tenth of the documents out; watch that number
print(ws.evaluate()["table"])         # both axes, against the model the stage started from
ws.fuse()                             # a plain checkpoint: AutoModelForCausalLM.from_pretrained
```

Every step, with what it costs and what can go wrong:
[docs/quickstart.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/quickstart.md); the
Python flow as a script:
[`examples/quickstart.py`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/quickstart.py).

## A second domain, and a chain

One extra step between domains. `extend` merges the finished stage into the model and folds the
domain's activations into `p(h)` as a sample-weighted mixture union — exact, no refit, and it reads
only the *new* domain — so the next stage adapts the right model and anchors against a `p(h)`
that describes it.

```bash
lfa extend   --workspace runs/my_domain
lfa train    --workspace runs/my_domain --corpus data/second_domain
lfa evaluate --workspace runs/my_domain
lfa fuse     --workspace runs/my_domain
```

```python
ws = Workspace.open("runs/my_domain")
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

λ from stage 2 on is the recipe's value times its `stage2_lambda_multiplier`; you do not set it
per domain. Details:
[docs/multi-domain-chains.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/multi-domain-chains.md).

## The walkthrough notebooks

**To watch it work rather than read about it**, open
[`examples/two_domain_walkthrough.ipynb`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/two_domain_walkthrough.ipynb):
Qwen3-0.6B adapted to Darwin, then to a Victorian cookbook, with every stage repeated with the
anchor off so the control sits beside each number. Across stage 2 the anchored model's Darwin
perplexity moves 17.07 → 18.60 while the unanchored one's goes to 33.56, having read no Darwin
either way. Recorded 2026-09-30 on one RTX 3090 with the self-generated artifact at the recorded
frame (2,500 documents × 2,048 tokens, K = 32) and the supplement on; one seed, one run per cell.
35.4 minutes of training and tables on that card once the artifact is in the store, and it
downloads what it needs and needs no API key.

A second notebook is optional and continues from the workspace the first leaves behind:
[`examples/what_the_anchor_does.ipynb`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/what_the_anchor_does.ipynb)
re-runs each control at its own best epoch count, so the gaps above can be split into what is
dose and what is anchor — at its own dose each control fits its new domain more closely than the
anchored run and keeps less of the rest — and then puts the three models to fixed probes, whose
answers, at this scale, do not show that difference.

## Documentation

| | |
|---|---|
| [docs/quickstart.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/quickstart.md) | the full pipeline, step by step |
| [docs/preparing-your-data.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/preparing-your-data.md) | formats, cleaning, corpus shapes, the supplement |
| [docs/the-artifact.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/the-artifact.md) | the self-generated build, the store, and a real-text artifact |
| [docs/concepts.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/concepts.md) | the building blocks, what the anchor does, what λ and μ are, how a run is read |
| [docs/recipes.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/recipes.md) | the shipped operating point field by field, and its couplings |
| [docs/multi-domain-chains.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/multi-domain-chains.md) | second and third domains; what `extend` does, and the `regenerate` route |
| [docs/model-integration-cookbook.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/model-integration-cookbook.md) | a model that is not Qwen3: adapter, artifact, λ |
| [docs/faq.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/faq.md) | GPU memory (8 GB cards included), self-generation cost, full weights, reading the general axis, what is not shipped |
| [`examples/two_domain_walkthrough.ipynb`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/two_domain_walkthrough.ipynb) | the runnable how-to: two domains one after the other, each stage repeated with the anchor off |
| [`examples/what_the_anchor_does.ipynb`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/what_the_anchor_does.ipynb) | optional, continues from it: the controls at their own best dose, and what the models say |

## Tests

From a checkout (the wheel ships the package and the examples, not the tests):

```bash
pytest -q                                    # no GPU, no corpus, no network
pytest tests/test_gpu_smoke.py -m gpu -q     # one stage on the card
```

## Provenance

This package is the method behind the LFA paper, ported from the research code and checked
against it: the sampler's draws replayed bit-for-bit, every anchor block bit-identical, and one
full run of the bundled recipe agreeing with the research run on every deterministic series
(per-epoch content loss within 0.191 %). [docs/verification.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/verification.md)
is the report, with the one deliberate divergence since (the corpus loader's per-epoch chunking)
and what it changes. Agreement with another implementation is not correctness, and nothing here
reproduces a published number: the paper's headline (domain perplexity 8.76 on Qwen3-0.6B at a
seed ΔPPL of −10.0 %) is the paper's measurement on the paper's corpus and instruments.

The recipe's lambda was tuned against an artifact fitted on real text; this package builds an
artifact from the model's own text instead, which matched it at every lambda tried — one model,
one seed, one domain.

The research code — every arm, ladder and retraction — is private and is not distributed. This is
what survived, ported, tested and documented.

## Citing

*Layerwise Function Anchoring: Preserving Sub-Module Functions on Sampled Hidden States for
Continual Domain Adaptation* — the LFA paper (2026); authors withheld for review.

Licensed under Apache-2.0 (see [LICENSE](https://github.com/sparkdoc/lfa-anchoring/blob/main/LICENSE)).
