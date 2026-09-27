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
(`--teacher-mode`; full-weight training moves the base, so there a real teacher is loaded). The statistic is a per-site mean, covariance and mixture — no text, no token ids,
nothing sequence-shaped. Everything runs locally: no judge, no API key.

**To watch it work rather than read about it**, open
[`examples/two_domain_walkthrough.ipynb`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/two_domain_walkthrough.ipynb):
Qwen3-0.6B adapted to Darwin, then to a Victorian cookbook, with every stage repeated with the
anchor off so the control sits beside each number. Across stage 2 the anchored model's Darwin
perplexity moves 17.45 → 18.92 while the unanchored one's goes to 31.45, having read no Darwin
either way. Recorded 2026-09-08, before 0.2.0, so on the raw books alone: a re-run today has the
model write and mix in its question-and-answer supplement first, so these are not a 0.2.0 run's.
19 minutes of training and tables on one RTX 3090, and it downloads what it needs and needs no API
key.

A second notebook is optional and continues from the workspace the first leaves behind:
[`examples/what_the_anchor_does.ipynb`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/what_the_anchor_does.ipynb)
re-runs each control at its own best epoch count, so the gaps above can be split into what is
dose and what is anchor, and then puts the three models to fixed probes — where the anchored
model brings natural selection to a question about island species and stops, while the control
drifts into biogeography on a question about Tokyo.

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

## The building blocks

| Block | What it is | Where it lives |
|---|---|---|
| **Model** | Any causal LM the package has an adapter for (Qwen3 today). Referenced by Hub id or path; never copied. | `lfa.adapters` |
| **Artifact** | The `p(h)` statistic for that model: per-site mean, covariance basis and K=32 mixture, int8, ~108 MB. Fetched by id with a checksum, or built from a seed corpus or from the model's own text. | `lfa.artifact` |
| **Recipe** | The tuned operating point (rank, λ, μ, epochs, schedule) *and what it was tuned against*, so a run that changes rank or artifact is told λ no longer means what it meant. | `lfa.Recipe` |
| **Corpus** | A flat directory of `.txt` files. `lfa prepare-domain` makes one from text, Markdown, HTML or PDF. | `lfa.corpus` |
| **Self-generation** | The model writes its own inputs: the seed corpus p(h) is estimated on (`init --artifact self-generated`, no download), the question-and-answer supplement `train` mixes into the domain at the recipe's token fraction, and, in a chain, a fresh p(h) from each stage's model (`artifact: regenerate`). Measured on one model and one seed (the LFA record's C12, C14, C15); the supplement's job is reachability, not protecting skills. | `lfa.selfgen` |
| **Workspace** | The state machine that holds the other four together across domains: which model the next stage adapts, which artifact version it anchors against, and a history entry per stage. | `lfa.Workspace` |
| **Train / Evaluate / Fuse / Extend** | The four operations on a workspace: adapt one domain; read the stage on both axes; export a plain checkpoint; fold the stage into the model *and* into `p(h)` for the next domain (or `regenerate-artifact`: fold it into the model and refit `p(h)` on that model's own text). | `lfa.train`, `lfa.evaluate` |

## Assemble them: one domain

Make the corpus, then four commands.

```bash
lfa prepare-domain ~/papers ~/notes.md --out data/my_domain      # .txt/.md/.html/.pdf -> .txt files

lfa init     runs/my_domain --model Qwen/Qwen3-0.6B --artifact qwen3-0.6b-gmm1543k-int8
lfa train    --workspace runs/my_domain --corpus data/my_domain
lfa evaluate --workspace runs/my_domain
lfa fuse     --workspace runs/my_domain
```

The same thing from Python, which the CLI calls into and decides nothing differently from:

```python
from lfa import Workspace

ws = Workspace.init("runs/my_domain", "Qwen/Qwen3-0.6B", artifact="qwen3-0.6b-gmm1543k-int8")
ws.train("data/my_domain")            # holds a tenth of the documents out; watch that number
print(ws.evaluate()["table"])         # both axes, against the model the stage started from
ws.fuse()                             # a plain checkpoint: AutoModelForCausalLM.from_pretrained
```

`init` records the model and the recipe and copies the artifact in as `artifacts/v1.pt`; it picks
the bundled recipe that names your model. `train` scores the held-out tenth after every epoch, and
warns at the end if that curve turned around — on a small corpus re-run with `--epochs <the
epoch it bottomed at>`. `evaluate` prints:

```
| metric               | before | after |     Δ% |
| -------------------- | -----: | ----: | -----: |
| general (WikiText-2) |  18.18 | 16.69 |  -8.2% |
| domain               |  23.30 | 12.78 | -45.1% |
```

Add `--compare-unanchored` for the λ = μ = 0 control as a third column, and `--n-windows none`
offline (the general axis reads WikiText-2 from the Hub). Full page:
[docs/quickstart.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/quickstart.md);
the Python flow as a script:
[`examples/quickstart.py`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/quickstart.py).

**No artifact download?**
`lfa init runs/my_domain --model Qwen/Qwen3-0.6B --artifact self-generated` has the model write
2,750 documents from its own document boundary and fits p(h) on them (by estimate about two
hours on an 8 GB card, half that on a 3090). On Qwen3-0.6B that artifact tied the published one
at every λ tried, one seed; on any other model it is the way to a first artifact, and λ is then
calibrated against it. `train` also writes the domain's
question-and-answer supplement with the model before training and mixes it in at 0.13 of training
tokens, the frame the shipped λ was tuned at; `--no-supplement` trains on the raw corpus alone and
says so.

## Assemble them: a second domain, and a chain

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

## The artifact

`lfa list-artifacts` shows what is published for Qwen3-0.6B: the recipe artifact
(`qwen3-0.6b-gmm1543k-int8`) and a ~1 MB diagonal one kept as a budget floor — a known-inferior
option that needs its own λ, not a cheaper equivalent. `init --artifact <id>` fetches and verifies
it; `lfa fetch-artifact <id> --dest artifacts/` fetches it once to share between workspaces. To
build one for another model, or to rebuild this one:

```bash
lfa prepare-seed-corpus --out data/seed.jsonl            # the 10:1 pretraining:instruction mix
lfa build-artifact --model <id> --corpus data/seed.jsonl --out artifacts/mine.pt
lfa build-artifact --model <id> --self-generated --out artifacts/mine.pt   # or: no download
```

[docs/rebuilding-the-artifact.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/rebuilding-the-artifact.md)
covers the build; [docs/adding-a-model.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/adding-a-model.md)
covers a model that is not Qwen3 (one adapter class, one artifact, one λ calibration). If a fetch
404s because the repository is still private, pass a local file **with its id** —
`--artifact /path/to/distribution_stats.pt --artifact-id qwen3-0.6b-gmm1543k-int8` — so the
recipe's λ is read against the right artifact; the quickstart explains why both are needed.

## Documentation

| | |
|---|---|
| [docs/quickstart.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/quickstart.md) | install, the corpus, the four commands, what a run costs, what a refusal means |
| [`examples/two_domain_walkthrough.ipynb`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/two_domain_walkthrough.ipynb) | the runnable how-to: two domains one after the other, each stage repeated with the anchor off |
| [`examples/what_the_anchor_does.ipynb`](https://github.com/sparkdoc/lfa-anchoring/blob/main/examples/what_the_anchor_does.ipynb) | optional, continues from it: the controls at their own best dose, and what the models say |
| [docs/concepts.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/concepts.md) | what the anchor does, what λ and μ are, how a run is read |
| [docs/recipes.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/recipes.md) | the shipped operating point field by field, and its couplings |
| [docs/multi-domain-chains.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/multi-domain-chains.md) | second and third domains; what `extend` does, and the `regenerate` route |
| [docs/adding-a-model.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/adding-a-model.md) | a model that is not Qwen3: adapter, artifact, λ |
| [docs/rebuilding-the-artifact.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/rebuilding-the-artifact.md) | the seed corpus, the self-generated route, and the build |
| [docs/faq.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/faq.md) | GPU memory (8 GB cards included), self-generation cost, full weights, reading the general axis, what is not shipped |
| [docs/verification.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/docs/verification.md) | what was checked against the research code, how, and what came out |
| [RELEASING.md](https://github.com/sparkdoc/lfa-anchoring/blob/main/RELEASING.md) | how the artifacts and a tag are cut |

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

The research code — every arm, ladder and retraction — is a private record and is not
distributed. This is what survived, ported, tested and documented.

## Citing

*Layerwise Function Anchoring: Preserving Sub-Module Functions on Sampled Hidden States for
Continual Domain Adaptation* — the LFA paper (2026); authors withheld for review.

Licensed under Apache-2.0 (see [LICENSE](https://github.com/sparkdoc/lfa-anchoring/blob/main/LICENSE)).
