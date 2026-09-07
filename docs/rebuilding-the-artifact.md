# Rebuilding the p(h) artifact

The artifact is the only large thing this package ships, and the only part of LFA that is not
data-free: it is built once from a general-purpose seed corpus, and every adaptation afterwards
samples from it instead of from text. You need to rebuild it for a **new model**
([adding-a-model.md](adding-a-model.md)), or if you want an artifact estimated on a different
corpus. You do not need to rebuild it for a new domain — that is what
[`lfa extend`](multi-domain-chains.md) is for.

Two commands:

```bash
lfa prepare-seed-corpus --out data/seed_corpus_10to1.jsonl
lfa build-artifact --model Qwen/Qwen3-0.6B \
                   --corpus data/seed_corpus_10to1.jsonl \
                   --out data/distributions/qwen3-0.6b/distribution_stats.pt \
                   --layer-group-size 7
```

## The seed corpus

10 : 1 pretraining : instruction, written as one JSONL with two row shapes — `{"text", "source"}`
and `{"prompt", "response", "source"}`. The instruction rows are kept as a pair rather than joined
because the artifact build renders them with the model's chat template: the hidden states of a
formatted exchange are not those of the same words run together.

Pretraining comes from RedPajama filtered into six sources; instruction from five datasets. Two
levels of number are involved and they are not the same:

* **Download targets** — `--n-pretraining 12000` (a per-source cap of 2,000) and
  `--n-instruction 20000`. These are what you ask for.
* **The realised corpus** — what the shipped `gmm1543k` artifact was actually estimated on:
  **9,663 pretraining documents and 966 instruction pairs**, which is `SHIPPED_COMPOSITION` in
  `lfa/seed_corpus.py`:

  | pretraining | | instruction | |
  |---|---:|---|---:|
  | `redpajama_stackexchange` | 2,000 | `alpaca` | 231 |
  | `redpajama_web` | 1,999 | `oasst2` | 224 |
  | `redpajama_book` | 1,998 | `ultrachat` | 212 |
  | `redpajama_wikipedia` | 1,995 | `dolly` | 203 |
  | `redpajama_arxiv` | 1,524 | `code_alpaca` | 96 |
  | `redpajama_github` | 147 | | |

  Two sources came up short of the 2,000 cap when that corpus was built, and the 10 : 1 mix then
  trimmed ~20,000 available instruction pairs down to the 966 the ratio allows.

### Checking a rebuild

`prepare-seed-corpus` writes a `.stats.json` sidecar next to the corpus with the realised
per-source counts. Diff it against the record:

```python
import json
from lfa.seed_corpus import SHIPPED_COMPOSITION

stats = json.load(open("data/seed_corpus_10to1.stats.json"))
shipped = {**SHIPPED_COMPOSITION["pretraining"], **SHIPPED_COMPOSITION["instruction"]}
for source, count in sorted(stats["source_distribution"].items()):
    print(f"{source:<26} {count:>6}   shipped {shipped.get(source, 0):>6}")
```

**What to expect, and what not to.** A rebuild does not reproduce the shipped corpus byte for byte
— the upstream datasets move and the sampling RNG is this module's own — and it need not reproduce
the per-source counts either. The scan is a head scan: each source collects candidates from the
start of the split until it has three times its allocation, so which sources come up short depends
on how the dataset is ordered *today*. A streaming spot-check of the first 2,000 rows on 2026-09-07
found 1,510 arXiv, 306 web, 31 book, 1 Wikipedia and no GitHub or StackExchange rows — an
arXiv-dense head, which is not the shape that produced the table above. So check that every source
is *present* and that the mix is roughly 10 : 1; treat a source at zero as a real failure (a
predicate that no longer matches the upstream metadata) and a moved count as expected drift.

This step needs network. There is no offline path: the corpus is a download.

### One scan, not six

`download_pretraining` walks the split **once**, building all six shortlists together, because a
RedPajama row is decoded on access and the six predicates are cheap beside that. Each source still
stops collecting at three times its allocation and the shortlists are still built in ascending
index order, so the rows are exactly the ones six separate passes gave — only the reading changed.
`tests/test_prepare.py::test_download_pretraining_scans_once_and_keeps_the_six_pass_result` pins
both halves of that.

## The build

`lfa build-artifact` runs the model over the corpus, records what arrives at each anchoring site,
and fits it. What is stored per site: the mean, a PCA basis spanning `--pca-variance` (0.95) of the
variance with its eigenvalues, and a `--gmm-k` (32) component mixture fitted in that basis, plus
per-dimension standard deviations — which the sampler uses for the **off-basis residual** it adds
back on every draw, so the sampled marginals are right rather than short.

What is deliberately *not* stored: the layer-0 `pre_qkv` lookup table. It is
`input_layernorm(embed_tokens(id))`, exactly reconstructible from the model's own weights, so only
the corpus **token frequencies** are kept (~600 KB against ~300 MB) and
`Sampler.build_embedding_lookup_from_model` rebuilds the table at load time.

The build also writes a `__meta__` block — model id, hidden size, layer count, site list, and
`n_samples_total` — and validates the finished artifact against the model before saving. Every
training run validates it again. `n_samples_total` is a **per-site** count, not a sum across sites:
every site sees the same token stream, and this is the number a later extension reads back as each
block's own count (the shipped Qwen3-0.6B artifact: 1,543,040 vectors per site, which is what the
`gmm1543k` in its name rounds).

### Memory: host RAM is the binding constraint

The bill is the **reservoirs** — the raw vectors retained per site for the mixture fit — at
`reservoir_size × D × itemsize`, where `D` is that site's own width. On Qwen3-0.6B (28 layers,
hidden 1024, `pre_o` 2048) at the default 200,000-vector reservoir in fp16:

| | |
|---|---|
| 56 sites of width 1024 | 0.41 GB each |
| 28 `pre_o` sites of width 2048 | 0.82 GB each |
| **all at once (`--layer-group-size` unset)** | **~46 GB** (~92 GB in fp32) |
| **`--layer-group-size 7`** | **~12 GB per group**, in four corpus passes |

So `--layer-group-size` is normally set; 7 is the value for a 28-layer model on a 64 GB host. The
float64 covariance accumulators add `D² × 8` bytes per site (~1.6 GB across all 84 sites of that
model), which is small beside the reservoirs but not nothing.

The GPU side is undemanding — the model is loaded in **float32**, deliberately: the artifact is a
second-moment estimate and bf16's 8-bit mantissa is a large error on a covariance.

⚠ **The fp16 reservoir has no range guard.** An activation above 65,504 would be stored as `inf`
and poison that site's fit. Nothing checks for it. If a model is suspected of large activations,
build with `dtype=torch.float32` (via `lfa.artifact.build.build_artifact`, which takes it as an
argument) and check the site statistics in the logs.

### Quantization

`build-artifact` writes blockwise-int8 by default (`--no-quantize` for full precision, which
doubles the file). Quantization is a storage format: it is applied to a shallow copy on save and
reconstructed on load, so nothing downstream knows whether the file was quantized. The shipped
Qwen3-0.6B artifact is ~108 MB int8 against ~226 MB in fp16.

## One known difference from the research code

`the research code` accumulates its running variance with `old_mean` computed **after** the batch has
already been folded into the running sum (`src/lra_distribution.py:1227-1240`: `self.sum_x +=` then
`old_mean = self.sum_x / self.n`), so its `std` is slightly off. This package's accumulator takes
`old_mean` before folding, which is the correct Chan update. The consequence is small but real: an
artifact built here has slightly different `std` values from the shipped one, and `std` enters the
sample stream through the off-basis residual term. It is a correction, not a divergence — but it is
why a locally-built artifact is not bit-identical to the shipped file, and why a λ calibrated
against one should be re-read against the other.

The `mean`, the PCA basis and the mixture are unaffected: those come from the sums, not from the
running variance.
