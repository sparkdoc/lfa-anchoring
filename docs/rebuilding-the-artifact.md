# Rebuilding the p(h) artifact

The artifact is the only large thing this package ships, and the only part of LFA that is not
data-free: it is built once from a general-purpose seed corpus — or from text the model writes
itself ([below](#the-self-generated-route)) — and every adaptation afterwards samples from it
instead of from text. You need to rebuild it for a **new model**
([adding-a-model.md](adding-a-model.md)), or if you want an artifact estimated on a different
corpus. You do not need to rebuild it for a new domain — that is what
[`lfa extend`](multi-domain-chains.md) is for.

Two commands over a downloaded seed corpus (or one, with `--self-generated`, over none):

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
float64 covariance accumulators add `D² × 8` bytes per site (8.39 MB at width 1024 and 33.55 MB at
2048, so 1.41 GB across all 84 sites of that model), which is small beside the reservoirs but not
nothing.

**The group size is part of the build, not only of its memory bill.** One torch generator is
shared across a group's sites, so grouping changes the reservoir draws and therefore the fitted
mixtures — not the exact moments (mean, covariance, basis), which come from sums over every vector.
Two builds at the same seed and different group sizes are two different artifacts. The artifact's
meta records the value used as `layer_group_size`, and a rebuild reproduces a file by passing
`--layer-group-size` with that recorded value. With `--self-generated` and no value given, the
build chooses the group from the model's config and the host RAM available when it starts, and
logs the choice: on a 28-layer Qwen3-0.6B at the 200,000-vector reservoir that was 7 (about
10.7 GiB of reservoirs per group against about 24 GiB available) on the machine this was written
on, and a host with less free memory chooses a smaller group — and so a different artifact.

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

## The self-generated route

```bash
lfa build-artifact --model Qwen/Qwen3-0.6B --self-generated --out artifacts/selfgen.pt
lfa init runs/my_domain --model Qwen/Qwen3-0.6B --artifact self-generated    # the same, into a workspace
```

The model writes the seed corpus itself and nothing is downloaded. The recorded frame — the one
behind the LFA record's C12 artifact, and the defaults of both commands — is:

* **2,500 raw documents** of up to 2,048 new tokens each, started from the model's document
  boundary (Qwen3's `<|endoftext|>`, its declared `generation_config.bos_token_id`), sampled at
  temperature 1.0 and top-p 1.0, stopped at the next boundary, **unfiltered**, seed 42;
* **no chat-format documents** (`--n-chat 0`): the launcher behind C12 intended 250 documents
  started from the bare user-turn header, but the record's own audit found the fitted corpus held
  none, so the frame is the 2,500 raw documents alone;
* the fit at **600k samples per site**, K = 32, PCA variance 0.95 (`--max-samples` defaults to
  600,000 on this route and to 1,500,000 over a seed corpus).

`top_k` is lifted explicitly (`top_k=0`, with `min_p=0.0` and `repetition_penalty=1.0`): Qwen3's
`generation_config.json` ships `top_k: 20`, so passing only temperature and top-p would sample
top-20 out of a 151,936-token vocabulary while looking untruncated.

The corpus is written beside the artifact, the `--out` path with its suffix replaced
(`artifacts/selfgen.corpus.jsonl` above; `artifacts/v1.corpus.jsonl` in a workspace), as `{"text",
"source"}` rows, `source` being `selfgen_raw` (or `selfgen_chatfmt` for the optional chat-format
share below), with `<corpus>.manifest.json` beside it: the writer's model id and checkpoint sha256,
the seed prefix and chat header used, the generation frame, the decoding settings, the counts (raw,
chat, empty), the corpus sha256 and the `lfa` version. The artifact's meta records `provenance:
"self-generated"`, that `corpus_sha256`, and the `layer_group_size` the fit used (see the memory
section above: the group is chosen from host RAM when none is given, and a rebuild passes the
recorded value). A corpus with fewer than 50 non-empty documents, or with more than 20 % of the
documents asked for coming out empty, is refused before any fit: an artifact fitted on it would fail
nowhere downstream. `--n-raw`, `--n-chat`, `--max-new-tokens` and `--max-samples` scale the frame
down for a smoke run (`init` takes the first three), which is then not the recorded frame.

`--n-chat N` adds N documents started from the bare user-turn header of the model's chat
template (`<|im_start|>user\n` for Qwen3, read from the template rather than hard-coded), header
kept, seed 43. It is an option outside the recorded frame, and what it does to the artifact is
unmeasured.

**What it is worth.** On Qwen3-0.6B the self-generated artifact matched the real-corpus artifact
at every λ tried and was at least as good as the published `gmm1543k` one at the recipe's λ — one
model, one seed, one domain (C12). That is why the bundled recipe carries
`calibrated_self_generated: true` and warns about nothing when a Qwen3-0.6B workspace uses one. On
any other model nothing has been measured: the route gives a first artifact, and λ is calibrated
against it ([adding-a-model.md](adding-a-model.md)).

A model without a chat template gets no chat-format share even when `--n-chat` asks for one; the
log says so. What each piece cost on an 8 GB card is in [faq.md](faq.md).

## What a locally-built artifact does not share with the shipped one

The shipped `qwen3-0.6b-gmm1543k-int8` file carries **no `embedding_lookup` entry**: 84 keys, all
of them site statistics. `Sampler.build_embedding_lookup_from_model` reconstructs the table from
the teacher at training time (layer-0 `pre_qkv` is `input_layernorm(embed_tokens(id))`, exact in
the model's own weights), but the *token frequencies* are not reconstructible from a model, so
layer 0 is sampled **uniformly over the vocabulary**. An artifact built here stores the
frequencies it counted while collecting, and samples layer 0 frequency-weighted.

So p(h) differs at that one site between a rebuilt artifact and the shipped one, and it is the
larger of the two differences on this page. Neither is wrong; the shipped behaviour must not be
"corrected" either, because changing which stream layer 0 draws from moves every sample after it
and would break the bit-identity check in `RELEASING.md` step 2 — which is the right outcome for a
change of that size, and the reason it is written down here instead.

## One known difference from the research code

The research code accumulates its running variance with `old_mean` computed **after** the batch has
already been folded into the running sum (`self.sum_x +=` and only then
`old_mean = self.sum_x / self.n`), so its `std` is slightly off. This package's accumulator takes
`old_mean` before folding, which is the correct Chan update. The consequence is small but real: an
artifact built here has slightly different `std` values from the shipped one, and `std` enters the
sample stream through the off-basis residual term. It is a correction, not a divergence — but it is
why a locally-built artifact is not bit-identical to the shipped file, and why a λ calibrated
against one should be re-read against the other.

The `mean`, the PCA basis and the mixture are unaffected: those come from the sums, not from the
running variance.
