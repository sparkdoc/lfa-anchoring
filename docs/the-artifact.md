# The p(h) artifact

The artifact is the only part of LFA that is not data-free: it is fitted once on text, and every
adaptation afterwards samples hidden states from it instead of from text. This package builds it
from text the model writes itself, so nothing is downloaded. You build one per **model**
([adding-a-model.md](adding-a-model.md) for a model that is not Qwen3), not per domain — a new
domain is what [`lfa extend`](multi-domain-chains.md) is for.

## The self-generated artifact

```bash
lfa init runs/my_domain --model Qwen/Qwen3-0.6B --artifact self-generated
lfa build-artifact --model Qwen/Qwen3-0.6B --self-generated --out artifacts/selfgen.pt
```

The first builds into the local store and copies the result into a new workspace; the second
builds a file at a path of your choosing, outside the store ([below](#a-file-outside-the-store)).
Both write the same artifact at the same frame.

### The frame

The recorded frame — the one the recipe's calibration was measured at, and the defaults of both
commands — is:

* **2,500 raw documents** of up to 2,048 new tokens each, started from the model's document
  boundary (Qwen3's `<|endoftext|>`, its declared `generation_config.bos_token_id`), sampled at
  temperature 1.0 and top-p 1.0, stopped at the next boundary, **unfiltered**, seed 42;
* **no chat-format documents** (`--n-chat 0`): the research launcher intended 250 documents
  started from the bare user-turn header, but the research run's own audit found the fitted corpus
  held none, so the frame is the 2,500 raw documents alone;
* the fit at **600k samples per site**, K = 32, PCA variance 0.95 (`--max-samples` defaults to
  600,000 on this route and to 1,500,000 over a seed corpus).

`top_k` is lifted explicitly (`top_k=0`, with `min_p=0.0` and `repetition_penalty=1.0`): Qwen3's
`generation_config.json` ships `top_k: 20`, so passing only temperature and top-p would sample
top-20 out of a 151,936-token vocabulary while looking untruncated.

The bundled recipe records this frame as `self_generated_frame` ([recipes.md](recipes.md)).
`--n-raw`, `--n-chat` and `--max-new-tokens` (both commands) and `--max-samples`, `--gmm-k` and
`--pca-variance` (`build-artifact`) change it — usually to scale it down for a trial — and the
result is then not the recorded frame: `train` warns, naming each field that differs from the recipe's.

`--n-chat N` adds N documents started from the bare user-turn header of the model's chat
template (`<|im_start|>user\n` for Qwen3, read from the template rather than hard-coded), header
kept, seed 43. It is an option outside the recorded frame, and what it does to the artifact is
unmeasured. A model without a chat template gets no chat-format share even when `--n-chat` asks
for one; the log says so.

**What it is worth.** On Qwen3-0.6B an artifact fitted on the model's own text at this frame
matched an artifact fitted on real text at every λ tried, and was at least as good at the recipe's
λ — one model, one seed, one domain. That real-text artifact is the one the recipe's λ was first
tuned against ([below](#advanced-an-artifact-fitted-on-real-text)); the recipe is now calibrated
against the self-generated one at this frame, and warns about nothing when a Qwen3-0.6B workspace
uses it. On any other model nothing has been measured: the route gives a first artifact, and λ is
calibrated against it ([adding-a-model.md](adding-a-model.md)).

### What it writes

The corpus is written as `{"text", "source"}` rows, `source` being `selfgen_raw` (or
`selfgen_chatfmt` for the optional chat-format share), with `<corpus>.manifest.json` beside it:
the writer's model id and checkpoint sha256, the seed prefix and chat header used, the generation
frame, the decoding settings, the counts (raw, chat, empty), the corpus sha256 and the `lfa`
version. In a workspace the corpus sits beside the artifact as `artifacts/v1.corpus.jsonl`; from
`build-artifact`, it is the `--out` path with its suffix replaced (`artifacts/selfgen.corpus.jsonl`
above).

The artifact's meta records `provenance: "self-generated"`, that `corpus_sha256`, the frame it was
built at (`selfgen_frame`, which is what `train` compares with the recipe's
`self_generated_frame`), and the `layer_group_size` the fit used. A corpus with fewer than 50
non-empty documents, or with more than 20 % of the documents asked for coming out empty, is
refused before any fit: an artifact fitted on it would fail nowhere downstream.

### What it costs

About 3 h 40 min for the full frame from a cold store, on one RTX 3090 (24 GB, 2026-09-30): about
80 minutes of generation, then 2 h 22 min of fitting — 22 min collecting hidden states and about
2 h 00 min fitting the mixtures on the GPU — on a host with 125 GiB of RAM. The artifact file is
110.3 MB, and the store entry, with its corpus, 126 MB. An 8 GB card has not been measured at this
frame; [faq.md](faq.md#how-long-does-self-generation-take) has the smaller pieces timed on one.
The generation survives an interruption ([below](#durability-and-resume)), so hours already spent
are not lost.

The GPU side is otherwise undemanding — the model is loaded in **float32** for the fit,
deliberately: the artifact is a second-moment estimate and bf16's 8-bit mantissa is a large error
on a covariance.

### Host RAM: the layer group

The fit's bill is the **reservoirs** — the raw vectors retained per site for the mixture fit — at
`reservoir_size × D × itemsize`, where `D` is that site's own width. On Qwen3-0.6B (28 layers,
hidden 1024, `pre_o` 2048) at the default 200,000-vector reservoir in fp16:

| | |
|---|---|
| 56 sites of width 1024 | 0.41 GB each |
| 28 `pre_o` sites of width 2048 | 0.82 GB each |
| **all at once (`--layer-group-size` unset over a seed corpus)** | **~46 GB** (~92 GB in fp32) |
| **`--layer-group-size 7`** | **~12 GB per group**, in four corpus passes |

The float64 covariance accumulators add `D² × 8` bytes per site (8.39 MB at width 1024 and
33.55 MB at 2048, so 1.41 GB across all 84 sites of that model), which is small beside the
reservoirs but not nothing.

On the self-generated route no value is needed: the build chooses the group from the model's
config and the host RAM available when it starts, and logs the choice. On a 28-layer Qwen3-0.6B at
the 200,000-vector reservoir that was 7 (about 10.7 GiB of reservoirs per group against about
24 GiB available) on the machine this was written on, and a host with less free memory chooses a
smaller group. On the host that built the recorded artifact (125 GiB of RAM, 2026-09-30) it was
28, every layer in one pass. It logged
`layer_group_size=28 for Qwen/Qwen3-0.6B: ~42.7 GiB of reservoirs per group against 117.9 GiB available`,
and the collection that followed logged `Collected 84 sites, 603973 samples at the thinnest site`.

**The group size is part of the build, not only of its memory bill.** One torch generator is
shared across a group's sites, so grouping changes the reservoir draws and therefore the fitted
mixtures — not the exact moments (mean, covariance, basis), which come from sums over every vector.
Two builds at the same seed and different group sizes are two different artifacts. The artifact's
meta records the value used as `layer_group_size`, and a rebuild reproduces a file by passing
`--layer-group-size` with that recorded value (`build-artifact`). The store's key leaves the group
size out: an entry is reused with whatever group its build chose.

### Durability and resume

The corpus is written batch by batch. Each finished batch's documents are appended to
`<corpus>.partial` and `<corpus>.progress.json` is rewritten (to a temporary file, then renamed)
with how far each share got — its batch index, the documents kept and the empties — and the frame
and writer it is building for. In a store entry these are `corpus.jsonl.partial` and
`corpus.jsonl.progress.json`. Every batch logs one line:

```
self-generated corpus: <n>/2500 documents (<e> empty)
```

* **Stopped part-way** (Ctrl-C, a crash, a reboot): run the same command again. Generation resumes
  at the next batch of each share, with the batch size the build started with and the same
  per-batch seed, and the empties already counted still count against the limit, so the refusal
  above applies to the whole build and not to the resumed tail. Rows appended after the last
  progress record are dropped on resume, so nothing is duplicated. Ctrl-C says that the same
  command resumes the build, and exits 130; under `lfa init` it also names the store entry
  that holds the partial build (the path `lfa list-artifacts` shows).
* **Finished**: the partial file is renamed to the corpus, the manifest written, and the progress
  file removed. The corpus hash is computed over the final file, so a resumed build's manifest is
  indistinguishable in form from an uninterrupted one's.
* **The fit fails or is interrupted after the corpus is complete**: the corpus is kept, and the
  next run fits it without generating again.
* **A different frame or writer** over a partial build is refused, naming `--rebuild` (in the
  store) or the files to delete (beside `--out`).

### The store

`lfa init --artifact self-generated` builds in the local store: `~/.cache/lfa/artifacts`, or
`$LFA_ARTIFACT_STORE` when it is set. An entry is a directory named for the model, its checkpoint
and the frame,

```
<model-slug>-<writer_sha256[:12]>-<frame_sha256[:12]>/
    artifact.pt
    corpus.jsonl  corpus.jsonl.manifest.json
    entry.json                                     # model id, frame, documents asked for
    corpus.jsonl.partial  corpus.jsonl.progress.json  .lock    # only while a build runs
```

`writer_sha256` hashes the checkpoint's weights, so an entry follows the model's weights rather
than its name. `frame_sha256` hashes the fields that shape the corpus and the fit (the document
counts and tokens, the seeds and filters, `max_samples`, `gmm_k`, `pca_variance`, the reservoir
size) and not the ones that change only speed or the host-RAM plan (the batch size, the device,
the layer group). A trial build is therefore its own entry and never stands in for the real one.

What `init` does with the entry:

* **finished**: copies `artifact.pt` in as `artifacts/v1.pt`, with the corpus and manifest beside
  it, and logs `Reused the self-generated artifact built <date> from <entry>`;
* **a complete corpus but no artifact** (an earlier fit failed): fits, then copies in;
* **a partial corpus**: resumes generation, fits, copies in;
* **nothing**: builds into the store, then copies in.

`lfa list-artifacts` prints one line per entry: the model, the frame (documents × tokens, K), its
state (`built`, `corpus complete, not fitted`, or `in progress: n/N documents`), the date it was
built, its size on disk and its path.

**Locking.** A build holds `<entry>/.lock`, which records its process id. A second build of the
same entry — the same `lfa init` in another terminal — is refused with a message naming the lock
and that process, and saying to run the same command again once it has finished. A lock whose
process is no longer running is taken over with a warning, and that build resumes. The store root
also holds `.store.lock`, which only serialises taking a lock.

**`--rebuild`** builds afresh even when the store has a match. The existing entry is moved aside to
`<entry>.replaced-<timestamp>/` — never deleted — and `lfa list-artifacts` leaves moved-aside
entries out. It is refused while a live build holds the entry. Delete a moved-aside directory
yourself when you no longer want it.

### Reusing a file

```bash
lfa init runs/second --model Qwen/Qwen3-0.6B --artifact runs/my_domain/artifacts/v1.pt
```

`--artifact` takes any artifact file and copies it in as `artifacts/v1.pt`. What the workspace
records comes from the file's meta: a self-generated file keeps its provenance and its id
(`self-generated:<corpus sha256[:12]>`), exactly as if it had been built at `init`, so the recipe
judges it by its frame and `extend` works. Any other file is recorded by the path you passed, with
no provenance, and the recipe notes that it is not the artifact its λ is calibrated against.

### A file outside the store

`lfa build-artifact --model <id> --self-generated --out <path>` writes the artifact at `--out` and
its corpus beside it, and keeps nothing in the store. It is durable the same way: its partial
corpus and progress file sit beside `--out` (`artifacts/selfgen.corpus.jsonl.partial` and
`artifacts/selfgen.corpus.jsonl.progress.json` above), and running the same command again resumes
it, or fits a complete corpus without generating. Nothing locks it, so do not run two builds with
the same `--out`. `lfa init --artifact <path>` then puts the file in a workspace.

### What the fit stores

`build-artifact` runs the model over the corpus, records what arrives at each anchoring site, and
fits it. What is stored per site: the mean, a PCA basis spanning `--pca-variance` (0.95) of the
variance with its eigenvalues, and a `--gmm-k` (32) component mixture fitted in that basis, plus
per-dimension standard deviations — which the sampler uses for the **off-basis residual** it adds
back on every draw, so the sampled marginals are right rather than short.

What is deliberately *not* stored: the layer-0 `pre_qkv` lookup table. It is
`input_layernorm(embed_tokens(id))`, exactly reconstructible from the model's own weights, so only
the corpus **token frequencies** are kept (~600 KB against ~300 MB) and
`Sampler.build_embedding_lookup_from_model` rebuilds the table at load time. Layer 0 is then
sampled frequency-weighted.

The build also writes a `__meta__` block — model id, hidden size, layer count, site list, and
`n_samples_total` — and validates the finished artifact against the model before saving. Every
training run validates it again. `n_samples_total` is a **per-site** count, not a sum across sites:
every site sees the same token stream, and this is the number a later extension reads back as each
block's own count.

`build-artifact` writes blockwise-int8 by default (`--no-quantize` for full precision, which
doubles the file). Quantization is a storage format: it is applied to a shallow copy on save and
reconstructed on load, so nothing downstream knows whether the file was quantized. The real-text
Qwen3-0.6B artifact the recipe's λ was tuned against is ~108 MB int8 against ~226 MB in fp16.

⚠ **The fp16 reservoir has no range guard.** An activation above 65,504 would be stored as `inf`
and poison that site's fit. Nothing checks for it. If a model is suspected of large activations,
build with `dtype=torch.float32` (via `lfa.artifact.build.build_artifact`, which takes it as an
argument) and check the site statistics in the logs.

## Advanced: an artifact fitted on real text

The route the recipe's λ was first tuned on: a downloaded seed corpus, then the same fit.

```bash
lfa prepare-seed-corpus --out data/seed_corpus_10to1.jsonl
lfa build-artifact --model Qwen/Qwen3-0.6B \
                   --corpus data/seed_corpus_10to1.jsonl \
                   --out data/distributions/qwen3-0.6b/distribution_stats.pt \
                   --layer-group-size 7
```

Over a seed corpus nothing chooses the layer group for you: `--layer-group-size` is normally set,
and 7 is the value for a 28-layer model on a 64 GB host ([the arithmetic](#host-ram-the-layer-group)).
This step needs network. There is no offline path: the corpus is a download.

**What the recipe says about it.** The bundled recipe is calibrated against the self-generated
artifact at its frame, so a workspace over an artifact fitted here is told, at every stage, that
it is not the artifact the recipe's λ is calibrated against and that λ should be calibrated
against held-out domain perplexity ([adding-a-model.md](adding-a-model.md)). It is a note, never a
refusal. Put the file in a workspace with `lfa init --artifact <path>`.

### The seed corpus

10 : 1 pretraining : instruction, written as one JSONL with two row shapes — `{"text", "source"}`
and `{"prompt", "response", "source"}`. The instruction rows are kept as a pair rather than joined
because the artifact build renders them with the model's chat template: the hidden states of a
formatted exchange are not those of the same words run together.

Pretraining comes from RedPajama filtered into six sources; instruction from five datasets. Two
levels of number are involved and they are not the same:

* **Download targets** — `--n-pretraining 12000` (a per-source cap of 2,000) and
  `--n-instruction 20000`. These are what you ask for.
* **The realised corpus** — the composition the recipe's λ was tuned on, which that real-text
  artifact (1,543,040 vectors per site) was estimated from: **9,663 pretraining documents and 966
  instruction pairs**, which is `SHIPPED_COMPOSITION` in `lfa/seed_corpus.py`:

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
per-source counts. Diff it against that composition:

```python
import json
from lfa.seed_corpus import SHIPPED_COMPOSITION

stats = json.load(open("data/seed_corpus_10to1.stats.json"))
shipped = {**SHIPPED_COMPOSITION["pretraining"], **SHIPPED_COMPOSITION["instruction"]}
for source, count in sorted(stats["source_distribution"].items()):
    print(f"{source:<26} {count:>6}   tuned on {shipped.get(source, 0):>6}")
```

**What to expect, and what not to.** A rebuild does not reproduce that corpus byte for byte — the
upstream datasets move and the sampling RNG is this module's own — and it need not reproduce the
per-source counts either. The scan is a head scan: each source collects candidates from the start
of the split until it has three times its allocation, so which sources come up short depends on
how the dataset is ordered *today*. A streaming spot-check of the first 2,000 rows on 2026-09-07
found 1,510 arXiv, 306 web, 31 book, 1 Wikipedia and no GitHub or StackExchange rows — an
arXiv-dense head, which is not the shape that produced the table above. So check that every source
is *present* and that the mix is roughly 10 : 1; treat a source at zero as a real failure (a
predicate that no longer matches the upstream metadata) and a moved count as expected drift.

### One scan, not six

`download_pretraining` walks the split **once**, building all six shortlists together, because a
RedPajama row is decoded on access and the six predicates are cheap beside that. Each source still
stops collecting at three times its allocation and the shortlists are still built in ascending
index order, so the rows are exactly the ones six separate passes gave — only the reading changed.
`tests/test_prepare.py::test_download_pretraining_scans_once_and_keeps_the_six_pass_result` pins
both halves of that.

### One known difference from the research code

The research code accumulates its running variance with `old_mean` computed **after** the batch has
already been folded into the running sum (`self.sum_x +=` and only then
`old_mean = self.sum_x / self.n`), so its `std` is slightly off. This package's accumulator takes
`old_mean` before folding, which is the correct Chan update. The consequence is small but real: the
real-text artifact the recipe's λ was tuned against was built by the research code, so an artifact
built here over the same corpus has slightly different `std` values, and `std` enters the sample
stream through the off-basis residual term. It is a correction, not a divergence — but it is why
an artifact built here is not bit-identical to one the research code built, and why a λ
calibrated against one should be re-read against the other.

The `mean`, the PCA basis and the mixture are unaffected: those come from the sums, not from the
running variance.
