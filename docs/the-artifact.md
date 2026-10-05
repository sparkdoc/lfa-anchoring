# The p(h) artifact

The artifact is the only part of LFA that is not data-free: it is fitted once on text, and every
adaptation afterwards samples hidden states from it instead of from text. This package fits it on
text the model writes itself, so no dataset is downloaded. It is made once per **model**
([model-integration-cookbook.md](model-integration-cookbook.md) for a model with no bundled recipe),
not per domain — a new domain is what [`lfa extend`](multi-domain-chains.md) is for — and once made
it can be published: when this package pins a published artifact for your model and frame,
`lfa init` downloads and verifies it instead of building ([below](#published-artifacts)).

## The self-generated artifact

```bash
lfa init runs/my_domain --model Qwen/Qwen3-0.6B --artifact self-generated
lfa build-artifact --model Qwen/Qwen3-0.6B --self-generated --out artifacts/selfgen.pt
```

The first puts the artifact in the local store — downloading a [published](#published-artifacts)
one when the package pins one for this model and frame, building it otherwise — and copies it into
a new workspace; the second always builds, to a path of your choosing, outside the store
([below](#a-file-outside-the-store)). Both give the same artifact at the same frame.

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

Each bundled recipe, `qwen3-0.6b` and `qwen3-1.7b`, records this frame as `self_generated_frame`
([recipes.md](recipes.md)).
`--n-raw`, `--n-chat` and `--max-new-tokens` (both commands) and `--max-samples`, `--gmm-k` and
`--pca-variance` (`build-artifact`) change it — usually to scale it down for a trial — and the
result is then not the recorded frame: `train` warns, naming each field that differs from the recipe's.

`--n-chat N` adds N documents started from the bare user-turn header of the model's chat
template (`<|im_start|>user\n` for Qwen3, read from the template rather than hard-coded), header
kept, seed 43. It is an option outside the recorded frame, and what it does to the artifact is
unmeasured. A model without a chat template gets no chat-format share even when `--n-chat` asks
for one; the log says so.

**What it is worth.** On Qwen3-0.6B an artifact fitted on the model's own text at this frame
matched an artifact fitted on real text at every λ tried, and was at least as good at the paper's
operating point — one model, one seed, one domain. That real-text artifact is the one the paper's
operating point was first tuned against ([below](#advanced-an-artifact-fitted-on-real-text)); the
comparison has been made on that model only. Each bundled recipe is calibrated against its own model's
self-generated artifact at this frame, and warns about nothing when a workspace over that model
uses it. On a model with no bundled recipe the route gives a first artifact, and λ is calibrated
against it ([model-integration-cookbook.md](model-integration-cookbook.md)).

### What it writes

The corpus is written as `{"text", "source"}` rows, `source` being `selfgen_raw` (or
`selfgen_chatfmt` for the optional chat-format share), with `<corpus>.manifest.json` beside it:
the writer's model id and checkpoint sha256, the seed prefix and chat header used, the generation
frame, the decoding settings, the counts (raw, chat, empty), the corpus sha256 and the `lfa`
version. In a workspace the corpus sits beside the artifact as `artifacts/v1.corpus.jsonl`; from
`build-artifact`, it is the `--out` path with its suffix replaced (`artifacts/selfgen.corpus.jsonl`
above).

The corpus is what the model writes from a bare document start, in whatever language it writes.
In the full-frame corpora behind the costs below (counted 2026-10-04), 1,043 of Qwen3-1.7B's 2,500
documents (41.7 %) are mostly CJK — more than 30 % of their characters in U+4E00–U+9FFF — and 154
of Qwen3-0.6B's 2,500 (6.2 %). The artifact describes the hidden states of that text, so it
describes the model as it writes.

The artifact's meta records `provenance: "self-generated"`, that `corpus_sha256`, the frame it was
built at (`selfgen_frame`, which is what `train` compares with the recipe's
`self_generated_frame`), and the `layer_group_size` the fit used. A corpus with fewer than 50
non-empty documents, or with more than 20 % of the documents asked for coming out empty, is
refused before any fit: an artifact fitted on it would fail nowhere downstream.

### What it costs

The full frame from a cold store, on one RTX 3090 (24 GB) in a host with 125 GiB of RAM:

| model | generation | collection and fit | total | layer group (RAM available) | artifact file |
|---|---:|---:|---:|---|---:|
| Qwen3-0.6B | 83 min | 2 h 28 min | 3 h 51 min | 28 of 28 (103.9 GiB) | 110.0 MB |
| Qwen3-1.7B | 87 min | 4 h 29 min | 5 h 56 min | 25 of 28 (115.0 GiB) | 243.7 MB |

Measured 2026-10-03/04, one build per model. The Qwen3-0.6B build ran alone on the host; the
Qwen3-1.7B build shared it for most of its run with a second Qwen3-1.7B build (at 1.5 M samples per
site). The Qwen3-0.6B fit was 27 min collecting hidden states and about 2 h fitting the mixtures.
Qwen3-1.7B's 84 sites are all 2,048 wide, twice the width of most of Qwen3-0.6B's, so its stored
statistics and its file are about twice as large; its group of 25 layers meant two corpus passes
([below](#host-ram-the-layer-group)). An 8 GB card has not been measured at this frame;
[faq.md](faq.md#how-long-does-self-generation-take) has the smaller pieces timed on one. The
generation survives an interruption ([below](#durability-and-resume)), so hours already spent are
not lost. A [published artifact](#published-artifacts) costs only its download: the artifact file,
its corpus and the manifest.

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
config and the host RAM available when it starts — the most layers whose reservoirs fit in half of
it — and logs the choice. On a 28-layer Qwen3-0.6B at the 200,000-vector reservoir that was 7
(about 10.7 GiB of reservoirs per group against about 24 GiB available) on the machine this was
written on, and a host with less free memory chooses a smaller group. On a host with 125 GiB of
RAM it was 28, every layer in one pass, logged as
`layer_group_size=28 for Qwen/Qwen3-0.6B: ~42.7 GiB of reservoirs per group against 103.9 GiB available`
and followed by `Collected 84 sites, 603008 samples at the thinnest site` (2026-10-04).

Qwen3-1.7B's sites are all 2,048 wide, so a layer's reservoirs cost about 2.3 GiB; on that host
the build chose 25 of its 28 layers at 115.0 GiB available (2026-10-03), two corpus passes. All 28
layers in one pass hold about 64 GiB of reservoirs, so the build chooses it only with about 128 GiB
available.

**The group size is a memory choice only.** Every site draws its reservoir from its own torch
generator, seeded from the build seed and the site's layer and name, so a site keeps the same
vectors whichever other layers share its pass. At a fixed seed, any group size gives the same
artifact — the same moments, the same reservoirs, the same fitted mixtures — and a host with less
RAM simply makes more corpus passes. The artifact's meta still records the value used as
`layer_group_size`, for the record. The store's key leaves the group size out, and that is safe:
the group a host chose is not among the things that can make two builds differ.

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

`lfa init --artifact self-generated` keeps its artifacts in the local store:
`~/.cache/lfa/artifacts`, or `$LFA_ARTIFACT_STORE` when it is set. An entry is a directory named
for the model, its checkpoint and the frame,

```
<model-slug>-<writer_sha256[:12]>-<frame_sha256[:12]>/
    artifact.pt
    corpus.jsonl  corpus.jsonl.manifest.json
    entry.json            # model id, frame, documents asked for, provenance (built or published)
    corpus.jsonl.partial  corpus.jsonl.progress.json  .lock    # only while a build runs
    artifact.partial.pt  *.download  .lock                     # only while a download runs
```

`writer_sha256` hashes the checkpoint's weights, so an entry follows the model's weights rather
than its name. `frame_sha256` hashes the fields that shape the corpus and the fit (the document
counts and tokens, the seeds and filters, `max_samples`, `gmm_k`, `pca_variance`, the reservoir
size) and not the ones that change only speed or the host-RAM plan (the batch size, the device,
the layer group). A trial build is therefore its own entry and never stands in for the real one.

What `init` does with the entry:

* **finished**: copies `artifact.pt` in as `artifacts/v1.pt`, with the corpus and manifest beside
  it, and logs `Reused the self-generated artifact built <date> from <entry>` (`downloaded <date>
  into <entry>` for a published one);
* **not finished, and the package pins a published artifact for this model, checkpoint and
  frame**: downloads and verifies it ([below](#published-artifacts)), then copies it in;
* **a complete corpus but no artifact** (an earlier fit failed): fits, then copies in;
* **a partial corpus**: resumes generation, fits, copies in;
* **nothing**: builds into the store, then copies in.

`lfa list-artifacts` prints one line per entry: the model, the frame (documents × tokens, K), its
state — `built <date>` for an entry built here, `published, downloaded <date>` for a downloaded
one, otherwise `corpus complete, not fitted`, `downloading` (a published artifact's files being
fetched, or left by a fetch whose process was killed) or `in progress: n/N documents` — its size on disk and
its path.

**Locking.** A build holds `<entry>/.lock`, which records its process id. A second build of the
same entry — the same `lfa init` in another terminal — is refused with a message naming the lock
and that process, and saying to run the same command again once it has finished. A lock whose
process is no longer running is taken over with a warning, and that build resumes. The store root
also holds `.store.lock`, which only serialises taking a lock.

**`--rebuild`** builds afresh, here, even when the store has a match, and never downloads. The
existing entry is moved aside to `<entry>.replaced-<timestamp>/` — never deleted — and
`lfa list-artifacts` leaves moved-aside entries out. It is refused while a live build holds the
entry. Delete a moved-aside directory yourself when you no longer want it.

### Published artifacts

An artifact is made once per model and frame, so one that has been made can be published and
fetched instead of built again. On a store miss, `lfa init --artifact self-generated` looks the
model up in the list of published artifacts pinned in the package, `lfa/artifact/published.json`.
A pin names exactly what the store keys an entry on — `model_id` (the Hub id, as you pass it to
`--model`), the checkpoint's `writer_sha256` and the `frame_sha256` — and the entry's three files,
each by an `https://` URL (a pin with any other scheme is refused when the list loads), sha256 of
the file and size in bytes:

| file | URL | sha256 of the file | size |
|---|---|---|---|
| `artifact.pt` | `artifact_url` | `artifact_file_sha256` | `artifact_size_bytes` |
| `corpus.jsonl` | `corpus_url` | `corpus_file_sha256` | `corpus_size_bytes` |
| `corpus.jsonl.manifest.json` | `manifest_url` | `manifest_file_sha256` | `manifest_size_bytes` |

A `*_file_sha256` hashes the file's bytes. It is not the `corpus_sha256` that the artifact's meta
and the manifest record, which hashes the corpus rows' text (what the build computes, and what the
workspace's artifact id is made from). Which models have a pin is that file in the release you
have installed; a model with no pin there is built.

On a match, `init` downloads the three files into the store entry under temporary names
(`artifact.partial.pt`, `corpus.jsonl.download`, `corpus.jsonl.manifest.json.download`), holding
the entry's lock as a build does and logging the artifact's progress every 10 %. It accepts them
only when

* each file's size and sha256 are the pinned ones;
* the artifact loads as one this release reads — the same check a stored entry passes
  (`built_with: lfa-anchoring`, a `format_version` this release reads, diagonal mixture heads);
* its meta says `provenance: "self-generated"`, names the model, carries the frame asked for as
  `selfgen_frame`, and names a corpus as `corpus_sha256`;
* the corpus rows hash, as the build hashes them, to that `corpus_sha256`;
* the manifest records the same `corpus_sha256`, the pinned `writer_sha256` and model, and the
  frame asked for.

Then the corpus and its manifest are renamed into place, and `artifact.pt` last, so an
`artifact.pt` in an entry always means a complete one. `entry.json` records
`provenance: "published"` with each file's URL and file sha256. The entry now has exactly a built
entry's layout and is reused exactly as one: the workspace gets `artifacts/v1.pt` with
`v1.corpus.jsonl` and its manifest beside it, and records the artifact as
`self-generated:<corpus sha256[:12]>`.

Nothing is fetched for a different model id (a local path to the same weights included), a
different snapshot of the weights (another `writer_sha256`), or a different frame (a trial build's
`--n-raw 60`, say): those are built. An unfinished local build in the entry is moved aside to
`<entry>.replaced-<timestamp>/` before the download; a build or download of the same entry still
running in another process is refused, as a second build is.

A download that fails — the network, an HTTP error, a timeout (60 s without data), a file cut
short, a size or sha256 that is not the pinned one, a file that fails any check above — is
refused in one line naming the URL and what failed. Nothing is kept: the temporary files are
removed, and so is the entry directory when nothing else is in it. `init` never falls back to an
hours-long build on its own: run the same command again to retry, or pass `--rebuild` to build
the artifact here instead (hours on one GPU). An interrupted download is discarded the same way,
and the same command starts it again. The download honours `HTTPS_PROXY` and `NO_PROXY` from the
environment.

### Reusing a file

```bash
lfa init runs/second --model Qwen/Qwen3-0.6B --artifact runs/my_domain/artifacts/v1.pt
```

`--artifact` takes an artifact this package built — another workspace's `artifacts/v1.pt`, or a
file `lfa build-artifact` wrote — and copies it in as `artifacts/v1.pt`. A file built anywhere else
is refused before the workspace is created: one with no `__meta__` block, one whose meta names
another builder in `built_with`, or one whose mixture heads are not diagonal — and so is one in an
artifact format this release does not read (see *Sharing and keeping artifacts* below). What the
workspace records comes from the file's meta: a self-generated file keeps its provenance and its id
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
variance with its eigenvalues, and a `--gmm-k` (32) component mixture with diagonal covariances
fitted in that basis, plus
per-dimension standard deviations — which the sampler uses for the **off-basis residual** it adds
back on every draw, so the sampled marginals are right rather than short.

What is deliberately *not* stored: the layer-0 `pre_qkv` lookup table. It is
`input_layernorm(embed_tokens(id))`, exactly reconstructible from the model's own weights, so only
the corpus **token frequencies** are kept (~600 KB against ~300 MB) and
`Sampler.build_embedding_lookup_from_model` rebuilds the table at load time. Layer 0 is then
sampled frequency-weighted.

The build also writes a `__meta__` block — model id, hidden size, layer count, site list,
`n_samples_total`, `built_with: lfa-anchoring`, which is what a file is accepted on, and
`format_version`, the layout it was written in — and validates the finished artifact against the
model before saving. Every training run validates it again. `n_samples_total` is a **per-site**
count, not a sum across sites: every site sees the same token stream. Each site's block also
carries its own `n_samples`, which is what an extension weights the new domain against; an artifact
without them is refused.

`build-artifact` writes blockwise-int8 by default (`--no-quantize` for full precision, which
doubles the file). Quantization is a storage format: it is applied to a shallow copy on save and
reconstructed on load, so nothing downstream knows whether the file was quantized. The real-text
Qwen3-0.6B artifact the paper's operating point was tuned against is ~108 MB int8 against ~226 MB in fp16.

⚠ **The fp16 reservoir has no range guard.** An activation above 65,504 would be stored as `inf`
and poison that site's fit. Nothing checks for it. If a model is suspected of large activations,
build with `dtype=torch.float32` (via `lfa.artifact.build.build_artifact`, which takes it as an
argument) and check the site statistics in the logs.

### Sharing and keeping artifacts

An artifact is portable across machines, people and releases of this package that share its format.
The meta records two things about the file: `format_version`, the stored layout
(`lfa.artifact.ARTIFACT_FORMAT`, 1 today), and `lfa_version`, which release wrote it — for the
record only, never checked. So an artifact someone else built with lfa-anchoring loads with your
copy, and so does one built with an earlier release, as long as the layout has not changed; a file
from before `format_version` was recorded is format 1. A release that changes the layout bumps the
format and says so in its changelog, and then refuses a file in a format it does not read with a
sentence naming both formats and how to build the artifact again. The store is checked the same
way: a matching entry in a format this release does not read is refused at `lfa init`, naming the
entry, and `--rebuild` builds it afresh. A [published artifact](#published-artifacts) is checked
the same way again after its download.

## Checking an artifact

`validate_against_model` checks that an artifact has the model's shapes, not that it describes the
model. `lfa probe-artifact` measures that, in minutes on one card: it prices the update directions
of a trained adapter under the artifact's samples and under the model's real activations, and
reports how far the two disagree. It is a report, never a verdict — the exit status says only that
the measurement ran.

```bash
lfa probe-artifact --model Qwen/Qwen3-1.7B --artifact runs/my_domain \
                   --adapter runs/witness/runs/stage1/final_model --output probe.json
```

`--artifact` takes a file or a workspace (its current artifact). `--adapter` is a saved PEFT
adapter trained on that model — the **witness** — and can be repeated. The real activations are
the model's on WikiText-2's test split, read from the Hub as `lfa evaluate` reads it.

**What it reports**, per probed site (default: five layers, first and last included) and as medians
by site class — linear (`pre_qkv` + `pre_o`) and MLP (`pre_mlp`). Each witness direction is one of
the adapter's LoRA deltas, or one of four random rank-32 directions matched to it in norm; its price
is the anchor's charge for that change of weights, `E‖Δf(h)‖²`, under the artifact's samples and
under real activations, and the probe reads their ratio:

* **LEVEL** — the geometric mean of the ratios: a uniform mis-scaling, which λ absorbs. Reported,
  not judged.
* **SHAPE** — the spread of the log ratios across directions: direction-dependent mispricing, which
  no scalar λ absorbs. This is the number to read.
* **floor** — SHAPE with a second, disjoint half of the real activations in place of the artifact:
  what real-against-real noise alone gives.
* **diagonal reference** — SHAPE of that second half with each feature column shuffled
  independently: every marginal kept, every correlation gone. What a perfect diagonal model of the
  real activations would give; it does not depend on the artifact.

**The witness must come from an unanchored run** (λ = μ = 0). An adapter trained anchored against
an artifact moves into the directions that artifact underprices, and makes it look worse than it
is: on a Qwen3-0.6B artifact (2026-10-03), a witness trained anchored at λ = 100,000 against a
sibling build of the same corpus read median SHAPE 0.238 (linear) and 0.606 (MLP), where an
unanchored one read 0.056 and 0.108. The probe reads each
adapter's run `config.json` and says when a witness was anchored or its history is unknown.
[model-integration-cookbook.md](model-integration-cookbook.md#probe-it) has the commands for a
witness run.

**Reading the numbers.** There is no threshold. Read a model's numbers against those recorded for
the two bundled models, measured with the same witness recipe:

| median SHAPE, linear / MLP | Qwen3-0.6B | Qwen3-1.7B |
|---|---:|---:|
| floor | 0.018 / 0.017 | about 0.03 / 0.02 |
| the self-generated artifact | 0.053–0.055 / 0.095–0.106 | 0.058–0.067 / 0.137–0.139 |
| diagonal reference | 0.139 / 0.064 | about 0.14 / 0.06 |

Witness recipe, both models: one adapter from an unanchored run (λ = μ = 0) on the walkthrough's
Darwin training text, 4 epochs, the `qwen3-0.6b` recipe's other fields (rank 32, batch 6 × 512).
The probe at its defaults (WikiText-2, `--n-real 30000 --n-model 30000 --n-random 4`, layers 0, 7,
14, 20, 27) on each model's self-generated artifact at the recorded frame; the artifact rows span
three probe seeds (0, 1, 2; the command uses 0), the Qwen3-0.6B floor and reference are at seed 0.
One witness per model, one RTX 3090, 2026-10-03/04. Qwen3-1.7B's LEVEL read about 0.97 (linear) and
0.76 (MLP). A Qwen3-1.7B artifact fitted on 1.5 M samples per site instead of 600k read the same
(0.060–0.064 / 0.136–0.141), so the frame's 600k per site is enough for it.

* **A collapsed artifact** (a point mass, a degenerate corpus) shows as SHAPE many times these.
* **A scale or layer error** (another model's artifact, layers out of order) shows in LEVEL, far
  from 1, while SHAPE may look ordinary.
* **The artifact against the diagonal reference is not a pass mark.** Both models' MLP class reads
  above its reference here. With another unanchored witness (20 epochs on another domain), another
  build of the Qwen3-0.6B artifact at the same frame read 0.034 against a reference of 0.093, below
  it (mean of 3 probe seeds, 2026-10-03). A comparison that flips with the witness says nothing on
  its own. More witnesses (repeat `--adapter`) steady the medians.
* The probe catches a broken artifact; it does not show that a sound one anchors well. Only a λ
  calibration shows that ([model-integration-cookbook.md](model-integration-cookbook.md#5-calibrate-λ)).

## Advanced: an artifact fitted on real text

The route the paper's Qwen3-0.6B operating point was first tuned on: a downloaded seed corpus,
then the same fit.

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

**What the recipe says about it.** Each bundled recipe is calibrated against the self-generated
artifact at its frame, so a workspace over an artifact fitted here is told, at every stage, that
it is not the artifact the recipe's λ is calibrated against and that λ should be calibrated
against held-out domain perplexity ([model-integration-cookbook.md](model-integration-cookbook.md)). It is a note, never a
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
* **The realised corpus** — the composition the paper's Qwen3-0.6B operating point was tuned on, which that real-text
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
real-text artifact the paper's operating point was tuned against was built by the research code, so an artifact
built here over the same corpus has slightly different `std` values, and `std` enters the sample
stream through the off-basis residual term. It is a correction, not a divergence — but it is why
an artifact built here is not bit-identical to one the research code built, and why a λ
calibrated against one should be re-read against the other.

The `mean`, the PCA basis and the mixture are unaffected: those come from the sums, not from the
running variance.
