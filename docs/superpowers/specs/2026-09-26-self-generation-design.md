# Self-generation in lfa-anchoring: design

Date: 2026-09-26. Status: approved in conversation section by section; awaiting written review.

## 1. Purpose

A user of this package has a model and a domain corpus, and nothing else. Today the package still
needs two things it did not make: a published p(h) artifact (or a Hub download of RedPajama and five
instruction datasets to build one), and a question-and-answer supplement it has no way to write. The
research record (mr-fusion claims C12, C14, C15) shows the model can supply both from its own text.
This work ports that capability so that, with no dataset download and no API key, a user can:

1. build the base artifact from text the model writes unconditionally;
2. have the model write a domain supplement, mixed into training at the recipe's token fraction;
3. in a chain, refit the artifact from each stage's own model instead of extending it.

**What the evidence supports, and what it does not.** The supplement's measured job is
*reachability*: it makes the new knowledge answerable in QA form (C12). A supplement written in a
skill's style does not protect that skill (C14), and LFA's ~10-point GSM8K loss is not repaired by
self-generated inputs (mr-fusion `docs/self_generated_data_2026-09.md` §0.0 items 4–5). The thing that
holds skills at base in the record is self-generated rehearsal, a replay method, which is out of scope
here. Every document this work touches says "reachability", never "retention".

**Scope of every number quoted.** One model (Qwen3-0.6B), one seed, one domain for C12/C14; rank 4,
one seed, three domains for C15. Judge `gpt-5.6-luna@medium`. The package docs state this scope
wherever a number appears and quote nothing as the package's own measurement.

## 2. Decisions taken in conversation

| decision | choice | why |
|---|---|---|
| scope | artifact, supplement, regenerate-per-stage chains; **not** SSR rehearsal | SSR is a second training objective, not an LFA capability |
| artifact role | equivalent calibrated artifact for Qwen3-0.6B; the only route for other models | C12: tie at every λ tried; nothing measured elsewhere |
| mix point | train-time mixing, `supplement_fraction` recipe field, default 0.13 for `qwen3-0.6b` | λ was tuned at f=0.13; today's f=0 runs are off that frame |
| missing supplement | `train` generates it, caches it, records its hash; `--no-supplement` opts out | keeps the four-command story; makes f=0 a choice, not an accident |
| writer model in a chain | the stage's entry (fused) model | what C15 did; the writer is the model being anchored |
| testing | GPU tier primary, sized for an 8 GB card; fast tier for logic only | owner: users have at least a minimal GPU; this machine (RTX 2070, 8 GB) is the acceptance machine |
| architecture | approach A: `lfa/selfgen/` subpackage, three thin integration points | keeps generation out of the Hub downloader and the file converter; every piece testable alone |

## 3. Architecture

```
lfa/selfgen/
  __init__.py
  generate.py          one batched sampling loop; seed prefix; burn-in; repeat filter
  artifact_corpus.py   the unconditional corpus at the recorded frame -> JSONL + manifest
  supplement.py        passages -> template -> pairs -> JSONL + manifest
lfa/artifact/build.py  + build_artifact_self_generated()
lfa/artifact/schema.py + meta field `provenance`, `corpus_sha256`
lfa/corpus.py          + supplement prefix selection; enable_thinking=False in chat rendering
lfa/recipe.py          + supplement_fraction, calibrated_self_generated; warnings() reads meta
lfa/workspace.py       + init(artifact="self-generated"); train() supplement step;
                         regenerate_artifact(); chain `artifact:` field
lfa/models.py          toolchain check demoted to a warning
lfa/cli.py             + flags and commands (§8)
```

Data flow, single domain, fully self-supplied:

```
model ──generate.py──► artifacts/v1_corpus.jsonl ──build_artifact──► artifacts/v1.pt
model + training-side docs ──supplement.py──► <domain>/.lfa/supplement.jsonl
raw train docs + supplement prefix at f ──corpus.py──► ChunkedCorpus ──train──► stage model
```

### 3.1 `lfa/selfgen/generate.py`

`generate_texts(model, tokenizer, prompts, *, max_new_tokens, temperature, top_p, stop_token_ids,
generator, batch_size) -> list[str]`: batched sampling with an explicit attention mask, a private
`torch.Generator` seeded by the caller, and decoding that stops at any id in `stop_token_ids`.

Ported as written from `scripts/prepare_selfgen_corpus.py` (mr-fusion): `pick_seed_prefix` (declared
sequence-start id, then BOS, then EOS, then newline), `drop_burn_in` (default 0, as measured),
`clean_raw` (cut at the next document boundary), `passes_filters(min_chars, max_repeat_ratio)`.

### 3.2 `lfa/selfgen/artifact_corpus.py`

`write_artifact_corpus(model, tokenizer, out_path, *, n_raw=2500, n_chat=250, max_new_tokens=2048,
seed=42, chat_seed=43, batch_size, chat_share=True) -> Manifest`.

The recorded frame (`scripts/_w11_artifact_corpus.sh`):

* **raw share**: `n_raw` documents from the seed prefix, temperature 1.0, top-p 1.0, up to 2,048 new
  tokens, stopped at the document boundary, **unfiltered** (`min_chars=1`, `max_repeat_ratio=1.0`),
  seed 42, `source: selfgen_raw`;
* **chat-format share**: `n_chat` documents started from the bare user-turn header (`<|im_start|>user\n`
  for Qwen3, taken from the tokenizer's chat template, not hard-coded), header **kept**, seed 43,
  `source: selfgen_chatfmt`. Off (`n_chat=0`) for the per-stage regenerate route (§3.6), which C15 ran
  without it.

Output: one JSONL of `{"text", "source"}` rows, and `<out>.manifest.json` with writer model id and
checkpoint sha256, decoding settings, counts, rejection tally (zero at this frame), corpus sha256,
`lfa` version.

Refusals: no CUDA; no adapter for the model; fewer than `min_docs` (default 50) non-empty documents
after generation; more than 20 % empty documents. A degenerate corpus fits an artifact that fails
nowhere downstream, so these are errors, not warnings.

### 3.3 Artifact build and meta

`build_artifact_self_generated(model_id, out_path, *, corpus_path=None, n_raw, n_chat, max_samples=
600_000, gmm_k=32, pca_variance=0.95, seed, device, ...) -> Path`: writes the corpus to
`corpus_path` (default `<out>.corpus.jsonl`), then calls the unchanged `build_artifact` with the
measured frame (600k samples per site, K=32, variance 0.95, `scripts/_w11_selfgen_artifact.sh`).

`make_meta` gains `provenance: "self-generated" | None` and `corpus_sha256: str | None`. Everything
built from real text leaves both absent. `load_artifact` tolerates their absence (older files).

### 3.4 Recipe coupling

`Recipe.supplement_fraction: float = 0.13` (the only frame measured; a recipe sets `0.0` to opt out
at the recipe level) and `Recipe.calibrated_self_generated: bool = False`, `True` in
`recipes/qwen3-0.6b.yaml`. Both are recorded in the workspace's stage entry like every other recipe
field.

`Recipe.warnings(rank, artifact_id, artifact_meta=None)`: a self-generated artifact (meta
`provenance == "self-generated"`) whose `model_id` equals the recipe's, on a recipe with the flag
set, is calibrated: no warning, and the docs cite C12 with its scope. Any other self-generated
artifact gets the existing artifact-swap text plus one sentence: "calibrate λ against held-out domain
perplexity; the adding-a-model page says how."

### 3.5 Workspace: init and train

`Workspace.init(path, model, artifact="self-generated", ...)`: a third branch beside the published id
and the local path. Builds into `artifacts/v1.pt` with the corpus at `artifacts/v1_corpus.jsonl`;
the existing rollback guard removes what init created if the build raises. Records
`artifact_id = "self-generated:<corpus sha256[:12]>"` and `artifact_provenance`.

`Workspace.train(corpus, ..., supplement: bool | str | Path = True)`:

1. resolve the recipe; if `supplement_fraction > 0` and `supplement is not False`:
   * `supplement` a path: use that JSONL;
   * else look for `<corpus>/.lfa/supplement.jsonl` whose manifest matches (`corpus_sha256` of the
     training-side documents, writer checkpoint sha256, template sha256, `lfa` major.minor); reuse
     on match, else regenerate;
2. `supplement=False`: train at f=0; the recipe warning fires ("λ was calibrated at
   supplement_fraction 0.13; this run mixes none");
3. history entry gains `supplement: {manifest path, n_pairs_available, n_pairs_used,
   target_fraction, achieved_fraction, under_target, writer_sha256}`.

### 3.6 Workspace: regenerate and chain

`Workspace.regenerate_artifact(*, n_raw=2500, max_samples=600_000, ...)`: sibling of `extend()`.
Requires a fused stage model (same `StageOrderError` as `extend`). Writes the corpus from the fused
model (`n_chat=0`, the C15 frame), fits from scratch with no base component and no merge, writes
`artifacts/v{N+1}.pt` and `artifacts/v{N+1}_corpus.jsonl`, bumps `artifact_version`, records
`artifact_route: "regenerate"` (extend records `"extend"`; earlier histories without the key read
as extend).

Chain spec gains one **top-level** field `artifact: extend | regenerate` (default `extend`), applied
at every stage boundary. A per-domain `artifact` key is rejected by `_validate_domains`. Under
`regenerate`, `chain` calls `regenerate_artifact()` then `train()`, so the supplement writer inside
`train` and the artifact writer are the same fused checkpoint.

`stage2_lambda_multiplier` applies unchanged on both routes. The docs say it is a starting point on
either, that C15 ported λ at 2x at rank 4 while the recipe's default is 3x, and that a regenerated
artifact is not the calibrated one: the off-calibration warning fires on stage 2 under `regenerate`
and names C15's scope.

### 3.7 `lfa/selfgen/supplement.py`

`write_supplement(model, tokenizer, documents: list[tuple[name, text]], out_path, *,
domain_description, pairs_per_passage=6, passage_chars=4000, min_passage_chars=200,
max_new_tokens=1024, temperature=0.7, top_p=0.8, batch_size=16, min_answer_chars=40,
max_answer_chars=100_000, seed=42) -> Manifest`.

* **Passages**: the research chunker `_chunk_document` (paragraph boundaries, ~4,000 chars) ported
  verbatim so passages are the ones the evidence was measured on.
* **Template**: `GENERATE_PROMPT` from `scripts/prepare_domain_qa.py`, with one change: "about a
  philosophy text" becomes "about a text on {domain_description}". `domain_description` defaults to
  the corpus directory's name with `_`/`-` as spaces; `--domain-description` overrides. **Recorded
  deviation** from the C12 frame; the manifest carries the rendered template and its sha256.
* **Not ported**: `--enforce-spec` (comparison-only device, owner 2026-09-10); the `reasoning` and
  `instruction` modes (C14: not the lever); the contamination `screen` stage (a user has no eval set).
* **Parser**: `parse_qa_pairs` as written: JSON objects first, salvaging complete objects from a
  truncated array, then `Question:/Answer:` markers; `parse_assistant_turn` strips the chat wrapper.
* **Filters**: answer length in `[min_answer_chars, max_answer_chars]`; exact-duplicate pairs dropped.
* **Output**: `{"prompt", "response", "source_doc"}` rows, in generation order (the prefix rule in
  §3.8 depends on the order being fixed), and `<out>.manifest.json` with writer checkpoint sha256,
  template sha256 and text, decoding settings, passage count, pairs kept, rejection tally, the
  training-side corpus sha256, `lfa` version.
* **Refusals**: zero passages before the model loads; zero pairs after generation (message names the
  passage count and the tally).

### 3.8 `lfa/corpus.py`: mixing and rendering

`build_datasets(path, tokenizer, ..., val_fraction, supplement=None, supplement_fraction=0.0, seed)`:

1. load raw documents; split held-out **first**, exactly as today (seed-deterministic), so the
   held-out perplexity stays the raw-text number and is comparable with every existing run;
2. the supplement writer only ever sees the training side (§3.7 is handed that list);
3. `select_supplement_prefix(raw_train_tokens, pair_tokens, target) -> Selection(n_used,
   achieved_fraction, under_target)`: the research `select_qa_for_fraction` ported verbatim (prefix
   whose achieved token share `used / (raw + used)` is closest to the target; compare the last count
   under with the first over);
4. render the selected pairs through the chat template with `enable_thinking=False` (Qwen3 then
   carries the empty think block the research training format carries); `_extract_text` passes the
   same flag for any prompt/response record;
5. shuffle pairs into the training documents under the run seed; `under_target` is a warning with the
   achieved number, never a refusal.

### 3.9 Environment fixes

* `check_gpu_toolchain`: a one-line **warning** instead of `MissingBuildToolchain`, because a real
  Qwen3-0.6B LoRA step and generation ran on this machine without headers (probe 2026-09-26).
  `LFA_SKIP_TOOLCHAIN_CHECK=1` silences it. The FAQ sentence changes accordingly.
* 8 GB cards: measured on the RTX 2070 (bf16, r32, 512 tokens): micro-batch 1 / 2 / 3 peak
  2.45 / 3.52 / 4.59 GiB at 1.08 / 1.98 / 2.94 s; batch 6 overflows on the fp32 logits. The FAQ
  documents `batch_size: 3, gradient_accumulation_steps: 2` as the recipe geometry on 8 GB. No
  automatic memory probing: the training frame must not depend on the card.
* bf16 on Turing: perplexity 9.67 (bf16) vs 9.70 (fp32) vs 9.68 (fp16) on one sentence; greedy
  outputs agree. Recorded in the FAQ so a user with an older card knows the dtype is sound.

## 4. Error handling summary

| condition | behaviour |
|---|---|
| no CUDA / no adapter / model cannot load | refusal before any generation, existing messages |
| degenerate artifact corpus (§3.2) | refusal naming counts |
| self-generated artifact off the calibrated model | warning (recipe) |
| no supplement and `supplement_fraction > 0` | generate (default) |
| `supplement=False` | train at f=0 + recipe warning |
| supplement pool short of target | warning with achieved fraction |
| zero pairs | refusal with tally |
| regenerate before a fused stage | `StageOrderError` |
| per-domain `artifact:` key in chain YAML | `ValueError` at validation, before anything trains |
| toolchain headers missing | warning, once |

## 5. Testing

**GPU tier is primary** (`-m gpu`, Qwen3-0.6B, sized for 8 GB, full frame reachable by parameter;
run on the RTX 2070 before each section is called done, output shown in the report):

* `test_selfgen_gpu.py`: 16 raw + 4 chat documents; raw ones start after the seed prefix and end at
  the boundary; manifest counts and hashes; refusal on a forced-empty run.
* artifact: build from that corpus at 20k samples/site, K=4; sample from it; meta provenance.
* supplement: real 3-passage corpus; pairs parse; manifest; reuse on matching hashes; regenerate on a
  changed writer hash.
* train: one epoch at `supplement_fraction 0.13` on a tiny corpus, batch 3 / ga 2; history carries
  the achieved fraction, the pair count, the writer hash; `supplement=False` warns.
* chain: two stages, two tiny corpora, `artifact: regenerate`; v2 has no base component; history has
  the route and writer hash; stage 2 trained against v2.

**Fast tier** (logic only, no fake-generator duplicates of the above): mix plan and prefix
precedence on a stub tokenizer; the chunker on paragraph boundaries; the parser on truncated JSON and
marker text (fixtures from the research tests); `select_supplement_prefix` on the research fixtures;
held-out split excludes written pairs; recipe warning in all four (model matches × flag) cells; CLI
mutual exclusion; init rollback when the injected builder raises; chain-spec field validation;
stage-order refusal; toolchain warning path.

## 6. Documentation

| file | change |
|---|---|
| `README.md` | one paragraph; one building-blocks row ("Self-generation") |
| `docs/quickstart.md` | the no-download route as an alternative first command; `--no-supplement` |
| `docs/rebuilding-the-artifact.md` | self-generated section beside the seed corpus; recorded frame; C12 equivalence with scope |
| `docs/adding-a-model.md` | self-generation as the way to a first artifact; calibrate λ after |
| `docs/recipes.md` | `supplement_fraction`, `calibrated_self_generated`; λ was tuned at 0.13 |
| `docs/concepts.md` | what the supplement does (reachability) and does not (skill retention: C14, GSM8K) |
| `docs/multi-domain-chains.md` | the regenerate route beside extend; C15 with its drift numbers and scope |
| `docs/faq.md` | 8 GB geometry; bf16 on Turing; toolchain check now a warning |
| `docs/verification.md` | untouched; nothing here was checked bit-for-bit against research code |
| `RELEASING.md` | changelog for 0.2.0 with the two recorded deviations |

Notebooks untouched (a fully self-supplied notebook is a follow-up needing a 3090 execution).

## 7. Version

0.2.0. No new published asset.

## 8. CLI surface

```
lfa build-artifact --model <id> --self-generated --out <file> [--n-raw --n-chat --max-samples]
lfa init <ws> --model <id> --artifact self-generated
lfa prepare-supplement --workspace <ws> --corpus <dir> [--domain-description ...]
lfa train --workspace <ws> --corpus <dir> [--no-supplement | --supplement <file>]
lfa regenerate-artifact --workspace <ws>
lfa chain domains.yaml --workspace <ws>      # YAML: artifact: extend | regenerate
```

`--self-generated` and `--corpus` are mutually exclusive on `build-artifact`.

## 9. Out of scope

SSR rehearsal; reasoning/instruction supplement modes; the contamination screen; automatic memory
probing; a third notebook; any claim that the supplement protects skills.
