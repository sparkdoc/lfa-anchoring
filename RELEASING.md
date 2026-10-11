# Releasing

What is released is the **package**: a tag, an sdist and a wheel. The p(h) artifacts it can
download are pinned in `lfa/artifact/published.json` (model id, checkpoint and frame, then each
file's URL, sha256 and size); `lfa init --artifact self-generated` fetches a pinned one on a store
miss and builds from the model's own text otherwise
([docs/the-artifact.md](docs/the-artifact.md#published-artifacts)).

Nothing here is published automatically. Every step below is a command to run and a thing to look
at; a line marked ✅ records the rehearsal it was checked in.

## 0. Preconditions

```bash
pytest -q                                    # the default suite, all green
pytest tests/test_gpu_smoke.py -m gpu -q     # on the card you will use
```

The dev venv should have the `[html]` extra installed, or one prepare-domain test skips instead of
running:

```bash
pip install -e '.[dev,html]'
pytest -q -rs | grep SKIP     # the only expected skip is the one meaningful WITHOUT the extra
```

Check `LICENSE` against the canonical text (it is the Apache 2.0 text with the copyright line
filled in; nothing else may differ):

```bash
curl -sS https://www.apache.org/licenses/LICENSE-2.0.txt \
  | diff - LICENSE          # ✅ only line 190, the "Copyright [yyyy] [name]" placeholder
```

Check `lfa/__init__.py::__version__` and `pyproject.toml::version` agree and are what you mean to
release.

Did this release change the artifact layout (a meta or site field renamed, added as required or
removed, a head type, an encoding)? Then bump `ARTIFACT_FORMAT` (`lfa/artifact/schema.py`) and add
a changelog line saying which artifacts it refuses. A release that keeps the layout leaves it
alone, so every artifact built with an earlier release of the same format still loads.

## 1. Tag and build the package

```bash
git tag -a v0.2.0 -m "lfa-anchoring 0.2.0"
pip install -e ".[dev]"    # `build` is in the dev extra; `python -m build` needs it installed
rm -rf build dist          # stale build/lib can be swept into a wheel
python -m build            # sdist + wheel
```

Check what the sdist carries (`MANIFEST.in` governs it): `lfa/`, `docs/`, `examples/`, `tests/`,
`LICENSE`, `README.md`, `RELEASING.md`, `constraints-tested.txt`, counted with
`tar -tzf dist/*.tar.gz | grep -v '/$' | wc -l`. ✅ 2026-09-08 at 0.1.0: 73 files, 240 KiB; files
have been added and removed since, so re-measure rather than trusting the figure.

The **wheel** carries the package, the bundled recipes (`lfa/recipes/qwen3-0.6b.yaml`,
`lfa/recipes/qwen3-1.7b.yaml`), the pins `lfa/artifact/published.json`, and `lfa/examples/` — the
top-level `examples/` directory, mapped into the package by `[tool.setuptools.package-dir]` so
that a pip-installed user has the two scripts the documentation sends them to
(`python -m lfa.examples.quickstart`). ✅ 34 files at 0.1.0; re-measure it too. Docs and tests are
deliberately sdist-only; what carries them to a PyPI reader is `[project.urls]` and the README's
absolute links. The metadata should read `License-Expression: Apache-2.0` with
`dist-info/licenses/LICENSE` present, and should carry four `Project-URL:` lines.

## 2. Install clean and run everything

```bash
python -m venv /tmp/lfa-release
/tmp/lfa-release/bin/pip install -c constraints-tested.txt dist/lfa_anchoring-0.2.0-*.whl
/tmp/lfa-release/bin/lfa --help
(cd /tmp && /tmp/lfa-release/bin/python -m lfa.examples.quickstart --help)   # the wheel ships these
```

Run the last line outside the checkout: `-m` puts the working directory first on `sys.path`, so
from the checkout the source tree's `lfa/` (which has no `examples` package) shadows the wheel's.

**Install with `-c constraints-tested.txt`**, which is what the README and the quickstart tell a
user to do. Without it pip resolves the newest torch and peft it can, and the numbers in
`docs/verification.md` stop holding: that harness is exact by design, so a minor version change
moves it. ✅ 2026-09-08: an unconstrained venv (torch 2.14.0, peft 0.20.0) failed the extension's
matched-initialization check that passes at the pinned versions.

Rehearse on an interpreter *without* development headers too: `lfa` warns once
rather than refusing when `Python.h` or a compiler is missing, because the package's own training
and generation paths ran without them (a torch path that JIT-compiles would still fail in gcc). A
distribution `python3` without its `-dev` package is exactly the machine a new user brings, so the
release should be seen to run there and to print the warning once.

Then, from a checkout with that venv:

```bash
pytest -q                                      # the default suite
CUDA_VISIBLE_DEVICES=0 pytest tests/test_gpu_smoke.py -m gpu -q
```

That is the whole of what this repository can run, and it is deliberate: everything here needs only
this package, a CPU and (for the last line) a card. A user who has nothing else gets a clean
`pytest -q`.

**The port verification is not in this repository and is not a gate on this step.** It compares this
package against the research implementation, so it needs both at once plus the base model, the
recipe adapter, the corpora, the artifacts and a matched reference run — gigabytes that are not
distributed. It lives with the research code, and
[`docs/verification.md`](docs/verification.md) is its report. Re-run it there when the release
changes anything the two implementations share. Since 2026-09-08 this package's per-epoch chunk
offset rotates the chunk boundaries rather than discarding each document's leading tokens, so the
per-epoch loader and full-run comparisons need the two loaders matched by hand on a branch;
everything read at offset 0 still compares. `docs/verification.md` opens with what that changes.

## 3. Publish

Push the tag, and publish the package release with the sdist and the wheel.

---

## Changelog

Newest first. A release that changes what a run computes says so in its first sentence; one that
does not says that too, because "nothing moved" is the claim a reader most needs to be able to
trust.

### 0.2.0

**What a default run computes moves.** `train` now mixes into the training side a
question-and-answer supplement the stage's entry model writes from the domain, at the recipe's
`supplement_fraction`; `--no-supplement` trains on the raw corpus alone, as every 0.1.x run did,
and warns that it is off the frame. The held-out split is taken before anything is mixed, so
the domain number stays a raw-text measurement. The learning-rate schedule now spans the steps a
run actually takes, so the end of every cosine run moves too (below).

`lfa init --artifact self-generated` builds the artifact from the model's own text, or, when the
package pins a published one for that exact model, checkpoint and frame, downloads it with its
corpus and verifies it (`--rebuild` always builds here); either way it is kept in a local store
for reuse (`lfa list-artifacts`). A build resumes after an interruption and keeps its corpus if
the fit fails. `--artifact` is required; the separate download command and the flag that named a
published artifact by id are gone. `prepare-domain --supplement --model`
and `prepare-supplement --model` prepare the supplement with the data. The recipe is calibrated
against the self-generated artifact at `self_generated_frame`.

Self-generation: `init --artifact self-generated` / `build-artifact --self-generated` (the model
writes the seed corpus p(h) is estimated on; no download), the supplement `train` now writes
with the entry model and mixes in at the recipe's `supplement_fraction` (0.13, the frame the
shipped λ was tuned at; every earlier companion run trained at 0), `prepare-supplement`,
`regenerate-artifact` and the chain's `artifact: regenerate` route. Four recorded deviations
from the research frame: the template's wording is domain-neutral (it says "about a text on
<domain>" where the research one said "a philosophy text", its rule against exam-style
questions no longer names thought experiments, and it drops the research example "Chalmers
argues that..." without naming another, since a small writer copies the example's name); the
question template adds a rule against citing the passage, and pairs that cite it anyway are
dropped; the answers are the model's own chat answers to the passage and the question, written in
a second pass (the native-format bullet below); and the contamination
screen against an evaluation set is not ported. The measured
supplement result was on the research wording and its one-pass template. The default domain description skips a generic
directory name for the nearest named ancestor (`data/darwin/train` → `darwin`). The toolchain
check is a warning. Recipe fields `supplement_fraction`, `self_generated_frame`; artifact meta
fields `provenance`, `corpus_sha256`, `selfgen_frame`; history keys `supplement`,
`artifact_route`. Evidence scope: one model, one seed (the self-generated artifact, the
supplement); rank 4, one seed (the regenerate route).

* **Qwen3-1.7B** is a bundled model: recipe `qwen3-1.7b`, for `Qwen/Qwen3-1.7B`, calibrated against
  its own self-generated artifact at the recorded frame ([docs/recipes.md](docs/recipes.md)). The
  `--model` value picks the bundled recipe whose `model_id` it is.
* **Both recipes' λ is 160,000** (`lambda_qkv` = `lambda_mlp`; the `qwen3-0.6b` recipe's was
  100,000 in 0.1.x). Read on the walkthrough's Darwin text on this release's frame — the two-pass
  supplement, its pairs shuffled, the schedule over the real step count — with each λ kept at its
  own dose (a 15-epoch run, then a re-run at its turn; one seed, perplexity;
  [docs/recipes.md](docs/recipes.md#how-λ-was-chosen)). Qwen3-0.6B: by the cookbook's §5 first
  rule, an interior pick (WikiText-2 16.15, held-out Darwin 17.29 at 4 epochs; base 17.93 and
  30.12). Qwen3-1.7B: by the least behaviour drift, a stated departure from the rule. There the
  rule picked the weakest rung at every extension (64,000, 25,600, 10,240), because WikiText-2
  kept falling as λ fell and a 3-epoch run with no anchor was itself 16 % below base on it; the
  shift in the base model's own answers to 40 general prompts was +0.346 nats for the unanchored
  control and +0.089 to +0.134 for every rung probed (10,240 was not), lowest at 160,000
  (WikiText-2 12.19, held-out Darwin 13.15 at 5 epochs; base 15.09 and 21.45). An earlier
  calibration in this release cycle, on the one-pass frame with every rung at a common 15 epochs,
  picked 1,000,000 on both models; the cookbook's §9 keeps it as its worked example. The frame
  alone does not explain the move: read at a common 15 epochs, this frame's own runs still pick
  1,000,000 for Qwen3-1.7B and land on the 2,500,000 end rung for Qwen3-0.6B. `epochs` stays 15:
  at 160,000 the Darwin text turns at epoch 4 (Qwen3-0.6B) or 5 (Qwen3-1.7B), the run warns, and
  the re-run at the turn is the model to keep. `stage2_lambda_multiplier` stays 3: on Qwen3-0.6B
  at 160,000, 3× beat 1× on all three axes at the recipe dose; with each stage 2 re-run at its own
  turn, 1× edges 3× by the cookbook's rule (3×'s cookery gain is 88.9 % of 1×'s, against a 90 %
  bar; one seed, 3× not at its own lowest epoch). Qwen3-1.7B at 160,000 gives the same verdicts by
  a different margin: 3× ahead on all three axes at the recipe dose (1× ended cookery above its
  start); with each stage 2 re-run at its own turn both qualify (3×'s cookery gain 96.9 % of 1×'s)
  and 1× edges it on Darwin (14.22 against 14.57; one seed, 3× again not at its own lowest epoch).
  `docs/tuning.md` §4 says why WikiText-2 below base is not read as retention on a chat-tuned
  model. The paper's operating point (λ = 100,000, calibrated on the research corpus) is
  unchanged, as the paper's. `TrainConfig` and `Recipe` default to the new value, and the
  small-corpus warning cites the new measurements.
* **`init` warns when the recipe names another model** than the workspace's: λ does not port
  between models. A warning, never a refusal — a local path to the same weights is another id.
* **`lfa probe-artifact`** (`lfa.probe.probe_artifact`) prices a trained adapter's update
  directions under an artifact's samples and under the model's real WikiText-2 activations, and
  reports LEVEL and SHAPE per site beside a real-against-real floor and a diagonal reference. It
  gives numbers and no verdict; the exit status says only that it ran. Read with a witness adapter
  from an unanchored run ([docs/the-artifact.md](docs/the-artifact.md#checking-an-artifact)).
* **`chat_turn_end`** (`lfa.selfgen.generate`): the supplement writer reads the token that closes
  an assistant turn off the writer's chat template (`<|im_end|>` under ChatML, `<|eot_id|>` under
  Llama 3) and stops and cuts there. `TrainConfig.model_id` is required.
* **Text without spaces is not degenerate.** The self-generated corpus's degeneracy filter counts
  repeated 8-grams of whitespace-separated words; a text with fewer than eight such words has no
  8-gram, so only the length floor can reject it. A paragraph in a script written without spaces
  is one or two such words.
* **The artifact meta also records `layer_group_size`**, for the record. The self-generated build
  chooses the group from host RAM when none is given, and the choice does not change the artifact.
* **Per-site reservoir generators.** Each site's reservoir draws come from its own generator,
  seeded from the build seed and the site, so at a fixed seed every layer group size builds the
  same artifact (one generator used to be shared across a pass's sites, which made the draws
  depend on how layers were grouped), and a slot drawn twice in one batch now goes to the later
  vector every run instead of to whichever write landed last.
* **The native-format supplement writer.** The writer asks for questions only (a JSON array of
  strings), then answers each in one ordinary chat turn holding the passage and "Using the
  passage, answer: <question>", thinking off, with no length, sentence, format or voice
  instruction. Why: in the format-drift diagnostic (Qwen3-0.6B, the walkthrough's Darwin data,
  four epochs, one seed, one greedy decode per prompt), a model trained on the one-pass writer's
  short third-person answers answered in that style everywhere, under the anchor: 0 of 40 general
  prompts with any markdown (base 39 of 40), 0 of 20 Darwin prompts (base 19 of 20), 0 of the
  judge's 80 Darwin questions (base 78 of 80), 223-character general answers (base 1,095), and 6
  of the 40 general prompts declined as not in "the text". Trained on this writer's supplement,
  the model answered 36 of 40, 15 of 20 and 66 of 80 with markdown, wrote 625-character general
  answers and declined none; its base-answer NLL shift was +0.106 against the one-pass writer's
  +0.222, its held-out Darwin perplexity 19.36 against 19.26, and WikiText-2 16.06 against 16.18.
  On correctness and completeness (`gpt-5.6-luna` at medium effort, 80 Darwin questions, blinded,
  frozen rubric, 0–4) it scored 1.337 against the one-pass writer's 1.575 (−0.237, paired
  p = 0.071: not separable at this n; zero-rates 0.350 and 0.338); the base model's level was
  1.050. A variant with the
  passage as a system message and the question alone kept more of the format but was behind the
  one-pass writer on the judged score (correctness and completeness, p = 0.010; correctness
  alone p = 0.007), which is why this turn ships. The measured arms
  mixed their supplement's unshuffled prefix (next bullet), under the filter as it stood before
  this release's guards and extended patterns. An instruction not to mention the passage
  suppressed the model's format (markdown in 8 % and 14 % of the base model's answers under two
  such wordings, against 72 % under this prompt), so source references are left to a filter. It
  departs further from the research's supplement, whose measured judge result was on the one-pass
  question-and-answer template. A question or answer that cites its source ("according to the
  passage", "the text states", "the text also notes", "in the context of the text"; not "in the
  text of the treaty" or "in the passage from Genesis 1") is dropped (`refers_to_source`; on the
  Darwin text 750, 355 of the 1,591 questions and 395 of the 1,235 answers), as is an answer that
  had not ended its turn at the 1,024-token cap (`truncated_answer`), and `leaked_json` now checks
  an answer only for the writer's whole JSON shape; `long_answer` (a 100,000-character guard the
  token cap made dead) is gone. The question parser reads an array to its own closing bracket,
  prefers an array whose items all ask, takes a single-quoted array, and salvages only literals
  ending in `?` from a broken one. Rows record `source_index` (the document) and `passage_index`
  (the passage within it). Longer answers mean fewer pairs at the same token share (162 of 840
  written, against 384, on the Darwin text, unshuffled). Writing it took 13 min 48 s on one RTX
  3090 with Qwen3-0.6B for 263 passages of the Darwin text (the seed-43 timing run, 2026-10-09: 862 pairs), timed from the first
  logged batch of questions to the closing `supplement:` line. The bundled recipes' λ was calibrated with this
  writer's supplement, its pairs shuffled (next bullet). Against no supplement (Qwen3-0.6B at
  λ 160,000, 4 epochs, two seeds, the same judge and 80 questions, answers up to 1,024 tokens) it
  scored +0.350 on correctness and completeness (paired p = 0.0013; 1.350 and 1.375 against 0.938
  and 1.087, zero-rates 0.36–0.38 against 0.48–0.53): mostly the model ending its answers, since
  without it 6 and 11 of 80 answers looped on to the cap, and on the 66 questions where none did
  the gap was +0.106 (p = 0.27). Its cost is the format: headings in 0.00–0.03 of general answers
  (base 0.45), shorter answers, and 7–8 of 10 format instructions followed (base 10 of 10).
* **What a run mixes in changes: the supplement's pairs are shuffled before the prefix is
  taken.** The research mixer shuffles its pool under its seed and then takes the prefix that
  reaches the token fraction; earlier builds of this package took the prefix of the file as
  written, which is passage by passage, so every run mixed only the first passages' pairs. The
  pairs are now shuffled under the run's `seed` (the recipe's, 42), and the history's
  `supplement` entry records `shuffle_seed`. On the Darwin text this writer's pairs mixed in went
  from 162 pairs drawn from 51 of the 265 passages (32 of 173 documents) to 158 pairs drawn
  from 125 passages (98 documents).
  The format-drift diagnostic's arms used the unshuffled prefix; the walkthrough notebooks'
  recordings (2026-10-11, at 208e25e) used the shuffle.
* **The supplement** lives under `<workspace>/supplements/<corpus sha256[:12]>/`, or beside the
  corpus in `<corpus>.supplement/` when it was prepared with the data, and is reused while the
  training side's hash, the writer checkpoint's hash, the question and answer templates' hashes,
  the filters' hash and the domain description match;
  `prepare-supplement --force` rewrites it. Its manifest records `chat_template_applied` (a writer
  without a chat template is warned about and is outside the recorded frame); duplicates are keyed
  on the question alone, as in the research writer.
* **`evaluate --compare-unanchored`**: the λ = μ = 0 control mixes the stage's own supplement at
  the stage's fraction, so it is still the same run without the anchor.
* **`prepare-domain --split-chars N`** cuts each file into documents of about N characters at
  paragraph boundaries (floor 500; 3,500 suggested for a book), so one long file can hold documents
  out. A corpus that cannot hold any document out is refused before the supplement is written, and a
  sentence-overlap check refuses text already in `--out` or given twice.
* **General text cleaning** before the markup steps: line breaks normalised, invisible and control
  characters removed, Unicode space variants made a plain space, NFC last. Content — a licence, a
  table of contents, an index — is the user's to cut; `docs/preparing-your-data.md` says what and
  how, with a Project Gutenberg checklist. A corpus prepared under 0.1.x re-prepares to different
  text wherever those characters occur, and so to a different training-side hash.
* **Every run ends with its held-out verdict** — turned by 10 % or more (a warning), 1–10 % (a
  re-run is likely better), under 1 % (optional), still falling, diverged, or no curve — kept under
  `held_out` in the stage's history entry and restated by `evaluate` under its table. A run trained
  at an earlier same-frame run's turn epoch that ends still falling is told to stop and to choose
  between the two on both axes (`rerun_of`).
* **`train --lambda` / `--mu`** set the stage's values exactly, with no stage multiplier on top; a
  resume without them keeps the run's own, and an explicit one that differs is refused. A repeat of
  a stage writes `runs/stage{N}_run{k}` instead of overwriting.
* **`evaluate --run NAME` / `fuse --run NAME`** read or export a named earlier run of the latest
  stage, and `fuse` says when it replaces an export. `evaluate --compare-unanchored` prints a dose
  note, with the commands to train the control at its own best epoch, when the control turned.
* **The supplement drops questions that carry the writer's JSON field syntax, and answers written
  in its JSON shape** (`leaked_json`, counted with the other rejections); the filters' hash and both templates' hashes
  join the reuse key, so every supplement written before this release is rewritten.
  `n_train_docs` counts domain documents only.
* **The learning-rate schedule spans the run.** Its length was the first epoch's optimizer steps
  times the epochs, as in the research code, but each epoch is cut at its own chunk offset and a
  later one can hold more chunks, so runs outlasted their schedule (566 steps against 516 planned on
  a four-epoch Darwin run, Qwen3-0.6B) and PyTorch's cosine rose again past its end (a
  fifteen-epoch run ended at about 5 % of the peak, rising). The length is now the real count,
  summed epoch by epoch (`ChunkedCorpus.chunk_count`, `lfa.train.run_optimizer_steps`), so the
  cosine ends at its floor on the last step; the `LR schedule:` log line states it, and a resume
  no longer depends on how the corpus it is handed happens to be chunked. The bundled recipes' λ
  was calibrated under the new sizing.
* **`evaluate --corpus` says what it scored.** A named corpus was logged as "trained on -- a fit,
  not a held-out measurement" whatever it was; it is now "scored whole as named -- held out only if
  this text was kept out of training", or a fit when it is the stage's own training corpus.
* New flags not named above: `train --supplement JSONL` (mix in a prepared supplement instead of
  writing one) and `--domain-description TEXT`; `init` / `build-artifact --self-generated` take
  `--n-raw`, `--n-chat` and `--max-new-tokens`, and `build-artifact` `--gen-seed`.
  `build-artifact --max-samples` has no fixed default: 600,000 on the self-generated route, 1,500,000
  over a seed corpus as before.
* Messages: download and store sizes in decimal MB; the toolchain warning names the running
  Python's `-dev` package; the `--lambda` note names the flag on the CLI and the argument from
  Python; the corpus-shape note leads with the run's own corpus.
* **CI** runs the default test tier on Python 3.11, 3.12 and 3.13 (CPU) on every push to `main`
  and every pull request; the default tier runs with `HF_HUB_OFFLINE` set (`tests/conftest.py`).
  `CONTRIBUTING.md` and a bug-report template are new.
* **Not ported**: `--enforce-spec` (a comparison-only device) and the reasoning and instruction
  supplement modes (a supplement written in a skill's mode left that skill no better).
  Self-generated rehearsal, a replay method, is out of scope.
* **Only artifacts this package built are read.** `init --artifact PATH`, `extend_artifact`,
  `load_artifact`, `Sampler` and `validate_against_model` refuse a file with no `__meta__` block,
  one whose `built_with` is not `lfa-anchoring`, or one with a mixture head that is not diagonal.
  `extend` refuses a base artifact without per-site `n_samples`.
* **Artifacts record their format.** The `__meta__` block carries `format_version` (1: the layout
  every release has written, so a file without the field, from 0.1.x, is format 1 and loads), and
  a file in a format this release does not read is refused with a sentence naming both formats.
  `lfa_version` stays a record and is not checked.
* **Removed since 0.1.x**: `extend_artifact(base_n=)`, `lfa.merge.annotate_count`,
  `lfa.models.MissingBuildToolchain`, `make_meta(built_with=)`, `TorchGMM(covariance_type=)`
  (it fits diagonal covariances only), and the reading of
  full-covariance, whitened and top-m GMM heads (`Sampler.sample_gmm`, `fit_domain_gmm`,
  `merge_gmm_blocks`).
* Docs: README (the five-command pipeline, the Models table), `docs/quickstart.md` (choosing a
  model, then the full pipeline, step by step), `docs/preparing-your-data.md` (new: formats,
  corpus shapes, the supplement), `docs/the-artifact.md` (renamed and rewritten: the
  self-generated build, the store, published artifacts, per-model cost, checking an artifact with
  `lfa probe-artifact`, a real-text artifact), `docs/model-integration-cookbook.md` (replaces
  `docs/adding-a-model.md`: the procedure for a model with no bundled recipe, Qwen3-1.7B worked through), `docs/recipes.md`,
  `docs/concepts.md` (the building blocks; what the supplement does: reachability; what it does
  not: protect skills), `docs/multi-domain-chains.md` (the regenerate route), `docs/faq.md` (8 GB cards, bf16 on Turing,
  the toolchain warning, what self-generation costs, memory per model), `docs/tuning.md` (new: the
  dose, the control at its own dose, another λ). `docs/verification.md`
  gains only a dated header: nothing in 0.2.0 was checked bit-for-bit against the research code.

### 0.1.1

**No number moves.** Under LoRA a run no longer loads a second copy of the model: PEFT keeps the
base weight of every module it adapts frozen, so the student already holds the teacher and
`lfa.models.AdapterDisabledTeacher` reads it there with the adapters switched off. Verified
bit-identical on GPU against the 0.1.0 path at the shipped Qwen3-0.6B recipe — 392 of 392 adapter
tensors `torch.equal` after 58 optimizer steps, every per-micro-batch loss term equal over all 116
recorded rows, with `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
— and the same protocol separates a seed-changed control at 0 of 392, so the comparison can see a
difference.

* **`teacher_mode`** — new `TrainConfig` field, `Workspace.train(teacher_mode=...)` argument and
  `lfa train --teacher-mode` flag, with values `auto | separate | adapter_disabled`. `auto` (the
  default) is `adapter_disabled` for a LoRA run and `separate` for `--full-weight`, where the base
  weights move and there is no teacher to read; `adapter_disabled` is refused there rather than
  approximated. The **resolved** mode is recorded in the run's `config.json` and in the workspace
  history entry, so a finished run says which teacher it trained against.
* **Peak GPU memory falls by the whole resident teacher** — 1.11 GiB on Qwen3-0.6B (596 M
  parameters in bfloat16), on allocated, reserved and `nvidia-smi` alike, at no cost in step time
  (−0.2 %, inside a 0.8 % run-to-run spread, paired at the shipped recipe on an RTX 3090).
* `Workspace.train`'s unanchored control inherits the stage's teacher mode, so a control is still
  the same run without the anchor rather than the same run set up differently.
* Docs: the README's data-free paragraph, the FAQ's memory answer, the quickstart's cost note and
  `docs/recipes.md`'s per-run overrides.

### 0.1.0

First release: the package, the two published p(h) artifacts (retired in 0.2.0), the bundled
Qwen3-0.6B recipe, and the two example notebooks. Rehearsed against `docs/verification.md`.

---

### Already done, and how to re-check it

| | how to verify |
|---|---|
| `license = "Apache-2.0"` + `license-files` (PEP 639), `setuptools>=77` | the wheel's `METADATA` says `License-Expression: Apache-2.0` |
| `MANIFEST.in`, including `constraints-tested.txt` | `tar tzf dist/*.tar.gz` |
| `LICENSE` is the canonical Apache 2.0 text | the `diff` in step 0 |
| the `[html]` extra installed in the dev venv | `pytest -q -rs` shows no `needs the [html] extra` skip |
