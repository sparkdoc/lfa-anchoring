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
the domain number stays a raw-text measurement.

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
`regenerate-artifact` and the chain's `artifact: regenerate` route. Two recorded deviations
from the research frame: the template's wording is domain-neutral (it says "about a text on
<domain>" where the research one said "a philosophy text", its rule against exam-style
questions no longer names thought experiments, and it drops the research example "Chalmers
argues that..." without naming another, since a small writer copies the example's name),
and the contamination screen against an evaluation set is not ported. The measured
supplement result was on the research wording. The default domain description skips a generic
directory name for the nearest named ancestor (`data/darwin/train` → `darwin`). The toolchain
check is a warning. Recipe fields `supplement_fraction`, `self_generated_frame`; artifact meta
fields `provenance`, `corpus_sha256`, `selfgen_frame`; history keys `supplement`,
`artifact_route`. Evidence scope: one model, one seed (the self-generated artifact, the
supplement); rank 4, one seed (the regenerate route).

* **Qwen3-1.7B** is a bundled model: recipe `qwen3-1.7b`, for `Qwen/Qwen3-1.7B`, calibrated against
  its own self-generated artifact at the recorded frame ([docs/recipes.md](docs/recipes.md)). The
  `--model` value picks the bundled recipe whose `model_id` it is.
* **The `qwen3-0.6b` recipe's λ is 1,000,000** (`lambda_qkv` = `lambda_mlp`; 100,000 in 0.1.x),
  calibrated like the Qwen3-1.7B recipe's on the walkthrough's Darwin text (one seed, perplexity;
  [docs/recipes.md](docs/recipes.md#how-λ-was-chosen)). At 100,000 that corpus over-trained at the
  recipe's 15 epochs and ended worse than the base model on both axes. The paper's operating point
  (λ = 100,000, calibrated on the research corpus) is unchanged, as the paper's. `TrainConfig` and
  `Recipe` default to the new value, and the small-corpus warning cites the new measurements.
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
* **The supplement** lives under `<workspace>/supplements/<corpus sha256[:12]>/`, or beside the
  corpus in `<corpus>.supplement/` when it was prepared with the data, and is reused while the
  training side's hash, the writer checkpoint's hash, the template's hash and the domain
  description match;
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
* **The supplement drops pairs that carry the writer's JSON field syntax** (`leaked_json`, counted
  with the other rejections); the filters' hash joins the reuse key, so a supplement written before
  this release is rewritten. `n_train_docs` counts domain documents only.
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
