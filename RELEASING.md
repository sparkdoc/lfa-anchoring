# Releasing

Two things are released and they are separate: the **artifacts** (large binary assets, tagged
`artifacts-v1`, whose SHA-256s are then pasted into the registry) and the **package** (a tag, an
sdist and a wheel). The artifacts come first, because the package's registry has to name them.

Nothing here is published automatically. Every step below is a command to run and a thing to look
at; the steps marked ✅ were rehearsed on 2026-09-07 and their measured output is recorded.

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

## 1. Re-save the artifacts with a `__meta__` block ✅

The two shipped artifacts were built by the research code and carry no meta block: no model id, no
hidden size, no layer count, no sample count. Every one of those is what
`validate_against_model` and a continual extension read, so the released files get one. Re-saving
is a load-and-save round trip — the statistics are untouched.

```python
# release_artifacts.py — run in the companion's venv, from the companion's root
from pathlib import Path

from lfa.artifact.schema import (EMBEDDING_LOOKUP_KEY, LM_HEAD_SITE, META_KEY, SITES,
                                 load_artifact, make_meta, save_artifact)

SRC = Path("/path/to/research-repo")
OUT = Path("release"); OUT.mkdir(exist_ok=True)

# These statistics were collected by the research code, not by this package. `built_with` says so;
# `lfa_version`, which make_meta fills in itself, records who wrote the block.
RESEARCH_BUILT = "research code; meta block added by lfa-anchoring"

# --- the recipe artifact: correlated basis + K=32 mixture per site, blockwise int8
params = load_artifact(SRC / "data/distributions/qwen3-0.6b-gmm1543k-int8/distribution_stats.pt")
params[META_KEY] = make_meta(model_id="Qwen/Qwen3-0.6B", hidden_size=1024, num_layers=28,
                             sites=list(SITES) + [LM_HEAD_SITE], n_samples_total=1_543_040,
                             built_with=RESEARCH_BUILT)
save_artifact(params, OUT / "qwen3-0.6b-gmm1543k-int8.pt", quantize=True)

# --- the diagonal budget floor, WITHOUT its ~312 MB layer-0 lookup table: that table is
#     input_layernorm(embed_tokens(id)), exactly reconstructible from the model, and the sampler
#     rebuilds it. The token frequencies stay (~600 KB), so frequency-weighted L_embed still works.
diagonal = load_artifact(SRC / "data/distributions/qwen3-0.6b-1200k-10to1-uniform-simple"
                            "/distribution_stats.pt")
lookup = diagonal[EMBEDDING_LOOKUP_KEY]
diagonal[EMBEDDING_LOOKUP_KEY] = {"token_frequencies": lookup["token_frequencies"]}
diagonal[META_KEY] = make_meta(model_id="Qwen/Qwen3-0.6B", hidden_size=1024, num_layers=28,
                               sites=list(SITES) + [LM_HEAD_SITE], n_samples_total=1_200_000,
                               built_with=RESEARCH_BUILT)
save_artifact(diagonal, OUT / "qwen3-0.6b-diagonal.pt", quantize=True)
```

`n_samples_total` is a **per-site** count, never a sum across sites: 1,543,040 for the recipe
artifact (what `gmm1543k` rounds) and 1,200,000 for the diagonal one. `built_with` defaults to
`"lfa-anchoring"`, which would be a false provenance here — these statistics are the research
code's — so both calls pass it explicitly.

Rehearsed output: `qwen3-0.6b-gmm1543k-int8.pt` **112.8 MB**, `qwen3-0.6b-diagonal.pt` **1.1 MB**
(dropping a `(151936, 1024)` table). Those are the sizes to expect; if the diagonal file comes out
at ~300 MB the table did not get dropped.

## 2. Prove the meta block changed nothing ✅

The sampler is the one thing checked at zero tolerance: 85 draws of one anchoring step, replayed
against the research implementation's recorded stream (`docs/verification.md`). Run it **against
the re-saved file**.

That check is in the port-verification harness, which lives with the research code because it needs
both implementations — this repository has the report, not the suite. Run it from the research
checkout, staging a root that carries the re-saved artifact where the fixture expects one:

```bash
STAGE=$(mktemp -d)
mkdir -p "$STAGE/data/distributions/qwen3-0.6b-gmm1543k-int8" "$STAGE/outputs/lra/qwen3-0.6b"
ln -s /path/to/lfa-anchoring/release/qwen3-0.6b-gmm1543k-int8.pt \
      "$STAGE/data/distributions/qwen3-0.6b-gmm1543k-int8/distribution_stats.pt"
ln -s /path/to/research-repo/outputs/lra/qwen3-0.6b/original \
      "$STAGE/outputs/lra/qwen3-0.6b/original"

cd /path/to/research-repo && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 LFA_RESEARCH_ROOT=$STAGE LFA_ANCHORING=/path/to/lfa-anchoring \
  pytest tests/lfa_port_verification/equivalence -m equivalence -k sampler_replays -q
```

Rehearsed: **1 passed**. And it is a real check, not a vacuous one — pointing the same staged path
at the *diagonal* artifact instead makes it fail, as it must.

A failure here means the re-save changed the sample stream. Do not widen anything: diff what the
round trip did to the file.

## 3. Hash and upload

```bash
sha256sum release/qwen3-0.6b-gmm1543k-int8.pt release/qwen3-0.6b-diagonal.pt
```

Create the asset release **`artifacts-v1`** and upload both `.pt` files to it. The URLs the
registry will hold are of the form

```
https://github.com/<org>/lfa-anchoring/releases/download/artifacts-v1/<file>.pt
```

## 4. Fill in the registry

In `lfa/artifact/fetch.py::ARTIFACTS`, for both entries:

* replace `<org>` in `_RELEASE_BASE` with the real organisation -- and in the same commit, the
  same placeholder in `pyproject.toml`'s `[project.urls]` and in `README.md`'s links, which are
  absolute because the README is the PyPI long description and PyPI does not rewrite relative
  ones (`grep -rn "<org>" --include="*.toml" --include="*.md" --include="*.py" .` finds all of
  them);
* replace `"sha256": PLACEHOLDER_SHA256` with the digest from step 3;
* check `size_mb` against the file you actually uploaded.

Until that is done, `fetch_artifact` raises `ArtifactNotPublished` rather than downloading
something it cannot verify, and `lfa list-artifacts` says `not published yet` — which is the
correct behaviour for every commit before this one. Do not paste digests for files you have not
run step 2 against.

**The gate for this step is a test, not a memory.** It fails until both halves are done:

```bash
pytest -m release -q          # <org> anywhere that ships, and placeholder checksums
```

It is deselected from the default suite on purpose — the repository is *supposed* to carry the
placeholder before a release — and it is in step 6's list below as well, so nothing is published
without it having been run. `tests/test_release.py::test_the_scan_has_teeth` runs in the default
suite and proves the scanner can actually see, since the gate itself is expected to be red until
today.

Verify:

```bash
lfa list-artifacts                                     # both rows say "published"
cd "$(mktemp -d)" && lfa fetch-artifact qwen3-0.6b-gmm1543k-int8 --dest .
lfa fetch-artifact qwen3-0.6b-gmm1543k-int8 --dest .    # again: verified, not re-downloaded
```

`fetch_artifact` downloads to `<name>.part`, hashes it, and only then moves it into place, so an
interrupted fetch never leaves something that looks like an artifact.

## 5. Tag and build the package

```bash
git tag -a v0.2.0 -m "lfa-anchoring 0.2.0"
pip install -e ".[dev]"    # `build` is in the dev extra; `python -m build` needs it installed
python -m build            # sdist + wheel
```

Check what the sdist carries (`MANIFEST.in` governs it): `lfa/`, `docs/`, `examples/`, `tests/`,
`LICENSE`, `README.md`, `RELEASING.md`, `constraints-tested.txt`. Rehearsed 2026-09-08 at 0.1.0:
**73 files, 240 KiB** (`tar -tzf dist/*.tar.gz | grep -v '/$' | wc -l`; the same listing is 82
lines with the directory entries counted). ⚠ 0.1.1 adds `tests/test_teacher_mode.py`, and 0.2.0
adds `lfa/selfgen/` and four `tests/test_selfgen_*.py` files, so that count has moved and has
**not** been re-measured — re-measure rather than trusting the figure, which is what this paragraph
has always said.

The **wheel** was 34 files at 0.1.0 (0.2.0 adds the four modules of `lfa/selfgen/`; re-measure it
too): the package, `lfa/recipes/qwen3-0.6b.yaml`, and `lfa/examples/` — the
top-level `examples/` directory, mapped into the package by `[tool.setuptools.package-dir]` so
that a pip-installed user has the two scripts the documentation sends them to
(`python -m lfa.examples.quickstart`). Docs and tests are deliberately sdist-only; what carries
them to a PyPI reader is `[project.urls]` and the README's absolute links. The metadata should
read `License-Expression: Apache-2.0` with `dist-info/licenses/LICENSE` present, and should carry
four `Project-URL:` lines with no `<org>` left in them.

## 6. Install clean and run everything

```bash
python -m venv /tmp/lfa-release
/tmp/lfa-release/bin/pip install -c constraints-tested.txt dist/lfa_anchoring-0.2.0-*.whl
/tmp/lfa-release/bin/lfa --help
/tmp/lfa-release/bin/python -m lfa.examples.quickstart --help     # the wheel ships these
```

**Install with `-c constraints-tested.txt`**, which is what the README and the quickstart tell a
user to do, and which this step omitted until the 0.1.0 release rehearsal caught it. Without it pip
resolves the newest torch and peft it can, and the numbers in `docs/verification.md` stop holding:
that harness compares against streams the research code produced under a specific build, so it is
exact by design and a minor version change moves it. Measured on 2026-09-08: an unconstrained venv
drew torch 2.14.0 and peft 0.20.0 against the tested 2.10.0 and 0.18.1, and the extension's
matched-initialization check failed there while passing at the pinned versions. That is the
constraints file doing its job, not a defect — but a release verified in an unpinned venv proves
less than it looks like it does.

Re-running the port verification against *newer* dependencies is a separate and worthwhile
exercise. It answers "does the port still match the research code on today's torch", which is a
real question. It is not what step 6 is for, it happens in the research checkout rather than here,
and its failures are not release blockers.

Rehearse on an interpreter *without* development headers too: since 0.2.0 `lfa` warns once
rather than refusing when `Python.h` or a compiler is missing, because the package's own training
and generation paths ran without them (a torch path that JIT-compiles would still fail in gcc). A
distribution `python3` without its `-dev` package is exactly the machine a new user brings, so the
release should be seen to run there and to print the warning once.

Then, from a checkout with that venv:

```bash
pytest -q                                      # the default suite
pytest -m release -q                           # <org> filled in, checksums real (step 4's gate)
CUDA_VISIBLE_DEVICES=0 pytest tests/test_gpu_smoke.py -m gpu -q
```

That is the whole of what this repository can run, and it is deliberate: everything here needs only
this package, a CPU and (for the last line) a card. A user who has nothing else gets a clean
`pytest -q`.

**The port verification is not in this repository and is not a gate on this step.** It compares this
package against the research implementation, so it needs both at once plus the base model, the
recipe adapter, the corpora, the artifacts and a matched reference run — gigabytes that are not
distributed. It lives with the research code, as `tests/lfa_port_verification/` there, and
[`docs/verification.md`](docs/verification.md) is its report. Re-run it there when the release
changes anything the two implementations share; step 2 above is the one part of it a release
*always* re-runs, because a re-saved artifact is a new file.

**One part of it no longer compares.** Since 2026-09-08 this package's per-epoch chunk offset
rotates the chunk boundaries rather than discarding each document's leading tokens, so its training
stream deliberately differs from the research code's for any corpus with documents longer than one
chunk — and the setting that reproduced the old stream has been removed rather than kept as a
compatibility shim. Everything read at offset 0 (the corpus counts, the first collated batch) is
unaffected and still compares; the per-epoch loader and full-run comparisons need the two loaders
matched by hand on a branch. `docs/verification.md` opens with what that changes.

## 7. Publish

Push the tag, publish the package release with the sdist and the wheel, and leave the artifacts
where step 3 put them: the registry now points at them by digest, so the two releases are joined by
the checksum rather than by their tags.

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

Self-generation: `init --artifact self-generated` / `build-artifact --self-generated` (the model
writes the seed corpus p(h) is estimated on; no download), the supplement `train` now writes
with the entry model and mixes in at the recipe's `supplement_fraction` (0.13, the frame the
shipped λ was tuned at; every earlier companion run trained at 0), `prepare-supplement`,
`regenerate-artifact` and the chain's `artifact: regenerate` route. Two recorded deviations
from the research frame: the template says "about a text on <domain>" (was "a philosophy
text"), and the contamination screen against an evaluation set is not ported. The toolchain
check is a warning. Recipe fields `supplement_fraction`, `calibrated_self_generated`; artifact
meta fields `provenance`, `corpus_sha256`; history keys `supplement`, `artifact_route`.
Evidence scope: one model, one seed (C12, C14); rank 4, one seed (C15).

* **The artifact meta also records `layer_group_size`.** One torch generator is shared across a
  group's sites, so grouping changes the reservoir draws and the fitted mixtures (not the exact
  moments). The self-generated build chooses the group from host RAM when none is given; a rebuild
  reproduces a file by passing `--layer-group-size` with the recorded value.
* **The supplement** lives under `<workspace>/supplements/<corpus sha256[:12]>/` and is reused
  while the training side's hash, the writer checkpoint's hash and the template's hash match;
  `prepare-supplement --force` rewrites it. Its manifest records `chat_template_applied` (a writer
  without a chat template is warned about and is outside the recorded frame); duplicates are keyed
  on the question alone, as in the research writer.
* **`evaluate --compare-unanchored`**: the λ = μ = 0 control mixes the stage's own supplement at
  the stage's fraction, so it is still the same run without the anchor.
* **Not ported**: `--enforce-spec` (a comparison-only device) and the reasoning and instruction
  supplement modes (C14: not the lever). Self-generated rehearsal, a replay method, is out of
  scope.
* Docs: README, quickstart, `docs/rebuilding-the-artifact.md` (the self-generated route),
  `docs/adding-a-model.md`, `docs/recipes.md`, `docs/concepts.md` (what the supplement does:
  reachability; what it does not: protect skills), `docs/multi-domain-chains.md` (the regenerate
  route), `docs/faq.md` (8 GB cards, bf16 on Turing, the toolchain warning, what self-generation
  costs). `docs/verification.md` is unchanged: nothing in 0.2.0 was checked bit-for-bit against
  the research code.

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

First release: the package, the two published p(h) artifacts, the bundled Qwen3-0.6B recipe, and
the two example notebooks. Rehearsed against `docs/verification.md`.

---

### Already done, and how to re-check it

| | how to verify |
|---|---|
| `license = "Apache-2.0"` + `license-files` (PEP 639), `setuptools>=77` | the wheel's `METADATA` says `License-Expression: Apache-2.0` |
| `MANIFEST.in`, including `constraints-tested.txt` | `tar tzf dist/*.tar.gz` |
| no `<org>` placeholder left in anything that ships, and real artifact checksums | `pytest -m release -q` |
| `LICENSE` is the canonical Apache 2.0 text | the `diff` in step 0 |
| the `[html]` extra installed in the dev venv | `pytest -q -rs` shows no `needs the [html] extra` skip |
