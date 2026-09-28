# Usable Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new user goes from nothing to an adapted model on one page — a self-generated p(h)
artifact (built once, durable, reused from a local store), domain data prepared with the optional
question-and-answer supplement, train, evaluate, fuse — with no published artifact anywhere and
documentation written for that user.

**Architecture:** Four code units change or appear: the durable corpus writer
(`lfa/selfgen/artifact_corpus.py`), a local artifact store (`lfa/artifact/store.py`, new) that
`Workspace.init` goes through for `--artifact self-generated`, a supplement cache layer
(`lfa/supplements.py`, new) shared by the workspace and a workspace-free data-prep path, and the
recipe's calibration reference (`lfa/recipe.py` + the bundled YAML). The artifact registry and
fetch path are deleted. Then a wording sweep, the docs rewrite, the examples, and a handoff for the
GPU checks this pass does not run.

**Tech Stack:** Python ≥ 3.11, torch, transformers, pytest (`filterwarnings = error`), argparse.

**Spec:** `docs/superpowers/specs/2026-09-28-usable-pipeline-design.md`

## Global Constraints

- **No GPU runs in this pass.** Run only the fast tier: `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest -q`
  from the repo root. GPU-marked tests may be written; they are not run.
- Every commit uses the repo-local identity `sparkdoc <sparkdoc@users.noreply.github.com>`
  (already configured; never pass `--author`, never touch global git config).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Package version stays `0.2.0` (unreleased); its changelog in `RELEASING.md` gains this work.
- No shipped text (package, docs, README, examples, recipe YAML, CLI help, log and refusal
  messages) may contain `C1[0-9]` claim ids, "the LFA record", `mr-fusion`, `gmm1543k`, or
  `--artifact-id`. Exempt: `docs/verification.md`, `docs/superpowers/`, `tests/`.
- The supplement is described as making the domain's knowledge answerable/reachable. Never as a
  skill supplement or as protecting skills.
- Evidence statements carry their scope: "one model (Qwen3-0.6B), one seed, one domain".
- Match the surrounding code: docstrings in the same explanatory register, refusals as sentences
  ending in what to do instead, `logger = logging.getLogger(__name__)`.
- Remove `.worktrees/` leftovers before any run of `tests/test_release.py`.

## Review Focus

1. **A build interrupted between appending rows and writing the progress file** — resume must
   not duplicate or drop documents (Task 1 pins it: partial file longer than `rows_written` is
   truncated on resume).
2. **Two `lfa init --artifact self-generated` in two terminals for the same model** — the second
   must refuse with the lock's path and pid, not interleave writes into one corpus (Task 3).
3. **A user passing another workspace's `artifacts/v1.pt`** — provenance must survive, so the
   recipe stays silent and `extend` works (Task 4).
4. **`train` in a chain's stage 2 with a supplement prepared beside the corpus by the base
   model** — it must not reuse it (writer differs) and must write its own (Task 5).
5. **A trial build at `--n-raw 60`** reused silently later as if it were the recorded frame — the
   store key includes the frame, and the recipe warns naming the differing fields (Tasks 2, 3).

---

### Task 1: Durable self-generated corpus writer

**Files:**
- Modify: `lfa/selfgen/artifact_corpus.py` (whole module: `_write_share`, `write_artifact_corpus`,
  `SelfGenOptions`, module docstring)
- Test: `tests/test_selfgen_artifact_corpus.py`

**Interfaces:**
- Produces:
  - `SelfGenOptions.corpus_frame() -> dict` — fields `n_raw, n_chat, max_new_tokens, seed,
    chat_seed, min_chars, max_repeat_ratio, burn_in_tokens`.
  - `SelfGenOptions.artifact_frame() -> dict` — `corpus_frame()` plus `max_samples, gmm_k,
    pca_variance, reservoir_size`.
  - `frame_sha256(options: SelfGenOptions) -> str` — sha256 hex of
    `json.dumps(options.artifact_frame(), sort_keys=True)`.
  - `partial_path(corpus_path: Path) -> Path` = `<corpus>.partial`;
    `progress_path(corpus_path: Path) -> Path` = `<corpus>.progress.json`.
  - `write_artifact_corpus(model_id, out_path, options, *, generate=generate_texts, writer=None)
    -> dict` keeps its signature and return (the manifest); it now resumes and reuses.
  - `class CorpusFrameMismatch(ValueError)`.
  - `SelfGenOptions.frame()` is unchanged (still includes `batch_size`; the manifest keeps it).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_selfgen_artifact_corpus.py`,
  reusing its `_Tok`, `_Model`, `_gen_factory`)

```python
from lfa.selfgen.artifact_corpus import (CorpusFrameMismatch, frame_sha256, partial_path,
                                         progress_path)


def _deterministic(p, i, *, seed=None, batch_index=None):
    return f"doc seed {seed} batch {batch_index} item {i} with some words<|endoftext|>"


def _gen_with_kwargs(fail_on_call=None):
    """Deterministic in (seed, batch_index, position); raises KeyboardInterrupt on one call."""
    calls = []

    def generate(model, tokenizer, prompts, **kwargs):
        calls.append(kwargs["batch_index"])
        if fail_on_call is not None and len(calls) == fail_on_call:
            raise KeyboardInterrupt
        return [_deterministic(p, i, seed=kwargs["seed"], batch_index=kwargs["batch_index"])
                for i, p in enumerate(prompts)]
    generate.calls = calls
    return generate


def _options(**kw):
    base = dict(n_raw=10, n_chat=0, batch_size=3, min_docs=1)
    base.update(kw)
    return SelfGenOptions(**base)


def test_an_interrupted_build_resumes_to_the_same_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    whole = tmp_path / "whole" / "corpus.jsonl"
    write_artifact_corpus("stub", whole, _options(), generate=_gen_with_kwargs(),
                          writer=(_Model(), _Tok()))

    resumed = tmp_path / "resumed" / "corpus.jsonl"
    with pytest.raises(KeyboardInterrupt):
        write_artifact_corpus("stub", resumed, _options(), generate=_gen_with_kwargs(3),
                              writer=(_Model(), _Tok()))
    assert partial_path(resumed).is_file() and progress_path(resumed).is_file()
    assert not resumed.exists()                         # nothing final until it is complete
    second = _gen_with_kwargs()
    manifest = write_artifact_corpus("stub", resumed, _options(), generate=second,
                                     writer=(_Model(), _Tok()))

    assert second.calls[0] == 2                         # batches 0 and 1 were kept, not redone
    assert resumed.read_text() == whole.read_text()
    assert manifest["corpus_sha256"] == json.loads(
        (whole.parent / "corpus.jsonl.manifest.json").read_text())["corpus_sha256"]
    assert not partial_path(resumed).exists() and not progress_path(resumed).exists()


def test_rows_appended_after_the_last_progress_write_are_dropped_on_resume(tmp_path,
                                                                            monkeypatch):
    """A crash between the append and the progress write leaves extra lines; they go."""
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    whole = tmp_path / "whole" / "corpus.jsonl"
    write_artifact_corpus("stub", whole, _options(), generate=_gen_with_kwargs(),
                          writer=(_Model(), _Tok()))
    out = tmp_path / "torn" / "corpus.jsonl"
    with pytest.raises(KeyboardInterrupt):
        write_artifact_corpus("stub", out, _options(), generate=_gen_with_kwargs(3),
                              writer=(_Model(), _Tok()))
    with open(partial_path(out), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": "torn row", "source": "selfgen_raw"}) + "\n")
    write_artifact_corpus("stub", out, _options(), generate=_gen_with_kwargs(),
                          writer=(_Model(), _Tok()))
    assert out.read_text() == whole.read_text()


def test_empties_carry_across_a_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    calls, interrupt = [], {"on": True}

    def half_empty(model, tokenizer, prompts, **kwargs):
        calls.append(kwargs["batch_index"])
        if interrupt["on"] and len(calls) == 3:
            raise KeyboardInterrupt
        return ["<|endoftext|>" if i == 0 else f"text {kwargs['batch_index']} {i} ok<|endoftext|>"
                for i in range(len(prompts))]
    out = tmp_path / "corpus.jsonl"
    with pytest.raises(KeyboardInterrupt):
        write_artifact_corpus("stub", out, _options(max_empty_fraction=1.0), generate=half_empty,
                              writer=(_Model(), _Tok()))
    empties_before = json.loads(progress_path(out).read_text())["shares"]["selfgen_raw"]["empties"]
    assert empties_before == 2
    interrupt["on"] = False
    manifest = write_artifact_corpus("stub", out, _options(max_empty_fraction=1.0),
                                     generate=half_empty, writer=(_Model(), _Tok()))
    assert manifest["counts"]["empty"] >= empties_before + 1


def test_a_changed_frame_refuses_to_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    out = tmp_path / "corpus.jsonl"
    with pytest.raises(KeyboardInterrupt):
        write_artifact_corpus("stub", out, _options(), generate=_gen_with_kwargs(2),
                              writer=(_Model(), _Tok()))
    with pytest.raises(CorpusFrameMismatch, match="rebuild"):
        write_artifact_corpus("stub", out, _options(n_raw=11), generate=_gen_with_kwargs(),
                              writer=(_Model(), _Tok()))


def test_a_complete_corpus_is_reused_without_generating(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    out = tmp_path / "corpus.jsonl"
    first = write_artifact_corpus("stub", out, _options(), generate=_gen_with_kwargs(),
                                  writer=(_Model(), _Tok()))
    unused = _gen_with_kwargs()
    again = write_artifact_corpus("stub", out, _options(batch_size=5), generate=unused,
                                  writer=(_Model(), _Tok()))
    assert unused.calls == [] and again == first     # batch size does not change the frame


def test_a_complete_corpus_from_another_writer_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    out = tmp_path / "corpus.jsonl"
    write_artifact_corpus("stub", out, _options(), generate=_gen_with_kwargs(),
                          writer=(_Model(), _Tok()))
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "d" * 64)
    with pytest.raises(CorpusFrameMismatch, match="writer"):
        write_artifact_corpus("stub", out, _options(), generate=_gen_with_kwargs(),
                              writer=(_Model(), _Tok()))


def test_progress_is_logged_per_batch(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    with caplog.at_level("INFO"):
        write_artifact_corpus("stub", tmp_path / "c.jsonl", _options(), generate=_gen_with_kwargs(),
                              writer=(_Model(), _Tok()))
    lines = [r.message for r in caplog.records if "self-generated corpus:" in r.message]
    assert lines[0].startswith("self-generated corpus: 3/10 documents")
    assert lines[-1].startswith("self-generated corpus: 10/10 documents")


def test_the_frames_and_their_hash():
    o = SelfGenOptions()
    assert o.corpus_frame() == {"n_raw": 2500, "n_chat": 0, "max_new_tokens": 2048, "seed": 42,
                                "chat_seed": 43, "min_chars": 1, "max_repeat_ratio": 1.0,
                                "burn_in_tokens": 0}
    assert o.artifact_frame() == {**o.corpus_frame(), "max_samples": 600_000, "gmm_k": 32,
                                  "pca_variance": 0.95, "reservoir_size": 200_000}
    assert frame_sha256(o) == frame_sha256(SelfGenOptions(batch_size=8, device="cpu"))
    assert frame_sha256(o) != frame_sha256(SelfGenOptions(n_raw=60))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest tests/test_selfgen_artifact_corpus.py -q`
Expected: FAIL — `ImportError: cannot import name 'CorpusFrameMismatch'`.

- [ ] **Step 3: Implement**

In `lfa/selfgen/artifact_corpus.py`:

1. Module docstring: replace the first paragraph's research-path parenthetical with plain text:
   "The recorded frame: 2,500 raw documents ... seed 42, and nothing else. The artifact fitted on
   it at 600k samples per site, K=32, matched an artifact fitted on real text at every lambda
   tried and was at least as good at the recipe's lambda -- one model (Qwen3-0.6B), one seed,
   one domain." In the second paragraph replace "the record's own audit" with "the research
   run's own audit". Add a third paragraph: "The writer is durable: each finished batch is
   appended to ``<corpus>.partial`` and ``<corpus>.progress.json`` is rewritten, so an
   interrupted build resumes at the next batch with the same per-batch seed; a complete corpus
   with a matching frame and writer is reused without generating."

2. Add, after `_GENERATION_FIELDS`:

```python
_CORPUS_FIELDS = ("n_raw", "n_chat", "max_new_tokens", "seed", "chat_seed", "min_chars",
                  "max_repeat_ratio", "burn_in_tokens")
_FIT_FIELDS = ("max_samples", "gmm_k", "pca_variance", "reservoir_size")


class CorpusFrameMismatch(ValueError):
    """Raised when an existing corpus or partial build was written under another frame or writer."""


def partial_path(corpus_path) -> Path:
    corpus_path = Path(corpus_path)
    return corpus_path.with_name(corpus_path.name + ".partial")


def progress_path(corpus_path) -> Path:
    corpus_path = Path(corpus_path)
    return corpus_path.with_name(corpus_path.name + ".progress.json")


def frame_sha256(options: "SelfGenOptions") -> str:
    """What identifies a self-generated artifact's frame: the corpus and fit fields, not speed."""
    blob = json.dumps(options.artifact_frame(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()
```

   and on `SelfGenOptions`:

```python
    def corpus_frame(self) -> dict:
        """The fields that shape the corpus (not ``batch_size``: speed, not content frame)."""
        return {name: getattr(self, name) for name in _CORPUS_FIELDS}

    def artifact_frame(self) -> dict:
        """The corpus frame plus the fit fields: what the artifact was estimated at."""
        return {**self.corpus_frame(), **{name: getattr(self, name) for name in _FIT_FIELDS}}
```

   Add `import hashlib, os` and `__all__` entries `CorpusFrameMismatch`, `frame_sha256`,
   `partial_path`, `progress_path`.

3. Rewrite `_write_share` to be driven by a mutable per-share `state` dict
   (`{"batch_index": int, "kept": int, "empties": int}`) and a `sink(rows)` callback, generating
   until `state["kept"] >= n_docs` or `state["empties"] > max_empties`. Batch size is
   `min(batch_size, n_docs - state["kept"])` where `batch_size` is a parameter (the recorded one on
   resume). After each batch: `state["batch_index"] += 1`, `state["kept"] += len(rows)`, then
   `sink(rows)`. It returns nothing.

4. Rewrite `write_artifact_corpus`:

```python
    out_path = Path(out_path)
    writer_sha256 = checkpoint_sha256(model_id)          # before the hours of sampling, not after
    manifest_file = Path(str(out_path) + ".manifest.json")
    if out_path.is_file() and manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text())
        _check_reusable(manifest, options, writer_sha256, out_path)   # raises CorpusFrameMismatch
        logger.info("Self-generated corpus already complete at %s; not generating again",
                    out_path)
        return manifest
    progress = _load_progress(out_path, options, writer_sha256)       # fresh dict if none
    batch_size = progress["batch_size"]
    rows_on_disk = _truncate_partial(out_path, progress["rows_written"])
    ... load writer, prefix, markers, header exactly as today ...
    def sink(rows):
        with open(partial_path(out_path), "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        progress["rows_written"] += len(rows)
        _save_progress(out_path, progress)
        done = sum(s["kept"] for s in progress["shares"].values())
        empty = sum(s["empties"] for s in progress["shares"].values())
        logger.info("self-generated corpus: %d/%d documents (%d empty)", done, asked, empty)
```

   Run the raw share with `state=progress["shares"]["selfgen_raw"]`, then (if `n_chat`) the chat
   share with `max_empties=limit - raw_state["empties"]`. Then apply the two `DegenerateCorpus`
   checks exactly as today using the shares' `kept`/`empties` (the partial and progress files are
   left in place when refusing, so a knowing retry with a looser limit resumes). On success:
   read the partial file's rows, `os.replace(partial, out_path)`, compute `corpus_sha256` over
   the rows' texts, write the manifest (unchanged fields; `counts` from the shares), then
   `progress_path(out_path).unlink()`.

   Helpers (module-private):
   - `_load_progress(out_path, options, writer_sha256) -> dict`: when the progress file exists,
     read it; if its `frame_sha256 != frame_sha256(options)` or its `writer_sha256` differs,
     raise `CorpusFrameMismatch(f"{partial_path(out_path)} is a partial build under a different
     frame or writer. Re-run with the frame it was started with, or discard it: `lfa init ...
     --rebuild` (store) or delete {partial_path(out_path)} and {progress_path(out_path)}.")`.
     Otherwise return a fresh `{"frame_sha256", "writer_sha256", "batch_size":
     options.batch_size, "rows_written": 0, "shares": {"selfgen_raw": {"batch_index": 0, "kept":
     0, "empties": 0}, "selfgen_chatfmt": {...same}}}`. A fresh start also deletes a stray partial
     file.
   - `_truncate_partial(out_path, n) -> int`: keep only the first `n` lines of the partial file
     (rewrite via temp + `os.replace` when longer); return `n`.
   - `_save_progress(out_path, progress)`: write JSON to `<progress>.tmp` then `os.replace`.
   - `_check_reusable(manifest, options, writer_sha256, out_path)`: compare
     `{k: manifest["frame"][k] for k in _CORPUS_FIELDS}` with `options.corpus_frame()` and
     `manifest["writer_sha256"]` with `writer_sha256`; on mismatch raise `CorpusFrameMismatch`
     naming which (the word "writer" when the writer differs, "frame" otherwise) and the
     `--rebuild` / delete remedy.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest tests/test_selfgen_artifact_corpus.py -q`
Expected: all pass, including the pre-existing tests (update `test_the_recorded_frame_is_the_default`'s
comment, which says "The C12 corpus held no chat-format documents (the record's audit)", to
"The recorded corpus held no chat-format documents (the research run's audit)").

- [ ] **Step 5: Run the full fast tier and commit**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest -q` — expected: all pass.

```bash
git add lfa/selfgen/artifact_corpus.py tests/test_selfgen_artifact_corpus.py
git commit -m "Self-generated corpus: durable batches, resume at the next batch, reuse a complete corpus

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The artifact records its frame; the recipe is calibrated against the self-generated artifact

**Files:**
- Modify: `lfa/artifact/schema.py` (`make_meta`), `lfa/artifact/build.py` (`build_artifact`,
  `build_artifact_self_generated`), `lfa/recipe.py`, `lfa/recipes/qwen3-0.6b.yaml`
- Test: `tests/test_recipe.py`, `tests/test_schema.py`, `tests/test_artifact_build.py`

**Interfaces:**
- Consumes: `SelfGenOptions.artifact_frame()` (Task 1).
- Produces:
  - `make_meta(..., selfgen_frame: dict | None = None)` — stored under `"selfgen_frame"` when
    not `None`.
  - `build_artifact(..., selfgen_frame: dict | None = None)` pass-through.
  - `build_artifact_self_generated(...)` passes `selfgen_frame=options.artifact_frame()`.
  - `lfa.recipe.SELF_GENERATED_REFERENCE = "self-generated"`.
  - `lfa.recipe.RECORDED_SELF_GENERATED_FRAME = {"n_raw": 2500, "n_chat": 0, "max_new_tokens":
    2048, "max_samples": 600_000, "gmm_k": 32, "pca_variance": 0.95}`.
  - `Recipe.calibrated_artifact: str = "self-generated"`; `Recipe.self_generated_frame: dict`
    (default a copy of `RECORDED_SELF_GENERATED_FRAME`); `calibrated_self_generated` **removed**.
  - `Recipe.warnings(rank, artifact_id, artifact_meta=None) -> list[str]` (signature unchanged).
  - `Recipe.bundled_for(model_id: str) -> str | None` (classmethod; the logic now in
    `lfa.workspace._bundled_recipe_for`, which becomes a one-line delegate).

- [ ] **Step 1: Write the failing tests** (in `tests/test_recipe.py`; delete the tests that
  exercise `calibrated_self_generated` and replace them with these)

```python
from lfa.recipe import RECORDED_SELF_GENERATED_FRAME, SELF_GENERATED_REFERENCE


def _selfgen_meta(model_id, **frame):
    return {"provenance": "self-generated", "model_id": model_id,
            "selfgen_frame": {**RECORDED_SELF_GENERATED_FRAME, "seed": 42, **frame}}


def test_the_bundled_recipe_is_calibrated_against_the_self_generated_artifact():
    recipe = Recipe.load("qwen3-0.6b")
    assert recipe.artifact == SELF_GENERATED_REFERENCE
    assert recipe.calibrated_artifact == SELF_GENERATED_REFERENCE
    assert recipe.self_generated_frame == RECORDED_SELF_GENERATED_FRAME


def test_a_self_generated_artifact_at_the_recorded_frame_is_silent():
    recipe = Recipe.load("qwen3-0.6b")
    meta = _selfgen_meta(recipe.model_id)
    assert recipe.warnings(recipe.calibrated_rank, "self-generated:abc", meta) == []


def test_a_trial_frame_names_the_fields_that_differ():
    recipe = Recipe.load("qwen3-0.6b")
    meta = _selfgen_meta(recipe.model_id, n_raw=60, max_new_tokens=128)
    notes = recipe.warnings(recipe.calibrated_rank, "self-generated:abc", meta)
    assert len(notes) == 1
    assert "n_raw 60 (calibrated at 2500)" in notes[0]
    assert "max_new_tokens 128 (calibrated at 2048)" in notes[0]


def test_a_self_generated_artifact_with_no_recorded_frame_says_so():
    recipe = Recipe.load("qwen3-0.6b")
    meta = {"provenance": "self-generated", "model_id": recipe.model_id}
    assert "records no generation frame" in recipe.warnings(32, "x", meta)[0]


def test_another_models_self_generated_artifact_is_a_mismatch():
    recipe = Recipe.load("qwen3-0.6b")
    notes = recipe.warnings(32, "x", _selfgen_meta("other/model"))
    assert "'other/model'" in notes[0]


def test_a_real_text_artifact_against_a_self_generated_calibration_is_noted():
    recipe = Recipe.load("qwen3-0.6b")
    notes = recipe.warnings(32, "/data/mine.pt", {"model_id": recipe.model_id})
    assert len(notes) == 1 and "fitted on the model's own text" in notes[0]
    assert "/data/mine.pt" in notes[0]


def test_a_recipe_calibrated_against_a_named_artifact_still_compares_ids(tmp_path):
    recipe = Recipe(name="mine", model_id="m", artifact="/a.pt", calibrated_artifact="/a.pt")
    assert recipe.warnings(32, "/a.pt") == []
    assert "'/a.pt'" in recipe.warnings(32, "/b.pt")[0]


def test_bundled_for_finds_the_recipe_that_names_the_model():
    assert Recipe.bundled_for("Qwen/Qwen3-0.6B") == "qwen3-0.6b"
    assert Recipe.bundled_for("nobody/nothing") is None


def test_the_recorded_frame_agrees_with_selfgen_options():
    from lfa.selfgen.artifact_corpus import SelfGenOptions
    frame = SelfGenOptions().artifact_frame()
    assert {k: frame[k] for k in RECORDED_SELF_GENERATED_FRAME} == RECORDED_SELF_GENERATED_FRAME
```

   In `tests/test_schema.py` add:

```python
def test_make_meta_records_the_self_generated_frame():
    meta = make_meta("m", 8, 2, ["pre_qkv"], 10, provenance="self-generated",
                     selfgen_frame={"n_raw": 60})
    assert meta["selfgen_frame"] == {"n_raw": 60}
    assert "selfgen_frame" not in make_meta("m", 8, 2, ["pre_qkv"], 10)
```

   In `tests/test_artifact_build.py`, extend the existing self-generated build test (the one that
   checks `provenance`) with `assert meta["selfgen_frame"] == options.artifact_frame()`.

- [ ] **Step 2: Run to verify failure**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest tests/test_recipe.py tests/test_schema.py tests/test_artifact_build.py -q`
Expected: FAIL — `ImportError: cannot import name 'RECORDED_SELF_GENERATED_FRAME'`.

- [ ] **Step 3: Implement**

- `schema.make_meta`: add the `selfgen_frame` keyword; document it in the Args block ("the
  generation and fit frame of a self-generated artifact (:meth:`SelfGenOptions.artifact_frame`),
  which :meth:`lfa.recipe.Recipe.warnings` compares with the recipe's calibrated frame").
  Remove the docstring sentence about "The shipped qwen3-0.6b artifact is named for it:
  ``gmm1543k`` is ..." — replace with "It is what a continual extension reads back when the
  blocks carry no count of their own." Also replace the `built_with` sentence's
  "``RELEASING.md`` step 1 does" with "a meta block added to a file something else built should
  say so".
- `build.build_artifact`: add `selfgen_frame: dict | None = None`, pass to `make_meta`.
  `build_artifact_self_generated`: pass `selfgen_frame=options.artifact_frame()`; in its
  docstring replace "the frame of the artifact behind C12" with "the recorded frame".
- `recipe.py`:

```python
#: The value of ``calibrated_artifact`` (and of a bundled recipe's ``artifact``) that means "an
#: artifact fitted on this model's own text at :attr:`Recipe.self_generated_frame`".
SELF_GENERATED_REFERENCE = "self-generated"

#: The frame the shipped lambda's self-generated calibration was measured at. It agrees with
#: :class:`lfa.selfgen.artifact_corpus.SelfGenOptions`' defaults (a test pins that); it is
#: spelled out here so this module does not import the generation stack.
RECORDED_SELF_GENERATED_FRAME = {"n_raw": 2500, "n_chat": 0, "max_new_tokens": 2048,
                                 "max_samples": 600_000, "gmm_k": 32, "pca_variance": 0.95}
```

  Fields: `calibrated_artifact: str = SELF_GENERATED_REFERENCE`,
  `self_generated_frame: dict = dataclasses.field(default_factory=lambda:
  dict(RECORDED_SELF_GENERATED_FRAME))`; delete `calibrated_self_generated` (field, docstring
  entry and comment). `__post_init__` refuses a `self_generated_frame` that is not a dict or has
  a key outside `RECORDED_SELF_GENERATED_FRAME`'s plus `seed`, `chat_seed`, `min_chars`,
  `max_repeat_ratio`, `burn_in_tokens`, `reservoir_size`.
  Update the class docstring: `artifact` is "what the recipe anchors against by default: an
  artifact id, a path, or ``self-generated``"; `calibrated_artifact` is "what the lambdas were
  calibrated against: ``self-generated`` (an artifact fitted on this model's own text at
  :attr:`self_generated_frame`), or an id or path"; add `self_generated_frame`.

  `warnings` body for the artifact part:

```python
        meta = artifact_meta or {}
        self_generated = meta.get("provenance") == "self-generated"
        if self_generated and meta.get("model_id") not in (None, self.model_id):
            notes.append(
                f"this self-generated artifact describes {meta.get('model_id')!r}, not this "
                f"recipe's {self.model_id!r}: lambda is coupled to the p(h) artifact, so "
                "calibrate it against held-out domain perplexity for this model.")
        elif self.calibrated_artifact == SELF_GENERATED_REFERENCE and self_generated:
            frame = meta.get("selfgen_frame")
            if frame is None:
                notes.append(
                    "this self-generated artifact records no generation frame, so it cannot be "
                    "checked against the frame this recipe's lambda was calibrated at "
                    f"({_describe(self.self_generated_frame)}); rebuild it with this version "
                    "(`lfa init ... --artifact self-generated --rebuild`) to have it recorded.")
            else:
                differ = [f"{key} {frame.get(key)!r} (calibrated at {value!r})"
                          for key, value in self.self_generated_frame.items()
                          if frame.get(key) != value]
                if differ:
                    notes.append(
                        "this self-generated artifact was built at a different frame from the "
                        "one this recipe's lambda was calibrated at: " + ", ".join(differ)
                        + ". A trial-sized build is fine for trying the pipeline; for a real "
                        "run, build at the recorded frame or calibrate lambda against held-out "
                        "domain perplexity (docs/adding-a-model.md).")
        elif self.calibrated_artifact == SELF_GENERATED_REFERENCE:
            notes.append(
                f"this recipe's lambda ({quoted}) is calibrated against an artifact fitted on the "
                f"model's own text at {_describe(self.self_generated_frame)}, and you are "
                f"anchoring against {artifact_id!r}, which is not one. A different artifact "
                "prices the same function differently, so calibrate lambda against held-out "
                "domain perplexity (docs/adding-a-model.md), reading the frontier rather than "
                "a single point.")
        elif artifact_id != self.calibrated_artifact:
            ... the existing id-comparison note, unchanged ...
```

  with `_describe(frame)` a module helper rendering e.g. `"2500 documents x 2048 tokens, 600000
  samples per site, K=32"` (use `frame.get` so a partial user frame still renders).
  `Recipe.bundled_for(model_id)` holds the body of `workspace._bundled_recipe_for`
  (importing `yaml` is already done in recipe.py); `workspace._bundled_recipe_for` returns
  `Recipe.bundled_for(model_id)`.
  Replace the comment above the removed field and the module docstring's claim wording: no "C12".
- `lfa/recipes/qwen3-0.6b.yaml`:
  - header comment's first paragraph: "rank-32 LoRA ... over a p(h) artifact fitted on the
    model's own text (see `self_generated_frame` below), anchored at lambda = 100,000 ...".
  - the `The p(h) ARTIFACT.` bullet: "`calibrated_artifact` below records what these lambdas are
    priced against. They were tuned against an artifact fitted on real text (a 10:1
    pretraining-to-instruction seed corpus, 1.54 M hidden vectors per site); an artifact fitted
    on the model's own text at `self_generated_frame` matched it at every lambda tried and was at
    least as good at this lambda -- one model, one seed, one domain -- and it is the one this
    package builds. Any other artifact is a re-tune."
  - `artifact: self-generated`; `calibrated_artifact: self-generated`; add

```yaml
# the frame the self-generated calibration was measured at; `lfa init --artifact
# self-generated` builds at exactly this by default, and a build at any other frame is told so
self_generated_frame:
  n_raw: 2500
  n_chat: 0
  max_new_tokens: 2048
  max_samples: 600000
  gmm_k: 32
  pca_variance: 0.95
```

  - remove `calibrated_self_generated: true` and its comment; the `supplement_fraction: 0.13`
    comment becomes "# -- the question-and-answer supplement the stage's entry model writes from
    the domain is mixed in at this share of training tokens, the frame lambda was tuned at".

- [ ] **Step 4: Fix the other callers** — `grep -rn "calibrated_self_generated" lfa tests` must
  come back empty. In `tests/test_workspace.py` the call
  `tiny_recipe(base_dir, supplement_fraction=0.0, calibrated_self_generated=False)` drops the
  keyword and instead passes `calibrated_artifact="tiny"` (the test's intent: a recipe that does
  not treat self-generation as calibrated). In `lfa/workspace.py`, the two tests of provenance in
  `train` keep working because `warnings` still receives `self._artifact_meta()`.
  `tests/conftest.py::tiny_recipe` keeps `calibrated_artifact="tiny"` (the fixture artifact is
  real-text-shaped, and its tests compare by id).

- [ ] **Step 5: Run and commit**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest -q` — expected: all pass.

```bash
git add lfa/artifact/schema.py lfa/artifact/build.py lfa/recipe.py lfa/recipes/qwen3-0.6b.yaml lfa/workspace.py tests/
git commit -m "The artifact records its self-generated frame; the recipe is calibrated against it

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The local artifact store

**Files:**
- Create: `lfa/artifact/store.py`
- Test: `tests/test_artifact_store.py` (new)

**Interfaces:**
- Consumes: `SelfGenOptions`, `frame_sha256`, `CorpusFrameMismatch` (Task 1);
  `build_artifact_self_generated(model_id, out_path, options, *, corpus_path, generate, writer)`
  (existing; records `selfgen_frame` after Task 2); `checkpoint_sha256` (existing,
  `lfa.selfgen.generate`).
- Produces:
  - `STORE_ENV = "LFA_ARTIFACT_STORE"`; `store_root() -> Path` (env var, else
    `Path.home() / ".cache" / "lfa" / "artifacts"`).
  - `entry_dir(model_id: str, writer_sha256: str, options: SelfGenOptions) -> Path`:
    `store_root() / f"{_slug(model_id)}-{writer_sha256[:12]}-{frame_sha256(options)[:12]}"`,
    `_slug` = the model id with every run of characters outside `[A-Za-z0-9._-]` replaced by
    `--`, last 60 characters kept.
  - `class StoreLocked(RuntimeError)`.
  - `obtain_self_generated(model_id: str, options: SelfGenOptions, *, rebuild: bool = False,
    generate=generate_texts, writer=None) -> tuple[Path, dict]` — returns
    `(entry / "artifact.pt", corpus manifest dict)`.
  - `list_store() -> list[dict]` — one dict per entry directory (skipping `*.replaced-*`):
    `{"path", "model_id", "frame", "built_at", "size_mb", "state"}` where `state` is `"built"`,
    or `"in progress: <done>/<asked> documents"`, or `"corpus complete, not fitted"`.

- [ ] **Step 1: Write the failing tests** (`tests/test_artifact_store.py`)

```python
"""The local store self-generated artifacts are built into and reused from."""
import json
import os

import pytest

import lfa.artifact.store as store
from lfa.selfgen.artifact_corpus import SelfGenOptions


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
    monkeypatch.setattr(store, "checkpoint_sha256", lambda model_id: "a" * 64)
    return tmp_path / "store"


def _fake_build(calls, *, fail=False):
    def build(model_id, out_path, options, *, corpus_path, generate=None, writer=None):
        calls.append(out_path)
        corpus_path.write_text('{"text": "x", "source": "selfgen_raw"}\n')
        manifest = {"corpus_sha256": "e" * 64, "writer_sha256": "a" * 64,
                    "frame": options.frame()}
        corpus_path.with_name(corpus_path.name + ".manifest.json").write_text(json.dumps(manifest))
        if fail:
            raise MemoryError("fit ran out of host memory")
        out_path.write_bytes(b"artifact")
        return out_path
    return build


def test_the_store_lives_where_the_environment_says(isolated_store):
    assert store.store_root() == isolated_store


def test_a_first_build_lands_in_the_entry_and_a_second_reuses_it(isolated_store, monkeypatch,
                                                                 caplog):
    calls = []
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls))
    options = SelfGenOptions(n_raw=60)
    first, manifest = store.obtain_self_generated("Qwen/Qwen3-0.6B", options)
    assert first == store.entry_dir("Qwen/Qwen3-0.6B", "a" * 64, options) / "artifact.pt"
    assert first.read_bytes() == b"artifact" and manifest["corpus_sha256"] == "e" * 64
    assert json.loads((first.parent / "entry.json").read_text())["model_id"] == "Qwen/Qwen3-0.6B"
    with caplog.at_level("INFO"):
        second, _ = store.obtain_self_generated("Qwen/Qwen3-0.6B", SelfGenOptions(n_raw=60,
                                                                               batch_size=4))
    assert second == first and len(calls) == 1
    assert any("Reused the self-generated artifact" in r.message for r in caplog.records)


def test_a_different_frame_or_checkpoint_is_a_different_entry(isolated_store):
    a = store.entry_dir("m", "a" * 64, SelfGenOptions())
    assert a != store.entry_dir("m", "a" * 64, SelfGenOptions(n_raw=60))
    assert a != store.entry_dir("m", "b" * 64, SelfGenOptions())
    assert a.name.startswith("m-aaaaaaaaaaaa-")


def test_a_failed_fit_keeps_the_corpus_and_the_next_call_fits_without_regenerating(
        isolated_store, monkeypatch):
    calls = []
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls, fail=True))
    with pytest.raises(MemoryError):
        store.obtain_self_generated("m", SelfGenOptions(n_raw=60))
    entry = store.entry_dir("m", "a" * 64, SelfGenOptions(n_raw=60))
    assert (entry / "corpus.jsonl").is_file() and not (entry / "artifact.pt").exists()
    assert not (entry / ".lock").exists()
    assert store.list_store()[0]["state"] == "corpus complete, not fitted"
    # The real builder reuses a complete corpus (Task 1); here the retry just has to be routed
    # to the same entry and corpus path.
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls))
    path, _ = store.obtain_self_generated("m", SelfGenOptions(n_raw=60))
    assert path.is_file() and calls[0] == calls[1] == entry / "artifact.partial.pt"


def test_rebuild_moves_the_old_entry_aside(isolated_store, monkeypatch):
    calls = []
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build(calls))
    store.obtain_self_generated("m", SelfGenOptions(n_raw=60))
    store.obtain_self_generated("m", SelfGenOptions(n_raw=60), rebuild=True)
    assert len(calls) == 2
    assert len(list(isolated_store.glob("*.replaced-*"))) == 1
    assert len(store.list_store()) == 1


def test_a_held_lock_refuses(isolated_store, monkeypatch):
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    entry = store.entry_dir("m", "a" * 64, SelfGenOptions(n_raw=60))
    entry.mkdir(parents=True)
    (entry / ".lock").write_text(str(os.getpid()))           # this process: alive
    with pytest.raises(store.StoreLocked, match=str(os.getpid())):
        store.obtain_self_generated("m", SelfGenOptions(n_raw=60))


def test_a_stale_lock_is_taken_over(isolated_store, monkeypatch, caplog):
    monkeypatch.setattr(store, "build_artifact_self_generated", _fake_build([]))
    entry = store.entry_dir("m", "a" * 64, SelfGenOptions(n_raw=60))
    entry.mkdir(parents=True)
    (entry / ".lock").write_text("999999999")                # no such pid
    with caplog.at_level("WARNING"):
        path, _ = store.obtain_self_generated("m", SelfGenOptions(n_raw=60))
    assert path.is_file() and any("stale" in r.message for r in caplog.records)


def test_list_store_shows_a_build_in_progress(isolated_store):
    entry = store.entry_dir("m", "a" * 64, SelfGenOptions(n_raw=60))
    entry.mkdir(parents=True)
    (entry / "corpus.jsonl.progress.json").write_text(json.dumps(
        {"shares": {"selfgen_raw": {"kept": 20}, "selfgen_chatfmt": {"kept": 0}}}))
    (entry / "entry.json").write_text(json.dumps(
        {"model_id": "m", "frame": SelfGenOptions(n_raw=60).artifact_frame(), "asked": 60}))
    [row] = store.list_store()
    assert row["state"] == "in progress: 20/60 documents"
```

- [ ] **Step 2: Run to verify failure**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest tests/test_artifact_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lfa.artifact.store'`.

- [ ] **Step 3: Implement `lfa/artifact/store.py`**

Module docstring: "Where self-generated artifacts are built and found again. A self-generated
artifact costs hours to build and is fixed by three things: the model checkpoint (by its sha256),
the frame (:meth:`SelfGenOptions.artifact_frame`), and this package's builder. The store keys an
entry on the first two, so a second workspace over the same model at the same frame copies the
finished file in rather than building it again, and a build that stopped part-way resumes in the
same entry. ``$LFA_ARTIFACT_STORE`` moves it; the default is ``~/.cache/lfa/artifacts``."

Behaviour of `obtain_self_generated`:

```python
def obtain_self_generated(model_id, options, *, rebuild=False, generate=generate_texts,
                          writer=None):
    writer_sha256 = checkpoint_sha256(model_id)
    entry = entry_dir(model_id, writer_sha256, options)
    if rebuild and entry.exists():
        aside = entry.with_name(f"{entry.name}.replaced-{time.strftime('%Y%m%d-%H%M%S')}")
        entry.rename(aside)
        logger.info("Moved the previous store entry aside to %s", aside)
    artifact = entry / "artifact.pt"
    corpus = entry / "corpus.jsonl"
    if artifact.is_file():
        built = time.strftime("%Y-%m-%d", time.localtime(artifact.stat().st_mtime))
        logger.info("Reused the self-generated artifact built %s from %s", built, entry)
        return artifact, _manifest(corpus)
    entry.mkdir(parents=True, exist_ok=True)
    _write_entry_json(entry, model_id, writer_sha256, options)          # model id, frame, asked
    with _lock(entry):
        partial = entry / "artifact.partial.pt"
        build_artifact_self_generated(model_id, partial, options, corpus_path=corpus,
                                      generate=generate, writer=writer)
        os.replace(partial, artifact)
    logger.info("Self-generated artifact stored at %s", artifact)
    return artifact, _manifest(corpus)
```

`_lock(entry)` is a context manager: `os.open(entry / ".lock", O_CREAT | O_EXCL | O_WRONLY)` and
write the pid; on `FileExistsError` read the pid — if `_alive(pid)` (via `os.kill(pid, 0)`,
`ProcessLookupError` → dead, `PermissionError` → alive) raise `StoreLocked(f"{entry} is being
built by process {pid} (lock {entry / '.lock'}). Wait for it to finish, then run the same
command again: it will reuse the finished artifact.")`; otherwise log a WARNING containing
"stale lock", overwrite it, and proceed. The lock is removed in a `finally`.
`build_artifact_self_generated` in `lfa/artifact/build.py` must accept `generate` and `writer`
keywords and pass them through (it already does) — confirm, do not change its signature.
`entry.json` holds `{"model_id", "writer_sha256", "frame": options.artifact_frame(), "asked":
options.n_raw + options.n_chat, "lfa_version"}`; `list_store` reads it, the progress file, and
the presence of `corpus.jsonl` / `artifact.pt` to compute `state`, and reports `built_at` from
`artifact.pt`'s mtime (`None` when unbuilt) and `size_mb` rounded to an int. Export
`obtain_self_generated`, `list_store`, `store_root`, `entry_dir`, `StoreLocked`, `STORE_ENV`
from `lfa/artifact/__init__.py` (Task 4 removes the fetch exports from the same file).

- [ ] **Step 4: Run and commit**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest -q` — expected: all pass.

```bash
git add lfa/artifact/store.py lfa/artifact/__init__.py tests/test_artifact_store.py
git commit -m "A local store for self-generated artifacts: keyed on checkpoint and frame, locked, resumable

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: No published artifacts; `init` goes through the store

**Files:**
- Delete: `lfa/artifact/fetch.py`, `tests/test_fetch.py`
- Modify: `lfa/artifact/__init__.py`, `lfa/workspace.py` (`init`, class docstring,
  `_artifact_id`, `_resolve_base_n`, the no-artifact refusal in `train`, module docstring line
  22), `lfa/cli.py` (imports, `USER_FACING_ERRORS`, `_init`, `_list_artifacts`, parser for
  `init`/`fetch-artifact`/`list-artifacts`, the Ctrl-C message, module docstring),
  `examples/quickstart.py` and `examples/chain_three_domains.py` (only: drop the `--artifact-id`
  flag and the `artifact_id=` argument, make `--artifact` required — Task 8 rewrites their prose)
- Test: `tests/conftest.py`, `tests/test_workspace.py`, `tests/test_cli.py`,
  `tests/test_examples.py`, `tests/test_release.py`

**Interfaces:**
- Consumes: `obtain_self_generated`, `list_store`, `StoreLocked` (Task 3); `CorpusFrameMismatch`
  (Task 1).
- Produces:
  - `Workspace.init(path, model_id, artifact, recipe=None, selfgen=None, *, rebuild=False)`.
    `artifact` is `"self-generated"` or a path; no `fetch`, no `artifact_id`.
  - CLI: `lfa init PATH --model ID --artifact {self-generated|PATH} [--rebuild] [--recipe]
    [--n-raw] [--n-chat] [--max-new-tokens] [--device]`; `lfa list-artifacts` lists the store;
    no `fetch-artifact`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_workspace.py` (new tests; the fixture changes are Step 3):

```python
def test_init_self_generated_goes_through_the_store_and_copies_the_corpus_in(
        tmp_path, base_dir, monkeypatch):
    import lfa.artifact.store as store_module
    monkeypatch.setenv("LFA_ARTIFACT_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(store_module, "checkpoint_sha256", lambda m: "a" * 64)
    calls = []

    def fake_build(model_id, out_path, options, *, corpus_path, generate=None, writer=None):
        calls.append(out_path)
        corpus_path.write_text('{"text": "x", "source": "selfgen_raw"}\n')
        Path(str(corpus_path) + ".manifest.json").write_text(json.dumps(
            {"corpus_sha256": "e" * 64, "frame": options.frame()}))
        out_path.write_bytes(Path(TINY_ARTIFACT_PATH).read_bytes())
        return out_path
    monkeypatch.setattr(store_module, "build_artifact_self_generated", fake_build)

    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact="self-generated")
    assert (ws.path / "artifacts" / "v1.pt").is_file()
    assert (ws.path / "artifacts" / "v1.corpus.jsonl").is_file()
    assert (ws.path / "artifacts" / "v1.corpus.jsonl.manifest.json").is_file()
    assert ws.state["artifact_id"] == "self-generated:" + "e" * 12
    assert ws.state["artifact_provenance"] == "self-generated"

    again = Workspace.init(tmp_path / "ws2", str(base_dir), artifact="self-generated")
    assert len(calls) == 1 and again.state["artifact_id"] == ws.state["artifact_id"]


def test_a_failed_self_generated_build_leaves_no_workspace_but_keeps_the_store(
        tmp_path, base_dir, monkeypatch):
    import lfa.artifact.store as store_module
    monkeypatch.setenv("LFA_ARTIFACT_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(store_module, "checkpoint_sha256", lambda m: "a" * 64)

    def dies_in_the_fit(model_id, out_path, options, *, corpus_path, **_):
        corpus_path.write_text('{"text": "x"}\n')
        raise MemoryError("fit")
    monkeypatch.setattr(store_module, "build_artifact_self_generated", dies_in_the_fit)
    with pytest.raises(MemoryError):
        Workspace.init(tmp_path / "ws", str(base_dir), artifact="self-generated")
    assert not (tmp_path / "ws").exists()
    assert list((tmp_path / "store").glob("*/corpus.jsonl"))


def test_a_self_generated_file_passed_by_path_keeps_its_provenance(tmp_path, base_dir):
    params = torch.load(TINY_ARTIFACT_PATH, map_location="cpu", weights_only=False)
    params["__meta__"]["provenance"] = "self-generated"
    params["__meta__"]["corpus_sha256"] = "f" * 64
    reused = tmp_path / "from_another_workspace_v1.pt"
    torch.save(params, reused)
    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact=str(reused))
    assert ws.state["artifact_id"] == "self-generated:" + "f" * 12
    assert ws.state["artifact_provenance"] == "self-generated"


def test_a_plain_file_is_recorded_by_the_path_it_was_passed_as(tmp_path, base_dir):
    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact=TINY_ARTIFACT_PATH)
    assert ws.state["artifact_id"] == TINY_ARTIFACT_PATH
    assert ws.state["artifact_provenance"] is None


def test_an_artifact_that_is_neither_self_generated_nor_a_file_says_what_to_pass(
        tmp_path, base_dir):
    with pytest.raises(ValueError, match="self-generated"):
        Workspace.init(tmp_path / "ws", str(base_dir), artifact="qwen3-0.6b-some-id")
    assert not (tmp_path / "ws").exists()


def test_rebuild_is_passed_to_the_store(tmp_path, base_dir, monkeypatch):
    seen = {}

    def fake_obtain(model_id, options, *, rebuild=False, **_):
        seen["rebuild"] = rebuild
        raise RuntimeError("stop here")
    monkeypatch.setattr(workspace_module, "obtain_self_generated", fake_obtain)
    with pytest.raises(RuntimeError):
        Workspace.init(tmp_path / "ws", str(base_dir), artifact="self-generated", rebuild=True)
    assert seen == {"rebuild": True}
```

`TINY_ARTIFACT_PATH` is a module-level string set by an autouse module fixture from
`tiny_artifact` (see Step 3).

In `tests/test_cli.py`:

```python
def test_init_needs_an_artifact_and_its_help_names_self_generated(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["init", "ws", "--model", "m"])
    assert exit_info.value.code == 2
    assert "--artifact" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["init", "--help"])
    assert "self-generated" in capsys.readouterr().out


def test_there_is_no_fetch_artifact_subcommand(capsys):
    with pytest.raises(SystemExit):
        main(["fetch-artifact", "x"])
    assert "invalid choice" in capsys.readouterr().err


def test_list_artifacts_lists_the_store(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("lfa.cli.list_store", lambda: [
        {"path": tmp_path / "e", "model_id": "Qwen/Qwen3-0.6B",
         "frame": {"n_raw": 2500, "max_new_tokens": 2048, "gmm_k": 32},
         "built_at": "2026-09-28", "size_mb": 108, "state": "built"}])
    assert main(["list-artifacts"]) == 0
    out = capsys.readouterr().out
    assert "Qwen/Qwen3-0.6B" in out and "2500 documents x 2048 tokens" in out and "108 MB" in out


def test_list_artifacts_on_an_empty_store_says_how_to_fill_it(monkeypatch, capsys):
    monkeypatch.setattr("lfa.cli.list_store", lambda: [])
    assert main(["list-artifacts"]) == 0
    assert "--artifact self-generated" in capsys.readouterr().out
```

And a guard test (in `tests/test_workspace.py`):

```python
def test_nothing_imports_a_fetch_module():
    import importlib.util
    assert importlib.util.find_spec("lfa.artifact.fetch") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest tests/test_workspace.py tests/test_cli.py -q -x`
Expected: FAIL (the first new test fails: `init` still defaults and fetches).

- [ ] **Step 3: Implement**

`lfa/workspace.py`:
- Remove `from .artifact.fetch import ARTIFACTS, fetch_artifact`; add
  `from .artifact.store import obtain_self_generated`. Keep `SELF_GENERATED_ARTIFACT`; its comment
  becomes "The ``artifact`` value that makes :meth:`Workspace.init` build v1 from the model's own
  text, or reuse one built earlier (:mod:`lfa.artifact.store`)."
- `init(cls, path, model_id, artifact, recipe=None, selfgen=None, *, rebuild=False)`. Argument
  resolution, before anything is created:

```python
        self_generated = artifact == SELF_GENERATED_ARTIFACT
        source = None if self_generated else Path(artifact).expanduser()
        if source is not None and not source.is_file():
            raise ValueError(
                f"{artifact!r} is not an artifact file. Pass --artifact self-generated to have "
                "the model build one from its own text (reused from the local store when it has "
                "been built before), or the path to an artifact file, such as another "
                "workspace's artifacts/v1.pt.")
```

  Inside the existing `try`:

```python
            if self_generated:
                options = selfgen or SelfGenOptions()
                stored, manifest = obtain_self_generated(model_id, options, rebuild=rebuild)
                shutil.copyfile(stored, destination)
                for suffix in ("", ".manifest.json"):
                    shutil.copyfile(stored.parent / f"corpus.jsonl{suffix}",
                                    artifacts_dir / f"v1.corpus.jsonl{suffix}")
                artifact_id = f"{SELF_GENERATED}:{manifest['corpus_sha256'][:12]}"
                provenance = SELF_GENERATED
            else:
                shutil.copyfile(source, destination)
                meta = _meta_of(destination)
                if meta.get("provenance") == SELF_GENERATED and meta.get("corpus_sha256"):
                    artifact_id = f"{SELF_GENERATED}:{meta['corpus_sha256'][:12]}"
                    provenance = SELF_GENERATED
                else:
                    artifact_id, provenance = str(artifact), None
                logger.info("Artifact %s copied to %s", source, destination)
```

  The failure branch keeps removing only what this call created: `v1*` files new in
  `artifacts_dir` (for both routes now) and the directories it made. It never touches the store.
  `_meta_of(path)` is a module helper returning `dict(torch.load(path, map_location="cpu",
  weights_only=False).get(META_KEY) or {})`; `_artifact_meta` uses it. State gets
  `"artifact_provenance": provenance`. Delete the `fetch=False` "No p(h) artifact" branch.
  Rewrite the `init` docstring's Args for `artifact` (the two kinds; the store; reuse; the
  hours), add `rebuild` ("move the store's matching entry aside and build afresh"), drop `fetch`
  and `artifact_id`; Raises: `ValueError` for an `artifact` that is neither,
  `lfa.artifact.store.StoreLocked` for a build already running.
- Class docstring: `current_artifact / artifact_version` "(v1 = as put in place at init)";
  `artifact_id` "``"self-generated:<corpus sha256[:12]>"`` for an artifact fitted on the model's
  own text (built at init or passed as a file whose meta says so), else the path it was passed
  as. It is what :meth:`lfa.recipe.Recipe.warnings` is read against."; `artifact_provenance`
  "``"self-generated"`` when v1 was fitted on the model's own text, ``None`` otherwise."
- `_artifact_id` docstring: drop "registry id"; the body is unchanged.
- `_resolve_base_n`: delete the registry lookup; the refusal becomes: `f"{current} carries no
  per-site n_samples and no __meta__.n_samples_total, so the new domain cannot be weighted by its
  sample share. Every artifact this package builds records it; an artifact from elsewhere needs
  it added, or pass base_n to lfa.artifact.extend.extend_artifact directly."` and the docstring
  drops the registry sentence.
- `train`'s no-artifact `WorkspaceNotReady`: "This workspace has no p(h) artifact. Create a new
  workspace with `lfa init <path> --model <model> --artifact self-generated` (or --artifact
  <an artifact file>)."
- Module docstring line 22: "v1 as put in place at init".

`lfa/artifact/__init__.py`: drop the fetch imports and `__all__` entries and the docstring
sentence about fetch; add ":mod:`~lfa.artifact.store` keeps self-generated artifacts so they are
built once per model and frame."

`lfa/cli.py`:
- Imports: drop the fetch import; add `from .artifact.store import StoreLocked, list_store` and
  `CorpusFrameMismatch` from `.selfgen.artifact_corpus`. `USER_FACING_ERRORS`: remove
  `ArtifactNotPublished, ChecksumMismatch, DownloadFailed`; add `StoreLocked` (a `RuntimeError`,
  so it must be named; `CorpusFrameMismatch` is a `ValueError` and already covered — do not add
  it twice). Update the comment above the tuple accordingly ("an artifact build already running").
- Module docstring: drop `fetch_artifact` from the list and "an artifact that is not published
  yet" from the refusals.
- `_init`: pass `rebuild=args.rebuild`; drop `artifact_id`.
- Remove `_fetch_artifact` and its parser. `init` parser: `--artifact` `required=True`,
  `metavar="self-generated|PATH"`, help: "`self-generated` to have the model write its own corpus
  and fit p(h) on it (hours on an 8 GB card, once per model: it is kept in the local store,
  ~/.cache/lfa/artifacts or $LFA_ARTIFACT_STORE, and reused), or the path to an artifact file".
  Add `--rebuild` (`store_true`): "with --artifact self-generated: build afresh even when the
  store has a matching artifact (the old entry is moved aside, not deleted)". Remove
  `--artifact-id`.
- `_list_artifacts`:

```python
def _list_artifacts(args) -> int:
    entries = list_store()
    if not entries:
        print("No self-generated artifacts in the store yet. `lfa init <workspace> --model <id> "
              "--artifact self-generated` builds one and keeps it here for reuse.")
        return 0
    for entry in entries:
        frame = entry["frame"] or {}
        shape = (f"{frame.get('n_raw')} documents x {frame.get('max_new_tokens')} tokens, "
                 f"K={frame.get('gmm_k')}")
        built = entry["built_at"] or "-"
        print(f"{entry['model_id']}  {shape}  {entry['state']}  built {built}  "
              f"{entry['size_mb']} MB  {entry['path']}")
    return 0
```

  and the parser help: "show the self-generated artifacts in the local store".
- Ctrl-C message: append " A self-generated artifact build resumes where it stopped when the
  same `lfa init` or `lfa build-artifact` command is run again."

Tests:
- `tests/conftest.py`: delete the `registry` fixture. `tiny_recipe` is unchanged.
- `tests/test_workspace.py`:
  - Add near the top a module-scoped autouse fixture setting the module global
    `TINY_ARTIFACT_PATH = str(tiny_artifact[1])`, and change `new_workspace` to
    `Workspace.init(path, str(base_dir), artifact=TINY_ARTIFACT_PATH, **overrides)`.
  - Remove the `registry` parameter from every test and fixture signature (mechanical).
  - Where a test asserts that the recipe is silent (`warnings(...) == []` or no WARNING
    logged), pass `tiny_recipe(base_dir, calibrated_artifact=TINY_ARTIFACT_PATH)` — the
    workspace now records the path it was given as `artifact_id`.
  - Delete these tests (their subject no longer exists): `test_init_writes_the_state_the_history_
    and_the_first_artifact` is **kept** but rewritten against a path artifact;
    delete `test_a_local_file_does_not_shadow_a_published_artifact_id`,
    `test_a_local_copy_can_be_recorded_as_the_published_artifact_it_is`,
    `test_an_artifact_id_that_is_not_published_is_refused`,
    `test_an_artifact_id_that_contradicts_the_artifact_is_refused`,
    `test_an_artifact_that_cannot_be_fetched_leaves_no_workspace_behind` (replaced by the
    self-generated rollback test above), `test_init_without_fetching_leaves_no_artifact_and_says_
    what_to_run`, `test_the_documented_local_artifact_route_trains_extends_and_warns_about_
    nothing`, `test_the_same_route_without_the_artifact_id_warns_falsely_and_then_cannot_extend`,
    `test_extend_reads_the_base_count_from_the_registry_when_the_artifact_carries_none`,
    `test_init_self_generated_refuses_an_artifact_id_before_creating_anything`.
  - `test_a_workspace_that_was_already_there_survives_a_failed_init` and
    `test_a_refused_init_leaves_nothing_behind`: rewrite to fail through a self-generated build
    that raises (monkeypatch `workspace_module.obtain_self_generated` to raise) and a
    non-existent path respectively.
  - `test_extend_names_base_n_when_nothing_supplies_the_count`: match `"base_n"` in the new
    message.
  - The self-generated tests at the end of the file that patch
    `ws_module.build_artifact_self_generated` for **init** are rewritten to patch
    `ws_module.obtain_self_generated` with a fake returning `(path, manifest)`; the regenerate
    tests keep patching `build_artifact_self_generated` (regeneration does not use the store).
    `test_init_self_generated_rollback_removes_the_partial_files_the_build_left` becomes the
    store test above.
- `tests/test_cli.py`: drop `registry`, pass `--artifact <tiny path>` wherever it relied on the
  default; delete tests of `fetch-artifact` and `--artifact-id`.
- `tests/test_examples.py`: remove `"fetch-artifact"` from `SUBCOMMANDS`; delete
  `PRE_RELEASE_FALLBACK_SITES` and `test_the_local_artifact_route_is_documented_with_its_id`.
- `tests/test_release.py`: delete `test_the_artifact_registry_carries_real_checksums` and
  `test_todays_repository_is_where_the_gate_expects_it`; the `<org>` test's docstring and message
  drop the registry ("in the README and the metadata"); keep `scan_for` and
  `test_the_scan_has_teeth`; add `".worktrees"` to `SKIPPED_DIRS`.
- `examples/quickstart.py`, `examples/chain_three_domains.py`: `--artifact` required (no
  default), remove `--artifact-id` and `artifact_id=`; nothing else yet.

- [ ] **Step 4: Run and commit**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest -q` — expected: all pass;
`grep -rn "fetch_artifact\|ARTIFACTS\|artifact_id=\|ArtifactNotPublished" lfa tests examples/*.py`
— expected: no output.

```bash
git add -A lfa tests examples/quickstart.py examples/chain_three_domains.py
git commit -m "No published artifacts: init takes self-generated (built once, reused from the store) or a file

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Data preparation with the supplement

**Files:**
- Create: `lfa/supplements.py`
- Modify: `lfa/workspace.py` (`_supplement_for`, `prepare_supplement`,
  `_domain_description_for`), `lfa/cli.py` (`prepare-supplement`, `prepare-domain`)
- Test: `tests/test_supplements.py` (new), `tests/test_workspace.py` (supplement tests' patch
  targets), `tests/test_cli.py`

**Interfaces:**
- Consumes: `split_documents(path, val_fraction, seed)` (`lfa.corpus`), `write_supplement`,
  `template_sha256`, `SupplementOptions` (`lfa.selfgen.supplement`), `checkpoint_sha256`,
  `sha256_text` (`lfa.selfgen.generate`), `Recipe`, `Recipe.bundled_for` (Task 2).
- Produces:
  - `domain_description_for(corpus_path: Path) -> str` (moved from workspace, same body).
  - `beside_corpus(corpus_path: Path) -> Path` = `corpus_path.with_name(corpus_path.name +
    ".supplement")`.
  - `supplement_for(corpus_path: Path, writer_id: str, recipe: Recipe, *, write_root: Path,
    search_roots: list[Path], domain_description: str | None = None, device: str = "cuda:0",
    force: bool = False) -> tuple[Path, dict]`.
  - `prepare_supplement(corpus, model_id, *, recipe=None, domain_description=None,
    device="cuda:0", force=False) -> Path` — the workspace-free route; writes under
    `beside_corpus(corpus)`.

- [ ] **Step 1: Write the failing tests** (`tests/test_supplements.py`)

```python
"""The supplement cache: where a supplement is found, and which writer it must come from."""
import json

import pytest

import lfa.supplements as supplements
from conftest import make_corpus, tiny_recipe


def _fake_writer(calls):
    def write(model_id, documents, out_path, *, domain_description, options, corpus_sha256,
              device=None, **_):
        calls.append((model_id, out_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"prompt": "q", "response": "a"}) + "\n")
        manifest = {"corpus_sha256": corpus_sha256, "writer_id": model_id,
                    "writer_sha256": f"sha-of-{model_id}", "template_sha256": "t" * 64,
                    "domain_description": domain_description}
        (out_path.parent / (out_path.name + ".manifest.json")).write_text(json.dumps(manifest))
        return manifest
    return write


@pytest.fixture
def patched(monkeypatch):
    calls = []
    monkeypatch.setattr(supplements, "write_supplement", _fake_writer(calls))
    monkeypatch.setattr(supplements, "checkpoint_sha256", lambda m: f"sha-of-{m}")
    monkeypatch.setattr(supplements, "template_sha256", lambda: "t" * 64)
    return calls


def test_prepared_beside_the_corpus_with_the_model(tmp_path, base_dir, patched):
    corpus = make_corpus(tmp_path / "my_domain", "cookery")
    path = supplements.prepare_supplement(corpus, "base", recipe=tiny_recipe(base_dir,
                                                                             val_fraction=0.1))
    assert path.parent.parent == (tmp_path / "my_domain.supplement").resolve()
    assert path.name == "supplement.jsonl" and len(patched) == 1


def test_the_same_writer_finds_it_beside_the_corpus(tmp_path, base_dir, patched):
    corpus = make_corpus(tmp_path / "my_domain", "cookery")
    recipe = tiny_recipe(base_dir, val_fraction=0.1)
    prepared = supplements.prepare_supplement(corpus, "base", recipe=recipe)
    found, _ = supplements.supplement_for(
        corpus, "base", recipe, write_root=tmp_path / "ws" / "supplements",
        search_roots=[tmp_path / "ws" / "supplements", supplements.beside_corpus(corpus)],
        device="cpu")
    assert found == prepared and len(patched) == 1


def test_a_different_writer_writes_its_own(tmp_path, base_dir, patched):
    corpus = make_corpus(tmp_path / "my_domain", "cookery")
    recipe = tiny_recipe(base_dir, val_fraction=0.1)
    supplements.prepare_supplement(corpus, "base", recipe=recipe)
    found, manifest = supplements.supplement_for(
        corpus, "stage1_fused", recipe, write_root=tmp_path / "ws" / "supplements",
        search_roots=[tmp_path / "ws" / "supplements", supplements.beside_corpus(corpus)],
        device="cpu")
    assert found.is_relative_to(tmp_path / "ws" / "supplements")
    assert manifest["writer_id"] == "stage1_fused" and len(patched) == 2


def test_a_different_domain_description_writes_again(tmp_path, base_dir, patched):
    corpus = make_corpus(tmp_path / "my_domain", "cookery")
    recipe = tiny_recipe(base_dir, val_fraction=0.1)
    supplements.prepare_supplement(corpus, "base", recipe=recipe)
    supplements.prepare_supplement(corpus, "base", recipe=recipe,
                                   domain_description="Victorian cookery")
    assert len(patched) == 2


def test_no_recipe_anywhere_is_refused(tmp_path, patched):
    corpus = make_corpus(tmp_path / "my_domain", "cookery")
    with pytest.raises(ValueError, match="--recipe"):
        supplements.prepare_supplement(corpus, "nobody/nothing")


def test_the_default_description_is_the_directory_name(tmp_path):
    assert supplements.domain_description_for(tmp_path / "victorian_cookery-1861") == \
        "victorian cookery 1861"
```

In `tests/test_cli.py`:

```python
def test_prepare_domain_with_a_supplement_needs_a_model(tmp_path, capsys):
    src = tmp_path / "src.txt"
    src.write_text("word " * 400)
    assert main(["prepare-domain", str(src), "--out", str(tmp_path / "out"),
                 "--supplement"]) == 2
    assert "--model" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_prepare_domain_with_a_supplement_writes_both(tmp_path, monkeypatch):
    src = tmp_path / "src.txt"
    src.write_text("word " * 400)
    seen = {}
    monkeypatch.setattr("lfa.cli.prepare_supplement",
                        lambda corpus, model_id, **kw: seen.update(corpus=corpus, model=model_id,
                                                                   **kw) or tmp_path / "s.jsonl")
    assert main(["prepare-domain", str(src), "--out", str(tmp_path / "out"), "--supplement",
                 "--model", "Qwen/Qwen3-0.6B", "--domain-description", "old books"]) == 0
    assert list((tmp_path / "out").glob("*.txt"))
    assert seen["model"] == "Qwen/Qwen3-0.6B" and seen["domain_description"] == "old books"


def test_prepare_supplement_with_a_model_needs_no_workspace(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("lfa.cli.prepare_supplement",
                        lambda corpus, model_id, **kw: seen.update(model=model_id) or tmp_path)
    assert main(["prepare-supplement", "--corpus", str(tmp_path), "--model", "m"]) == 0
    assert seen == {"model": "m"}
```

- [ ] **Step 2: Run to verify failure**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest tests/test_supplements.py tests/test_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lfa.supplements'`.

- [ ] **Step 3: Implement**

`lfa/supplements.py` — module docstring: "The question-and-answer supplement's cache: where a
supplement is looked for, and when one found is the right one. A supplement is the entry model's
own question-and-answer pairs over the training side of a corpus; its measured effect is on
whether the domain's knowledge can be reached when the model is asked about it, not on
protecting skills. It is keyed on the training side's hash, the writer checkpoint's hash, the
template's hash and the domain description, so it is reused only when all four match. A
supplement prepared with the data (`lfa prepare-domain --supplement --model ...`) sits beside the
corpus; one written by `train` sits in the workspace; `train` looks in both."

```python
def supplement_for(corpus_path, writer_id, recipe, *, write_root, search_roots,
                   domain_description=None, device="cuda:0", force=False):
    corpus_path = Path(corpus_path).expanduser().resolve()
    train_docs, _ = split_documents(corpus_path, recipe.val_fraction, recipe.seed)
    corpus_hash = sha256_text(train_docs)
    writer_hash = checkpoint_sha256(str(writer_id))
    description = domain_description or domain_description_for(corpus_path)
    if not force:
        for root in search_roots:
            out = Path(root) / corpus_hash[:12] / "supplement.jsonl"
            manifest_path = Path(str(out) + ".manifest.json")
            if out.is_file() and manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                if (manifest.get("corpus_sha256") == corpus_hash
                        and manifest.get("writer_sha256") == writer_hash
                        and manifest.get("template_sha256") == template_sha256()
                        and manifest.get("domain_description") == description):
                    logger.info("Supplement reused: %s", out)
                    return out, manifest
    out = Path(write_root) / corpus_hash[:12] / "supplement.jsonl"
    logger.info("Writing the supplement for %s with %s (%d training documents)", corpus_path,
                writer_id, len(train_docs))
    manifest = write_supplement(str(writer_id), train_docs, out, domain_description=description,
                                options=SupplementOptions(), corpus_sha256=corpus_hash,
                                device=device)
    return out, manifest


def prepare_supplement(corpus, model_id, *, recipe=None, domain_description=None,
                       device="cuda:0", force=False):
    corpus_path = Path(corpus).expanduser().resolve()
    if recipe is None:
        recipe = Recipe.bundled_for(model_id)
        if recipe is None:
            raise ValueError(
                f"No bundled recipe names {model_id!r}, and the supplement is written from the "
                "recipe's training side (its held-out fraction and seed). Pass --recipe <name or "
                "YAML path>.")
    if not isinstance(recipe, Recipe):
        recipe = Recipe.load(recipe)
    root = beside_corpus(corpus_path)
    path, _ = supplement_for(corpus_path, model_id, recipe, write_root=root, search_roots=[root],
                             domain_description=domain_description, device=device, force=force)
    return path
```

`lfa/workspace.py`: `_supplement_for` becomes

```python
        return supplement_for(
            corpus_path, str(self.state["current_model"]), recipe,
            write_root=self.path / "supplements",
            search_roots=[self.path / "supplements", beside_corpus(corpus_path)],
            domain_description=domain_description, device=_primary_device(placement),
            force=force)
```

with its docstring updated to say it looks in the workspace, then beside the corpus, and writes
into the workspace; the writer is the workspace's current model, so a chain's later stage never
reuses a supplement the base model wrote. Replace "(the C15 protocol)" with "(each stage's pairs
come from the model that stage starts from)". Remove the now-unused imports
(`write_supplement`, `template_sha256`, `SupplementOptions`, `checkpoint_sha256` if nothing else
in workspace uses it — `grep` first) and replace `_domain_description_for` uses with
`domain_description_for` imported from `lfa.supplements`.

`tests/test_workspace.py`: the supplement tests patch `lfa.supplements.write_supplement`,
`lfa.supplements.checkpoint_sha256`, `lfa.supplements.template_sha256` instead of the
`ws_module` names. Add one test: a supplement prepared beside the corpus with
`prepare_supplement(corpus, str(base_dir), recipe=...)` is reused by `ws.train(...)` on a fresh
workspace over `base_dir` (the fake writer is called once in total).

`lfa/cli.py`:
- `from .supplements import prepare_supplement`.
- `prepare-supplement` parser: add `--model ID` ("write with this model, without a workspace;
  the file lands beside the corpus in <corpus>.supplement/, where `train` finds it"); help of the
  subcommand: "have a model write the question-and-answer supplement for a corpus, to inspect
  before training (train writes it itself otherwise)".
- `_prepare_supplement`:

```python
def _prepare_supplement(args) -> int:
    if args.model:
        print(prepare_supplement(args.corpus, args.model, recipe=args.recipe,
                                 domain_description=args.domain_description,
                                 device=args.device, force=args.force))
        return 0
    print(_open(args).prepare_supplement(args.corpus, recipe=args.recipe,
                                         domain_description=args.domain_description,
                                         device=args.device, force=args.force))
    return 0
```

- `prepare-domain` parser: add `--supplement` (`store_true`, "also have --model write the
  question-and-answer supplement for the prepared corpus; it makes the domain answerable when
  the model is asked about it, and `train` mixes it in"), `--model`, `--recipe`,
  `--domain-description`, and `_add_device(domain)`.
- `_prepare_domain`:

```python
def _prepare_domain(args) -> int:
    if args.supplement and not args.model:
        raise ValueError("--supplement needs --model: the supplement is written by the model you "
                         "will train, e.g. --model Qwen/Qwen3-0.6B.")
    # No report of its own: `prepare_domain` already logs what it wrote and what it skipped.
    prepare_domain(args.inputs, args.out, min_length=args.min_length, combine=args.combine)
    if args.supplement:
        print(prepare_supplement(args.out, args.model, recipe=args.recipe,
                                 domain_description=args.domain_description, device=args.device))
    return 0
```

- [ ] **Step 4: Run and commit**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest -q` — expected: all pass.

```bash
git add lfa/supplements.py lfa/workspace.py lfa/cli.py tests/
git commit -m "Data prep writes the supplement beside the corpus; train finds it there when the writer matches

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Wording sweep in the package, with a test that keeps it swept

**Files:**
- Modify: every file under `lfa/` that matches the scan (at the time of writing:
  `lfa/selfgen/__init__.py`, `artifact_corpus.py`, `generate.py`, `supplement.py`, `lfa/corpus.py`,
  `lfa/cli.py`, `lfa/workspace.py`, `lfa/recipe.py`, `lfa/artifact/build.py`,
  `lfa/artifact/schema.py`, `lfa/quantize.py`, `lfa/seed_corpus.py`, `lfa/recipes/qwen3-0.6b.yaml`)
- Create: `tests/test_wording.py`

**Interfaces:**
- Produces: `tests/test_wording.py::SCANNED_ROOTS` (a list of repo-relative roots) and
  `FORBIDDEN` (list of `(label, regex)`), which Tasks 7 and 8 extend.

- [ ] **Step 1: Write the failing test** (`tests/test_wording.py`)

```python
"""Shipped text is written for a user of this package, not for a reader of the research record.

The research repository, its claim ids and its retired artifact are not things a user can open,
so no shipped file names them. `docs/verification.md` is a dated record of the port and
`docs/superpowers/` holds development documents; both are exempt.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where shipped text lives. Tasks that sweep more of the tree extend this.
SCANNED_ROOTS = ["lfa"]

EXEMPT = {"docs/verification.md"}
EXEMPT_DIRS = {"docs/superpowers", "__pycache__", ".worktrees"}
SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".ipynb", ".in"}

#: Built by parts so this file does not match itself.
FORBIDDEN = [
    ("a research claim id", re.compile(r"\bC1[0-9]\b")),
    ("the research record", re.compile(r"\bthe LFA " + r"record\b", re.IGNORECASE)),
    ("the research repository", re.compile("mr-" + "fusion")),
    ("the retired published artifact", re.compile("gmm" + "1543k")),
    ("the retired --artifact-id flag", re.compile("--artifact" + "-id")),
]


def _shipped_files():
    for root in SCANNED_ROOTS:
        base = REPO_ROOT / root
        paths = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in paths:
            relative = path.relative_to(REPO_ROOT).as_posix()
            if (not path.is_file() or path.suffix not in SUFFIXES or relative in EXEMPT
                    or any(relative == d or relative.startswith(d + "/") or f"/{d}/" in relative
                           for d in EXEMPT_DIRS)):
                continue
            yield relative, path.read_text(encoding="utf-8", errors="replace")


def test_no_shipped_file_speaks_in_the_research_records_terms():
    hits = []
    for relative, text in _shipped_files():
        for number, line in enumerate(text.splitlines(), 1):
            for label, pattern in FORBIDDEN:
                if pattern.search(line):
                    hits.append(f"{relative}:{number}: {label}: {line.strip()[:100]}")
    assert not hits, "\n".join(hits)


def test_the_scan_sees_a_planted_hit(tmp_path, monkeypatch):
    planted = REPO_ROOT / "lfa" / "_planted_for_the_wording_test.py"
    planted.write_text("# see C12 in mr-" + "fusion\n")
    try:
        hits = [rel for rel, text in _shipped_files() if "C12" in text]
        assert "lfa/_planted_for_the_wording_test.py" in hits
    finally:
        planted.unlink()
```

- [ ] **Step 2: Run to verify failure**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest tests/test_wording.py -q`
Expected: FAIL, listing every current hit under `lfa/`.

- [ ] **Step 3: Sweep** — for each hit, rewrite the sentence so it states what was measured and
  at what scale, with no id and no repository path. Patterns to apply:
  - "C12" / "(C12: …)" → "(one model, Qwen3-0.6B; one seed; one domain)" beside the claim the
    sentence already makes; "the artifact behind C12" → "the recorded artifact".
  - "C14" → the plain statement: "a supplement written in a skill's mode left that skill no
    better (instruction following and reasoning; one model, one seed, one domain)".
  - "C15" / "the C15 protocol" / "the C15 route (rank 4, one seed)" → "regenerating the
    artifact from each stage's model (measured on one configuration: rank 4, one seed)".
    This includes `REGENERATED_NOTE` in `lfa/workspace.py`.
  - "mr-fusion ``scripts/...``" / "mr-fusion `prepare_domain_qa...`" → "the research code".
  - "the LFA record" / "the record" (in `lfa/`) → "the research runs".
  - `lfa/selfgen/__init__.py` docstring second sentence → "The evidence behind this is one model
    (Qwen3-0.6B) and one seed; every number the package documentation quotes about it carries
    that scope."
  - `lfa/quantize.py`, `lfa/seed_corpus.py` mentions of `gmm1543k`: describe the artifact as
    "the real-text artifact the recipe's lambda was tuned against" (seed_corpus's
    `SHIPPED_COMPOSITION` keeps its name and values; only prose changes).
  - "arm" in code comments is research slang: replace with "run" or "configuration" where it
    appears in `lfa/models.py`, `lfa/quantize.py`.

- [ ] **Step 4: Run and commit**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest -q` — expected: all pass.

```bash
git add lfa tests/test_wording.py
git commit -m "Package text states what was measured, not which research claim it was

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: The docs a new user reads

**Files:**
- Rewrite: `README.md`, `docs/quickstart.md`
- Create: `docs/preparing-your-data.md`
- Rename + rewrite: `docs/rebuilding-the-artifact.md` → `docs/the-artifact.md` (`git mv`)
- Modify: `docs/concepts.md`, `docs/recipes.md`, `docs/faq.md`, `docs/adding-a-model.md`,
  `docs/multi-domain-chains.md`, `docs/verification.md` (header only), `RELEASING.md`
- Delete: `docs/superpowers/specs/2026-09-26-self-generation-design.md`,
  `docs/superpowers/plans/2026-09-26-self-generation.md`
- Modify: `tests/test_wording.py` (`SCANNED_ROOTS += ["README.md", "RELEASING.md", "docs"]`)

**Interfaces:**
- Consumes: the CLI surface of Tasks 4–5 exactly: `lfa init PATH --model ID --artifact
  self-generated [--rebuild] [--n-raw N] [--max-new-tokens N]`, `lfa list-artifacts`,
  `lfa prepare-domain INPUTS --out DIR [--supplement --model ID] [--domain-description TEXT]`,
  `lfa prepare-supplement --corpus DIR (--model ID | --workspace PATH)`, `lfa train --workspace
  PATH --corpus DIR [--no-supplement | --supplement FILE] [--epochs N]`, `lfa evaluate`,
  `lfa fuse`, `lfa extend`, `lfa regenerate-artifact`, `lfa chain`, `lfa build-artifact --model ID
  (--self-generated | --corpus JSONL) --out PATH`, `lfa prepare-seed-corpus`.

Every command shown in a doc must be one of these, spelled as here. Every number quoted must
already be in the current docs (move it, keep its date and scope); no new numbers. Keep the
existing register: plain declarative sentences, the caveat beside the number it qualifies.

- [ ] **Step 1: Extend the wording test's roots** — `SCANNED_ROOTS = ["lfa", "README.md",
  "RELEASING.md", "docs"]`. Run `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest tests/test_wording.py -q`;
  expected: FAIL listing the doc hits (this is the checklist for the steps below).

- [ ] **Step 2: `README.md`** — in this order, and nothing else:
  1. Title and the existing first two paragraphs (what LFA is, the loss), trimmed of the claim
     ids; the paragraph on the teacher-under-LoRA stays.
  2. **Install** (unchanged content).
  3. **The pipeline** — the five commands:

```bash
lfa init runs/my_domain --model Qwen/Qwen3-0.6B --artifact self-generated   # once per model
lfa prepare-domain ~/papers ~/notes.md --out data/my_domain \
    --supplement --model Qwen/Qwen3-0.6B                                     # your documents
lfa train    --workspace runs/my_domain --corpus data/my_domain
lfa evaluate --workspace runs/my_domain
lfa fuse     --workspace runs/my_domain
```

     followed by one paragraph per step, three sentences at most each: `init` has the model write
     2,500 documents of its own and fits p(h) on them — hours on an 8 GB card, not timed — and
     keeps the result in the local store so every later workspace over the same model reuses it
     (`lfa list-artifacts`); an interrupted build resumes when run again. `prepare-domain` turns
     text, Markdown, HTML or PDF into the corpus and, with `--supplement`, has the model write
     question-and-answer pairs over it, which make the domain's knowledge answerable when the
     model is asked about it (they do not protect skills; the anchor does that job); without
     `--supplement`, `train` writes them itself before the first epoch. `train`, `evaluate`,
     `fuse`: as the current README says, with the evaluate table.
     Then the same flow in Python (`Workspace.init(..., artifact="self-generated")`,
     `prepare_supplement` from `lfa.supplements`, `ws.train`, `ws.evaluate`, `ws.fuse`).
  4. **A second domain, and a chain** — the current section, unchanged except wording sweep.
  5. **The walkthrough notebooks** — the current two paragraphs about them, with the sentence
     "Recorded 2026-09-08 with the since-retired published artifact on the raw books alone; a
     run today builds its own artifact and mixes in the supplement, so the numbers will differ."
  6. **Documentation** table: quickstart ("the full pipeline, step by step"),
     preparing-your-data ("formats, cleaning, corpus shapes, the supplement"), the-artifact ("the
     self-generated build, the store, and a real-text artifact"), concepts, recipes,
     multi-domain-chains, adding-a-model, faq, the two notebooks. `docs/verification.md` and
     `RELEASING.md` are not in the table.
  7. **Tests**, **Provenance** (the current text, which is where `docs/verification.md` is
     linked, plus one sentence: "The recipe's lambda was tuned against an artifact fitted on real
     text; this package builds an artifact from the model's own text instead, which matched it at
     every lambda tried — one model, one seed, one domain."), **Citing**.
  Remove: "The building blocks" table (its content moves to concepts.md, Step 5), "The artifact"
  section, every `--artifact-id` and published-artifact sentence.

- [ ] **Step 3: `docs/quickstart.md`** — rewrite as **The full pipeline**, with these headed
  steps, each: the command, what it does, what it costs, what can go wrong and what to do.
  - **0. Install and check the card** — current Install and prerequisites text; `lfa --version`;
    the examples shipped in the wheel.
  - **1. Build (or reuse) the artifact** — `lfa init ... --artifact self-generated`; what it
    writes (2,500 documents from the bare document-start token, 600k samples per site, K = 32);
    where it lands (`artifacts/v1.pt`, the corpus beside it, the store at
    `~/.cache/lfa/artifacts` or `$LFA_ARTIFACT_STORE`); the log line per batch; Ctrl-C and
    resume (run the same command again); a failed fit keeps the corpus; `--rebuild`;
    `lfa list-artifacts`; reusing a file with `--artifact path/to/v1.pt`; a trial build
    (`--n-raw 60 --max-new-tokens 128`) for trying the pipeline, and that `train` then warns that
    lambda was calibrated at the recorded frame. Cost: "several hours on an 8 GB card; not timed"
    plus the pointer to faq.md for the timed pieces. The evidence sentence with its scope.
  - **2. Prepare your data** — the `prepare-domain` command with and without `--supplement`, two
    sentences, and a pointer to preparing-your-data.md for everything else.
  - **3. Train** — the current item 2 text (held-out tenth, watch the curve, `--epochs`,
    `--resume`), with the supplement paragraph shortened to: train uses a supplement prepared
    with the data when the same model wrote it, writes one otherwise, `--no-supplement` and
    `--supplement FILE`.
  - **4. Evaluate** — current item 3 plus "What it prints" (with its date/predates-0.2.0 note).
  - **5. Fuse** — current item 4.
  - **6. A next domain** — `extend` then `train`, or `lfa chain domains.yaml`; pointer to
    multi-domain-chains.md.
  - **What it costs** and **When something is refused** — current sections; in the refusal list
    replace "an artifact that is not published" with "an artifact build already running for the
    same model and frame" and add "a partial build under a different frame".
  - **Next** — links.
  Delete: "Get a p(h) artifact" and the private-repository note.

- [ ] **Step 4: `docs/preparing-your-data.md`** (new) — sections:
  - **From your files to a corpus** — `lfa prepare-domain` inputs (`.txt`/`.md` pass through,
    `.html` needs `[html]`, `.pdf` needs `[pdf]`), `--min-length` (default 1000) and why,
    `--combine` (for inspection, not training), output layout (flat directory of `.txt`, one
    document per file), host RAM (about eight times the corpus size, refused up front).
  - **Three shapes that train badly** — move the thresholds and remedies from faq.md (leave a
    one-line pointer in faq.md).
  - **The supplement** — what it is (the model reads each training-side passage and writes six
    question-and-answer pairs from a fixed template); what it is for (reaching the domain's
    knowledge when asked, the recipe mixes it at 0.13 of training tokens, the frame lambda was
    tuned at) and not for (it does not protect skills: a supplement written in a skill's mode
    left the skill no better, instruction following and reasoning, one model, one seed, one
    domain; the anchor is what keeps skills); the held-out tenth stays raw text.
  - **Preparing it with the data** — `prepare-domain --supplement --model`, or
    `prepare-supplement --corpus DIR --model ID` for a corpus you already have; where it lands
    (`<corpus>.supplement/<hash>/supplement.jsonl` + manifest); how to read it; when `train`
    reuses it (same training side, same writer checkpoint, same template, same description) and
    when it writes its own (a chain's later stage, whose model is the fused one);
    `--domain-description` and its default (the directory name).
  - **Bringing your own** — `train --supplement FILE`, the JSONL format (one object per line
    with `prompt` and `response`), rendered through the model's chat template.
  - **Training without one** — `--no-supplement` and the warning it prints.
  - **What it costs** — the pointer to faq.md's timed generation pieces.

- [ ] **Step 5: The other pages**
  - `git mv docs/rebuilding-the-artifact.md docs/the-artifact.md`; restructure: **The
    self-generated artifact** first (the frame, the cost pieces, layer grouping from host RAM,
    durability and resume, the store and `--rebuild`, reusing a file, `build-artifact
    --self-generated --out` for a file outside the store, which resumes the same way beside its
    output), then **Advanced: an artifact fitted on real text** (the current seed-corpus content,
    retitled; the `SHIPPED_COMPOSITION` table stays as "the composition the recipe's lambda was
    tuned on"; the recipe will note that such an artifact is not its calibration reference).
    Remove the "shipped file carries no `embedding_lookup` entry" subsection and every
    published-artifact reference.
  - `concepts.md`: add the building-blocks table from the README (Model, Artifact — "built from
    the model's own text, kept in the local store", Recipe, Corpus, Supplement, Workspace, the
    operations), sweep claim ids / "the record" / published artifact; the supplement paragraph
    states reachability-not-skills as in Global Constraints.
  - `recipes.md`: the artifact coupling section says the recipe is calibrated against the
    self-generated artifact at `self_generated_frame`, with where the lambda was tuned (real
    text) and the matching evidence and scope; document `self_generated_frame`; remove
    `calibrated_self_generated`.
  - `faq.md`, `adding-a-model.md`, `multi-domain-chains.md`: sweep; links to
    `rebuilding-the-artifact.md` become `the-artifact.md`; adding-a-model's artifact step is
    `lfa init --artifact self-generated` on the new model, then calibrate lambda.
  - `docs/verification.md`: add a first paragraph, italic: "*A dated record of the 0.1 port
    (2026-09-07). It predates the self-generated default and the retirement of the published
    artifact it names; the checks it reports are about the method's code, which those changes did
    not touch.*"
  - `RELEASING.md`: delete §1–§4 (the artifact release: re-save, hashes, upload, registry) and
    renumber; delete the `<org>`/registry step; keep preconditions, tagging and the package
    build/upload; trim each "✅ rehearsed on 2026-09-07" record to one line; add to the 0.2.0
    changelog: "No published artifacts: `lfa init --artifact self-generated` builds one from the
    model's own text and keeps it in a local store for reuse (`lfa list-artifacts`); the build
    resumes after an interruption and keeps its corpus if the fit fails. `--artifact` is
    required; `fetch-artifact` and `--artifact-id` are gone. `prepare-domain --supplement --model`
    and `prepare-supplement --model` prepare the supplement with the data. The recipe is
    calibrated against the self-generated artifact at `self_generated_frame`."
  - `git rm docs/superpowers/specs/2026-09-26-self-generation-design.md
    docs/superpowers/plans/2026-09-26-self-generation.md`.

- [ ] **Step 6: Check links and commands, run, commit**

Run: `grep -rn "rebuilding-the-artifact\|fetch-artifact\|list-artifacts.*published" README.md docs RELEASING.md examples lfa`
Expected: no output.
Run the link check (expected output: `0 broken`):

```bash
python3 - <<'EOF'
import re
from pathlib import Path
broken = []
for doc in [Path("README.md"), *sorted(Path("docs").glob("*.md"))]:
    for target in re.findall(r"\]\(([^)#\s]+)", doc.read_text()):
        if target.startswith("https://github.com/sparkdoc/lfa-anchoring/blob/main/"):
            path = Path(target.split("/blob/main/", 1)[1])
        elif re.match(r"[a-z]+://", target):
            continue
        else:
            path = (doc.parent / target)
        if not path.exists():
            broken.append(f"{doc}: {target}")
print("\n".join(broken)); print(f"{len(broken)} broken")
EOF
```
Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest -q` — expected: all pass.

```bash
git add -A README.md RELEASING.md docs tests/test_wording.py
git commit -m "Docs for a new user: the full pipeline, preparing your data, the self-generated artifact

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Examples, notebooks' setup, and the GPU pipeline test (written, not run)

**Files:**
- Modify: `examples/quickstart.py`, `examples/chain_three_domains.py`, `examples/domains.yaml`,
  `examples/two_domain_walkthrough.ipynb`, `examples/what_the_anchor_does.ipynb`,
  `tests/test_notebook.py` (docstring and any setup assertion that names the fetched artifact),
  `tests/test_wording.py` (`SCANNED_ROOTS += ["examples"]`)
- Create: `tests/test_pipeline_gpu.py`

- [ ] **Step 1: Extend the wording test** — add `"examples"` to `SCANNED_ROOTS`; run it;
  expected: FAIL listing the example hits.

- [ ] **Step 2: Example scripts** — module docstrings show the pipeline with
  `--artifact self-generated` (store reuse explained in one sentence), and the quickstart's shows
  data preparation with `lfa prepare-domain ... --supplement --model ...`; `--artifact` stays
  required with help "self-generated, or an artifact file". Remove every paragraph about the
  published assets not existing. `tests/test_examples.py::test_the_quickstart_example_trains_
  evaluates_and_exports` passes `--artifact <tiny path>` already and must stay green.

- [ ] **Step 3: Notebooks** — edit the JSON with a small Python script (load with `json`, edit
  cell sources, dump with `indent=1` and a trailing newline, matching the file's current
  formatting): the setup cell that fetches `qwen3-0.6b-gmm1543k-int8` becomes
  `Workspace.init(..., artifact="self-generated")` with a comment that the first run builds it
  (hours) and later runs reuse it from the store; any cell passing `artifact_id` drops it. Insert
  a markdown cell after the title: "**Recorded outputs.** The outputs below were recorded on
  2026-09-08 on one RTX 3090 with the since-retired published artifact, on the raw books with no
  supplement. Run today, the notebook builds its own artifact and mixes in the supplement, so
  the numbers will differ; the prose around each table says what to look for, not which number
  to expect." Do not execute the notebooks and do not touch output cells.
  `tests/test_notebook.py`: its docstring's "fetches the artifact" → "builds or reuses the
  self-generated artifact"; its default-suite structural tests must pass (run them).

- [ ] **Step 4: `tests/test_pipeline_gpu.py`** — marked `gpu`, not run in this pass:

```python
"""The whole pipeline on the card, at a trial frame: build, prepare with the supplement, train,
evaluate, fuse; then a second workspace that must reuse the stored artifact.

    pytest tests/test_pipeline_gpu.py -m gpu -q
"""
import pytest

from lfa.cli import main

pytestmark = pytest.mark.gpu

MODEL = "Qwen/Qwen3-0.6B"


def test_the_pipeline_end_to_end_at_a_trial_frame(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("LFA_ARTIFACT_STORE", str(tmp_path / "store"))
    src = tmp_path / "src"
    src.mkdir()
    for i in range(12):
        (src / f"doc{i}.txt").write_text(
            f"Document {i}. " + "The lighthouse keeper logged the tides and the weather. " * 60)
    ws = tmp_path / "ws"

    assert main(["init", str(ws), "--model", MODEL, "--artifact", "self-generated",
                 "--n-raw", "60", "--max-new-tokens", "128"]) == 0
    assert main(["prepare-domain", str(src), "--out", str(tmp_path / "corpus"),
                 "--supplement", "--model", MODEL]) == 0
    assert list((tmp_path / "corpus.supplement").glob("*/supplement.jsonl"))
    with caplog.at_level("INFO"):
        assert main(["train", "--workspace", str(ws), "--corpus", str(tmp_path / "corpus"),
                     "--epochs", "1"]) == 0
    assert any("Supplement reused" in r.message for r in caplog.records)
    assert any("different frame" in r.message for r in caplog.records)   # the trial-frame note
    assert main(["evaluate", "--workspace", str(ws), "--n-windows", "5"]) == 0
    assert main(["fuse", "--workspace", str(ws)]) == 0

    caplog.clear()
    with caplog.at_level("INFO"):
        assert main(["init", str(tmp_path / "ws2"), "--model", MODEL, "--artifact",
                     "self-generated", "--n-raw", "60", "--max-new-tokens", "128"]) == 0
    assert any("Reused the self-generated artifact" in r.message for r in caplog.records)
```

  Verify only that it is collected and deselected by default:
  `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest tests/test_pipeline_gpu.py -q` → "1 deselected" (or skipped),
  and `python -c "import ast,sys; ast.parse(open('tests/test_pipeline_gpu.py').read())"`.

- [ ] **Step 5: Run and commit**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest -q` — expected: all pass.

```bash
git add examples tests/test_notebook.py tests/test_wording.py tests/test_pipeline_gpu.py
git commit -m "Examples and notebooks build their own artifact; a GPU pipeline test for the follow-up

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: The handoff for GPU verification

**Files:**
- Create: `docs/superpowers/handoffs/2026-09-28-gpu-verification.md`

- [ ] **Step 1: Write the handoff** with these sections, filled from the repository state at the
  time of writing (commit hash from `git rev-parse HEAD`):
  1. **State** — the commit it was written against; that this pass ran the fast tier only
     (paste the final `pytest -q` summary line); that no GPU test has run against this code.
  2. **GPU tier** — `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest -m gpu -q` (all GPU tests, including
     `tests/test_pipeline_gpu.py`), and what each new assertion proves.
  3. **The full-frame build** — `lfa init runs/verify --model Qwen/Qwen3-0.6B --artifact
     self-generated` with `time`; after about a third of the documents (watch the per-batch log),
     Ctrl-C, then the same command again: confirm it resumes at the next batch and the finished
     corpus has 2,500 rows; record the wall time for generation and fit separately and the host
     RAM layer-group choice from the log; `lfa list-artifacts`.
  4. **Examples and notebooks** — on an 8 GB card use batch 3 × gradient accumulation 2 (a
     recipe copy with `batch_size: 3`, `gradient_accumulation_steps: 2`); run
     `examples/quickstart.py`, `examples/chain_three_domains.py`, and both notebooks
     (`pytest tests/test_notebook.py -m notebook`), save the executed notebooks.
  5. **Docs to update from the measurements** — replace "not timed" in README, quickstart,
     the-artifact and faq with the measured build cost (card named); replace the notebooks'
     "Recorded outputs" cell and every number README and quickstart quote from them, labelled
     with card, artifact (self-generated, recorded frame) and supplement; `grep -rn "not timed"`
     must come back empty.
  6. **Owner items** — push `main` (`git push origin main`); delete the remote `artifacts-v1`
     tag and its GitHub release if it exists (`git push origin :refs/tags/artifacts-v1`, and the
     release in the GitHub UI) and the local tag (`git tag -d artifacts-v1`) — decide first, it is
     outward-facing; the parked item: per-site reservoir generators in `lfa/artifact/collect.py`
     so layer grouping stops changing the fitted mixtures.
  7. **Last step** — `git rm -r docs/superpowers` and commit, once every item above is done.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/handoffs/2026-09-28-gpu-verification.md
git commit -m "Handoff: the GPU verification this pass did not run

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 3 (controller, not an implementer):** update the memory file
  `project_lfa_anchoring_companion.md`: the published artifacts are retired (no Monday rebuild);
  the store and durable build exist; the handoff path; `docs/superpowers/` holds this round's
  spec, plan and handoff until the GPU session deletes it.
