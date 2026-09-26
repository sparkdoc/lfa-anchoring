# Self-Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A user with only a model and a domain corpus can build the p(h) artifact from the model's own text, have the model write the domain supplement that training mixes in at the recipe's token fraction, and in a chain refit the artifact from each stage's own model.

**Architecture:** A new `lfa/selfgen/` subpackage holds one batched sampling loop (`generate.py`), the unconditional artifact corpus at the recorded research frame (`artifact_corpus.py`) and the passage-to-pairs supplement writer (`supplement.py`). Three thin integration points wire it in: `build_artifact_self_generated` and `Workspace.init(artifact="self-generated")` for the artifact, a supplement step inside `Workspace.train` with token-fraction mixing in `lfa/corpus.py`, and `Workspace.regenerate_artifact` plus a chain-wide `artifact:` field. Two environment fixes ride along: the toolchain check becomes a warning, and the 8 GB geometry is documented.

**Tech Stack:** Python ≥ 3.11, torch ≥ 2.2, transformers 4.56–4.x, peft 0.18, pytest ≥ 8. GPU tests run on Qwen/Qwen3-0.6B (cached locally) on the RTX 2070, 8 GB.

**Spec:** `docs/superpowers/specs/2026-09-26-self-generation-design.md` (read §1a amendments first).

## Global Constraints

- Every number quoted in docs is the research record's, scoped "one model (Qwen3-0.6B), one seed" (C12/C14) or "rank 4, one seed" (C15); never the package's own measurement.
- Docs say **reachability**, never **retention**, for what the supplement does (spec §1).
- Recorded research frames, verbatim: artifact corpus 2,500 raw × 2,048 tokens from the seed prefix at T=1.0/top-p=1.0, unfiltered, seed 42, plus 250 chat-header documents, seed 43; artifact fit 600,000 samples/site, K=32, PCA variance 0.95; supplement 6 pairs per ~4,000-char passage, T=0.7/top-p=0.8, 1,024 new tokens, batch 16, answers ≥ 40 chars; mixing target 0.13 by training tokens.
- Every truncation knob is passed explicitly on every `generate` call: `top_k=0, min_p=0.0, repetition_penalty=1.0`.
- Two recorded deviations from the research frame: the template says "about a text on {domain_description}" (was "about a philosophy text"); the contamination `screen` stage is not ported. `--enforce-spec` and the reasoning/instruction modes are not ported.
- The held-out split is taken from raw documents **before** mixing, with the same shuffle as today; the supplement writer sees only the training side.
- Warnings policy in `pyproject.toml` is `error` scoped to `lfa`/`tests`; a new warning this package emits must be a `logger.warning`, not `warnings.warn`, unless a test pins it.
- GPU tests: `pytestmark = pytest.mark.gpu`, sized for 8 GB (Qwen3-0.6B: micro-batch ≤ 3 at 512 tokens, or batch 1 at ≤ 128 tokens in tests), `HF_HUB_OFFLINE=1`, run as `LFA_SKIP_TOOLCHAIN_CHECK=1 pytest tests/test_selfgen_gpu.py -m gpu -q` outside the sandbox. Fast tests never fake a generator to re-test what a GPU test covers.
- Commit after every task with the repo-local identity (already `sparkdoc <sparkdoc@users.noreply.github.com>`); end commit messages with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Run `pytest -q` (fast tier) before every commit; it must stay green. Environment for this machine: `HOME=$TMPDIR HF_HUB_OFFLINE=1 LFA_SKIP_TOOLCHAIN_CHECK=1` are needed only inside the agent sandbox; outside it plain `pytest -q` works.

## Review Focus

1. **A corpus with one giant document.** The training side is one document, so one passage set and one held-out document of zero. `write_supplement` must still produce pairs and `load_corpus` must still refuse a split that leaves nothing to train on (test in Task 8: `test_a_single_document_corpus_trains_on_it_and_holds_nothing_out`).
2. **A model whose tokenizer has no chat template.** `chat_user_header` and pair rendering must fall back (plain join, no header) rather than raise; the artifact corpus then has no chat share and says so (Task 3: `test_a_tokenizer_without_a_chat_template_gives_no_header`).
3. **A supplement pool larger than the target.** Only a prefix is used; the pairs beyond it are recorded as available-but-unused, never silently dropped from the manifest (Task 8: `test_a_pool_larger_than_the_target_uses_a_prefix_and_reports_the_rest`).
4. **Re-running `train` after editing the corpus.** The corpus hash changes, so a stale supplement must be regenerated, not reused (Task 10: `test_an_edited_corpus_regenerates_the_supplement`).
5. **`regenerate` on a chain whose recipe has `calibrated_self_generated=False`.** The off-calibration warning must fire at stage 2 and name C15's scope, and the chain must continue (Task 11: `test_regenerate_warns_off_calibration_and_continues`).

---

### Task 1: Toolchain check becomes a warning

**Files:**
- Modify: `lfa/models.py:99-152` (`MissingBuildToolchain`, `check_gpu_toolchain`)
- Modify: `tests/test_models.py:239-300`
- Modify: `docs/faq.md` (the "How do I use a model that is not Qwen3?" page is untouched; the toolchain sentence lives in `README.md:52-54` and `docs/quickstart.md` Install section)

**Interfaces:**
- Consumes: nothing new.
- Produces: `check_gpu_toolchain(device) -> None` now logs `logger.warning(...)` once per process instead of raising; `MissingBuildToolchain` stays defined (still in `cli.USER_FACING_ERRORS`) but is no longer raised anywhere.

- [ ] **Step 1: Rewrite the four toolchain tests to pin the warning**

Replace `test_a_cuda_run_without_python_headers_is_refused_before_anything_loads`, `test_the_preflight_also_wants_a_compiler`, `test_the_preflight_can_be_switched_off_for_a_machine_that_never_compiles` and `test_resolving_a_cuda_device_runs_the_preflight` in `tests/test_models.py` with:

```python
def test_a_cuda_run_without_python_headers_warns_once_and_proceeds(no_headers, caplog):
    """A real Qwen3-0.6B LoRA step and a generation ran on an RTX 2070 with no headers
    (2026-09-26), so the missing toolchain is a warning about a path some torch builds take,
    not a refusal."""
    import lfa.models as models_module

    models_module._toolchain_warned = False
    with caplog.at_level("WARNING", logger="lfa.models"):
        check_gpu_toolchain("cuda:0")
        check_gpu_toolchain("cuda:0")

    messages = [r.getMessage() for r in caplog.records if "Python development headers" in r.getMessage()]
    assert len(messages) == 1                            # once per process
    assert "python3-dev" in messages[0]
    assert TOOLCHAIN_CHECK_OFF in messages[0]
    assert resolve_device("cuda:0") == "cuda:0"          # and the resolver proceeds


def test_the_preflight_also_mentions_a_missing_compiler(monkeypatch, caplog):
    import lfa.models as models_module

    monkeypatch.delenv(TOOLCHAIN_CHECK_OFF, raising=False)
    monkeypatch.setattr(models_module.shutil, "which", lambda name: None)
    models_module._toolchain_warned = False
    with caplog.at_level("WARNING", logger="lfa.models"):
        check_gpu_toolchain({"": "cuda:0"})
    assert any("C compiler" in r.getMessage() for r in caplog.records)


def test_the_preflight_can_be_switched_off(no_headers, monkeypatch, caplog):
    import lfa.models as models_module

    monkeypatch.setenv(TOOLCHAIN_CHECK_OFF, "1")
    models_module._toolchain_warned = False
    with caplog.at_level("WARNING", logger="lfa.models"):
        check_gpu_toolchain("cuda:0")
    assert not [r for r in caplog.records if "headers" in r.getMessage()]
```

Keep `test_a_cpu_run_needs_no_toolchain` as is.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_models.py -k "toolchain or preflight" -q`
Expected: FAIL (`MissingBuildToolchain` raised; `_toolchain_warned` missing).

- [ ] **Step 3: Demote the check**

In `lfa/models.py`, replace the body of `check_gpu_toolchain` from `if not missing: return` onward, and add the module flag:

```python
#: Set once the toolchain warning has been printed; the check runs on every workspace operation.
_toolchain_warned = False


def check_gpu_toolchain(device: str | dict) -> None:
    """Warn, once, when a CUDA run might need to compile and this machine cannot.

    Some torch paths (inductor, custom triton kernels) JIT-compile a small CUDA shim on the
    first kernel launch, which needs ``Python.h`` and a C compiler. The package's own training
    and generation paths do not: a Qwen3-0.6B LoRA stage and an unconditional generation ran on
    an RTX 2070 under a Python with no development headers (2026-09-26). So a missing
    toolchain is reported once, as a warning naming the remedy, and the run proceeds.
    """
    global _toolchain_warned
    if os.environ.get(TOOLCHAIN_CHECK_OFF) or _toolchain_warned:
        return
    if not _single_device(device).startswith("cuda"):
        return

    header = Path(sysconfig.get_paths()["include"]) / "Python.h"
    compiler = next((found for candidate in (os.environ.get("CC"), "gcc", "cc", "clang")
                     if candidate and (found := shutil.which(candidate))), None)
    missing = []
    if not header.is_file():
        missing.append(f"the Python development headers ({header} does not exist)")
    if compiler is None:
        missing.append("a C compiler on PATH (gcc, cc or clang)")
    if not missing:
        return

    _toolchain_warned = True
    logger.warning(
        "This machine cannot compile for the GPU: %s is missing for %s. This package's own "
        "training and generation paths ran without it, but a torch path that JIT-compiles "
        "(torch.compile, custom triton kernels) would fail in gcc mid-run. Install your "
        "distribution's development package for this interpreter (`python3-dev` / "
        "`python3.13-dev`, plus `build-essential`) if that happens. Set %s=1 to silence this.",
        " and ".join(missing), sys.executable, TOOLCHAIN_CHECK_OFF,
    )
```

Update the `MissingBuildToolchain` docstring to say it is kept for callers that catch it and is no longer raised. Update `resolve_device`'s docstring: remove the `MissingBuildToolchain` from `Raises`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_models.py -q`
Expected: PASS.

- [ ] **Step 5: Update the two doc sentences**

`README.md` Install blockquote: replace "torch compiles a small CUDA shim on the first kernel launch. `lfa` checks before loading anything and says so in one line." with "some torch paths compile a small CUDA shim on the first kernel launch; the package's own paths do not, so `lfa` warns once if the headers are missing and proceeds." Make the same change in `docs/quickstart.md` Install section (grep `python3-dev`).

- [ ] **Step 6: Run the fast tier and commit**

```bash
pytest -q
git add lfa/models.py tests/test_models.py README.md docs/quickstart.md
git commit -m "Toolchain check is a warning, not a refusal: a real GPU stage ran without headers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Artifact meta carries provenance

**Files:**
- Modify: `lfa/artifact/schema.py:67-100` (`make_meta`)
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces: `make_meta(..., provenance: str | None = None, corpus_sha256: str | None = None)`; both keys present in the dict (value `None` when absent). Constant `SELF_GENERATED = "self-generated"` in `lfa/artifact/schema.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_schema.py`:

```python
def test_meta_carries_provenance_when_given_and_none_otherwise():
    from lfa.artifact.schema import SELF_GENERATED, make_meta

    plain = make_meta("m", 32, 2, ["pre_qkv"], 10)
    assert plain["provenance"] is None and plain["corpus_sha256"] is None

    selfgen = make_meta("m", 32, 2, ["pre_qkv"], 10, provenance=SELF_GENERATED,
                        corpus_sha256="ab" * 32)
    assert selfgen["provenance"] == "self-generated"
    assert selfgen["corpus_sha256"] == "ab" * 32
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_schema.py::test_meta_carries_provenance_when_given_and_none_otherwise -q`
Expected: FAIL, `ImportError: SELF_GENERATED`.

- [ ] **Step 3: Implement**

In `lfa/artifact/schema.py`, after `EMBEDDING_LOOKUP_KEY`:

```python
#: ``__meta__["provenance"]`` of an artifact fitted on text the model wrote itself.
SELF_GENERATED = "self-generated"
```

Extend `make_meta`'s signature and dict:

```python
def make_meta(
    model_id: str,
    hidden_size: int,
    num_layers: int,
    sites: list[str],
    n_samples_total: int | None,
    built_with: str = "lfa-anchoring",
    provenance: str | None = None,
    corpus_sha256: str | None = None,
) -> dict:
    ...
    return {
        "model_id": model_id,
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "sites": list(sites),
        "n_samples_total": None if n_samples_total is None else int(n_samples_total),
        "built_with": built_with,
        "lfa_version": __version__,
        # What text the statistics were collected on. `None` is real text (the seed corpus, a
        # domain); SELF_GENERATED is text the model wrote, and `corpus_sha256` then names it.
        "provenance": provenance,
        "corpus_sha256": corpus_sha256,
    }
```

Add to the docstring `Args`: `provenance: :data:`SELF_GENERATED` for an artifact fitted on the model's own text; ``None`` for everything fitted on real text. corpus_sha256: the hash of the corpus file the statistics were collected on, when it is a generated one.`

- [ ] **Step 4: Run the schema tests**

Run: `pytest tests/test_schema.py tests/test_artifact_build.py -q`
Expected: PASS (existing callers pass no new args).

- [ ] **Step 5: Commit**

```bash
git add lfa/artifact/schema.py tests/test_schema.py
git commit -m "Artifact meta records provenance and corpus hash

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `lfa/selfgen/generate.py`, the sampling loop

**Files:**
- Create: `lfa/selfgen/__init__.py`, `lfa/selfgen/generate.py`
- Test: `tests/test_selfgen_generate.py` (fast), `tests/test_selfgen_gpu.py` (GPU, created here, extended by later tasks)

**Interfaces:**
- Produces:
  - `load_writer(model_id: str, device: str = "cuda:0") -> tuple[nn.Module, tokenizer]` — bf16 on CUDA, fp32 on CPU, `eval()`, tokenizer `padding_side="left"`, pad token guaranteed (via `lfa.models.load_tokenizer`).
  - `pick_seed_prefix(tokenizer, model=None) -> str`
  - `chat_user_header(tokenizer) -> str | None` — the text a user turn opens with (`"<|im_start|>user\n"` on Qwen3); `None` when the tokenizer has no chat template.
  - `boundary_markers(tokenizer, seed_prefix: str) -> tuple[str, ...]`
  - `clean_raw(text: str, markers) -> str`
  - `drop_burn_in(text, tokenizer, n_tokens) -> str`
  - `passes_filters(text, *, min_chars, max_repeat_ratio) -> bool`
  - `generate_texts(model, tokenizer, prompts: list[str], *, max_new_tokens, temperature, top_p, stop_token_ids: list[int], seed: int, batch_index: int = 0) -> list[str]` — returns the decoded **new** tokens with special tokens kept.
  - `checkpoint_sha256(model_id: str) -> str` — sha256 over every `*.safetensors` file (sorted) under `lfa.models.resolve_model_path(model_id)`.
  - `sha256_text(parts: Iterable[str]) -> str`

- [ ] **Step 1: Write the failing fast tests**

Create `tests/test_selfgen_generate.py`:

```python
"""The pieces of the sampling loop that need no model: prefix precedence, boundaries, filters."""
from types import SimpleNamespace

import pytest

from lfa.selfgen.generate import (boundary_markers, chat_user_header, clean_raw, drop_burn_in,
                                  passes_filters, pick_seed_prefix, sha256_text)


class _Tok:
    """A stub tokenizer: decode by id, optional bos/eos, optional chat template."""

    def __init__(self, bos=None, eos=None, template=True):
        self.bos_token, self.eos_token = bos, eos
        self._template = template

    def decode(self, ids, **_):
        return {7: "<|endoftext|>", 8: "<s>"}.get(ids[0], "?")

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
        if not self._template:
            raise ValueError("no chat template")
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def test_seed_prefix_prefers_the_models_declared_start_token():
    model = SimpleNamespace(generation_config=SimpleNamespace(bos_token_id=7))
    assert pick_seed_prefix(_Tok(bos="<s>"), model) == "<|endoftext|>"


def test_seed_prefix_falls_back_to_bos_then_eos_then_newline():
    assert pick_seed_prefix(_Tok(bos="<s>", eos="</s>")) == "<s>"
    assert pick_seed_prefix(_Tok(eos="</s>")) == "</s>"
    assert pick_seed_prefix(_Tok()) == "\n"


def test_chat_user_header_is_what_precedes_the_user_content():
    assert chat_user_header(_Tok()) == "<|im_start|>user\n"


def test_a_tokenizer_without_a_chat_template_gives_no_header():
    assert chat_user_header(_Tok(template=False)) is None


def test_clean_raw_cuts_at_the_first_boundary_marker():
    markers = boundary_markers(_Tok(), "<|endoftext|>")
    assert "<|endoftext|>" in markers and "<|im_start|>" in markers
    assert clean_raw("alpha beta<|im_start|>user\ngamma<|endoftext|>", markers) == "alpha beta"
    assert clean_raw("no marker here", markers) == "no marker here"


def test_drop_burn_in_removes_the_first_n_tokens_and_empties_short_text():
    class T(_Tok):
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}
        def decode(self, ids, **_):
            return " ".join(ids)
    assert drop_burn_in("a b c d", T(), 2) == "c d"
    assert drop_burn_in("a b", T(), 2) == ""
    assert drop_burn_in("a b", T(), 0) == "a b"


def test_passes_filters_rejects_short_and_looping_text():
    prose = "the quick brown fox jumps over the lazy dog and keeps running far away " * 3
    loop = "one two three four five six seven " * 20
    assert passes_filters(prose, min_chars=10, max_repeat_ratio=0.3)
    assert not passes_filters("too short", min_chars=200, max_repeat_ratio=1.0)
    assert not passes_filters(loop, min_chars=10, max_repeat_ratio=0.3)
    assert passes_filters(loop, min_chars=10, max_repeat_ratio=1.0)   # the unfiltered frame


def test_sha256_text_is_order_sensitive_and_stable():
    assert sha256_text(["a", "b"]) == sha256_text(["a", "b"])
    assert sha256_text(["a", "b"]) != sha256_text(["b", "a"])
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_selfgen_generate.py -q`
Expected: FAIL, `ModuleNotFoundError: lfa.selfgen`.

- [ ] **Step 3: Implement the module**

Create `lfa/selfgen/__init__.py`:

```python
"""Self-generation: the model supplies its own inputs.

Three things a user otherwise has to bring from outside -- the seed corpus p(h) is estimated on,
the question-and-answer supplement that makes a domain reachable, and a fresh p(h) for each stage
of a chain -- can be written by the model itself. The research record behind this
(mr-fusion claims C12, C14, C15) is one model (Qwen3-0.6B) and one seed; every number the
package documentation quotes about it carries that scope.
"""

from .artifact_corpus import SelfGenOptions, write_artifact_corpus  # noqa: F401  (Task 4)
from .supplement import write_supplement  # noqa: F401  (Task 9)

__all__ = ["SelfGenOptions", "write_artifact_corpus", "write_supplement"]
```

(Until Tasks 4 and 9 exist, leave the two imports commented out and `__all__ = []`; uncomment in those tasks.)

Create `lfa/selfgen/generate.py`:

```python
"""One batched sampling loop, and the text-level helpers around it.

Ported from the research generator (mr-fusion ``scripts/prepare_selfgen_corpus.py``). The one
rule that matters most is in :func:`generate_texts`: **every truncation knob is passed
explicitly**. ``model.generate`` inherits any knob it is not given from the checkpoint's
``generation_config.json``, and Qwen3's ships ``top_k: 20`` -- so passing only temperature and
top-p yields top-20 sampling out of a 151,936-token vocabulary while looking untruncated
(measured in the research record, 2026-09-09: lifting it took distinct sampled tokens from 817
to 1,326 at a fixed seed). ``top_k=0`` and ``min_p=0.0`` disable truncation;
``repetition_penalty=1.0`` keeps the chain a true sample of the model's distribution.

Reproducibility: ``transformers.generate`` takes no private generator, so each batch seeds
torch's global RNG from ``(seed, batch_index)``; a rebuild with the same seed and batch size
draws the same text.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch

from ..models import load_tokenizer, resolve_model_path

__all__ = [
    "load_writer", "pick_seed_prefix", "chat_user_header", "boundary_markers", "clean_raw",
    "drop_burn_in", "passes_filters", "generate_texts", "checkpoint_sha256", "sha256_text",
]

_HEADER_MARK = "␟"   # a character no template contains, to find where the content goes


def load_writer(model_id: str, device: str = "cuda:0"):
    """The model that writes: bf16 on an accelerator, fp32 on CPU, in eval mode, left-padded."""
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(model_id)
    tokenizer.padding_side = "left"                       # batched generation needs left padding
    dtype = torch.float32 if device.startswith("cpu") else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(resolve_model_path(model_id), dtype=dtype,
                                                 device_map=device).eval()
    return model, tokenizer


def pick_seed_prefix(tokenizer, model=None) -> str:
    """The token a free-running sample starts from, chosen without assuming a packing convention.

    Precedence: the model's DECLARED sequence-start id (Qwen3 sets
    ``generation_config.bos_token_id`` to ``<|endoftext|>`` while ``tok.bos_token`` is None),
    then the tokenizer's BOS, then its EOS, then a bare newline.
    """
    if model is not None:
        bid = getattr(getattr(model, "generation_config", None), "bos_token_id", None)
        if isinstance(bid, (list, tuple)):
            bid = bid[0] if bid else None
        if bid is not None:
            return tokenizer.decode([bid])
    for attr in ("bos_token", "eos_token"):
        token = getattr(tokenizer, attr, None)
        if token:
            return token
    return "\n"


def chat_user_header(tokenizer) -> str | None:
    """What a user turn opens with under the tokenizer's chat template, or ``None`` without one."""
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": _HEADER_MARK}], tokenize=False,
            add_generation_prompt=False)
    except Exception:                                     # no template, or one that rejects it
        return None
    head, sep, _ = rendered.partition(_HEADER_MARK)
    return head if sep else None


def boundary_markers(tokenizer, seed_prefix: str) -> tuple[str, ...]:
    """Where a raw continuation ends: the document boundary, and any chat-template opening."""
    markers = [seed_prefix]
    for token in (getattr(tokenizer, "eos_token", None),):
        if token and token not in markers:
            markers.append(token)
    header = chat_user_header(tokenizer)
    if header:
        # The first special token of the header, e.g. "<|im_start|>", not the whole header.
        first = header.split("\n", 1)[0]
        for candidate in (first.split("user")[0], first):
            if candidate and candidate not in markers:
                markers.append(candidate)
                break
    return tuple(markers)


def clean_raw(text: str, markers: Iterable[str]) -> str:
    """Truncate a raw continuation at the first boundary marker."""
    cut = len(text)
    for marker in markers:
        index = text.find(marker)
        if index != -1:
            cut = min(cut, index)
    return text[:cut].strip()


def drop_burn_in(text: str, tokenizer, n_tokens: int) -> str:
    """Discard the first ``n_tokens`` of a sample so the record does not depend on the prefix.

    An empirical decorrelation, not a mixing guarantee; the recorded frame uses 0.
    """
    if n_tokens <= 0:
        return text
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= n_tokens:
        return ""
    return tokenizer.decode(ids[n_tokens:])


def passes_filters(text: str, *, min_chars: int, max_repeat_ratio: float) -> bool:
    """Length floor plus a degeneracy filter: the repeated fraction of 8-grams, 1 - distinct/total."""
    if len(text) < min_chars:
        return False
    tokens = text.split()
    if len(tokens) < 8:
        return False
    grams = Counter(tuple(tokens[i:i + 8]) for i in range(len(tokens) - 7))
    total = sum(grams.values())
    return (1.0 - len(grams) / total) <= max_repeat_ratio


def generate_texts(model, tokenizer, prompts: list[str], *, max_new_tokens: int,
                   temperature: float, top_p: float, stop_token_ids: list[int], seed: int,
                   batch_index: int = 0) -> list[str]:
    """Sample one continuation per prompt; returns the NEW tokens, special tokens kept."""
    torch.manual_seed(seed * 1_000_003 + batch_index)
    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, top_k=0, min_p=0.0, repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_token_ids,
        )
    new = out[:, encoded["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new, skip_special_tokens=False)


def checkpoint_sha256(model_id: str) -> str:
    """One hash over every ``*.safetensors`` file of the checkpoint, in sorted order."""
    root = Path(resolve_model_path(model_id))
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.safetensors")):
        digest.update(path.name.encode())
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def sha256_text(parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()
```

- [ ] **Step 4: Run the fast tests**

Run: `pytest tests/test_selfgen_generate.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Write the GPU test module with its first test**

Create `tests/test_selfgen_gpu.py`:

```python
"""Self-generation on the card: Qwen3-0.6B writes, the package fits, trains and chains.

Sized for an 8 GB card (RTX 2070, 2026-09-26). Run with::

    LFA_SKIP_TOOLCHAIN_CHECK=1 HF_HUB_OFFLINE=1 pytest tests/test_selfgen_gpu.py -m gpu -q
"""
from __future__ import annotations

import pytest
import torch

from lfa.selfgen.generate import (boundary_markers, chat_user_header, clean_raw, generate_texts,
                                  load_writer, pick_seed_prefix)

pytestmark = pytest.mark.gpu

MODEL = "Qwen/Qwen3-0.6B"
DEVICE = "cuda:0"


@pytest.fixture(scope="module")
def writer():
    model, tokenizer = load_writer(MODEL, DEVICE)
    yield model, tokenizer
    del model
    torch.cuda.empty_cache()


def test_qwen3_writes_from_its_document_boundary_and_stops_at_the_next(writer):
    model, tokenizer = writer
    prefix = pick_seed_prefix(tokenizer, model)
    assert prefix == "<|endoftext|>"
    assert chat_user_header(tokenizer) == "<|im_start|>user\n"
    markers = boundary_markers(tokenizer, prefix)
    stop = [tokenizer.convert_tokens_to_ids(t) for t in ("<|endoftext|>", "<|im_start|>")]

    texts = generate_texts(model, tokenizer, [prefix] * 4, max_new_tokens=64, temperature=1.0,
                           top_p=1.0, stop_token_ids=stop, seed=42)
    again = generate_texts(model, tokenizer, [prefix] * 4, max_new_tokens=64, temperature=1.0,
                           top_p=1.0, stop_token_ids=stop, seed=42)

    assert len(texts) == 4 and all(clean_raw(t, markers) for t in texts)
    assert texts == again                                  # seeded per batch
```

- [ ] **Step 6: Run the GPU test outside the sandbox**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 HF_HUB_OFFLINE=1 pytest tests/test_selfgen_gpu.py -m gpu -q`
Expected: PASS (1 test, under 30 s).

- [ ] **Step 7: Commit**

```bash
pytest -q
git add lfa/selfgen tests/test_selfgen_generate.py tests/test_selfgen_gpu.py
git commit -m "selfgen: the sampling loop, seed prefix, boundaries and filters

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `lfa/selfgen/artifact_corpus.py`, the unconditional corpus

**Files:**
- Create: `lfa/selfgen/artifact_corpus.py`
- Modify: `lfa/selfgen/__init__.py` (uncomment the import)
- Test: `tests/test_selfgen_artifact_corpus.py` (fast), `tests/test_selfgen_gpu.py` (add one)

**Interfaces:**
- Produces:
  - `@dataclass SelfGenOptions(n_raw=2500, n_chat=250, max_new_tokens=2048, batch_size=32, seed=42, chat_seed=43, min_docs=50, max_empty_fraction=0.2, max_samples=600_000, gmm_k=32, pca_variance=0.95, layer_group_size=7, reservoir_size=200_000, device="cuda:0")` with `def frame(self) -> dict` (the generation fields only) and `def build_kwargs(self) -> dict` (the fit fields).
  - `class DegenerateCorpus(ValueError)`.
  - `write_artifact_corpus(model_id: str, out_path: Path, options: SelfGenOptions, *, generate=generate_texts, writer=None) -> dict` returning the manifest (also written to `<out_path>.manifest.json`). `writer` is an optional preloaded `(model, tokenizer)`.

- [ ] **Step 1: Write the failing fast tests**

Create `tests/test_selfgen_artifact_corpus.py`:

```python
"""The artifact corpus writer, with the model replaced by an injected generator."""
import json

import pytest

from lfa.selfgen.artifact_corpus import DegenerateCorpus, SelfGenOptions, write_artifact_corpus


class _Tok:
    bos_token = None
    eos_token = "<|endoftext|>"
    pad_token_id = 0

    def decode(self, ids, **_):
        return "<|endoftext|>"

    def apply_chat_template(self, messages, **_):
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"

    def convert_tokens_to_ids(self, token):
        return {"<|endoftext|>": 1, "<|im_start|>": 2}.get(token, 3)

    def __call__(self, text, **_):
        return {"input_ids": [0] * len(text.split())}


class _Model:
    device = "cpu"
    generation_config = None


def _gen_factory(script):
    calls = []

    def generate(model, tokenizer, prompts, **kwargs):
        calls.append((list(prompts), kwargs))
        return [script(p, i) for i, p in enumerate(prompts)]
    generate.calls = calls
    return generate


def test_writes_raw_and_chat_shares_with_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    gen = _gen_factory(lambda p, i: f"{p}some generated prose number {i} that goes on<|endoftext|>tail")
    options = SelfGenOptions(n_raw=5, n_chat=2, batch_size=4, min_docs=1)

    manifest = write_artifact_corpus("stub", tmp_path / "corpus.jsonl", options,
                                     generate=gen, writer=(_Model(), _Tok()))

    rows = [json.loads(l) for l in (tmp_path / "corpus.jsonl").read_text().splitlines()]
    assert [r["source"] for r in rows].count("selfgen_raw") == 5
    assert [r["source"] for r in rows].count("selfgen_chatfmt") == 2
    raw = [r for r in rows if r["source"] == "selfgen_raw"][0]["text"]
    assert raw.startswith("some generated") and "<|endoftext|>" not in raw    # prefix stripped, cut
    chat = [r for r in rows if r["source"] == "selfgen_chatfmt"][0]["text"]
    assert chat.startswith("<|im_start|>user\n")                             # header KEPT
    assert manifest["counts"] == {"raw": 5, "chat": 2, "empty": 0}
    assert manifest["frame"]["n_raw"] == 5 and manifest["writer_sha256"] == "c" * 64
    assert manifest["corpus_sha256"] and (tmp_path / "corpus.jsonl.manifest.json").is_file()
    # the recorded decoding frame, on every call
    assert all(k["temperature"] == 1.0 and k["top_p"] == 1.0 for _, k in gen.calls)
    # the two shares are seeded apart
    seeds = {k["seed"] for _, k in gen.calls}
    assert seeds == {42, 43}


def test_the_recorded_frame_is_the_default():
    o = SelfGenOptions()
    assert (o.n_raw, o.n_chat, o.max_new_tokens, o.seed, o.chat_seed) == (2500, 250, 2048, 42, 43)
    assert (o.max_samples, o.gmm_k, o.pca_variance) == (600_000, 32, 0.95)


def test_too_many_empty_documents_is_a_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    gen = _gen_factory(lambda p, i: "<|endoftext|>" if i % 2 else f"{p}text {i} here now")
    options = SelfGenOptions(n_raw=8, n_chat=0, batch_size=8, min_docs=1, max_empty_fraction=0.2)

    with pytest.raises(DegenerateCorpus, match="empty"):
        write_artifact_corpus("stub", tmp_path / "c.jsonl", options, generate=gen,
                              writer=(_Model(), _Tok()))
    assert not (tmp_path / "c.jsonl").exists()


def test_too_few_documents_is_a_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.artifact_corpus.checkpoint_sha256", lambda m: "c" * 64)
    gen = _gen_factory(lambda p, i: f"{p}fine text {i}")
    options = SelfGenOptions(n_raw=3, n_chat=0, batch_size=3, min_docs=50)

    with pytest.raises(DegenerateCorpus, match="50"):
        write_artifact_corpus("stub", tmp_path / "c.jsonl", options, generate=gen,
                              writer=(_Model(), _Tok()))
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_selfgen_artifact_corpus.py -q`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `lfa/selfgen/artifact_corpus.py`:

```python
"""The corpus p(h) is estimated on, written by the model with no input data.

The recorded frame (mr-fusion ``scripts/_w11_artifact_corpus.sh``, the artifact behind C12):
2,500 raw documents of up to 2,048 tokens started from the model's document-boundary token at
temperature 1.0 and top-p 1.0, **unfiltered**, seed 42; plus 250 documents started from the
bare user-turn header with the header kept, seed 43, so the corpus carries the chat-format share
the real seed corpus has. The artifact fitted on it at 600k samples per site, K=32, tied the
shipped artifact at every lambda tried -- one model, one seed.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .. import __version__
from .generate import (boundary_markers, chat_user_header, checkpoint_sha256, clean_raw,
                       generate_texts, load_writer, passes_filters, pick_seed_prefix,
                       sha256_text)

logger = logging.getLogger(__name__)

__all__ = ["SelfGenOptions", "DegenerateCorpus", "write_artifact_corpus"]

_GENERATION_FIELDS = ("n_raw", "n_chat", "max_new_tokens", "batch_size", "seed", "chat_seed",
                      "min_chars", "max_repeat_ratio", "burn_in_tokens")


class DegenerateCorpus(ValueError):
    """Raised when the generated corpus is too small or too empty to fit an artifact on."""


@dataclass
class SelfGenOptions:
    """The generation frame and the fit frame of a self-generated artifact. Defaults are the record."""

    # -- generation
    n_raw: int = 2500
    n_chat: int = 250
    max_new_tokens: int = 2048
    batch_size: int = 32
    seed: int = 42
    chat_seed: int = 43
    min_chars: int = 1                 # unfiltered, as recorded
    max_repeat_ratio: float = 1.0
    burn_in_tokens: int = 0
    min_docs: int = 50
    max_empty_fraction: float = 0.2
    # -- fit
    max_samples: int = 600_000
    gmm_k: int = 32
    pca_variance: float = 0.95
    layer_group_size: int | None = 7
    reservoir_size: int = 200_000
    device: str = "cuda:0"

    def frame(self) -> dict:
        return {name: getattr(self, name) for name in _GENERATION_FIELDS}

    def build_kwargs(self) -> dict:
        return dict(max_samples=self.max_samples, gmm_k=self.gmm_k,
                    pca_variance=self.pca_variance, layer_group_size=self.layer_group_size,
                    reservoir_size=self.reservoir_size, device=self.device)


def _stop_ids(tokenizer, markers) -> list[int]:
    ids = []
    for marker in markers:
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(token_id, int) and token_id >= 0 and token_id not in ids:
            ids.append(token_id)
    return ids or [tokenizer.eos_token_id]


def _write_share(model, tokenizer, *, prefix: str, keep_prefix: bool, n_docs: int, seed: int,
                 source: str, options: SelfGenOptions, markers, generate) -> tuple[list[dict], int]:
    """Generate ``n_docs`` documents from ``prefix``; returns (rows, empties)."""
    rows, empties, batch_index = [], 0, 0
    stop = _stop_ids(tokenizer, markers)
    while len(rows) < n_docs:
        size = min(options.batch_size, n_docs - len(rows))
        texts = generate(model, tokenizer, [prefix] * size, max_new_tokens=options.max_new_tokens,
                         temperature=1.0, top_p=1.0, stop_token_ids=stop, seed=seed,
                         batch_index=batch_index)
        batch_index += 1
        for text in texts:
            body = clean_raw(text, markers)
            if not body:
                empties += 1
                if empties > options.max_empty_fraction * n_docs + size:
                    break                                 # the refusal below decides
                continue
            if not passes_filters(body, min_chars=options.min_chars,
                                  max_repeat_ratio=options.max_repeat_ratio):
                empties += 1
                continue
            rows.append({"text": (prefix + body) if keep_prefix else body, "source": source})
        if empties > options.max_empty_fraction * max(n_docs, 1) and empties >= size:
            break
    return rows, empties


def write_artifact_corpus(model_id: str, out_path, options: SelfGenOptions, *,
                          generate=generate_texts, writer=None) -> dict:
    """Write the unconditional corpus to ``out_path`` (JSONL) and its manifest beside it.

    Args:
        writer: an already-loaded ``(model, tokenizer)``; loaded with :func:`load_writer` otherwise.
        generate: the sampling function (injectable for tests).

    Raises:
        DegenerateCorpus: fewer than ``options.min_docs`` documents, or more than
            ``options.max_empty_fraction`` of the draws came out empty.
    """
    out_path = Path(out_path)
    model, tokenizer = writer if writer is not None else load_writer(model_id, options.device)
    prefix = pick_seed_prefix(tokenizer, model)
    markers = boundary_markers(tokenizer, prefix)

    raw, raw_empty = _write_share(model, tokenizer, prefix=prefix, keep_prefix=False,
                                  n_docs=options.n_raw, seed=options.seed, source="selfgen_raw",
                                  options=options, markers=markers, generate=generate)
    chat, chat_empty = [], 0
    header = chat_user_header(tokenizer)
    if options.n_chat and header:
        chat, chat_empty = _write_share(model, tokenizer, prefix=header, keep_prefix=True,
                                        n_docs=options.n_chat, seed=options.chat_seed,
                                        source="selfgen_chatfmt", options=options,
                                        markers=markers, generate=generate)
    elif options.n_chat:
        logger.warning("%s has no chat template: the artifact corpus carries no chat-format "
                       "share", model_id)

    rows = raw + chat
    empties = raw_empty + chat_empty
    asked = options.n_raw + (options.n_chat if header else 0)
    if empties > options.max_empty_fraction * max(asked, 1):
        raise DegenerateCorpus(
            f"{empties} of {asked + empties} draws from {model_id} came out empty or degenerate "
            f"(limit {options.max_empty_fraction:.0%}). An artifact fitted on such a corpus "
            "fails nowhere downstream, so this is a refusal: check the seed prefix "
            f"({prefix!r}) and the model, or raise max_empty_fraction knowingly.")
    if len(rows) < options.min_docs:
        raise DegenerateCorpus(
            f"only {len(rows)} documents were generated; at least {options.min_docs} are "
            "needed to fit p(h) on (options.min_docs).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "kind": "artifact-corpus",
        "model_id": model_id,
        "writer_sha256": checkpoint_sha256(model_id),
        "seed_prefix": prefix,
        "chat_header": header,
        "frame": options.frame(),
        "decoding": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0},
        "counts": {"raw": len(raw), "chat": len(chat), "empty": empties},
        "corpus_sha256": sha256_text(r["text"] for r in rows),
        "lfa_version": __version__,
    }
    Path(str(out_path) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Self-generated corpus: %d raw + %d chat documents -> %s", len(raw), len(chat),
                out_path)
    return manifest
```

Uncomment the `artifact_corpus` import in `lfa/selfgen/__init__.py` and add `"SelfGenOptions", "write_artifact_corpus"` to `__all__`.

- [ ] **Step 4: Run the fast tests**

Run: `pytest tests/test_selfgen_artifact_corpus.py tests/test_selfgen_generate.py -q`
Expected: PASS.

- [ ] **Step 5: Add the GPU test**

Append to `tests/test_selfgen_gpu.py`:

```python
from lfa.selfgen.artifact_corpus import SelfGenOptions, write_artifact_corpus

SMALL = SelfGenOptions(n_raw=16, n_chat=4, max_new_tokens=128, batch_size=8, min_docs=10,
                       max_samples=20_000, gmm_k=4, layer_group_size=7, reservoir_size=5_000,
                       device=DEVICE)


@pytest.fixture(scope="module")
def small_corpus(tmp_path_factory, writer):
    out = tmp_path_factory.mktemp("selfgen") / "corpus.jsonl"
    manifest = write_artifact_corpus(MODEL, out, SMALL, writer=writer)
    return out, manifest


def test_the_artifact_corpus_has_both_shares_and_a_manifest(small_corpus):
    import json
    out, manifest = small_corpus
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert manifest["counts"]["raw"] == 16 and manifest["counts"]["chat"] == 4
    assert manifest["seed_prefix"] == "<|endoftext|>"
    assert all(r["text"].startswith("<|im_start|>user\n") for r in rows
               if r["source"] == "selfgen_chatfmt")
    assert not any("<|endoftext|>" in r["text"] for r in rows)
    assert len(manifest["writer_sha256"]) == 64
```

- [ ] **Step 6: Run the GPU tests outside the sandbox**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 HF_HUB_OFFLINE=1 pytest tests/test_selfgen_gpu.py -m gpu -q`
Expected: PASS (2 tests, about a minute).

- [ ] **Step 7: Commit**

```bash
pytest -q
git add lfa/selfgen tests/test_selfgen_artifact_corpus.py tests/test_selfgen_gpu.py
git commit -m "selfgen: the unconditional artifact corpus at the recorded frame

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `build_artifact_self_generated` and `lfa build-artifact --self-generated`

**Files:**
- Modify: `lfa/artifact/build.py` (add function; thread `provenance`/`corpus_sha256` into `make_meta`)
- Modify: `lfa/cli.py:145-151` (`_build_artifact`), `lfa/cli.py` build-artifact parser
- Test: `tests/test_artifact_build.py`, `tests/test_cli.py`, `tests/test_selfgen_gpu.py`

**Interfaces:**
- Produces: `build_artifact(..., provenance: str | None = None, corpus_sha256: str | None = None)` (passed through to `make_meta`); `build_artifact_self_generated(model_id, out_path, options: SelfGenOptions | None = None, *, corpus_path=None, quantize=True, seed=0, writer=None, generate=generate_texts) -> Path` writing the corpus to `corpus_path` (default `out_path.with_suffix(".corpus.jsonl")`).

- [ ] **Step 1: Write the failing fast tests**

Append to `tests/test_artifact_build.py`:

```python
def test_build_artifact_self_generated_writes_the_corpus_then_fits_with_provenance(tmp_path,
                                                                                  monkeypatch):
    """The corpus lands beside the artifact, and the meta names it."""
    import lfa.artifact.build as build_module
    from lfa.artifact.build import build_artifact_self_generated
    from lfa.selfgen.artifact_corpus import SelfGenOptions

    seen = {}

    def fake_corpus(model_id, out_path, options, *, generate, writer):
        Path(out_path).write_text('{"text": "generated"}\n')
        return {"corpus_sha256": "d" * 64}

    def fake_build(model_id, corpus_path, out_path, **kwargs):
        seen.update(kwargs, corpus=str(corpus_path))
        Path(out_path).write_bytes(b"pt")
        return Path(out_path)

    monkeypatch.setattr(build_module, "write_artifact_corpus", fake_corpus)
    monkeypatch.setattr(build_module, "build_artifact", fake_build)

    out = build_artifact_self_generated("m", tmp_path / "art.pt",
                                        SelfGenOptions(max_samples=123, gmm_k=4))

    assert out == tmp_path / "art.pt"
    assert seen["corpus"] == str(tmp_path / "art.corpus.jsonl")
    assert seen["max_samples"] == 123 and seen["gmm_k"] == 4
    assert seen["provenance"] == "self-generated" and seen["corpus_sha256"] == "d" * 64
```

Append to `tests/test_cli.py` (and add `"build-artifact"` is already in `SUBCOMMANDS`):

```python
def test_build_artifact_refuses_both_a_corpus_and_self_generated(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["build-artifact", "--model", "m", "--out", "x.pt", "--corpus", "c.jsonl",
              "--self-generated"])
    assert exit_info.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_build_artifact_needs_one_of_them(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["build-artifact", "--model", "m", "--out", "x.pt"])
    assert exit_info.value.code == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_artifact_build.py -k self_generated tests/test_cli.py -k build_artifact -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `lfa/artifact/build.py`, add to `build_artifact`'s signature `provenance: str | None = None, corpus_sha256: str | None = None,` (after `dtype`), document them, and pass them into `make_meta(...)`. Then add:

```python
from ..selfgen.artifact_corpus import SelfGenOptions, write_artifact_corpus
from ..selfgen.generate import generate_texts
from .schema import SELF_GENERATED


def build_artifact_self_generated(
    model_id: str,
    out_path,
    options: SelfGenOptions | None = None,
    *,
    corpus_path=None,
    quantize: bool = True,
    seed: int = 0,
    writer=None,
    generate=generate_texts,
) -> Path:
    """Write a corpus with the model itself, then fit p(h) on it at the recorded frame.

    The corpus lands at ``corpus_path`` (default ``<out>.corpus.jsonl``) with its manifest, so
    what the artifact was fitted on is inspectable. The artifact's meta carries
    ``provenance = "self-generated"`` and the corpus hash; :meth:`lfa.recipe.Recipe.warnings`
    reads both.

    ``options`` defaults to :class:`SelfGenOptions`, the frame of the artifact behind C12
    (2,500 + 250 documents; 600k samples per site; K=32). Scale it down for a smoke run.
    """
    options = options or SelfGenOptions()
    out_path = Path(out_path)
    corpus_path = Path(corpus_path) if corpus_path else out_path.with_suffix(".corpus.jsonl")
    manifest = write_artifact_corpus(model_id, corpus_path, options, generate=generate,
                                     writer=writer)
    return build_artifact(model_id, corpus_path, out_path, quantize=quantize, seed=seed,
                          provenance=SELF_GENERATED, corpus_sha256=manifest["corpus_sha256"],
                          **options.build_kwargs())
```

In `lfa/cli.py`:

```python
from .artifact.build import build_artifact, build_artifact_self_generated
from .selfgen.artifact_corpus import SelfGenOptions


def _build_artifact(args) -> int:
    # `--max-samples` has no argparse default: the two routes were measured at different
    # counts (1,500,000 over the seed corpus, 600,000 over the self-generated one).
    if args.self_generated:
        options = SelfGenOptions(n_raw=args.n_raw, n_chat=args.n_chat,
                                 max_new_tokens=args.max_new_tokens, seed=args.gen_seed,
                                 max_samples=args.max_samples or 600_000, gmm_k=args.gmm_k,
                                 pca_variance=args.pca_variance,
                                 layer_group_size=args.layer_group_size, device=args.device)
        print(build_artifact_self_generated(args.model, args.out, options,
                                            quantize=args.quantize, seed=args.seed))
        return 0
    print(build_artifact(args.model, args.corpus, args.out,
                         max_samples=args.max_samples or 1_500_000,
                         pca_variance=args.pca_variance, gmm_k=args.gmm_k,
                         layer_group_size=args.layer_group_size, quantize=args.quantize,
                         device=args.device, seed=args.seed))
    return 0
```

In the parser, change `--max-samples` to `type=int, default=None` with help "hidden vectors to collect per site (default: 1,500,000 with --corpus, 600,000 with --self-generated)". Then replace the `--corpus` argument with a mutually exclusive group and add the generation flags:

```python
    source = build.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", metavar="JSONL",
                        help="the seed corpus (see `lfa prepare-seed-corpus`)")
    source.add_argument("--self-generated", dest="self_generated", action="store_true",
                        help="write the corpus with the model itself first: 2,500 documents "
                             "from its document boundary plus 250 chat-format ones, then fit "
                             "at 600k samples per site (the frame of the C12 artifact). No "
                             "download.")
    build.add_argument("--n-raw", dest="n_raw", type=int, default=2500, metavar="N",
                       help="--self-generated: raw documents to write (default: %(default)s)")
    build.add_argument("--n-chat", dest="n_chat", type=int, default=250, metavar="N",
                       help="--self-generated: chat-format documents (default: %(default)s)")
    build.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=2048,
                       metavar="N", help="--self-generated: tokens per document (default: "
                                        "%(default)s)")
    build.add_argument("--gen-seed", dest="gen_seed", type=int, default=42, metavar="N",
                       help="--self-generated: the writer's seed (default: %(default)s)")
```

- [ ] **Step 4: Run the fast tests**

Run: `pytest tests/test_artifact_build.py tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Add the GPU test**

Append to `tests/test_selfgen_gpu.py`:

```python
from lfa.artifact.build import build_artifact_self_generated
from lfa.artifact.schema import load_artifact
from lfa.sampler import Sampler


@pytest.fixture(scope="module")
def small_artifact(tmp_path_factory, writer):
    out = tmp_path_factory.mktemp("selfgen_art") / "art.pt"
    build_artifact_self_generated(MODEL, out, SMALL, writer=writer)
    return out


def test_a_self_generated_artifact_fits_carries_provenance_and_samples(small_artifact):
    params = load_artifact(small_artifact)
    meta = params["__meta__"]
    assert meta["provenance"] == "self-generated" and len(meta["corpus_sha256"]) == 64
    assert meta["model_id"] == MODEL and meta["num_layers"] == 28
    assert small_artifact.with_suffix(".corpus.jsonl").is_file()

    sampler = Sampler(small_artifact, device=DEVICE, seed=0)
    draw = sampler.sample_best(5, "pre_mlp", 8)
    assert draw is not None and draw.shape == (8, 1024)
```

- [ ] **Step 6: Run the GPU tests outside the sandbox**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 HF_HUB_OFFLINE=1 pytest tests/test_selfgen_gpu.py -m gpu -q`
Expected: PASS (3 tests; the build loads Qwen3-0.6B in fp32, about 2.4 GB, four corpus passes).

- [ ] **Step 7: Commit**

```bash
pytest -q
git add lfa/artifact/build.py lfa/cli.py tests/test_artifact_build.py tests/test_cli.py tests/test_selfgen_gpu.py
git commit -m "build-artifact --self-generated: corpus from the model, then the recorded fit

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Recipe fields and the provenance-aware warning

**Files:**
- Modify: `lfa/recipe.py:40-130, 235-275`, `lfa/recipes/qwen3-0.6b.yaml`
- Test: `tests/test_recipe.py`

**Interfaces:**
- Produces: `Recipe.supplement_fraction: float = 0.13`, `Recipe.calibrated_self_generated: bool = False`; `Recipe.warnings(rank, artifact_id, artifact_meta: dict | None = None) -> list[str]`; `Recipe.to_train_config` unchanged (the fraction is consumed by the workspace, not the trainer).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_recipe.py`:

```python
import dataclasses


def _recipe(**overrides):
    """The bundled Qwen3 point with fields overridden (it is calibrated by declaration)."""
    return dataclasses.replace(Recipe.load("qwen3-0.6b"), **overrides)


SELFGEN_META = {"model_id": "Qwen/Qwen3-0.6B", "provenance": "self-generated"}


def test_a_self_generated_artifact_on_the_calibrated_model_and_flag_is_silent():
    recipe = _recipe(model_id="Qwen/Qwen3-0.6B", calibrated_self_generated=True)
    assert recipe.warnings(recipe.calibrated_rank, "self-generated:abc", SELFGEN_META) == []


def test_a_self_generated_artifact_without_the_flag_warns_to_calibrate():
    recipe = _recipe(model_id="Qwen/Qwen3-0.6B", calibrated_self_generated=False)
    notes = recipe.warnings(recipe.calibrated_rank, "self-generated:abc", SELFGEN_META)
    assert len(notes) == 1 and "calibrate lambda" in notes[0] and "adding-a-model" in notes[0]


def test_a_self_generated_artifact_for_another_model_warns_even_with_the_flag():
    recipe = _recipe(model_id="Qwen/Qwen3-0.6B", calibrated_self_generated=True)
    meta = {**SELFGEN_META, "model_id": "someone/other-model"}
    notes = recipe.warnings(recipe.calibrated_rank, "self-generated:abc", meta)
    assert len(notes) == 1 and "self-generated" in notes[0]


def test_a_real_text_artifact_keeps_the_existing_swap_warning():
    recipe = _recipe(model_id="Qwen/Qwen3-0.6B", calibrated_self_generated=True)
    notes = recipe.warnings(recipe.calibrated_rank, "other-artifact",
                            {"model_id": "Qwen/Qwen3-0.6B", "provenance": None})
    assert len(notes) == 1 and "coupled to the p(h) artifact" in notes[0]


def test_supplement_fraction_is_validated_and_defaults_to_the_measured_frame():
    assert _recipe().supplement_fraction == 0.13
    with pytest.raises(ValueError, match="supplement_fraction"):
        _recipe(supplement_fraction=1.0)


def test_the_bundled_qwen3_recipe_is_calibrated_for_self_generation():
    from lfa import Recipe
    recipe = Recipe.load("qwen3-0.6b")
    assert recipe.calibrated_self_generated is True and recipe.supplement_fraction == 0.13
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_recipe.py -q`
Expected: FAIL on the six new tests.

- [ ] **Step 3: Implement**

In `Recipe` (after `val_fraction`):

```python
    # -- the supplement: the question-and-answer pairs the entry model writes from the domain,
    #    mixed in at this share of training TOKENS. 0.13 is the frame the shipped lambda was
    #    tuned at; 0.0 trains on the raw corpus alone (and is off that frame).
    supplement_fraction: float = 0.13
```

and in the calibration block:

```python
    # True when an artifact fitted on this model's OWN text is a calibrated substitute for
    # `calibrated_artifact` (Qwen3-0.6B: C12, a tie at every lambda tried; one seed).
    calibrated_self_generated: bool = False
```

In `__post_init__`:

```python
        if not 0.0 <= self.supplement_fraction < 1.0:
            raise ValueError(
                f"supplement_fraction must be in [0, 1), got {self.supplement_fraction}: it is "
                "the share of training TOKENS the written pairs make up.")
```

Extend `warnings`:

```python
    def warnings(self, rank: int, artifact_id: str, artifact_meta: dict | None = None) -> list[str]:
        ...
        meta = artifact_meta or {}
        self_generated = meta.get("provenance") == "self-generated"
        if self_generated:
            if meta.get("model_id") == self.model_id and self.calibrated_self_generated:
                pass                                       # the calibrated substitute
            elif meta.get("model_id") == self.model_id:
                notes.append(
                    "this artifact was fitted on the model's own text and this recipe does not "
                    "record self-generation as calibrated: calibrate lambda against held-out "
                    "domain perplexity (docs/adding-a-model.md, 'Calibrating λ'), reading the "
                    "frontier rather than a single point."
                )
            else:
                notes.append(
                    f"this self-generated artifact describes {meta.get('model_id')!r}, not this "
                    f"recipe's {self.model_id!r}: lambda is coupled to the p(h) artifact, so "
                    "calibrate it against held-out domain perplexity for this model."
                )
        elif artifact_id != self.calibrated_artifact:
            notes.append(  # the existing artifact-swap text, unchanged
                ...
            )
```

Add to `lfa/recipes/qwen3-0.6b.yaml` under the calibration record:

```yaml
# -- self-generation (C12: one model, one seed). An artifact fitted on this model's own text tied
#    the gmm1543k artifact at every lambda tried, so it is a calibrated substitute here; the
#    supplement the entry model writes is mixed in at the fraction the lambda was tuned at.
calibrated_self_generated: true
supplement_fraction: 0.13
```

Add a docstring line for each field in the `Args` list.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_recipe.py tests/test_workspace.py -q`
Expected: PASS (existing callers pass no `artifact_meta`).

- [ ] **Step 5: Commit**

```bash
pytest -q
git add lfa/recipe.py lfa/recipes/qwen3-0.6b.yaml tests/test_recipe.py
git commit -m "Recipe: supplement_fraction and calibrated_self_generated; warnings read provenance

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: `Workspace.init(artifact="self-generated")` and `lfa init --artifact self-generated`

**Files:**
- Modify: `lfa/workspace.py:316-455` (init), `lfa/workspace.py:906-916` (`_artifact_id`), `lfa/cli.py` (`_init`, init parser)
- Test: `tests/test_workspace.py`, `tests/test_cli.py`, `tests/test_selfgen_gpu.py`

**Interfaces:**
- Consumes: `build_artifact_self_generated`, `SelfGenOptions`.
- Produces: `Workspace.init(..., artifact="self-generated", selfgen: SelfGenOptions | None = None)`; state keys `artifact_id = "self-generated:<sha[:12]>"`, `artifact_provenance = "self-generated"`; `Workspace._artifact_meta() -> dict` (the `__meta__` of `current_artifact`, `{}` if none); `Workspace.train` passes it to `recipe.warnings` (done here so the warning fires from this task on).

- [ ] **Step 1: Write the failing fast tests**

Append to `tests/test_workspace.py`:

```python
SELF_GENERATED = "self-generated"


def test_init_self_generated_builds_into_v1_and_records_provenance(tmp_path, base_dir,
                                                                    tiny_artifact, monkeypatch):
    import lfa.workspace as ws_module
    _, fixture = tiny_artifact

    def fake_build(model_id, out_path, options, **kwargs):
        Path(out_path).write_bytes(fixture.read_bytes())
        Path(out_path).with_suffix(".corpus.jsonl").write_text('{"text": "x"}\n')
        Path(str(Path(out_path).with_suffix(".corpus.jsonl")) + ".manifest.json").write_text(
            '{"corpus_sha256": "%s"}' % ("e" * 64))
        return Path(out_path)
    monkeypatch.setattr(ws_module, "build_artifact_self_generated", fake_build)

    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact=SELF_GENERATED)

    assert (tmp_path / "ws" / "artifacts" / "v1.pt").is_file()
    assert (tmp_path / "ws" / "artifacts" / "v1.corpus.jsonl").is_file()
    assert ws.state["artifact_id"] == "self-generated:" + "e" * 12
    assert ws.state["artifact_provenance"] == SELF_GENERATED


def test_init_self_generated_rolls_back_when_the_build_raises(tmp_path, base_dir, monkeypatch):
    import lfa.workspace as ws_module

    def failing(model_id, out_path, options, **kwargs):
        raise RuntimeError("no card")
    monkeypatch.setattr(ws_module, "build_artifact_self_generated", failing)

    with pytest.raises(RuntimeError, match="no card"):
        Workspace.init(tmp_path / "ws", str(base_dir), artifact=SELF_GENERATED)
    assert not (tmp_path / "ws").exists()
```

Append to `tests/test_cli.py`:

```python
def test_init_self_generated_is_routed_to_the_builder(tmp_path, base_dir, tiny_artifact,
                                                       monkeypatch, capsys):
    import lfa.workspace as ws_module
    _, fixture = tiny_artifact
    seen = {}

    def fake_build(model_id, out_path, options, **kwargs):
        seen["n_raw"] = options.n_raw
        Path(out_path).write_bytes(fixture.read_bytes())
        Path(out_path).with_suffix(".corpus.jsonl").write_text('{"text": "x"}\n')
        Path(str(Path(out_path).with_suffix(".corpus.jsonl")) + ".manifest.json").write_text(
            '{"corpus_sha256": "%s"}' % ("f" * 64))
        return Path(out_path)
    monkeypatch.setattr(ws_module, "build_artifact_self_generated", fake_build)

    assert main(["init", str(tmp_path / "ws"), "--model", str(base_dir),
                 "--artifact", "self-generated", "--n-raw", "7"]) == 0
    assert seen["n_raw"] == 7
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_workspace.py -k self_generated tests/test_cli.py -k self_generated -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `lfa/workspace.py`, import `from .artifact.build import build_artifact_self_generated`, `from .artifact.schema import SELF_GENERATED`, `from .selfgen.artifact_corpus import SelfGenOptions`. Add a module constant `SELF_GENERATED_ARTIFACT = "self-generated"`. In `init`, add parameter `selfgen: SelfGenOptions | None = None` and a branch **before** `if artifact in ARTIFACTS:`:

```python
        self_generated = artifact == SELF_GENERATED_ARTIFACT
        if self_generated:
            if artifact_id is not None:
                raise ValueError("artifact_id is for a local copy of a published artifact; a "
                                 "self-generated one names itself by its corpus hash.")
            artifact_id, source = None, None
        elif artifact in ARTIFACTS:
            ...  # unchanged
```

In the `try:` block, add a first branch:

```python
            if self_generated:
                options = selfgen or SelfGenOptions()
                build_artifact_self_generated(model_id, destination, options,
                                              corpus_path=artifacts_dir / "v1.corpus.jsonl")
                manifest = json.loads((artifacts_dir / "v1.corpus.jsonl.manifest.json").read_text())
                artifact_id = f"{SELF_GENERATED}:{manifest['corpus_sha256'][:12]}"
            elif source is not None:
                ...
```

Extend the rollback: after the `for directory in reversed(created): rmdir` loop, remove any partial files the build left (`for leftover in artifacts_dir.glob("v1*"): leftover.unlink(missing_ok=True)` **before** the rmdir loop, guarded by `if self_generated`). Add `"artifact_provenance": SELF_GENERATED if self_generated else None` to `state`. Document `artifact="self-generated"` and `selfgen` in the docstring.

Add:

```python
    def _artifact_meta(self) -> dict:
        """The ``__meta__`` block of the current artifact (``{}`` when there is none or it has none)."""
        path = self.state.get("current_artifact")
        if not path or not Path(path).is_file():
            return {}
        params = torch.load(path, map_location="cpu", weights_only=False)
        return dict(params.get(META_KEY) or {})
```

(import `META_KEY` from `.artifact.schema`). In `train`, change the warning loop to `for note in resolved.warnings(config.lora_rank, self._artifact_id(), self._artifact_meta()):`.

In `lfa/cli.py`, `_init` builds `SelfGenOptions` from `args.n_raw, args.n_chat, args.max_new_tokens, args.device` when `args.artifact == "self-generated"` and passes `selfgen=`; add `--n-raw`, `--n-chat`, `--max-new-tokens` (same defaults and help as Task 5) and `_add_device(init)` to the init parser; update `--artifact` help: "a published artifact id, a path to an artifact file, or `self-generated` to build one from the model's own text (no download)".

- [ ] **Step 4: Run the fast tests**

Run: `pytest tests/test_workspace.py tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Add the GPU test**

Append to `tests/test_selfgen_gpu.py`:

```python
from lfa.workspace import Workspace


def test_init_self_generated_on_qwen3_records_the_corpus_hash(tmp_path):
    ws = Workspace.init(tmp_path / "ws", MODEL, artifact="self-generated", selfgen=SMALL)
    assert ws.state["artifact_id"].startswith("self-generated:")
    assert (tmp_path / "ws" / "artifacts" / "v1.corpus.jsonl.manifest.json").is_file()
    assert ws._artifact_meta()["provenance"] == "self-generated"
```

- [ ] **Step 6: Run the GPU tests outside the sandbox**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 HF_HUB_OFFLINE=1 pytest tests/test_selfgen_gpu.py -m gpu -q`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
pytest -q
git add lfa/workspace.py lfa/cli.py tests/test_workspace.py tests/test_cli.py tests/test_selfgen_gpu.py
git commit -m "init --artifact self-generated: build v1 from the model's own text

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Mixing in `lfa/corpus.py`

**Files:**
- Modify: `lfa/corpus.py:451-476` (`_extract_text`), `lfa/corpus.py:526-568` (`load_corpus`)
- Test: `tests/test_corpus.py`

**Interfaces:**
- Produces:
  - `split_documents(path, val_fraction: float, seed: int) -> tuple[list[str], list[str]]` — the seed-shuffled raw documents, training side then held-out side (the exact split `load_corpus` makes; raises the existing "holds out all" `ValueError`).
  - `@dataclass Selection(n_used: int, achieved_fraction: float, under_target: bool)`; `select_supplement_prefix(raw_tokens: list[int], pair_tokens: list[int], target: float) -> Selection`.
  - `render_pair(tokenizer, prompt: str, response: str) -> str` — chat template with `enable_thinking=False`, plain join fallback.
  - `load_supplement(path, tokenizer) -> list[str]` — rendered pair texts in file order.
  - `load_corpus(..., supplement=None, supplement_fraction: float = 0.0)`; the returned training `ChunkedCorpus` gets `.supplement_report: dict | None` = `{"n_available", "n_used", "target_fraction", "achieved_fraction", "under_target"}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_corpus.py`:

```python
from lfa.corpus import (load_corpus, load_supplement, render_pair, select_supplement_prefix,
                        split_documents)


def test_zero_fraction_selects_no_pairs():
    r = select_supplement_prefix([100, 100], [30, 30, 30], 0.0)
    assert (r.n_used, r.achieved_fraction, r.under_target) == (0, 0.0, False)


def test_picks_the_count_closest_to_the_target():
    # total_raw=200, target=0.2 -> needed=50; n=1: 30/230=0.130; n=2: 60/260=0.231 -> 2
    r = select_supplement_prefix([100, 100], [30, 30, 30], 0.2)
    assert r.n_used == 2 and r.achieved_fraction == pytest.approx(60 / 260, abs=1e-4)
    assert r.under_target is False


def test_a_pool_too_small_takes_all_and_flags_under_target():
    r = select_supplement_prefix([100], [10, 10], 0.5)
    assert r.n_used == 2 and r.under_target is True
    assert r.achieved_fraction == pytest.approx(20 / 120, abs=1e-4)


def test_an_empty_pool_with_a_positive_target_is_under():
    r = select_supplement_prefix([100], [], 0.25)
    assert (r.n_used, r.achieved_fraction, r.under_target) == (0, 0.0, True)


def test_a_fraction_at_or_above_one_is_rejected():
    with pytest.raises(ValueError):
        select_supplement_prefix([100], [10], 1.0)


def test_a_pool_larger_than_the_target_uses_a_prefix_and_reports_the_rest(tmp_path, tiny_model):
    _, tokenizer = tiny_model
    docs = tmp_path / "docs"; docs.mkdir()
    for i in range(4):
        (docs / f"d{i}.txt").write_text("raw document text " * 20)
    supp = tmp_path / "supplement.jsonl"
    supp.write_text("".join('{"prompt": "q%d?", "response": "a%d."}\n' % (i, i) for i in range(40)))

    train, _ = load_corpus(docs, tokenizer, max_length=64, val_fraction=0.0, seed=0,
                           supplement=supp, supplement_fraction=0.1)

    report = train.supplement_report
    assert report["n_available"] == 40 and 0 < report["n_used"] < 40
    assert abs(report["achieved_fraction"] - 0.1) < 0.05 and report["under_target"] is False


def test_the_held_out_split_is_taken_from_raw_documents_before_mixing(tmp_path, tiny_model):
    _, tokenizer = tiny_model
    docs = tmp_path / "docs"; docs.mkdir()
    for i in range(10):
        (docs / f"d{i}.txt").write_text(f"raw document {i} " * 20)
    supp = tmp_path / "supplement.jsonl"
    supp.write_text('{"prompt": "q?", "response": "PAIRTEXT."}\n' * 5)

    train_docs, held = split_documents(docs, 0.2, seed=3)
    train, val = load_corpus(docs, tokenizer, max_length=64, val_fraction=0.2, seed=3,
                             supplement=supp, supplement_fraction=0.3)

    assert len(held) == 2 and val.report["n_docs"] == 2
    assert not any("PAIRTEXT" in t for t in held)
    assert train.report["n_docs"] == 8 + train.supplement_report["n_used"]


def test_a_single_document_corpus_trains_on_it_and_holds_nothing_out(tmp_path, tiny_model):
    _, tokenizer = tiny_model
    docs = tmp_path / "docs"; docs.mkdir()
    (docs / "only.txt").write_text("one long document " * 50)
    train, val = load_corpus(docs, tokenizer, max_length=64, val_fraction=0.1, seed=0)
    assert train.report["n_docs"] == 1 and val is not None and val.report["n_docs"] == 0


def test_render_pair_uses_the_non_thinking_template_when_there_is_one():
    class T:
        def apply_chat_template(self, messages, tokenize=False, enable_thinking=None, **_):
            assert enable_thinking is False
            return "<u>" + messages[0]["content"] + "</u><a>" + messages[1]["content"] + "</a>"
    assert render_pair(T(), "q", "a") == "<u>q</u><a>a</a>"

    class NoTemplate:
        def apply_chat_template(self, *a, **k):
            raise ValueError("none")
    assert render_pair(NoTemplate(), "q", "a") == "q\na"
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_corpus.py -q`
Expected: FAIL, `ImportError`.

- [ ] **Step 3: Implement**

In `lfa/corpus.py`, add (after `_extract_text`):

```python
def render_pair(tokenizer, prompt: str, response: str) -> str:
    """One question-and-answer pair as a full chat turn, ``enable_thinking=False``.

    Qwen3 then inserts the empty ``<think>\\n\\n</think>`` block that the research training
    format carries (mr-fusion ``prepare_domain_qa.qa_to_chat_text``), so a written pair is
    trained in the format the model answers in. Without a template the two are joined plainly.
    """
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=False, enable_thinking=False)
    except Exception:
        logger.debug("Chat template failed for a pair; joining plainly.")
        return (prompt + "\n" + response).strip()


def load_supplement(path, tokenizer) -> list[str]:
    """The written pairs as training documents, in file order (the prefix rule needs it)."""
    texts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("prompt"):
                texts.append(render_pair(tokenizer, str(record["prompt"]),
                                         str(record.get("response", ""))))
    return texts


@dataclass
class Selection:
    """How many pairs a token-fraction target selects, and what share they actually make."""
    n_used: int
    achieved_fraction: float
    under_target: bool


def select_supplement_prefix(raw_tokens: list[int], pair_tokens: list[int],
                             target: float) -> Selection:
    """The pair prefix whose token share ``used / (raw + used)`` is closest to ``target``.

    Ported verbatim from the research mixer (``prepare_domain_qa.select_qa_for_fraction``):
    pairs are added in order, the largest count under the need is compared with the first count
    over it, and the closer one wins.
    """
    if not (0.0 <= target < 1.0):
        raise ValueError(f"target must be in [0, 1): got {target}")
    total_raw = sum(raw_tokens)
    if target == 0.0:
        return Selection(0, 0.0, under_target=False)
    if not pair_tokens or total_raw == 0:
        return Selection(0, 0.0, under_target=True)

    needed = target / (1.0 - target) * total_raw
    cumsums = list(itertools.accumulate(pair_tokens))
    k = 0
    for i, c in enumerate(cumsums):
        if c <= needed:
            k = i + 1
        else:
            break

    def achieved(n: int) -> float:
        used = cumsums[n - 1] if n > 0 else 0
        return used / (total_raw + used)

    candidates = [k] + ([k + 1] if k < len(pair_tokens) else [])
    best = min(candidates, key=lambda n: abs(achieved(n) - target))
    return Selection(best, achieved(best), under_target=achieved(best) < target - 1e-9)


def split_documents(path, val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """The seed-shuffled raw documents, training side then held-out side.

    This is the one split: the trainer, ``evaluate`` and the supplement writer all read it, so
    a pair is never written from a document the stage is scored on.
    """
    texts = load_texts(path)
    if not texts:
        raise ValueError(f"No texts found in {path}")
    random.Random(seed).shuffle(texts)
    if val_fraction <= 0.0:
        return texts, []
    split_idx = int(len(texts) * (1 - val_fraction))
    if split_idx == 0:
        raise ValueError(
            f"val_fraction={val_fraction} holds out all {len(texts)} document(s) found in "
            f"{path}, leaving nothing to train on. Lower it, or pass val_fraction=0.0 to train "
            "on everything and read the domain number as a fit rather than a measurement.")
    return texts[:split_idx], texts[split_idx:]
```

Rewrite `load_corpus`:

```python
def load_corpus(path, tokenizer, max_length=512, stride=0, val_fraction=0.0, seed=42,
                keep_short_whole=True, supplement=None, supplement_fraction=0.0):
    """Read ``path`` and build the training corpus, optionally holding documents out and
    mixing a written supplement in.

    The held-out split is taken from the RAW documents first (:func:`split_documents`), so the
    held-out perplexity stays a raw-text number comparable across runs; then a prefix of the
    supplement is chosen for ``supplement_fraction`` of training tokens
    (:func:`select_supplement_prefix`) and shuffled into the training side under ``seed``. The
    training corpus's ``supplement_report`` says what was used.
    """
    train_texts, val_texts = split_documents(path, val_fraction, seed)

    report = None
    if supplement is not None and supplement_fraction > 0.0:
        pairs = load_supplement(supplement, tokenizer)
        raw_tokens = [len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in train_texts]
        pair_tokens = [len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in pairs]
        chosen = select_supplement_prefix(raw_tokens, pair_tokens, supplement_fraction)
        report = {"n_available": len(pairs), "n_used": chosen.n_used,
                  "target_fraction": supplement_fraction,
                  "achieved_fraction": chosen.achieved_fraction,
                  "under_target": chosen.under_target}
        if chosen.under_target:
            logger.warning("Supplement pool short of the target: %d pairs give a token share of "
                           "%.3f against %.3f asked", chosen.n_used, chosen.achieved_fraction,
                           supplement_fraction)
        train_texts = train_texts + pairs[:chosen.n_used]
        random.Random(seed + 1).shuffle(train_texts)

    def build(subset):
        corpus = ChunkedCorpus(subset, tokenizer, max_length=max_length, stride=stride,
                               keep_short_whole=keep_short_whole)
        corpus.supplement_report = None
        return corpus

    train = build(train_texts)
    train.supplement_report = report
    if val_fraction <= 0.0:
        return train, None
    return train, build(val_texts)
```

(Add `import itertools` and `from dataclasses import dataclass`; `ChunkedCorpus.__init__` must tolerate an empty `texts` list for the one-document case: check `ChunkedCorpus` and, if it raises on zero documents, make `report["n_docs"] == 0` a valid empty corpus rather than an error, since `val_fraction=0.1` on one document holds out none.) Also change `_extract_text`'s chat rendering to pass `enable_thinking=False` for consistency.

- [ ] **Step 4: Run the corpus tests**

Run: `pytest tests/test_corpus.py tests/test_workspace.py tests/test_evaluate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
pytest -q
git add lfa/corpus.py tests/test_corpus.py
git commit -m "corpus: split first, then mix a supplement prefix at a token fraction

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: `lfa/selfgen/supplement.py`, the writer

**Files:**
- Create: `lfa/selfgen/supplement.py`
- Modify: `lfa/selfgen/__init__.py`
- Test: `tests/test_selfgen_supplement.py` (fast), `tests/test_selfgen_gpu.py`

**Interfaces:**
- Produces:
  - `GENERATE_TEMPLATE: str` (the research template with `{domain}`, `{n}`, `{passage}` slots), `render_template(domain_description, n, passage) -> str`, `template_sha256() -> str`.
  - `chunk_document(text, target_chars) -> list[str]`, `chunk_passages(text, size=4000, min_size=200) -> list[str]`.
  - `parse_assistant_turn(decoded: str) -> str`, `parse_qa_pairs(text) -> list[dict]`.
  - `@dataclass SupplementOptions(pairs_per_passage=6, passage_chars=4000, min_passage_chars=200, max_new_tokens=1024, temperature=0.7, top_p=0.8, batch_size=16, min_answer_chars=40, max_answer_chars=100_000, seed=42)`.
  - `class NoPairsWritten(ValueError)`.
  - `write_supplement(model_id, documents: list[str], out_path, *, domain_description: str, options=None, corpus_sha256: str, generate=generate_texts, writer=None, device="cuda:0") -> dict` (manifest, also written to `<out>.manifest.json`).

- [ ] **Step 1: Write the failing fast tests**

Create `tests/test_selfgen_supplement.py`:

```python
"""The supplement writer's text logic: passages, the template, the forgiving parser."""
import json

import pytest

from lfa.selfgen.supplement import (NoPairsWritten, SupplementOptions, chunk_document,
                                    chunk_passages, parse_qa_pairs, render_template,
                                    write_supplement)


def test_parses_a_clean_json_array():
    out = parse_qa_pairs('[{"question": "What is qualia?", "answer": "Subjective experience."}]')
    assert out == [{"question": "What is qualia?", "answer": "Subjective experience."}]


def test_parses_json_embedded_in_chatter():
    text = 'Sure! Here you go:\n[{"question": "Q1?", "answer": "A1."}]\nHope that helps.'
    assert parse_qa_pairs(text) == [{"question": "Q1?", "answer": "A1."}]


def test_parses_question_answer_markers_when_json_fails():
    text = "Question: What is the hard problem?\nAnswer: Explaining why experience exists.\n"
    assert parse_qa_pairs(text) == [
        {"question": "What is the hard problem?", "answer": "Explaining why experience exists."}]


def test_parses_multiple_marker_pairs():
    text = "Q: First question?\nA: First answer.\n\nQ: Second question?\nA: Second answer."
    out = parse_qa_pairs(text)
    assert [p["question"] for p in out] == ["First question?", "Second question?"]
    assert out[1]["answer"] == "Second answer."


def test_rejects_empty_or_malformed():
    assert parse_qa_pairs("") == []
    assert parse_qa_pairs("just some prose with no pairs at all") == []
    assert parse_qa_pairs('[{"question": "", "answer": "orphan"}]') == []


def test_salvages_complete_objects_from_a_truncated_array():
    text = '[{"question": "Q1?", "answer": "A1."}, {"question": "Q2?", "answer": "A2."}, {"ques'
    assert [p["question"] for p in parse_qa_pairs(text)] == ["Q1?", "Q2?"]


def test_chunk_document_splits_and_merges_on_paragraphs():
    split = chunk_document("A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100, 150)
    assert [c[0] for c in split] == ["A", "B", "C"]
    merged = chunk_document("A" * 50 + "\n\n" + "B" * 50, 150)
    assert len(merged) == 1 and "A" * 50 in merged[0] and "B" * 50 in merged[0]


def test_chunk_passages_drops_runts():
    assert chunk_passages("y" * 150, size=1000, min_size=200) == []


def test_the_template_names_the_domain_and_keeps_the_rules():
    prompt = render_template("Victorian cookery", 6, "PASSAGE")
    assert "about a text on Victorian cookery" in prompt
    assert "write 6 diverse question-answer pairs" in prompt
    assert "Keep answers to 2-5 sentences" in prompt and "PASSAGE" in prompt
    assert "philosophy" not in prompt


class _Tok:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=None, **_):
        return "<u>" + messages[0]["content"] + "</u><a>"

    def convert_tokens_to_ids(self, token):
        return 2


class _Model:
    device = "cpu"


def test_write_supplement_records_pairs_and_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.supplement.checkpoint_sha256", lambda m: "a" * 64)
    docs = ["para one " * 60 + "\n\n" + "para two " * 60, "short " * 80]

    def generate(model, tokenizer, prompts, **kwargs):
        assert kwargs["temperature"] == 0.7 and kwargs["top_p"] == 0.8
        return ['[{"question": "Why?", "answer": "' + "Because of the passage. " * 3 + '"},'
                ' {"question": "Tiny?", "answer": "no"}]<|im_end|>'] * len(prompts)

    manifest = write_supplement("stub", docs, tmp_path / "s.jsonl", domain_description="tests",
                                options=SupplementOptions(passage_chars=400, min_passage_chars=10,
                                                          batch_size=4),
                                corpus_sha256="b" * 64, generate=generate,
                                writer=(_Model(), _Tok()))

    rows = [json.loads(l) for l in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert rows and all(set(r) == {"prompt", "response", "source_index"} for r in rows)
    assert all(len(r["response"]) >= 40 for r in rows)          # the short answer was dropped
    assert manifest["n_passages"] >= 2 and manifest["n_pairs"] == len(rows)
    assert manifest["rejected"]["short_answer"] >= 1
    assert manifest["writer_sha256"] == "a" * 64 and manifest["corpus_sha256"] == "b" * 64
    assert manifest["template_sha256"] and "tests" in manifest["template"]
    assert (tmp_path / "s.jsonl.manifest.json").is_file()


def test_no_pairs_at_all_is_a_refusal_naming_the_tally(tmp_path, monkeypatch):
    monkeypatch.setattr("lfa.selfgen.supplement.checkpoint_sha256", lambda m: "a" * 64)

    def generate(model, tokenizer, prompts, **kwargs):
        return ["nothing parseable here"] * len(prompts)

    with pytest.raises(NoPairsWritten, match="1 passage"):
        write_supplement("stub", ["text " * 100], tmp_path / "s.jsonl", domain_description="d",
                         options=SupplementOptions(min_passage_chars=10), corpus_sha256="b" * 64,
                         generate=generate, writer=(_Model(), _Tok()))


def test_no_passages_refuses_before_any_model_loads(tmp_path):
    with pytest.raises(NoPairsWritten, match="0 passages"):
        write_supplement("stub", ["tiny"], tmp_path / "s.jsonl", domain_description="d",
                         corpus_sha256="b" * 64, writer=None)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_selfgen_supplement.py -q`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `lfa/selfgen/supplement.py`:

```python
"""The domain supplement, written by the entry model from the domain's own passages.

Self-distillation of SKILL, not of knowledge: the writer reads each passage in context, so the
domain content comes from the corpus and only question-forming, answer construction and
third-person voice come from the model. The research record (C12) measured the resulting
supplement about 0.1 below a frontier-written one on the judge, and found its job to be
*reachability* -- the new knowledge becomes answerable in question-and-answer form -- not the
protection of any skill (C14). One model, one seed.

Two deliberate deviations from the research frame: the template says "about a text on
{domain}" where the research one said "about a philosophy text", and there is no contamination
screen against an evaluation set (a user has none). The comparison-only `--enforce-spec`
reminder and the reasoning/instruction modes are not ported.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import __version__
from .generate import checkpoint_sha256, generate_texts, load_writer

logger = logging.getLogger(__name__)

__all__ = ["GENERATE_TEMPLATE", "SupplementOptions", "NoPairsWritten", "render_template",
           "template_sha256", "chunk_document", "chunk_passages", "parse_assistant_turn",
           "parse_qa_pairs", "write_supplement"]

# The research template (mr-fusion `prepare_domain_qa.GENERATE_PROMPT`) with the one change
# recorded in the module docstring.
GENERATE_TEMPLATE = """You are creating training data that teaches a small language model to \
ANSWER QUESTIONS about a text on {domain} in a helpful assistant's voice.

From the passage below, write {n} diverse question-answer pairs.

Rules:
- QUESTIONS: natural, information-seeking questions a curious student might ask. Do NOT write \
exam-style questions that name specific thought experiments or arguments (avoid "How does the \
author use the X thought experiment to argue Y"). Prefer "What is...", "Why does...", "What is \
the relationship between...", "What does the author mean by...".
- ANSWERS: written in the THIRD PERSON as a knowledgeable assistant, referring to the author by \
name where the passage names them (e.g., "Chalmers argues that..."). NEVER answer in the first \
person as the author ("I argue..."). Ground every answer strictly in the passage; do not invent. \
Keep answers to 2-5 sentences.
- Vary the difficulty and type.

Passage:
\"\"\"
{passage}
\"\"\"

Return ONLY a JSON array of objects: [{{"question": "...", "answer": "..."}}, ...]"""

_OBJ = re.compile(r'\{\s*"question"\s*:\s*"(.*?)"\s*,\s*"answer"\s*:\s*"(.*?)"\s*\}', re.S)
_MARK = re.compile(r'(?:^|\n)\s*(?:Question|Q)\s*[:.\)]\s*(.+?)\n\s*(?:Answer|A)\s*[:.\)]\s*(.+?)'
                   r'(?=\n\s*(?:Question|Q)\s*[:.\)]|\Z)', re.S | re.I)
_TURN_END = "<|im_end|>"


class NoPairsWritten(ValueError):
    """Raised when the corpus gives no passages, or the writer produced no usable pair."""


@dataclass
class SupplementOptions:
    """The recorded frame of the self-written supplement (C12)."""
    pairs_per_passage: int = 6
    passage_chars: int = 4000
    min_passage_chars: int = 200
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.8
    batch_size: int = 16
    min_answer_chars: int = 40
    max_answer_chars: int = 100_000
    seed: int = 42


def render_template(domain_description: str, n: int, passage: str) -> str:
    return GENERATE_TEMPLATE.format(domain=domain_description, n=n, passage=passage)


def template_sha256() -> str:
    return hashlib.sha256(GENERATE_TEMPLATE.encode("utf-8")).hexdigest()


def chunk_document(text: str, target_chars: int) -> list[str]:
    """Split on blank-line (paragraph) boundaries into ~``target_chars`` sections; ported verbatim."""
    paras = [p for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > target_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur.strip():
        chunks.append(cur)
    return chunks


def chunk_passages(text: str, size: int = 4000, min_size: int = 200) -> list[str]:
    return [c for c in chunk_document(text, size) if len(c) >= min_size]


def parse_assistant_turn(decoded: str) -> str:
    """The assistant's text up to the turn end; the whole string when there is no turn end."""
    return decoded.split(_TURN_END, 1)[0] if _TURN_END in decoded else decoded


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip('"')


def parse_qa_pairs(text: str) -> list[dict]:
    """JSON objects first (salvaging complete ones from a truncated array), then Q/A markers."""
    pairs = [{"question": _clean(q), "answer": _clean(a)} for q, a in _OBJ.findall(text or "")]
    if not pairs:
        pairs = [{"question": _clean(q), "answer": _clean(a)} for q, a in _MARK.findall(text or "")]
    return [p for p in pairs if p["question"] and p["answer"]]


def _prompt_for(tokenizer, passage: str, n: int, domain_description: str) -> str:
    body = render_template(domain_description, n, passage)
    try:
        return tokenizer.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    except Exception:
        return body + "\n"


def write_supplement(model_id: str, documents: list[str], out_path, *, domain_description: str,
                     options: SupplementOptions | None = None, corpus_sha256: str,
                     generate=generate_texts, writer=None, device: str = "cuda:0") -> dict:
    """Write question-and-answer pairs from ``documents`` (the TRAINING side only) to ``out_path``.

    Raises:
        NoPairsWritten: no passage of at least ``min_passage_chars`` (before any model loads),
            or no pair survived parsing and the length filter.
    """
    options = options or SupplementOptions()
    out_path = Path(out_path)
    passages = [(index, passage) for index, text in enumerate(documents)
                for passage in chunk_passages(text, options.passage_chars,
                                              options.min_passage_chars)]
    if not passages:
        raise NoPairsWritten(
            f"0 passages of at least {options.min_passage_chars} characters in "
            f"{len(documents)} training document(s): nothing to write a supplement from.")

    model, tokenizer = writer if writer is not None else load_writer(model_id, device)
    stop = [tokenizer.eos_token_id]
    end_id = tokenizer.convert_tokens_to_ids(_TURN_END)
    if isinstance(end_id, int) and end_id >= 0 and end_id not in stop:
        stop.append(end_id)

    rows, rejected, seen = [], {"short_answer": 0, "long_answer": 0, "duplicate": 0,
                                "unparseable_passage": 0}, set()
    for batch_index in range(0, len(passages), options.batch_size):
        batch = passages[batch_index:batch_index + options.batch_size]
        prompts = [_prompt_for(tokenizer, passage, options.pairs_per_passage, domain_description)
                   for _, passage in batch]
        outputs = generate(model, tokenizer, prompts, max_new_tokens=options.max_new_tokens,
                           temperature=options.temperature, top_p=options.top_p,
                           stop_token_ids=stop, seed=options.seed,
                           batch_index=batch_index // options.batch_size)
        for (source_index, _), output in zip(batch, outputs):
            pairs = parse_qa_pairs(parse_assistant_turn(output))
            if not pairs:
                rejected["unparseable_passage"] += 1
            for pair in pairs:
                length = len(pair["answer"])
                if length < options.min_answer_chars:
                    rejected["short_answer"] += 1
                    continue
                if length > options.max_answer_chars:
                    rejected["long_answer"] += 1
                    continue
                key = (pair["question"], pair["answer"])
                if key in seen:
                    rejected["duplicate"] += 1
                    continue
                seen.add(key)
                rows.append({"prompt": pair["question"], "response": pair["answer"],
                             "source_index": source_index})
        logger.info("supplement: %d/%d passages -> %d pairs", min(batch_index + len(batch),
                    len(passages)), len(passages), len(rows))

    if not rows:
        raise NoPairsWritten(
            f"{len(passages)} passage(s) were sent to {model_id} and no usable pair came back "
            f"(rejected: {rejected}). Check that the model follows the template; a chat model "
            "with a chat template is expected.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "kind": "supplement",
        "model_id": model_id,
        "writer_sha256": checkpoint_sha256(model_id),
        "corpus_sha256": corpus_sha256,
        "domain_description": domain_description,
        "template": render_template(domain_description, options.pairs_per_passage, "{passage}"),
        "template_sha256": template_sha256(),
        "options": asdict(options),
        "decoding": {"temperature": options.temperature, "top_p": options.top_p, "top_k": 0,
                     "min_p": 0.0},
        "n_documents": len(documents),
        "n_passages": len(passages),
        "n_pairs": len(rows),
        "rejected": rejected,
        "lfa_version": __version__,
    }
    Path(str(out_path) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
```

Uncomment the `supplement` import in `lfa/selfgen/__init__.py`.

- [ ] **Step 4: Run the fast tests**

Run: `pytest tests/test_selfgen_supplement.py -q`
Expected: PASS (12 tests).

- [ ] **Step 5: Add the GPU test**

Append to `tests/test_selfgen_gpu.py`:

```python
from lfa.selfgen.supplement import SupplementOptions, write_supplement

DARWIN = """On the Origin of Species was published in 1859. Darwin argued that species change over
time through a process he called natural selection, in which individuals better suited to their
environment leave more offspring.

He drew on his observations of finches in the Galapagos, whose beaks differed from island to
island according to the food available, and on the practice of animal breeders, who select for
traits deliberately.

The book provoked immediate controversy, but by the 1870s most naturalists accepted that
evolution had occurred, even where they doubted that natural selection was its main cause.
""" * 3


@pytest.fixture(scope="module")
def small_supplement(tmp_path_factory, writer):
    out = tmp_path_factory.mktemp("supp") / "supplement.jsonl"
    manifest = write_supplement(MODEL, [DARWIN], out, domain_description="natural history",
                                options=SupplementOptions(passage_chars=800, batch_size=4,
                                                          max_new_tokens=512),
                                corpus_sha256="0" * 64, writer=writer)
    return out, manifest


def test_qwen3_writes_parseable_pairs_from_a_real_passage(small_supplement):
    import json
    out, manifest = small_supplement
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert manifest["n_passages"] >= 2 and manifest["n_pairs"] == len(rows) >= 2
    assert all(len(r["response"]) >= 40 and r["prompt"].strip() for r in rows)
    assert "about a text on natural history" in manifest["template"]
```

- [ ] **Step 6: Run the GPU tests outside the sandbox**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 HF_HUB_OFFLINE=1 pytest tests/test_selfgen_gpu.py -m gpu -q`
Expected: PASS (5 tests).

- [ ] **Step 7: Commit**

```bash
pytest -q
git add lfa/selfgen tests/test_selfgen_supplement.py tests/test_selfgen_gpu.py
git commit -m "selfgen: the supplement writer, the template, the forgiving parser

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: The supplement step in `Workspace.train`, and the CLI

**Files:**
- Modify: `lfa/workspace.py` (`train`, `_run_training`, new `_supplement_for`, `prepare_supplement`)
- Modify: `lfa/cli.py` (`_train`, train parser, new `prepare-supplement` subcommand, `USER_FACING_ERRORS`)
- Test: `tests/test_workspace.py`, `tests/test_cli.py` (`SUBCOMMANDS`), `tests/test_selfgen_gpu.py`

**Interfaces:**
- Consumes: `split_documents`, `sha256_text`, `write_supplement`, `checkpoint_sha256`, `template_sha256`, `load_corpus(..., supplement=, supplement_fraction=)`.
- Produces:
  - `Workspace.train(..., supplement: bool | str | Path = True, domain_description: str | None = None)`.
  - `Workspace.prepare_supplement(corpus, *, recipe=None, domain_description=None, device=DEFAULT_DEVICE, force=False) -> Path` — writes (or reuses) `<ws>/supplements/<corpus sha[:12]>/supplement.jsonl` from the workspace's **current model**.
  - `Workspace._supplement_for(corpus_path, recipe, *, domain_description, device) -> tuple[Path, dict]` — reuse-or-write.
  - History entry key `"supplement": {"path", "manifest", "n_pairs_available", "n_pairs_used", "target_fraction", "achieved_fraction", "under_target", "writer_sha256"} | None`.
  - CLI: `lfa train --no-supplement | --supplement FILE`, `--domain-description TEXT`; `lfa prepare-supplement --workspace WS --corpus DIR [--domain-description] [--force]`.

- [ ] **Step 1: Write the failing fast tests**

Append to `tests/test_workspace.py` (these use the tiny model, so `write_supplement` is monkeypatched; what is tested is the workspace's decisions):

```python
def _fake_supplement_writer(calls):
    def write(model_id, documents, out_path, *, domain_description, options, corpus_sha256,
              generate=None, writer=None, device="cuda:0"):
        calls.append(dict(model_id=model_id, n_docs=len(documents), domain=domain_description,
                          corpus_sha256=corpus_sha256))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text('{"prompt": "q?", "response": "%s", "source_index": 0}\n'
                                  % ("an answer " * 8) * 6)
        manifest = {"writer_sha256": "w" * 64, "corpus_sha256": corpus_sha256,
                    "template_sha256": "t" * 64, "n_pairs": 6, "lfa_version": "0.2.0"}
        Path(str(out_path) + ".manifest.json").write_text(json.dumps(manifest))
        return manifest
    return write


@pytest.fixture
def supplement_writer(monkeypatch):
    import lfa.workspace as ws_module
    calls = []
    monkeypatch.setattr(ws_module, "write_supplement", _fake_supplement_writer(calls))
    monkeypatch.setattr(ws_module, "checkpoint_sha256", lambda m: "w" * 64)
    monkeypatch.setattr(ws_module, "template_sha256", lambda: "t" * 64)
    return calls


def test_train_writes_the_supplement_from_the_training_side_and_records_it(tmp_path, registry,
                                                                            base_dir, corpus_a,
                                                                            supplement_writer):
    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact="tiny")
    recipe = tiny_recipe(base_dir, val_fraction=0.25, supplement_fraction=0.2)

    entry = ws.train(corpus_a, recipe=recipe, device="cpu")

    assert len(supplement_writer) == 1
    assert supplement_writer[0]["n_docs"] == 6                      # 8 docs, 2 held out
    assert supplement_writer[0]["model_id"] == str(base_dir)        # the stage's entry model
    assert supplement_writer[0]["domain"] == "domain a"             # from the directory name
    supp = entry["supplement"]
    assert supp["n_pairs_available"] == 6 and 0 < supp["n_pairs_used"] <= 6
    assert supp["target_fraction"] == 0.2 and supp["writer_sha256"] == "w" * 64
    assert Path(supp["path"]).is_relative_to(tmp_path / "ws" / "supplements")
    assert entry["n_val_docs"] == 2                                 # held-out count untouched


def test_a_matching_supplement_is_reused_not_rewritten(tmp_path, registry, base_dir, corpus_a,
                                                        supplement_writer):
    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact="tiny")
    recipe = tiny_recipe(base_dir, supplement_fraction=0.2)
    ws.train(corpus_a, recipe=recipe, device="cpu")
    ws.train(corpus_a, recipe=recipe, device="cpu")                # a repeat of the stage
    assert len(supplement_writer) == 1


def test_an_edited_corpus_regenerates_the_supplement(tmp_path, registry, base_dir,
                                                     supplement_writer):
    from conftest import make_corpus
    corpus = make_corpus(tmp_path / "domain_x", "geology")
    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact="tiny")
    recipe = tiny_recipe(base_dir, supplement_fraction=0.2)
    ws.train(corpus, recipe=recipe, device="cpu")
    (corpus / "doc_0.txt").write_text("a different document about geology " * 6)
    ws.train(corpus, recipe=recipe, device="cpu")
    assert len(supplement_writer) == 2
    assert supplement_writer[0]["corpus_sha256"] != supplement_writer[1]["corpus_sha256"]


def test_supplement_false_trains_at_zero_and_warns(tmp_path, registry, base_dir, corpus_a,
                                                    supplement_writer, caplog):
    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact="tiny")
    recipe = tiny_recipe(base_dir, supplement_fraction=0.2)
    with caplog.at_level("WARNING", logger="lfa.workspace"):
        entry = ws.train(corpus_a, recipe=recipe, device="cpu", supplement=False)
    assert entry["supplement"] is None and not supplement_writer
    assert any("supplement_fraction 0.2" in r.getMessage() and "mixes none" in r.getMessage()
               for r in caplog.records)


def test_a_supplement_path_is_used_as_given(tmp_path, registry, base_dir, corpus_a,
                                             supplement_writer):
    given = tmp_path / "mine.jsonl"
    given.write_text('{"prompt": "q?", "response": "%s"}\n' % ("word " * 10) * 3)
    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact="tiny")
    entry = ws.train(corpus_a, recipe=tiny_recipe(base_dir, supplement_fraction=0.2),
                     device="cpu", supplement=given)
    assert not supplement_writer and entry["supplement"]["path"] == str(given)
    assert entry["supplement"]["n_pairs_available"] == 3


def test_prepare_supplement_writes_once_and_names_the_file(tmp_path, registry, base_dir,
                                                            corpus_a, supplement_writer):
    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact="tiny")
    first = ws.prepare_supplement(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")
    second = ws.prepare_supplement(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")
    assert first == second and first.name == "supplement.jsonl" and len(supplement_writer) == 1
    ws.prepare_supplement(corpus_a, recipe=tiny_recipe(base_dir), device="cpu", force=True)
    assert len(supplement_writer) == 2
```

Update `SUBCOMMANDS` in `tests/test_cli.py` to include `"prepare-supplement"`, and add:

```python
def test_train_no_supplement_and_supplement_file_are_exclusive(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["train", "--corpus", "c", "--no-supplement", "--supplement", "s.jsonl"])
    assert exit_info.value.code == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_workspace.py -k supplement tests/test_cli.py -k "supplement or subcommand" -q`
Expected: FAIL.

- [ ] **Step 3: Implement the workspace side**

Imports in `lfa/workspace.py`: `from .corpus import load_corpus, split_documents`, `from .selfgen.generate import checkpoint_sha256, sha256_text`, `from .selfgen.supplement import SupplementOptions, template_sha256, write_supplement`.

Add helpers:

```python
def _domain_description_for(corpus_path: Path) -> str:
    """The corpus directory's name with `_`/`-` as spaces; what the template says the text is on."""
    name = corpus_path.stem if corpus_path.is_file() else corpus_path.name
    return re.sub(r"[_\-]+", " ", name).strip() or "the domain"
```

and the method:

```python
    def _supplement_for(self, corpus_path: Path, recipe: Recipe, *, domain_description: str | None,
                        device: str | dict, force: bool = False) -> tuple[Path, dict]:
        """The supplement for ``corpus_path``: reused when its manifest matches, else written.

        A match is the training side's corpus hash, the writer checkpoint's hash and the
        template's hash. The writer is the workspace's CURRENT model -- the stage's entry model
        -- so in a chain the fused model writes the next domain's pairs (the C15 protocol).
        """
        train_docs, _ = split_documents(corpus_path, recipe.val_fraction, recipe.seed)
        corpus_hash = sha256_text(train_docs)
        writer_id = str(self.state["current_model"])
        writer_hash = checkpoint_sha256(writer_id)
        directory = self.path / "supplements" / corpus_hash[:12]
        out = directory / "supplement.jsonl"
        manifest_path = Path(str(out) + ".manifest.json")
        if out.is_file() and manifest_path.is_file() and not force:
            manifest = json.loads(manifest_path.read_text())
            if (manifest.get("corpus_sha256") == corpus_hash
                    and manifest.get("writer_sha256") == writer_hash
                    and manifest.get("template_sha256") == template_sha256()):
                logger.info("Supplement reused: %s", out)
                return out, manifest
        description = domain_description or _domain_description_for(corpus_path)
        logger.info("Writing the supplement for %s with %s (%d training documents)", corpus_path,
                    writer_id, len(train_docs))
        manifest = write_supplement(writer_id, train_docs, out, domain_description=description,
                                    options=SupplementOptions(), corpus_sha256=corpus_hash,
                                    device=_primary_device(resolve_device(device)))
        return out, manifest

    def prepare_supplement(self, corpus, *, recipe: Recipe | str | Path | None = None,
                           domain_description: str | None = None,
                           device: str | dict = DEFAULT_DEVICE, force: bool = False) -> Path:
        """Write (or reuse) the supplement ``train`` would write for ``corpus``, and return its path."""
        corpus_path = Path(corpus).expanduser().resolve()
        resolved = self._resolve_recipe(recipe)
        path, _ = self._supplement_for(corpus_path, resolved, domain_description=domain_description,
                                       device=device, force=force)
        return path
```

In `train`: add parameters `supplement: bool | str | Path = True, domain_description: str | None = None`. After the recipe is resolved and before `_run_training`:

```python
        supplement_path, supplement_manifest = None, None
        if resolved.supplement_fraction > 0.0 and supplement is not False:
            if supplement is True:
                supplement_path, supplement_manifest = self._supplement_for(
                    corpus_path, resolved, domain_description=domain_description, device=device)
            else:
                supplement_path = Path(supplement).expanduser().resolve()
                manifest_file = Path(str(supplement_path) + ".manifest.json")
                supplement_manifest = (json.loads(manifest_file.read_text())
                                       if manifest_file.is_file() else {})
        elif resolved.supplement_fraction > 0.0:
            logger.warning("Training on the raw corpus alone: this recipe's lambda was "
                           "calibrated at supplement_fraction %g and this run mixes none.",
                           resolved.supplement_fraction)
```

Pass `supplement_path` and `resolved.supplement_fraction` into `_run_training` (new keyword params `supplement=None, supplement_fraction=0.0`), which forwards them to `load_corpus(...)`. `_run_training` returns `dataset.supplement_report` inside `counts` as `counts["supplement_report"]`. Build the history entry:

```python
            "supplement": None if supplement_path is None else {
                "path": str(supplement_path),
                "manifest": str(supplement_path) + ".manifest.json",
                "n_pairs_available": report["n_available"],
                "n_pairs_used": report["n_used"],
                "target_fraction": report["target_fraction"],
                "achieved_fraction": report["achieved_fraction"],
                "under_target": report["under_target"],
                "writer_sha256": (supplement_manifest or {}).get("writer_sha256"),
            },
```

where `report = corpus_counts.pop("supplement_report")`. Document both new parameters in the docstring.

- [ ] **Step 4: Implement the CLI side**

In `lfa/cli.py`, `_train` passes `supplement=(False if args.no_supplement else (args.supplement or True)), domain_description=args.domain_description`. Parser:

```python
    supplement = train.add_mutually_exclusive_group()
    supplement.add_argument("--no-supplement", dest="no_supplement", action="store_true",
                            help="train on the raw corpus alone (the recipe's lambda was "
                                 "calibrated with the written supplement mixed in; this warns)")
    supplement.add_argument("--supplement", metavar="JSONL",
                            help="a prompt/response JSONL to mix in instead of writing one")
    train.add_argument("--domain-description", dest="domain_description", metavar="TEXT",
                       help="what the template says the text is on (default: the corpus "
                            "directory's name)")
```

Add the subcommand:

```python
def _prepare_supplement(args) -> int:
    print(_open(args).prepare_supplement(args.corpus, recipe=args.recipe,
                                         domain_description=args.domain_description,
                                         device=args.device, force=args.force))
    return 0

    # ------------------------------------------------------------------- prepare-supplement
    supp = subcommands.add_parser(
        "prepare-supplement",
        help="have the workspace's current model write the question-and-answer supplement "
             "for a corpus, to inspect before training (train writes it itself otherwise)")
    _add_workspace(supp)
    supp.add_argument("--corpus", required=True, metavar="DIR")
    supp.add_argument("--recipe", metavar="NAME|PATH")
    supp.add_argument("--domain-description", dest="domain_description", metavar="TEXT")
    supp.add_argument("--force", action="store_true", help="rewrite an existing supplement")
    _add_device(supp)
    supp.set_defaults(handler=_prepare_supplement)
```

Add `NoPairsWritten` and `DegenerateCorpus` to `USER_FACING_ERRORS` (import from `lfa.selfgen.supplement` and `lfa.selfgen.artifact_corpus`).

- [ ] **Step 5: Run the fast tests**

Run: `pytest tests/test_workspace.py tests/test_cli.py tests/test_evaluate.py -q`
Expected: PASS. (`evaluate` rebuilds the held-out split with `load_corpus` without a supplement, which is unchanged.)

- [ ] **Step 6: Add the GPU test**

Append to `tests/test_selfgen_gpu.py`:

```python
from lfa import Recipe


def _small_recipe():
    return Recipe(name="qwen3-small", model_id=MODEL, artifact="self-generated",
                  lora_rank=4, lora_alpha=8, lambda_qkv=1000.0, lambda_mlp=1000.0, mu=0.05,
                  n_anchor_samples=4, epochs=1, checkpoint_mode="none", learning_rate=1e-4,
                  warmup_steps=1, batch_size=1, gradient_accumulation_steps=2,
                  sequence_length=128, seed=42, val_fraction=0.25, supplement_fraction=0.13,
                  calibrated_rank=4, calibrated_self_generated=True)


@pytest.fixture(scope="module")
def darwin_corpus(tmp_path_factory):
    directory = tmp_path_factory.mktemp("domain") / "natural_history"
    directory.mkdir()
    for i in range(4):
        (directory / f"doc_{i}.txt").write_text(DARWIN.replace("1859", str(1859 + i)))
    return directory


def test_a_stage_trains_with_the_self_written_supplement_mixed_in(tmp_path, darwin_corpus):
    ws = Workspace.init(tmp_path / "ws", MODEL, artifact="self-generated", selfgen=SMALL)

    entry = ws.train(darwin_corpus, recipe=_small_recipe(), device=DEVICE)

    supp = entry["supplement"]
    assert supp["n_pairs_available"] >= 2 and supp["n_pairs_used"] >= 1
    assert 0.0 < supp["achieved_fraction"] < 0.5
    assert len(supp["writer_sha256"]) == 64
    assert entry["n_val_docs"] == 1 and entry["n_train_docs"] == 3 + supp["n_pairs_used"]
    assert (tmp_path / "ws" / "supplements").is_dir()
```

- [ ] **Step 7: Run the GPU tests outside the sandbox**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 HF_HUB_OFFLINE=1 pytest tests/test_selfgen_gpu.py -m gpu -q`
Expected: PASS (6 tests). Peak memory under 5 GB; if the stage OOMs, lower `sequence_length` to 64 in `_small_recipe`.

- [ ] **Step 8: Commit**

```bash
pytest -q
git add lfa/workspace.py lfa/cli.py lfa/selfgen/supplement.py tests/test_workspace.py tests/test_cli.py tests/test_selfgen_gpu.py
git commit -m "train writes and mixes the supplement at the recipe's fraction; prepare-supplement

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: `regenerate_artifact` and the chain's `artifact:` field

**Files:**
- Modify: `lfa/workspace.py` (`extend` records the route; new `regenerate_artifact`; `chain`)
- Modify: `lfa/cli.py` (`regenerate-artifact` subcommand; `chain` unchanged)
- Test: `tests/test_workspace.py`, `tests/test_cli.py` (`SUBCOMMANDS`), `tests/test_selfgen_gpu.py`

**Interfaces:**
- Produces: `Workspace.regenerate_artifact(*, selfgen: SelfGenOptions | None = None, device=DEFAULT_DEVICE) -> Path`; state key `artifact_route: "extend" | "regenerate"` set by both operations and copied into the next stage's history entry as `"artifact_route"`; chain spec top-level `artifact: extend | regenerate` (`CHAIN_ARTIFACT_ROUTES = ("extend", "regenerate")`); `Workspace.chain(..., selfgen: SelfGenOptions | None = None)`.

- [ ] **Step 1: Write the failing fast tests**

Append to `tests/test_workspace.py`:

```python
def _fake_regenerate_builder(monkeypatch, tiny_artifact):
    import lfa.workspace as ws_module
    _, fixture = tiny_artifact
    calls = []

    def fake_build(model_id, out_path, options, *, corpus_path=None, **kwargs):
        calls.append(dict(model_id=model_id, n_chat=options.n_chat))
        Path(out_path).write_bytes(fixture.read_bytes())
        Path(corpus_path).write_text('{"text": "x"}\n')
        Path(str(corpus_path) + ".manifest.json").write_text(
            '{"corpus_sha256": "%s", "writer_sha256": "%s"}' % ("9" * 64, "8" * 64))
        return Path(out_path)
    monkeypatch.setattr(ws_module, "build_artifact_self_generated", fake_build)
    return calls


def test_regenerate_before_a_trained_stage_is_a_stage_order_error(tmp_path, registry, base_dir):
    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact="tiny")
    with pytest.raises(StageOrderError, match="Nothing to"):
        ws.regenerate_artifact(device="cpu")


def test_regenerate_writes_v2_from_the_fused_model_with_no_chat_share(tmp_path, registry,
                                                                       base_dir, corpus_a,
                                                                       tiny_artifact, monkeypatch):
    calls = _fake_regenerate_builder(monkeypatch, tiny_artifact)
    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact="tiny")
    ws.train(corpus_a, recipe=tiny_recipe(base_dir, supplement_fraction=0.0), device="cpu")

    out = ws.regenerate_artifact(device="cpu")

    assert out == tmp_path / "ws" / "artifacts" / "v2.pt" and out.is_file()
    assert (tmp_path / "ws" / "artifacts" / "v2.corpus.jsonl").is_file()
    assert calls[0]["model_id"] == str(tmp_path / "ws" / "models" / "stage1_fused")
    assert calls[0]["n_chat"] == 0                                  # the C15 frame
    assert ws.state["artifact_version"] == 2 and ws.state["artifact_route"] == "regenerate"
    assert ws.state["pending_extend"] is False


def test_the_next_stage_records_the_route_it_anchored_under(tmp_path, registry, base_dir,
                                                             corpus_a, corpus_b, tiny_artifact,
                                                             monkeypatch):
    _fake_regenerate_builder(monkeypatch, tiny_artifact)
    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact="tiny")
    recipe = tiny_recipe(base_dir, supplement_fraction=0.0)
    ws.train(corpus_a, recipe=recipe, device="cpu")
    ws.regenerate_artifact(device="cpu")
    entry = ws.train(corpus_b, recipe=recipe, device="cpu")
    assert entry["artifact_route"] == "regenerate" and entry["artifact_version"] == 2


def test_a_chain_spec_accepts_a_top_level_artifact_route_and_refuses_a_per_domain_one(
        tmp_path, registry, base_dir, corpus_a, corpus_b, tiny_artifact, monkeypatch):
    calls = _fake_regenerate_builder(monkeypatch, tiny_artifact)
    ws = _chain_workspace(tmp_path, base_dir)
    spec = tmp_path / "domains.yaml"

    spec.write_text(yaml.safe_dump({"artifact": "regenerate", "domains": [
        {"name": "a", "corpus": str(corpus_a)}, {"name": "b", "corpus": str(corpus_b)}]}))
    entries = ws.chain(spec, device="cpu", need=NEED, k_domain=K_DOMAIN)
    assert len(entries) == 2 and len(calls) == 1 and entries[1]["artifact_route"] == "regenerate"

    spec.write_text(yaml.safe_dump({"artifact": "sideways", "domains": [
        {"name": "a", "corpus": str(corpus_a)}]}))
    with pytest.raises(ValueError, match="extend, regenerate"):
        ws.chain(spec, device="cpu", need=NEED, k_domain=K_DOMAIN)

    spec.write_text(yaml.safe_dump({"domains": [
        {"name": "a", "corpus": str(corpus_a), "artifact": "regenerate"}]}))
    with pytest.raises(ValueError, match="top level"):
        ws.chain(spec, device="cpu", need=NEED, k_domain=K_DOMAIN)


def test_regenerate_warns_off_calibration_and_continues(tmp_path, registry, base_dir, corpus_a,
                                                        corpus_b, tiny_artifact, monkeypatch,
                                                        caplog):
    _fake_regenerate_builder(monkeypatch, tiny_artifact)
    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact="tiny")
    recipe = tiny_recipe(base_dir, supplement_fraction=0.0, calibrated_self_generated=False)
    ws.train(corpus_a, recipe=recipe, device="cpu")
    ws.regenerate_artifact(device="cpu")
    with caplog.at_level("WARNING", logger="lfa.workspace"):
        entry = ws.train(corpus_b, recipe=recipe, device="cpu")
    assert entry["stage"] == 2
    assert any("C15" in r.getMessage() and "rank 4" in r.getMessage() for r in caplog.records)
```

(`_chain_workspace` is the helper the existing chain tests use; if its recipe carries `supplement_fraction` > 0, pass a recipe with `supplement_fraction=0.0` or patch the writer as in Task 10. The regenerated fixture artifact has no `provenance` in its meta because the fake copies the tiny fixture; make the fake **also** rewrite the meta: load with `torch.load`, set `params["__meta__"]["provenance"] = "self-generated"` and `["model_id"] = model_id`, save. Do that in `_fake_regenerate_builder`.)

Add `"regenerate-artifact"` to `SUBCOMMANDS` in `tests/test_cli.py`.

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_workspace.py -k "regenerate or route" tests/test_cli.py -k subcommand -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `lfa/workspace.py`:

```python
CHAIN_ARTIFACT_ROUTES = ("extend", "regenerate")
```

In `extend`, add `artifact_route="extend"` to the `self.state.update(...)`. Add:

```python
    def regenerate_artifact(self, *, selfgen: SelfGenOptions | None = None,
                            device: str | dict = DEFAULT_DEVICE) -> Path:
        """Fold the last stage into the model, then fit p(h) FROM SCRATCH on the fused model's
        own text: the alternative to :meth:`extend` between two domains.

        The C15 frame: 2,500 documents from the document-boundary token, no chat-format share,
        600k samples per site, K=32, no base component and no merge. The record (rank 4, one
        seed, three domains) found a chain anchored this way as good as one on the real seed
        corpus on every judge, perplexity and skill benchmark, with the generated text drifting
        toward the last domain (two-thirds after the first, six-sevenths after the second).

        Raises:
            StageOrderError: nothing has been trained since the last extension or regeneration.
        """
        if not self.state["pending_extend"]:
            raise StageOrderError(
                "Nothing to regenerate from: no stage has been trained since the last extension "
                "or regeneration. Train a domain first (`lfa train --corpus ...`).")
        entry = self.history[-1]
        stage = self.state["stage"]
        adapter_dir = Path(self.state["last_stage_adapter"])
        if (adapter_dir / "adapter_config.json").is_file():
            fused = fuse(adapter_dir, str(self.state["last_stage_base"]),
                         self.path / "models" / f"stage{stage}_fused", dtype=_stage_dtype(entry))
        else:
            fused = adapter_dir
            logger.info("Stage %d trained full weights; its checkpoint is the fused model", stage)

        options = dataclasses.replace(selfgen or SelfGenOptions(), n_chat=0,
                                      device=_primary_device(resolve_device(device)))
        version = self.state["artifact_version"] + 1
        out = self.path / "artifacts" / f"v{version}.pt"
        build_artifact_self_generated(str(fused), out, options,
                                      corpus_path=self.path / "artifacts" / f"v{version}.corpus.jsonl")

        self.state.update(current_model=str(fused), current_artifact=str(out),
                          artifact_version=version, pending_extend=False,
                          artifact_route="regenerate")
        self._save_state()
        logger.info("Stage %d folded in: model %s, artifact v%d regenerated from its own text",
                    stage, fused, version)
        return out
```

In `train`'s history entry add `"artifact_route": self.state.get("artifact_route")` (`None` for stage 1). In the off-calibration warning loop, when `self.state.get("artifact_route") == "regenerate"` and the recipe emits the self-generated note, append to it: `" A regenerated artifact is the C15 route (rank 4, one seed); the stage multiplier is a starting point there, not a calibrated constant."` — implement by post-processing the notes list in `train`:

```python
        notes = resolved.warnings(config.lora_rank, self._artifact_id(), self._artifact_meta())
        if self.state.get("artifact_route") == "regenerate":
            notes = [n + " A regenerated artifact is the C15 route (rank 4, one seed): the "
                     "stage multiplier is a starting point there, not a calibrated constant."
                     if "self-generated" in n or "own text" in n else n for n in notes]
        for note in notes:
            logger.warning(note)
```

In `chain`: read `route = spec.get("artifact", "extend")`; refuse with `ValueError(f"{spec_path}: 'artifact' must be one of extend, regenerate; got {route!r}.")` if not in `CHAIN_ARTIFACT_ROUTES`; add `"artifact"` handling to `_refuse_unknown_domain_keys` so a per-domain `artifact` key raises `ValueError(f"{spec_path}: domain {position} carries 'artifact', which is a chain-wide choice: put `artifact: {value}` at the spec's top level ...")`; between stages call `self.regenerate_artifact(selfgen=selfgen, device=device)` when `route == "regenerate"` else `self.extend(...)`. Add `selfgen` to `chain`'s signature and docstring; document the YAML field in the docstring's example.

In `lfa/cli.py`:

```python
def _regenerate_artifact(args) -> int:
    print(_open(args).regenerate_artifact(device=args.device))
    return 0

    # --------------------------------------------------------------------- regenerate-artifact
    regen = subcommands.add_parser(
        "regenerate-artifact",
        help="fold the trained stage into the model and fit a fresh p(h) on that model's own "
             "text (the alternative to extend; the C15 route)")
    _add_workspace(regen)
    _add_device(regen)
    regen.set_defaults(handler=_regenerate_artifact)
```

- [ ] **Step 4: Run the fast tests**

Run: `pytest tests/test_workspace.py tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Add the GPU test**

Append to `tests/test_selfgen_gpu.py`:

```python
import yaml


def test_a_two_stage_chain_under_regenerate_refits_v2_from_the_fused_model(tmp_path, darwin_corpus):
    second = tmp_path / "cookery"
    second.mkdir()
    for i in range(4):
        (second / f"recipe_{i}.txt").write_text(
            ("Take a pound of flour and rub in the butter. Add the eggs one at a time, beating "
             "well, and bake in a moderate oven for forty minutes.\n\n") * 12)
    recipe_path = _small_recipe().save(tmp_path / "small.yaml")
    ws = Workspace.init(tmp_path / "ws", MODEL, artifact="self-generated", selfgen=SMALL,
                        recipe=str(recipe_path))                # a chain uses the workspace's recipe
    spec = tmp_path / "domains.yaml"
    spec.write_text(yaml.safe_dump({"artifact": "regenerate", "domains": [
        {"name": "darwin", "corpus": str(darwin_corpus)}, {"name": "cookery", "corpus": str(second)}]}))

    entries = ws.chain(spec, device=DEVICE, selfgen=SMALL)

    assert [e["stage"] for e in entries] == [1, 2]
    assert entries[1]["artifact_route"] == "regenerate" and entries[1]["artifact_version"] == 2
    meta = ws._artifact_meta()
    assert meta["provenance"] == "self-generated"
    assert meta["model_id"] == str(tmp_path / "ws" / "models" / "stage1_fused")
    assert (tmp_path / "ws" / "artifacts" / "v2.corpus.jsonl.manifest.json").is_file()
```

- [ ] **Step 6: Run the GPU tests outside the sandbox**

Run: `LFA_SKIP_TOOLCHAIN_CHECK=1 HF_HUB_OFFLINE=1 pytest tests/test_selfgen_gpu.py -m gpu -q`
Expected: PASS (7 tests; the chain fuses once and builds one small artifact from the fused checkpoint, a few minutes).

- [ ] **Step 7: Commit**

```bash
pytest -q
git add lfa/workspace.py lfa/cli.py tests/test_workspace.py tests/test_cli.py tests/test_selfgen_gpu.py
git commit -m "regenerate-artifact and the chain's artifact: route (the C15 protocol)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: Documentation, README, FAQ, changelog, version 0.2.0

**Files:**
- Modify: `lfa/__init__.py` (`__version__ = "0.2.0"`), `pyproject.toml` (`version = "0.2.0"`)
- Modify: `README.md`, `docs/quickstart.md`, `docs/rebuilding-the-artifact.md`, `docs/adding-a-model.md`, `docs/recipes.md`, `docs/concepts.md`, `docs/multi-domain-chains.md`, `docs/faq.md`, `RELEASING.md`
- Test: `tests/test_cli.py` (version test reads `__version__`, no change), `tests/test_release.py` (check it for a pinned version string and update if so)

**Interfaces:** none new; every sentence below is copied into the named section.

- [ ] **Step 1: Bump the version**

`lfa/__init__.py`: `__version__ = "0.2.0"`. `pyproject.toml`: `version = "0.2.0"`. Run `grep -rn "0\.1\.1" tests/ RELEASING.md pyproject.toml lfa/` and update any pinned occurrence that names the *current* version (leave changelog history).

- [ ] **Step 2: README**

In the building-blocks table add a row after **Corpus**:

```
| **Self-generation** | The model writes its own inputs: the seed corpus p(h) is estimated on (`init --artifact self-generated`, no download), the question-and-answer supplement `train` mixes into the domain at the recipe's token fraction, and, in a chain, a fresh p(h) from each stage's model (`artifact: regenerate`). Measured on one model and one seed (the LFA record's C12, C14, C15); the supplement's job is reachability, not skill retention. | `lfa.selfgen` |
```

After "Assemble them: one domain", add one paragraph:

> **No artifact download?** `lfa init runs/my_domain --model Qwen/Qwen3-0.6B --artifact self-generated` has the model write 2,750 documents from its own document boundary and fits p(h) on them (about two hours on an 8 GB card, half that on a 3090). On Qwen3-0.6B that artifact tied the published one at every λ tried, one seed; on any other model it is the way to a first artifact, and λ is then calibrated against it. `train` also writes the domain's question-and-answer supplement with the model before training and mixes it in at 0.13 of training tokens, the frame the shipped λ was tuned at; `--no-supplement` trains on the raw corpus alone and says so.

- [ ] **Step 3: quickstart.md**

In "Get a p(h) artifact", add a subsection "Or let the model write one" with the `init --artifact self-generated` command, the cost line (RTX 2070: ~2 h generation + ~40 min fit; RTX 3090: about half), and the scope sentence. In "The four commands", after the `train` description add: "`train` first writes the supplement: the entry model reads each training-side passage and writes six question-and-answer pairs from a fixed template, cached under `supplements/` and reused while the corpus, the writer and the template are unchanged. `--no-supplement` opts out; `lfa prepare-supplement` writes it ahead of time to inspect." Add `--no-supplement`, `--supplement`, `--domain-description` to "What it prints"'s flag list if one exists.

- [ ] **Step 4: rebuilding-the-artifact.md**

Add a section "## The self-generated route" before "## What a locally-built artifact does not share with the shipped one", stating: the recorded frame (2,500 raw × 2,048 tokens from `<|endoftext|>` at T=1.0/top-p=1.0 unfiltered, seed 42; 250 from the bare user header, seed 43; 600k samples/site, K=32, variance 0.95); the command; the manifest fields; the C12 result ("tied the gmm1543k artifact at every λ tried; Qwen3-0.6B, one seed, one domain") and the note that `top_k` is lifted explicitly because Qwen3's generation config would otherwise sample top-20.

- [ ] **Step 5: adding-a-model.md**

In "## 2. The artifact", add first: "The shortest route is `lfa build-artifact --model <id> --self-generated --out artifacts/<name>.pt`: the model writes its own seed corpus and no dataset is downloaded. It needs the model to expose a document-boundary token (`generation_config.bos_token_id`, BOS or EOS) and, for the chat-format share, a chat template; without one the share is skipped and the log says so. The recipe will warn that λ is uncalibrated against it, which is true: go to §3." Keep the seed-corpus route as the alternative.

- [ ] **Step 6: recipes.md**

In "### Data" document `supplement_fraction` (0.13; "the shipped λ was tuned with a question-and-answer supplement at 0.13 of training tokens; every companion run before 0.2.0 trained at 0, off that frame"). In "### Calibration record" document `calibrated_self_generated` (true; C12 scope). In "## The couplings, and what the warnings mean" add the two new warning texts and when each fires.

- [ ] **Step 7: concepts.md**

Add "## What the supplement does, and does not do" after "## Reading a run: two axes, never one": reachability (C12: correctness and domain accuracy move, completeness barely); not skill protection (C14: reasoning-style and instruction-style supplements protected neither GSM8K nor IFEval in any method); LFA's own ~10-point GSM8K loss is not repaired by self-generated inputs; the thing that holds skills at base in the record is self-generated rehearsal, a replay method this package does not implement. Scope line.

- [ ] **Step 8: multi-domain-chains.md**

Add "## The regenerate route" after "## What `extend` does": what `regenerate_artifact` does (fuse, then a fresh p(h) from the fused model's own text, no merge), the YAML field, the C15 finding with its drift numbers (domain-term rates 0.5 / 68.2 / 2.5 % and 2.7 / 9.6 / 86.1 % across the two later domains; the chain matched the real-seed chain on every judge, perplexity and skill benchmark; rank 4, one seed), and the λ note (C15 ported λ at 2x from stage 2; the recipe's 3x is a starting point on either route; the stage-2 warning under `regenerate`).

- [ ] **Step 9: faq.md**

In "## How much GPU memory does a run need?" add an 8 GB paragraph: "On an RTX 2070 (8 GB, 2026-09-26) a Qwen3-0.6B rank-32 step at 512 tokens peaks at 2.45 / 3.52 / 4.59 GiB for micro-batch 1 / 2 / 3 and overflows at 6 on the fp32 logits (6 × 512 × 151,936 floats). Set `batch_size: 3` and `gradient_accumulation_steps: 2` to keep the recipe's 6 × 512 geometry; about 6 s per optimizer step. bfloat16 is sound on Turing: perplexity 9.67 against fp32's 9.70 on one sentence, greedy outputs equal, at about half fp16 speed." Add a question "## The run warned that this machine cannot compile for the GPU" with the Task 1 text. Add "## How long does self-generation take?" with the 2070 numbers (16 documents × 512 tokens in 54 s; the full 2,750-document corpus about two hours; a supplement at 6 pairs per 4,000-character passage roughly one passage per 4 s at batch 16).

- [ ] **Step 10: RELEASING.md changelog**

Add above `### 0.1.1`:

```
### 0.2.0

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
```

- [ ] **Step 11: Run everything and commit**

```bash
pytest -q
LFA_SKIP_TOOLCHAIN_CHECK=1 HF_HUB_OFFLINE=1 pytest tests/test_selfgen_gpu.py tests/test_gpu_smoke.py -m gpu -q   # outside the sandbox
git add -A
git commit -m "0.2.0: self-generation documented; FAQ gains the 8 GB geometry and the toolchain note

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage.** §3.1 → Task 3; §3.2 → Task 4; §3.3 → Tasks 2, 5; §3.4 → Task 6; §3.5 → Tasks 7, 10; §3.6 → Task 11; §3.7 → Task 9; §3.8 → Task 8; §3.9 → Tasks 1, 12; §4 errors: each row has an owning task; §5 tests: GPU tier in Tasks 3, 4, 5, 7, 9, 10, 11; §6 docs → Task 12; §7 version → Task 12; §8 CLI → Tasks 5, 7, 10, 11.
- **Types.** `SelfGenOptions` (Task 4) is what Tasks 5, 7, 11 consume; `Selection`/`select_supplement_prefix`/`split_documents`/`load_supplement` (Task 8) are what Task 10 consumes; `write_supplement(model_id, documents, out_path, *, domain_description, options, corpus_sha256, generate, writer, device)` is defined in Task 9 and consumed in Task 10.
- **Review Focus** items 1–5 each have a named test in Tasks 8, 3, 8, 10, 11.
