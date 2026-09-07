#!/usr/bin/env python3
"""The equivalence run: this package trains what the research code trains.

The companion's bundled ``qwen3-0.6b`` recipe is run end to end on one domain, and the checkpoint
it produces is compared against the checkpoint an **the research code run of the identical configuration**
produced -- same corpus, same int8 artifact, same seed, same fifteen epochs of cosine, same loader
frame. What is asserted is that the two implementations land in the same place, not that either
lands on a number recorded in a paper: the reference is re-measured beside the companion, on the
same instrument, on the same machine.

The instrument is the research code's ``scripts/eval_domain_perplexity.py``, shelled out to *in
the research code's own virtualenv* and pointed at both checkpoints in turn:

* **domain** = ``direct_perplexity.overall.mean_perplexity`` over the held-out chat-formatted Q&A
  set (never comparable with held-out raw-prose perplexity, which is a different quantity), and
* **seed** = WikiText-2 over the full test split, read as drift against the base model.

The companion's own metrics are reported beside them, un-banded: :func:`lfa.evaluate.
domain_perplexity` on the stage's held-out documents and :func:`lfa.evaluate.wikitext2_perplexity`
with ``n_windows=0``. The WikiText-2 pair should agree closely (it is the same computation); the
two domain numbers should not, because they measure different text.

Tolerances live in ``expected.json`` and are the only numbers in this directory that are written
down; the reference's measured values are written *out*, to ``reference.json``, by this script.

the research code is read-only here. Every byte this script writes goes under ``--out``: the workspace,
the training run, and both scoring directories (``--output`` is passed explicitly so that nothing
lands in the checkout the corpus, the artifact and the reference run came from).

Idempotent, because the training run is over an hour of GPU time: a workspace that already carries
a trained stage is reused -- but only after its recorded frame is checked against the one being
asked for -- and a scoring directory that already holds a result for the same checkpoint and Q&A
file is not rescored. Delete ``--out`` to start over.

Usage::

    python tests/acceptance/run_recipe.py --out tests/acceptance/_runs/2026-09-07-equiv

Run it on ONE pinned card (the default ``--cuda-visible-devices 0``); the anchoring forward is
launch-bound at this model size and a second tenant on the card silently corrupts the timing.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

#: The research checkout. The companion lives inside it, so its grandparent is the research code's root;
#: ``--the research code`` (or ``LFA_RESEARCH_ROOT``) overrides that for a checkout kept elsewhere.
DEFAULT_RESEARCH_ROOT = Path(__file__).resolve().parents[3]

MODEL_ID = "Qwen/Qwen3-0.6B"
RECIPE = "qwen3-0.6b"

#: The corpus both runs train on: raw Chalmers prose plus a generated Q&A supplement. Lambda is
#: coupled to corpus composition, so this is part of the configuration the two runs share, not a
#: detail of one of them.
CORPUS_REL = "data/domain/chalmers_qa0.15"

#: The int8 p(h) artifact both runs anchor against, and the registry id it is a copy of.
ARTIFACT_REL = "data/distributions/qwen3-0.6b-gmm1543k-int8/distribution_stats.pt"
ARTIFACT_ID = "qwen3-0.6b-gmm1543k-int8"

#: The held-out direct-QA set the domain number is measured on.
QA_REL = ("data/domain_perplexity_questions/chalmers/"
          "domain_perplexity_qa_direct_20260310_123856.jsonl")

#: The the research code run this one is compared against, trained from
#: ``scripts/_lfa_companion_reference.sh`` (15 epochs, cosine over 15, --keep-short-whole, int8
#: artifact, seed 42). ``final_model/`` is its checkpoint and ``training_history.json`` its curve.
REFERENCE_REL = "outputs/lra/qwen3-0.6b/chalmers/judge_search/gmm_r32_lam100000_e15cos_keepshort"

EXPECTED_FILE = Path(__file__).with_name("expected.json")

#: What the two runs must agree on before their outputs mean anything: the research code's ``config.json``
#: key on the left, the companion's :class:`lfa.train.TrainConfig` attribute on the right. These
#: are the *frame* -- corpus geometry, optimizer geometry, anchoring strength, loader -- not the
#: thing under comparison, which is the two implementations of the same objective.
FRAME_FIELDS = {
    "lora_rank": "lora_rank",
    "lora_alpha": "lora_alpha",
    "freeze_embed": "freeze_embed",
    "lambda_qkv": "lambda_qkv",
    "lambda_mlp": "lambda_mlp",
    "mu": "mu",
    "anchor_end_ratio": "anchor_end_ratio",
    "anchor_schedule": "anchor_schedule",
    "n_anchor_samples": "n_anchor_samples",
    "num_epochs": "num_epochs",
    "learning_rate": "learning_rate",
    "batch_size": "batch_size",
    "gradient_accumulation_steps": "gradient_accumulation_steps",
    "warmup_steps": "warmup_steps",
    "weight_decay": "weight_decay",
    "sequence_length": "sequence_length",
    "seed": "seed",
    "keep_short_whole": "keep_short_whole",
    "val_fraction": "val_fraction",
}


class MissingInput(RuntimeError):
    """A required read-only input is not on this machine (the test turns this into a skip)."""


class FrameMismatch(RuntimeError):
    """The reference run and the companion run are not the same configuration."""


class ReusedRunDiffers(RuntimeError):
    """The run already in ``--out`` was made under a different frame than the one asked for."""


# ==============================================================================================
# Inputs
# ==============================================================================================

def resolve_inputs(research: Path, corpus: str | Path | None = None,
                   artifact: str | Path | None = None,
                   qa_file: str | Path | None = None,
                   reference: str | Path | None = None) -> dict[str, Path]:
    """Locate everything this run reads, or say precisely which piece is missing.

    Raises:
        MissingInput: naming the path that is absent. The corpora, artifacts and reference run are
            gigabytes and are not distributed with the companion, so a machine without the
            research checkout should skip this run rather than fail it. The reference run's
            ``final_model/`` is included: until it is there, there is nothing to compare against.
    """
    research = Path(research).expanduser().resolve()
    reference_dir = (Path(reference).expanduser().resolve() if reference
                     else research / REFERENCE_REL)
    wanted = {
        "research": (research, "the the research code checkout"),
        "python": (research / ".venv" / "bin" / "python", "the research code's virtualenv"),
        "scorer": (research / "scripts" / "eval_domain_perplexity.py", "the research code's scorer"),
        "metrics": (research / "scripts" / "comparison_metrics.py", "the research code's extractors"),
        "corpus": (Path(corpus).expanduser().resolve() if corpus else research / CORPUS_REL,
                   "the Chalmers corpus with its Q&A supplement"),
        "artifact": (Path(artifact).expanduser().resolve() if artifact
                     else research / ARTIFACT_REL, "the p(h) artifact"),
        "qa_file": (Path(qa_file).expanduser().resolve() if qa_file else research / QA_REL,
                    "the held-out direct-QA set"),
        "reference": (reference_dir, "the the research code reference run"),
        "reference_model": (reference_dir / "final_model", "the reference run's checkpoint"),
        "reference_config": (reference_dir / "config.json", "the reference run's config"),
        "reference_history": (reference_dir / "training_history.json",
                              "the reference run's training curve"),
        # Required because the corpus check reads its chunk counts back off it -- the research code logs
        # those two numbers rather than recording them in JSON. Declared here so a reference run
        # kept without its log is named in the first seconds, not two hours in.
        "reference_log": (reference_dir / "training.log", "the reference run's log"),
    }
    resolved = {}
    for key, (path, what) in wanted.items():
        if not path.exists():
            raise MissingInput(
                f"{what} is not at {path}. This run reads the research checkout (it is never "
                "written to); point --the research code at yours, or set LFA_RESEARCH_ROOT."
            )
        resolved[key] = path
    return resolved


def _load_module(path: Path, name: str):
    """Import a module from a file path without putting its directory on ``sys.path``.

    ``comparison_metrics`` is dependency-light by design, so it imports cleanly into this
    process; going through the file rather than the package keeps the research code's ``scripts/`` out of
    the import path, where its common module names would shadow the companion's own.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(payload) -> str:
    """A stable sha256 over a JSON-able value (used for the recipe and the reference's config)."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# ==============================================================================================
# The frame
# ==============================================================================================

def frame_differences(reference_config: dict, config) -> list[dict]:
    """Where the reference run's configuration and the companion's disagree.

    Empty means the two runs are the same experiment and their outputs are comparable. Anything
    in it means they are not, whatever the perplexities come out as -- which is the failure mode
    this whole comparison exists to be safe from.
    """
    differences = []
    for their_key, our_attr in FRAME_FIELDS.items():
        theirs = reference_config.get(their_key)
        ours = getattr(config, our_attr)
        same = (abs(theirs - ours) <= 1e-12 * max(1.0, abs(ours))
                if isinstance(ours, float) and isinstance(theirs, (int, float))
                else theirs == ours)
        if not same:
            differences.append({"field": their_key, "reference": theirs, "companion": ours})
    return differences


# ==============================================================================================
# The run
# ==============================================================================================

def train_stage(out: Path, recipe, corpus: Path, artifact: Path, *, device: str,
                keep_short_whole: bool,
                allow_code_change: bool = False) -> tuple[object, dict, float]:
    """Train stage 1 of a fresh workspace on ``corpus``, or reuse the one already at ``out``.

    A reused run is checked against what was asked for -- corpus, loader frame, held-out fraction,
    the recipe itself, **and the implementation that produced it** -- and refused if any of them
    moved (:class:`ReusedRunDiffers`). Silently re-scoring a run made under a different
    configuration, or by different code, is exactly the mistake this script is supposed to catch
    in others: the criterion would then certify an objective that never ran.

    Returns ``(workspace, history_entry, wall_clock_seconds)``; the wall clock is ``0.0`` for a
    reused run, which is what tells the caller not to quote it as a timing.
    """
    from lfa import Workspace

    workspace_dir = out / "workspace"
    if (workspace_dir / "workspace.json").exists():
        workspace = Workspace.open(workspace_dir)
        if workspace.history:
            entry = workspace.history[-1]
            _refuse_a_different_run(entry, recipe, corpus, keep_short_whole,
                                    allow_code_change=allow_code_change)
            print(f"[reuse] stage already trained at {entry['output_dir']}")
            return workspace, entry, 0.0
    else:
        # `fetch=False` and a local artifact path: the published registry digests are
        # placeholders, so a path is the supported route to the real file. `artifact_id` says
        # which published artifact that file is, so the recipe's calibration is read against the
        # id rather than against a path it has never heard of.
        workspace = Workspace.init(workspace_dir, MODEL_ID, artifact=str(artifact),
                                   recipe=RECIPE, fetch=False, artifact_id=ARTIFACT_ID)

    started = time.time()
    entry = workspace.train(corpus, recipe, keep_short_whole=keep_short_whole, device=device)
    return workspace, entry, time.time() - started


def _refuse_a_different_run(entry: dict, recipe, corpus: Path, keep_short_whole: bool, *,
                            allow_code_change: bool = False) -> None:
    """Raise unless the run already on disk is the run being asked for, by this code.

    The frame fields say the run trained the same experiment; ``implementation.code_digest``
    (:func:`lfa.workspace.code_identity`) says it was trained by the code now being certified.
    Without the second, a kept run plus an edited ``lfa/losses.py`` gives a green criterion for an
    objective that never executed -- which is what the criterion exists to make impossible.

    ``allow_code_change`` downgrades only the implementation difference to a warning, for
    re-scoring a kept run on purpose after a change known to be inert. It is never silent: the
    difference is printed and the caller records it in ``results.json``.
    """
    from lfa.workspace import code_identity

    trained_by = (entry.get("implementation") or {}).get("code_digest")
    now = code_identity()["code_digest"]
    if trained_by != now:
        made_by = ("code that recorded no identity (it predates the field)" if not trained_by
                   else f"lfa code {trained_by}")
        detail = (f"the run in this output directory was trained by {made_by}, and this is lfa "
                  f"code {now}")
        if not allow_code_change:
            raise ReusedRunDiffers(
                f"{detail}. Re-scoring it would report a criterion for an implementation that "
                "never ran. Retrain into a fresh --out, or pass --allow-code-change if you have "
                "established that the difference cannot touch this run."
            )
        print(f"[warn] --allow-code-change: {detail}", file=sys.stderr)

    asked = {
        "corpus": str(corpus),
        "keep_short_whole": keep_short_whole,
        "val_fraction": recipe.val_fraction,
        "recipe_digest": _digest(dataclasses.asdict(recipe)),
    }
    on_disk = {
        "corpus": entry["corpus"],
        "keep_short_whole": entry["keep_short_whole"],
        "val_fraction": entry["val_fraction"],
        "recipe_digest": _digest(entry["recipe"]),
    }
    moved = {key: (on_disk[key], value) for key, value in asked.items() if on_disk[key] != value}
    if moved:
        detail = "; ".join(f"{key}: on disk {have!r}, asked for {want!r}"
                           for key, (have, want) in moved.items())
        raise ReusedRunDiffers(
            f"the run already in this output directory was made under a different frame -- "
            f"{detail}. Re-scoring it would report numbers for a configuration nobody asked for. "
            "Point --out at a fresh directory, or delete this one."
        )


def score_with_research(inputs: dict[str, Path], checkpoint: Path, out_dir: Path,
                         env: dict[str, str]) -> dict:
    """Score ``checkpoint`` with the research instrument, in the research code's own virtualenv.

    This is the call ``scripts/judge_search.sh`` makes, with ``--output`` added so the result
    lands under this run rather than in the checkpoint directory. A result already in ``out_dir``
    is reused only when it names this same checkpoint and this same Q&A file -- the two fields the
    scorer itself records.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results = sorted(out_dir.glob("domain_perplexity_results_*.json"))
    if results:
        payload = json.loads(results[-1].read_text())
        scored, qa = Path(payload.get("model", "")), Path(payload.get("direct_qa_file", ""))
        same = (_same_path(scored, checkpoint, inputs["research"])
                and _same_path(qa, inputs["qa_file"], inputs["research"]))
        if not same:
            raise ReusedRunDiffers(
                f"{results[-1]} scored {payload.get('model')!r} against "
                f"{payload.get('direct_qa_file')!r}, but this run asked for {checkpoint} against "
                f"{inputs['qa_file']}. Delete that directory rather than reading its numbers as "
                "this checkpoint's."
            )
        print(f"[reuse] research scoring already at {results[-1]}")
    else:
        command = [
            str(inputs["python"]), "scripts/eval_domain_perplexity.py",
            "--model", str(checkpoint),
            "--qa-file", str(inputs["qa_file"]),
            "--wikitext-perplexity",
            "--output", str(out_dir),
        ]
        print("[run]", " ".join(command))
        subprocess.run(command, cwd=inputs["research"], env=env, check=True)
        results = sorted(out_dir.glob("domain_perplexity_results_*.json"))
        if not results:
            raise RuntimeError(f"the scorer wrote no result into {out_dir}")
        payload = json.loads(results[-1].read_text())

    metrics = _load_module(inputs["metrics"], "research_comparison_metrics")
    return {
        "checkpoint": str(checkpoint),
        "domain_direct_ppl": metrics.extract_domain_direct_ppl(payload),
        "seed_ppl": metrics.extract_seed_ppl_domain(payload),
        "results_file": str(results[-1]),
        "qa_file": str(inputs["qa_file"]),
    }


def _same_path(recorded: Path, wanted: Path, root: Path) -> bool:
    """Paths the scorer recorded are relative to the research code's root; ours are absolute."""
    recorded = recorded if recorded.is_absolute() else root / recorded
    return recorded.resolve() == Path(wanted).resolve()


def score_with_companion(workspace, *, device: str) -> dict:
    """The companion's own two axes, reported beside the research instrument's.

    ``Workspace.evaluate`` scores the stage's base model and the adapted model on the stage's
    held-out document split (the tenth of the corpus the run never trained on) and on the full
    WikiText-2 test split. The WikiText-2 numbers are the same quantity the research instrument
    reports -- and the base one is what both runs' seed *drift* is measured against, so it is not
    only decoration here.
    """
    scores = workspace.evaluate(n_windows=0, device=device)
    return {
        "domain_ppl_heldout_docs": scores["after"]["domain"],
        "base_domain_ppl_heldout_docs": scores["before"]["domain"],
        "wikitext2_ppl": scores["after"]["general"],
        "base_wikitext2_ppl": scores["before"]["general"],
        "table": scores["table"],
    }


# ==============================================================================================
# The comparison
# ==============================================================================================

def content_curve(history: list[dict]) -> list[float]:
    """Per-epoch content loss, in epoch order. Both repos write the key under the same name."""
    return [float(record["loss_content"]) for record in history]


def optimizer_steps(history: list[dict]) -> list[int]:
    """Per-epoch cumulative optimizer step, in epoch order. Both repos write ``global_step``.

    This is the deterministic backbone of the comparison. The number of optimizer steps an epoch
    takes is a function of the document set, the split, the chunker, that epoch's chunk offset, the
    batch size and the accumulation window, so any difference in those lands here -- as an integer,
    which no amount of floating-point drift can blur. It is a very tight *necessary* condition, not
    a proof of identity: at ``batch_size=6, ga=1`` the count is ``ceil(n_chunks / 6)``, so a
    difference of up to five chunks in an epoch would hide inside the ceiling. What rules that out
    is not this row but the source-level argument recorded in ``expected.json``: both loaders
    enumerate the corpus with the same sorted ``rglob``, shuffle it with the same Mersenne Twister
    under the same seed, and split it at the same index. This row is also what turns the learning
    rate into an assertion rather than a hand check, the schedule being a function of the step.
    """
    return [int(record["global_step"]) for record in history]


def held_out_curve(history: list[dict]) -> list[float]:
    """Per-epoch held-out loss, in epoch order, from either repo's history.

    the research code records it as ``eval.loss`` (``scripts/lra_run_experiment.py``'s
    ``evaluate_holdout``); the companion records it as ``val_loss``
    (:func:`lfa.train.validation_loss`). The two compute the same token-weighted cross-entropy
    over the same held-out text, so the fifteen-point curves are directly comparable.
    """
    values = []
    for record in history:
        if "val_loss" in record:
            values.append(float(record["val_loss"]))
        elif isinstance(record.get("eval"), dict) and "loss" in record["eval"]:
            values.append(float(record["eval"]["loss"]))
    return values


def held_out_tokens(history: list[dict]) -> list[int]:
    """Per-epoch held-out token count, from either repo's history.

    Exact equality of this integer says the two runs held out the same *quantity* of text under
    the same chunker -- a much tighter condition than the same number of documents, and one that
    catches a changed split, a changed offset rule or a changed sequence length. Like the step
    counts it is necessary rather than sufficient: equal token counts are arithmetically
    consistent with different text, and what excludes that is the source-level argument in
    ``expected.json``.
    """
    counts = []
    for record in history:
        if "val_tokens" in record:
            counts.append(int(record["val_tokens"]))
        elif isinstance(record.get("eval"), dict) and "tokens" in record["eval"]:
            counts.append(int(record["eval"]["tokens"]))
    return counts


#: the research code logs its chunk counts rather than recording them in JSON, so they are read back off
#: the run's own log. Both lines are written by ``scripts/lra_run_experiment.py`` for every run.
_REFERENCE_COUNT_LINES = {
    "train_chunks": re.compile(r"Training examples:\s*(\d+)"),
    "val_chunks": re.compile(r"Evaluation examples:\s*(\d+)"),
}


def reference_chunk_counts(log_path: Path) -> dict[str, int | None]:
    """The reference run's training and held-out chunk counts, read from its log."""
    text = Path(log_path).read_text(errors="replace")
    found = {}
    for key, pattern in _REFERENCE_COUNT_LINES.items():
        match = pattern.search(text)
        found[key] = int(match.group(1)) if match else None
    return found


def companion_chunk_counts(entry: dict) -> dict[str, int | None]:
    """The companion run's chunk counts: recorded if the run recorded them, else rebuilt.

    :meth:`lfa.workspace.Workspace.train` records ``n_train_chunks``/``n_val_chunks``. A run made
    before it did is not left unchecked: the split is rebuilt from the frame the entry itself
    records -- same corpus, seed, sequence length, held-out fraction and loader frame -- which is
    deterministic, so it reproduces the counts the run used.
    """
    if entry.get("n_train_chunks") is not None:
        return {"train_chunks": int(entry["n_train_chunks"]),
                "val_chunks": int(entry.get("n_val_chunks") or 0)}

    from lfa.corpus import load_corpus
    from lfa.models import load_tokenizer

    recipe = entry["recipe"]
    frame = entry.get("keep_short_whole")
    tokenizer = load_tokenizer(str(entry["base_model"]))
    train, held_out = load_corpus(
        entry["corpus"], tokenizer, max_length=recipe["sequence_length"],
        val_fraction=float(entry["val_fraction"]), seed=recipe["seed"],
        keep_short_whole=recipe["keep_short_whole"] if frame is None else bool(frame),
    )
    return {"train_chunks": len(train), "val_chunks": 0 if held_out is None else len(held_out)}


def check_equivalence(companion: dict, reference: dict, base_seed_ppl: float | None,
                      companion_history: list[dict], reference_history: list[dict],
                      frame: list[dict], counts: dict, expected: dict | None = None) -> list[dict]:
    """Compare the companion's run with the reference's, and say which comparison is the criterion.

    The rows come in three kinds, and the distinction is the point:

    ``frame``
        The two runs are the same experiment at all. Nothing below means anything without it.
    ``criterion``
        The deterministic evidence: optimizer steps per epoch, the corpus counts, the training
        curve and the held-out curve. These are what says the two implementations compute the same
        thing. They are exact or near-exact because they are not resampled quantities.
    ``report``
        One draw each of an instrument, at the end. They are printed with their deviations and
        carry **no verdict** (``ok`` is ``None``), because there is no measured band to hold them
        to: the anchor is a Monte-Carlo term drawn from independent RNG streams on the two sides
        (the research code's sampler uses torch's global generator, the companion's a private one), so two
        full runs are two draws of a stochastic objective whose spread nobody has measured. A band
        asserted on an unmeasured spread is a guess, and asserting one is the same mistake as
        asserting a difference smaller than the noise of the thing being measured. A large gap
        here means *investigate* -- a second seed on each side, which is also what would let a
        band be derived -- not *regression*.

    Returns one row per axis. A criterion row's missing measurement is a failure, not a skip -- it
    means an instrument produced no number for something that had to be compared; a report row
    with no number prints ``n/a`` and still decides nothing.

    Args:
        base_seed_ppl: the base model's WikiText-2 perplexity, which both drifts are read
            against. Measured in this same run by :func:`score_with_companion`.
        frame: the output of :func:`frame_differences`; a non-empty list fails its own row,
            because two runs of different configurations are not evidence about either.
        counts: ``{"companion": {...}, "reference": {...}}`` chunk counts, plus the companion's
            document counts under ``"documents"`` for the record.
    """
    expected = expected or json.loads(EXPECTED_FILE.read_text())
    checks = [{
        "kind": "frame",
        "name": "the two runs are the same configuration",
        "measured": len(frame),
        "detail": frame,
        "tolerance": "0 differing frame fields",
        "ok": not frame,
    }]

    # -- criterion 1: the optimizer steps themselves
    ours, theirs = optimizer_steps(companion_history), optimizer_steps(reference_history)
    mismatched = [{"epoch": i + 1, "companion": a, "reference": b}
                  for i, (a, b) in enumerate(zip(ours, theirs)) if a != b]
    checks.append({
        "kind": "criterion",
        "name": "optimizer steps per epoch",
        "measured": len(mismatched),
        "detail": {"companion": ours, "reference": theirs, "mismatched": mismatched},
        "tolerance": f"exact equality on all {len(theirs)} epochs",
        "ok": bool(theirs) and len(ours) == len(theirs) and not mismatched,
    })

    # -- criterion 2: the corpus the two runs actually saw
    our_counts, their_counts = counts.get("companion", {}), counts.get("reference", {})
    our_tokens = held_out_tokens(companion_history)
    their_tokens = held_out_tokens(reference_history)
    same_counts = [
        our_counts.get("train_chunks") is not None
        and our_counts.get("train_chunks") == their_counts.get("train_chunks"),
        our_counts.get("val_chunks") is not None
        and our_counts.get("val_chunks") == their_counts.get("val_chunks"),
        # The held-out token count is one constant per run, so it is compared as a value rather
        # than epoch by epoch: a run that stopped early should fail the step and curve rows, not
        # this one, which is about the corpus.
        bool(our_tokens) and bool(their_tokens) and set(our_tokens) == set(their_tokens),
    ]
    checks.append({
        "kind": "criterion",
        "name": "corpus: training chunks, held-out chunks, held-out tokens",
        "measured": sum(1 for ok in same_counts if not ok),
        "detail": {"companion": our_counts, "reference": their_counts,
                   "documents": counts.get("documents"),
                   "held_out_tokens": {"companion": sorted(set(our_tokens)),
                                       "reference": sorted(set(their_tokens))}},
        "tolerance": "exact equality on all three",
        "ok": all(same_counts),
    })

    # -- criterion 3: the training curve, every epoch
    checks.append(_curve_check(
        kind="criterion",
        name="content loss per epoch",
        ours=content_curve(companion_history), theirs=content_curve(reference_history),
        epochs=expected["content_curve"]["epochs"], tol_rel=expected["content_curve"]["tol_rel"],
    ))

    # -- criterion 4: the held-out curve, every epoch (absolute, in nats: it is a loss)
    ours, theirs = held_out_curve(companion_history), held_out_curve(reference_history)
    epochs = expected["held_out_curve"]["epochs"]
    tol_nats = expected["held_out_curve"]["tol_abs_nats"]
    pairs = list(zip(ours[:epochs], theirs[:epochs]))
    per_epoch = [{"epoch": i + 1, "companion": a, "reference": b, "deviation_nats": abs(a - b)}
                 for i, (a, b) in enumerate(pairs)]
    worst = max((row["deviation_nats"] for row in per_epoch), default=None)
    checks.append({
        "kind": "criterion",
        "name": f"held-out loss per epoch, 1-{epochs}",
        "measured": worst,
        "detail": per_epoch,
        "tolerance": f"every epoch within {tol_nats:.3g} nats (all {epochs} epochs present)",
        "ok": len(pairs) == epochs and worst is not None and worst <= tol_nats,
    })

    # -- report 1: the domain instrument, one draw, no verdict. `expected.json` carries no band
    # for either instrument row (`reported_not_asserted`), so there is nothing to read from it.
    ours, theirs = companion["domain_direct_ppl"], reference["domain_direct_ppl"]
    relative = None if ours is None or not theirs else abs(ours - theirs) / theirs
    checks.append({
        "kind": "report",
        "name": "REPORTED (one draw, not asserted): domain direct-QA perplexity",
        "measured": ours,
        "reference": theirs,
        "deviation": None if relative is None else relative * 100,
        "tolerance": "reported against %s; no band -- the spread of this quantity is unmeasured"
                     % ("n/a" if theirs is None else round(theirs, 4)),
        "ok": None,
    })

    # -- report 2: the preservation instrument, one draw, no verdict
    our_drift = _drift(companion["seed_ppl"], base_seed_ppl)
    their_drift = _drift(reference["seed_ppl"], base_seed_ppl)
    gap = None if our_drift is None or their_drift is None else abs(our_drift - their_drift)
    checks.append({
        "kind": "report",
        "name": "REPORTED (one draw, not asserted): WikiText-2 drift vs base %s" % (
            "n/a" if base_seed_ppl is None else f"{base_seed_ppl:.2f}"),
        "measured": our_drift,
        "reference": their_drift,
        "deviation": gap,
        "tolerance": "reported against %s pp; no band -- the spread of this quantity is "
                     "unmeasured" % ("n/a" if their_drift is None else round(their_drift, 3)),
        "ok": None,
    })
    return checks


def _curve_check(*, kind: str, name: str, ours: list[float], theirs: list[float], epochs: int,
                 tol_rel: float) -> dict:
    """One per-epoch relative-deviation row (used for the content curve)."""
    pairs = list(zip(ours[:epochs], theirs[:epochs]))
    per_epoch = [{"epoch": i + 1, "companion": a, "reference": b,
                  "deviation_pct": abs(a - b) / b * 100 if b else None}
                 for i, (a, b) in enumerate(pairs)]
    worst = max((row["deviation_pct"] for row in per_epoch if row["deviation_pct"] is not None),
                default=None)
    return {
        "kind": kind,
        "name": f"{name}, 1-{epochs}",
        "measured": worst,
        "detail": per_epoch,
        "tolerance": f"every epoch within {tol_rel * 100:.3g}% (all {epochs} epochs present)",
        "ok": len(pairs) == epochs and worst is not None and worst <= tol_rel * 100,
    }


def _drift(perplexity: float | None, base: float | None) -> float | None:
    """Percentage change from the base model's perplexity; ``None`` if either is missing."""
    if perplexity is None or not base:
        return None
    return (perplexity - base) / base * 100


#: What to do about a row, by kind. A criterion row failing is a regression: the two
#: implementations stopped computing the same thing. A `report` row cannot fail -- it carries no
#: verdict, because the spread of what it measures has never been measured -- see `expected.json`.
FAILURE_GUIDANCE = {
    "frame": ("the two runs are not the same experiment, so nothing below them means anything. "
              "Fix the recipe or point --reference at a matching run."),
    "criterion": ("this is the equivalence criterion and it is deterministic: a failure here is a "
                  "regression in what this package computes. Do not widen the tolerance -- find "
                  "the change."),
    "report": ("this row is REPORTED, not asserted, and decides nothing: it is one draw of an "
               "instrument on an objective sampled from independent RNG streams on the two "
               "sides, and the spread of that quantity has never been measured. A large gap "
               "means INVESTIGATE (a second seed on each side, which is also what would let a "
               "band be derived) rather than REGRESSION. Read the criterion rows for the "
               "verdict."),
}


def format_checks(checks: list[dict]) -> str:
    """The checks as lines, with a failing frame row expanded field by field.

    A row whose ``ok`` is ``None`` is a ``report`` row: it prints ``[----]`` and its deviation
    rather than a verdict, so that nobody reads a printed PASS off a number nothing asserted.

    The expansion is selected on the row's ``kind``, not on its display name: renaming a check
    should not silently drop the detail that makes its failure diagnosable.
    """
    lines = []
    for check in checks:
        measured = check["measured"]
        shown = "n/a" if measured is None else (f"{measured:.4f}"
                                                if isinstance(measured, float) else str(measured))
        verdict = "----" if check["ok"] is None else ("PASS" if check["ok"] else "FAIL")
        deviation = check.get("deviation")
        if check["ok"] is None and deviation is not None:
            shown = f"{shown} (deviation {deviation:.4g})"
        lines.append(f"  [{verdict}] {check['name']}: {shown} "
                     f"({check['tolerance']})")
        if check.get("kind") == "frame":
            for difference in check["detail"]:
                lines.append(f"        {difference['field']}: reference "
                             f"{difference['reference']!r} vs companion "
                             f"{difference['companion']!r}")
    return "\n".join(lines)


# ==============================================================================================
# Entry point
# ==============================================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--the research code", default=os.environ.get("LFA_RESEARCH_ROOT")
                        or str(DEFAULT_RESEARCH_ROOT),
                        help="the research checkout to read (never written to)")
    parser.add_argument("--out", required=True,
                        help="where the workspace, the run and the scoring output go")
    parser.add_argument("--device", default="cuda:0", help="device for training and evaluation")
    parser.add_argument("--cuda-visible-devices", default="0",
                        help="pin the run to one card; ignored if the variable is already set. "
                             "Pass an empty string to leave the environment alone.")
    parser.add_argument("--corpus", default=None,
                        help=f"override the corpus (default <the research code>/{CORPUS_REL})")
    parser.add_argument("--artifact", default=None,
                        help=f"override the p(h) artifact (default <the research code>/{ARTIFACT_REL})")
    parser.add_argument("--qa-file", default=None,
                        help=f"override the held-out QA set (default <the research code>/{QA_REL})")
    parser.add_argument("--reference", default=None,
                        help=f"override the reference run (default <the research code>/{REFERENCE_REL})")
    parser.add_argument("--skip-research-scoring", action="store_true",
                        help="train and report the companion's own metrics only")
    parser.add_argument("--allow-frame-mismatch", action="store_true",
                        help="train and score even though the reference run is a different "
                             "configuration; the difference is reported as a failing check")
    parser.add_argument("--allow-code-change", action="store_true",
                        help="re-score a kept run that was trained by different code; the "
                             "difference is printed and recorded in results.json instead of "
                             "refusing the reuse")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero when a checked number falls outside its tolerance "
                             "(the two instrument rows are reported, not checked, and cannot "
                             "decide this either way)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # The package logs through `logging` and configures no handler of its own (a library should
    # not). Without this an hour-long run is silent, and a stall is indistinguishable from
    # progress; with it, every epoch's losses and both perplexity computations are on the log.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Pinned before torch is imported anywhere below, and passed to the scorer's environment too.
    if args.cuda_visible_devices and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    # The machine's network is flaky and everything this run needs is cached; a Hub check that
    # hangs would strand an hour-long run.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    inputs = resolve_inputs(args.research, args.corpus, args.artifact, args.qa_file,
                            args.reference)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    from lfa import Recipe

    recipe = Recipe.load(RECIPE)
    reference_config = json.loads(inputs["reference_config"].read_text())
    reference_history = json.loads(inputs["reference_history"].read_text())

    # Checked BEFORE anything is trained, and fatal: two hours of GPU time spent comparing two
    # different experiments produces numbers, and numbers about nothing are worse than no numbers.
    frame = frame_differences(reference_config,
                              recipe.to_train_config(1, inputs["artifact"],
                                                     keep_short_whole=recipe.keep_short_whole))
    if frame and not args.allow_frame_mismatch:
        raise FrameMismatch(
            "the reference run and this recipe are not the same configuration, so comparing them "
            "would measure the difference rather than the implementations: "
            + "; ".join(f"{d['field']} reference {d['reference']!r} vs companion "
                        f"{d['companion']!r}" for d in frame)
            + ". Fix the recipe or point --reference at a matching run; --allow-frame-mismatch "
              "runs anyway and reports the difference as a failing check."
        )
    if frame:
        print("[warn] --allow-frame-mismatch: the reference run and this recipe differ on: "
              + ", ".join(difference["field"] for difference in frame), file=sys.stderr)

    workspace, entry, wall_clock = train_stage(
        out, recipe, inputs["corpus"], inputs["artifact"], device=args.device,
        keep_short_whole=recipe.keep_short_whole, allow_code_change=args.allow_code_change,
    )
    companion_history = json.loads(
        (Path(entry["output_dir"]) / "training_history.json").read_text())

    research = reference = None
    if not args.skip_research_scoring:
        research = score_with_research(inputs, Path(entry["adapter"]), out / "scoring_companion",
                                        dict(os.environ))
        reference = score_with_research(inputs, inputs["reference_model"],
                                         out / "scoring_reference", dict(os.environ))
    companion = score_with_companion(workspace, device=args.device)

    counts = {
        "companion": companion_chunk_counts(entry),
        "reference": reference_chunk_counts(inputs["reference_log"]),
        "documents": {"train": entry["n_train_docs"], "held_out": entry["n_val_docs"]},
    }

    reference_record = {
        "run": str(inputs["reference"]),
        "checkpoint": str(inputs["reference_model"]),
        "config_digest": _digest(reference_config),
        "config": {key: reference_config.get(key) for key in FRAME_FIELDS},
        "domain_direct_ppl": None if reference is None else reference["domain_direct_ppl"],
        "seed_ppl": None if reference is None else reference["seed_ppl"],
        "content_curve": content_curve(reference_history),
        "held_out_curve": held_out_curve(reference_history),
        "optimizer_steps": optimizer_steps(reference_history),
        "held_out_tokens": held_out_tokens(reference_history),
        "chunk_counts": counts["reference"],
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "instrument": None if reference is None else reference["results_file"],
    }
    (out / "reference.json").write_text(json.dumps(reference_record, indent=2))

    checks = check_equivalence(
        companion=research or {"domain_direct_ppl": None, "seed_ppl": None},
        reference=reference or {"domain_direct_ppl": None, "seed_ppl": None},
        base_seed_ppl=companion["base_wikitext2_ppl"],
        companion_history=companion_history,
        reference_history=reference_history,
        frame=frame,
        counts=counts,
    )

    from lfa import __version__ as lfa_version
    from lfa.workspace import code_identity

    # Provenance of the TRAINING, not of this process: a kept run re-scored later is scored by a
    # newer HEAD, and stamping the record with that HEAD is how a curve came to carry a commit it
    # was not trained under. `implementation` is written by `Workspace.train` at train time.
    implementation = entry.get("implementation") or {}
    scoring_code = code_identity()
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lfa_version": lfa_version,
        "companion_commit": implementation.get("git_revision"),
        "companion_code_digest": implementation.get("code_digest"),
        "companion_commit_is": ("the revision the stage was trained under" if implementation
                                else "unknown: the run recorded no implementation identity"),
        "scoring_process_commit": _git_revision(Path(__file__).resolve().parents[2]),
        "scoring_process_code_digest": scoring_code["code_digest"],
        # Not a bare flag: when the override fires, the record has to name WHICH implementations
        # were waved through, or "allowed" says that something was permitted without saying what.
        "reused_run_code_change_allowed": (
            {"trained_by": implementation.get("code_digest") or "unrecorded",
             "scored_by": scoring_code["code_digest"]}
            if args.allow_code_change else False),
        "research": str(inputs["research"]),
        "research_revision": _git_revision(inputs["research"]),
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "wall_clock_seconds": round(wall_clock, 1),
        "corpus": str(inputs["corpus"]),
        "artifact": str(inputs["artifact"]),
        "recipe": RECIPE,
        "keep_short_whole": entry["keep_short_whole"],
        "val_fraction": entry["val_fraction"],
        "n_train_docs": entry["n_train_docs"],
        "n_val_docs": entry["n_val_docs"],
        "epochs": entry["epochs"],
        "checkpoint": entry["adapter"],
        "content_curve": content_curve(companion_history),
        "held_out_curve": held_out_curve(companion_history),
        "optimizer_steps": optimizer_steps(companion_history),
        "held_out_tokens": held_out_tokens(companion_history),
        "chunk_counts": counts["companion"],
        "val_perplexity_curve": [record.get("val_perplexity") for record in companion_history],
        "learning_rate_curve": [record["learning_rate"] for record in companion_history],
        "research_instrument": research,
        "reference": reference_record,
        "companion_instrument": companion,
        "checks": checks,
        # `is not False`, not truthiness: a `report` row's `ok` is None and decides nothing.
        "passed": all(check["ok"] is not False for check in checks),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))

    print("\n" + companion["table"])
    print(f"\nEquivalence against {inputs['reference']}:")
    print(format_checks(checks))
    print(f"\nWrote {out / 'results.json'} and {out / 'reference.json'}")
    return 0 if (results["passed"] or not args.strict) else 1


def _git_revision(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


if __name__ == "__main__":
    sys.exit(main())
