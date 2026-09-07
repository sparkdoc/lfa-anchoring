"""The equivalence test: this package trains what the research code trains.

One test here is opt-in (``pytest -m acceptance``) because it is over an hour of GPU time and it
reads gigabytes the companion does not distribute -- the corpus, the p(h) artifact, the held-out
Q&A set and the the research code reference run all live in the research checkout. Without them it skips,
naming what is missing. The rest of this file is cheap, runs in the default suite, and checks the
comparison logic itself: that the tolerance file says what it is supposed to say, that the
comparison accepts a matching run and rejects a drifting one, and that a missing input is named.

What the GPU test asserts comes in three kinds, and the distinction is the point. The frame is
checked first (the two runs must be the same experiment at all). Then the CRITERION, which is
deterministic: per-epoch optimizer steps as exact integers, the corpus counts, and the two
fifteen-point curves. Then two REPORTED rows -- the research instrument's domain and WikiText-2
numbers, one draw each -- which are printed with their deviations and asserted on by nothing,
because the anchor is drawn from independent RNG streams on the two sides, two full runs are two
draws of a stochastic objective, and the spread of those draws has never been measured. A band on
an unmeasured spread would be a guess; the way to get one back is a second seed, not a wider
number.

The tolerances, and that reasoning, live in ``expected.json`` beside this file, which is the one
place they are described. They are not to be widened to accommodate a run: a number outside them
is a finding.

Set ``LFA_ACCEPTANCE_OUT`` to a directory to keep (or reuse) the run: ``run_recipe.py`` will not
retrain a workspace that already carries a trained stage, nor rescore a checkpoint it has already
scored -- but it refuses to reuse either if the frame it finds is not the frame being asked for.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
from pathlib import Path

import pytest

# Appended, not prepended: this module is imported at collection time even on a default run
# (which deselects the acceptance marker), and putting this directory first would let it shadow
# imports for the rest of the session.
sys.path.append(str(Path(__file__).parent))

import run_recipe  # noqa: E402  (needs the path entry above)


@pytest.mark.acceptance
def test_the_recipe_lands_where_the_research_code_lands(tmp_path):
    # NOT marked `gpu`, deliberately: `-m gpu` is how a maintainer runs the fast GPU tests, and a
    # marker that also selects an hour of training is a footgun. `-m acceptance` is the only way
    # to select this one. The marker's other job -- skipping on a CUDA-less machine
    # (`tests/conftest.py`) -- is two lines, and they are here instead.
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA (this run trains for about an hour on one card)")

    research = os.environ.get("LFA_RESEARCH_ROOT") or str(run_recipe.DEFAULT_RESEARCH_ROOT)
    try:
        run_recipe.resolve_inputs(research)
    except run_recipe.MissingInput as missing:
        pytest.skip(str(missing))

    out = Path(os.environ.get("LFA_ACCEPTANCE_OUT") or (tmp_path / "equivalence"))
    assert run_recipe.main(["--the research code", research, "--out", str(out)]) == 0

    results = json.loads((out / "results.json").read_text())
    # Reported whatever the verdict, so a failure carries its numbers rather than just a name.
    print(run_recipe.format_checks(results["checks"]))

    for check in results["checks"]:
        if check["ok"] is None:
            # A `report` row: printed above with its deviation, asserted on by nothing. See
            # `expected.json` for why there is no band to assert.
            continue
        # The guidance differs by kind, and this is the one place most readers will meet it: a
        # criterion miss is a regression; a report row cannot get here at all.
        assert check["ok"], (
            f"{check['name']}: measured {check['measured']}, outside {check['tolerance']}.\n"
            f"{run_recipe.FAILURE_GUIDANCE[check['kind']]}\n"
            f"The run is at {out / 'workspace'}; the reference is {results['reference']['run']}."
        )


def test_the_tolerance_file_carries_exactly_the_agreed_tolerances():
    """The tolerances themselves: a guard against a loosening going unnoticed.

    ``expected.json`` holds tolerances and nothing else -- the reference's measured values are
    written out by the harness, not written down here -- so this is the whole of what could be
    quietly widened to rescue a run.
    """
    expected = json.loads(run_recipe.EXPECTED_FILE.read_text())
    assert expected["optimizer_steps"] == {"exact": True}
    assert expected["corpus_counts"] == {"exact": True}
    assert expected["content_curve"] == {"epochs": 15, "tol_rel": 0.005}
    assert expected["held_out_curve"] == {"epochs": 15, "tol_abs_nats": 0.03}
    # The instrument rows carry no band at all -- not a wide one. Restoring a number here
    # without the seed measurement that would justify it is the change this line exists to catch.
    assert expected["domain_ppl"] == {"reported_not_asserted": True}
    assert expected["seed_drift"] == {"reported_not_asserted": True}
    assert set(expected) == {"_comment", "optimizer_steps", "corpus_counts", "content_curve",
                             "held_out_curve", "domain_ppl", "seed_drift"}


def test_the_tolerance_file_says_which_check_is_the_criterion():
    """The reasoning is the durable part: the next reader must not repeat it from scratch."""
    comment = " ".join(json.loads(run_recipe.EXPECTED_FILE.read_text())["_comment"]).lower()
    assert "independent rng streams" in comment          # the real dominant divergence term
    assert "reported, not asserted" in comment           # what the perplexities are now
    assert "second seed" in comment                      # what would give them a band back
    assert "exact integer equality" in comment           # what the criterion is
    assert "widened" in comment


# --- the comparison logic, on synthetic runs -------------------------------------------------

EPOCHS = 15
STEPS = [603 * (i + 1) for i in range(EPOCHS)]
CONTENT = [2.62 - 0.04 * i for i in range(EPOCHS)]
HELD_OUT = [2.63 - 0.008 * i for i in range(EPOCHS)]
TOKENS = 157366
COUNTS = {"companion": {"train_chunks": 3614, "val_chunks": 435},
          "reference": {"train_chunks": 3614, "val_chunks": 435},
          "documents": {"train": 1505, "held_out": 168}}


def companion_history(steps=None, content=None, held_out=None, tokens=TOKENS):
    """A companion `training_history.json`, as `lfa.train.train` writes it."""
    return [{"epoch": i + 1, "global_step": s, "loss_content": c, "val_loss": v,
             "val_perplexity": math.exp(v), "val_tokens": tokens}
            for i, (s, c, v) in enumerate(zip(steps or STEPS, content or CONTENT,
                                              held_out or HELD_OUT))]


def reference_history(steps=None, content=None, held_out=None, tokens=TOKENS):
    """An the research code `training_history.json`: same content key, `eval` instead of `val_*`."""
    return [{"epoch": i + 1, "global_step": s, "loss_content": c,
             "eval": {"loss": v, "tokens": tokens}}
            for i, (s, c, v) in enumerate(zip(steps or STEPS, content or CONTENT,
                                              held_out or HELD_OUT))]


def _checks(domain=8.9012, seed=16.6134, companion=None, reference=None, counts=None,
            reference_domain=8.90, reference_seed=16.61, base=18.18, frame=()):
    return run_recipe.check_equivalence(
        companion={"domain_direct_ppl": domain, "seed_ppl": seed},
        reference={"domain_direct_ppl": reference_domain, "seed_ppl": reference_seed},
        base_seed_ppl=base,
        companion_history=companion if companion is not None else companion_history(),
        reference_history=reference if reference is not None else reference_history(),
        frame=list(frame),
        counts=counts if counts is not None else COUNTS,
    )


def _verdicts(checks):
    return {check["name"]: check["ok"] for check in checks}


def _failed(checks):
    """The rows that carry a FAILING verdict. A `report` row (`ok is None`) carries none."""
    return [check["name"] for check in checks if check["ok"] is False]


def _reported(checks):
    return [check for check in checks if check["kind"] == "report"]


def test_a_run_that_matches_the_reference_passes_every_check():
    assert _failed(_checks()) == []
    assert [check["kind"] for check in _checks()] == [
        "frame", "criterion", "criterion", "criterion", "criterion", "report", "report"]


def test_the_two_instrument_rows_are_reported_and_assert_nothing():
    """The one recorded ruling this file has to keep honest.

    They were coarse *sanity checks* with a 2 % / 1.0 pp band, and the 2026-09-07 run consumed
    1.95 % of the 2 %. Since the spread of these quantities across seeds has never been measured,
    that band was never calibrated -- so the rows now carry no verdict at all rather than a
    tighter or a wider guess. Restoring a band needs a second seed, not a decision.
    """
    reported = _reported(_checks())
    assert len(reported) == 2
    assert all(check["ok"] is None for check in reported)
    assert all(check["name"].startswith("REPORTED (one draw, not asserted)") for check in reported)
    assert all("no band" in check["tolerance"] for check in reported)
    # And they are printed without a verdict, so nobody reads a PASS off an unasserted number.
    printed = run_recipe.format_checks(_checks())
    assert "REPORTED" in printed and "[----]" in printed


# --- the criterion bites ----------------------------------------------------------------------

def test_one_differing_optimizer_step_fails_the_criterion():
    """Any difference in documents, split, chunking or batching lands here as an integer."""
    drifted = list(STEPS)
    drifted[7] += 1
    checks = _checks(companion=companion_history(steps=drifted))
    assert _failed(checks) == ["optimizer steps per epoch"]
    [row] = [c for c in checks if c["name"] == "optimizer steps per epoch"]
    assert row["detail"]["mismatched"] == [{"epoch": 8, "companion": STEPS[7] + 1,
                                            "reference": STEPS[7]}]


def test_a_run_that_stopped_early_fails_the_criterion_rather_than_passing_on_a_prefix():
    checks = _checks(companion=companion_history(steps=STEPS[:10], content=CONTENT[:10],
                                                 held_out=HELD_OUT[:10]))
    # The corpus row is NOT among them: the corpus was fine, the run was short.
    assert _failed(checks) == ["optimizer steps per epoch",
                               "content loss per epoch, 1-15",
                               "held-out loss per epoch, 1-15"]


@pytest.mark.parametrize("counts, expected_failure", [
    ({**COUNTS, "companion": {"train_chunks": 3600, "val_chunks": 435}}, True),
    ({**COUNTS, "companion": {"train_chunks": 3614, "val_chunks": 430}}, True),
    ({**COUNTS, "companion": {"train_chunks": None, "val_chunks": None}}, True),
    (COUNTS, False),
])
def test_the_corpus_counts_must_match_exactly(counts, expected_failure):
    failed = _failed(_checks(counts=counts))
    corpus_row = ["corpus: training chunks, held-out chunks, held-out tokens"]
    assert (failed == corpus_row) is expected_failure


def test_a_different_held_out_token_count_fails_the_corpus_check():
    """Same number of documents, different text or chunking: this is where it shows."""
    checks = _checks(companion=companion_history(tokens=TOKENS - 512))
    assert _failed(checks) == ["corpus: training chunks, held-out chunks, held-out tokens"]


def test_the_content_curve_is_checked_on_every_epoch_not_a_window():
    """The last epoch is inside the check -- there is no window it drifts outside of."""
    late = list(CONTENT)
    late[14] *= 1.01                                    # 1 % at the final epoch: outside 0.5 %
    assert _failed(_checks(companion=companion_history(content=late))) == [
        "content loss per epoch, 1-15"]


def test_the_held_out_curve_is_checked_in_nats_on_every_epoch():
    late = list(HELD_OUT)
    late[11] += 0.05                                    # 0.05 nats: outside the 0.03 tolerance
    assert _failed(_checks(companion=companion_history(held_out=late))) == [
        "held-out loss per epoch, 1-15"]


def test_the_measured_agreement_of_the_real_run_clears_the_curve_tolerances():
    """The tolerances have headroom over what the two implementations actually did.

    Both curves come from the run of 2026-09-07 (report section 5): the worst content deviation
    was 0.191 % against 0.5 %, and the worst held-out gap 0.0105 nats against 0.03.
    """
    content_ours = [2.6201, 2.5176, 2.4745, 2.4323, 2.3968, 2.3654, 2.3305, 2.2981,
                    2.2629, 2.2310, 2.1748, 2.1389, 2.1093, 2.0806, 2.0715]
    content_theirs = [2.6188, 2.5173, 2.4712, 2.4322, 2.3947, 2.3660, 2.3270, 2.2957,
                      2.2596, 2.2300, 2.1760, 2.1348, 2.1104, 2.0779, 2.0716]
    held_ours = [2.6260, 2.5891, 2.5678, 2.5421, 2.5364, 2.5243, 2.5169, 2.5085,
                 2.5063, 2.5024, 2.5065, 2.5194, 2.5322, 2.5430, 2.5480]
    held_theirs = [2.6365, 2.5953, 2.5701, 2.5476, 2.5317, 2.5208, 2.5109, 2.5047,
                   2.5026, 2.4980, 2.5103, 2.5220, 2.5341, 2.5427, 2.5486]

    checks = _checks(companion=companion_history(content=content_ours, held_out=held_ours),
                     reference=reference_history(content=content_theirs, held_out=held_theirs))

    content = [c for c in checks if c["name"] == "content loss per epoch, 1-15"][0]
    held = [c for c in checks if c["name"] == "held-out loss per epoch, 1-15"][0]
    assert content["ok"] and content["measured"] == pytest.approx(0.191, abs=0.01)
    assert held["ok"] and held["measured"] == pytest.approx(0.0105, abs=0.001)


# --- the instrument rows, reported and unasserted --------------------------------------------

@pytest.mark.parametrize("domain, seed, deviations", [
    # 3 % over the reference's domain perplexity, and no WikiText-2 gap.
    (9.17, 16.61, (3.034, 0.0)),
    # +1.5 pp of WikiText-2 drift (16.61 -> 16.88 against base 18.18), and no domain gap.
    (8.90, 16.88, (0.0, 1.485)),
])
def test_a_run_that_drifts_on_an_instrument_reports_the_gap_and_fails_nothing(domain, seed,
                                                                             deviations):
    """A drift on either instrument is printed as a number to investigate, not as a verdict.

    Both of these would have failed under the old 2 % / 1.0 pp bands. Nothing about the run has
    changed; what changed is that a band nobody measured no longer decides whether it passed.
    """
    checks = _checks(domain=domain, seed=seed)
    assert _failed(checks) == []
    reported = _reported(checks)
    assert [check["deviation"] for check in reported] == [pytest.approx(deviations[0], abs=0.01),
                                                          pytest.approx(deviations[1], abs=0.01)]
    assert all(check["ok"] is None for check in reported)


def test_two_runs_of_different_configurations_are_not_evidence_about_either():
    """A frame difference fails on its own, whatever everything else came out as."""
    frame = [{"field": "lora_rank", "reference": 32, "companion": 16}]
    checks = _checks(frame=frame)
    assert checks[0]["ok"] is False and checks[0]["kind"] == "frame"
    assert _failed(checks) == ["the two runs are the same configuration"]
    assert "lora_rank" in run_recipe.format_checks(checks)


def test_a_missing_instrument_number_is_reported_as_absent_rather_than_as_agreement():
    """No number is not the same as a matching number -- but it is not a failure either.

    A criterion row missing its measurement fails (the corpus row does exactly that above); an
    instrument row prints `n/a` and decides nothing, which is all an unasserted row can do.
    """
    checks = _checks(domain=None, seed=None)
    assert _failed(checks) == []
    assert [check["measured"] for check in _reported(checks)] == [None, None]
    assert [check["deviation"] for check in _reported(checks)] == [None, None]
    assert "n/a" in run_recipe.format_checks(checks)


def test_the_reference_chunk_counts_are_read_off_its_log(tmp_path):
    log = tmp_path / "training.log"
    log.write_text("INFO   Training examples: 3614\nINFO   Evaluation examples: 435\n")
    assert run_recipe.reference_chunk_counts(log) == {"train_chunks": 3614, "val_chunks": 435}
    empty = tmp_path / "quiet.log"
    empty.write_text("nothing of interest\n")
    assert run_recipe.reference_chunk_counts(empty) == {"train_chunks": None, "val_chunks": None}


def test_recorded_chunk_counts_are_preferred_to_rebuilding_the_corpus():
    """A run that recorded them needs no corpus on disk to be checked."""
    entry = {"n_train_chunks": 3614, "n_val_chunks": 435, "corpus": "/gone",
             "base_model": "/gone", "val_fraction": 0.1,
             "recipe": {"sequence_length": 512, "seed": 42, "keep_short_whole": True}}
    assert run_recipe.companion_chunk_counts(entry) == {"train_chunks": 3614, "val_chunks": 435}


def test_a_frame_difference_is_found_field_by_field(tmp_path):
    """The comparison's first job: the two runs' own configs, compared where it matters."""
    from lfa import Recipe

    config = Recipe.load("qwen3-0.6b").to_train_config(1, tmp_path / "stats.pt")
    reference = {key: getattr(config, attr) for key, attr in run_recipe.FRAME_FIELDS.items()}

    assert run_recipe.frame_differences(reference, config) == []
    assert run_recipe.frame_differences({**reference, "lambda_mlp": 20000.0}, config) == [
        {"field": "lambda_mlp", "reference": 20000.0, "companion": 100000.0}]
    # A float that differs only in representation is not a difference.
    assert run_recipe.frame_differences({**reference, "learning_rate": 3e-4}, config) == []


def _reusable_entry(tmp_path, recipe):
    """A history entry as `Workspace.train` writes one, for the run the harness would reuse."""
    from lfa.workspace import code_identity

    return {
        "corpus": str(tmp_path / "corpus"),
        "keep_short_whole": True,
        "val_fraction": 0.1,
        "recipe": dataclasses.asdict(recipe),
        "implementation": code_identity(),
    }


def test_a_reused_run_under_a_different_frame_is_refused(tmp_path):
    """The reuse path re-scores an existing run; it must first check it is the same run."""
    from lfa import Recipe

    recipe = dataclasses.replace(Recipe.load("qwen3-0.6b"), val_fraction=0.1)
    entry = _reusable_entry(tmp_path, recipe)

    run_recipe._refuse_a_different_run(entry, recipe, tmp_path / "corpus", True)

    with pytest.raises(run_recipe.ReusedRunDiffers, match="keep_short_whole"):
        run_recipe._refuse_a_different_run(entry, recipe, tmp_path / "corpus", False)
    with pytest.raises(run_recipe.ReusedRunDiffers, match="corpus"):
        run_recipe._refuse_a_different_run(entry, recipe, tmp_path / "other", True)
    with pytest.raises(run_recipe.ReusedRunDiffers, match="recipe_digest"):
        run_recipe._refuse_a_different_run(
            entry, dataclasses.replace(recipe, lambda_qkv=20000.0), tmp_path / "corpus", True)


def test_a_reused_run_made_by_different_code_is_refused(tmp_path, capsys):
    """The hole this closes: a kept run plus an edited objective is a criterion for code that
    never executed.

    The frame fields say the run trained the same *experiment*; only the implementation digest
    says it was trained by the code being certified. The 2026-09-07 equivalence run was exactly
    this case -- scored at 17:57 from a stage trained at 16:13, with nine modules changed in
    between -- and nothing in the harness noticed.
    """
    from lfa import Recipe

    recipe = dataclasses.replace(Recipe.load("qwen3-0.6b"), val_fraction=0.1)
    corpus = tmp_path / "corpus"
    entry = _reusable_entry(tmp_path, recipe)

    other_code = {**entry, "implementation": {"code_digest": "0123456789abcdef",
                                              "git_revision": None}}
    with pytest.raises(run_recipe.ReusedRunDiffers, match="0123456789abcdef"):
        run_recipe._refuse_a_different_run(other_code, recipe, corpus, True)

    # A run made before the field existed cannot vouch for itself either.
    no_identity = {key: value for key, value in entry.items() if key != "implementation"}
    with pytest.raises(run_recipe.ReusedRunDiffers, match="recorded no identity"):
        run_recipe._refuse_a_different_run(no_identity, recipe, corpus, True)

    # ...and the deliberate override is loud rather than silent.
    run_recipe._refuse_a_different_run(other_code, recipe, corpus, True, allow_code_change=True)
    assert "--allow-code-change" in capsys.readouterr().err


def test_a_missing_input_is_named_rather_than_guessed(tmp_path):
    with pytest.raises(run_recipe.MissingInput, match="the research code checkout"):
        run_recipe.resolve_inputs(tmp_path / "nowhere")


def test_a_reference_run_without_its_log_is_named_up_front(tmp_path):
    """The corpus check reads the reference's chunk counts off its log, so the log is an input.

    Not declaring it is how a two-hour run ends in a FileNotFoundError after the training and both
    scorings are already spent.
    """
    research = tmp_path / "the research code"
    reference = research / run_recipe.REFERENCE_REL
    for path in (research / ".venv" / "bin", research / "scripts", reference / "final_model",
                 research / run_recipe.CORPUS_REL,
                 (research / run_recipe.ARTIFACT_REL).parent,
                 (research / run_recipe.QA_REL).parent):
        path.mkdir(parents=True, exist_ok=True)
    for path in (research / ".venv" / "bin" / "python",
                 research / "scripts" / "eval_domain_perplexity.py",
                 research / "scripts" / "comparison_metrics.py",
                 research / run_recipe.ARTIFACT_REL, research / run_recipe.QA_REL,
                 reference / "config.json", reference / "training_history.json"):
        path.write_text("{}")

    with pytest.raises(run_recipe.MissingInput, match="reference run's log"):
        run_recipe.resolve_inputs(research)

    (reference / "training.log").write_text("Training examples: 3614\nEvaluation examples: 435\n")
    assert run_recipe.resolve_inputs(research)["reference_log"].name == "training.log"


def test_a_frame_mismatch_refuses_before_anything_is_trained(tmp_path, monkeypatch):
    """`FrameMismatch` is raised by `main` ahead of `train_stage`, not reported after it."""
    calls = []
    monkeypatch.setattr(run_recipe, "train_stage",
                        lambda *a, **k: calls.append(a) or (None, {}, 0.0))
    monkeypatch.setattr(run_recipe, "frame_differences",
                        lambda *a, **k: [{"field": "lora_rank", "reference": 32,
                                          "companion": 16}])
    monkeypatch.setattr(run_recipe, "resolve_inputs", _fake_inputs(tmp_path))

    with pytest.raises(run_recipe.FrameMismatch, match="lora_rank"):
        run_recipe.main(["--out", str(tmp_path / "out"), "--cuda-visible-devices", ""])

    assert calls == [], "training started despite a frame mismatch"


def _fake_inputs(tmp_path):
    """Enough of `resolve_inputs`'s answer for `main` to reach the frame check."""
    reference = tmp_path / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    (reference / "config.json").write_text("{}")
    (reference / "training_history.json").write_text("[]")
    (reference / "training.log").write_text("")
    return lambda *a, **k: {
        "research": tmp_path, "python": tmp_path, "scorer": tmp_path, "metrics": tmp_path,
        "corpus": tmp_path, "artifact": tmp_path / "artifact.pt", "qa_file": tmp_path,
        "reference": reference, "reference_model": reference / "final_model",
        "reference_config": reference / "config.json",
        "reference_history": reference / "training_history.json",
        "reference_log": reference / "training.log",
    }


def test_the_chunk_counts_can_be_rebuilt_for_a_run_that_did_not_record_them(tmp_path, tiny_model):
    """The branch the 2026-09-07 run actually took: an entry from before the fields existed.

    The rebuild is deterministic in the frame the entry itself records, so it reproduces the
    counts the run used rather than guessing them -- which is why it can stand in for a recorded
    value rather than skipping the check.
    """
    model, tokenizer = tiny_model
    base = tmp_path / "base"
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(10):
        (corpus / f"doc_{i}.txt").write_text(("document %d about anchoring " % i) * 40)

    entry = {
        "corpus": str(corpus), "base_model": str(base), "val_fraction": 0.2,
        "keep_short_whole": True,
        "recipe": {"sequence_length": 64, "seed": 0, "keep_short_whole": True},
    }
    rebuilt = run_recipe.companion_chunk_counts(entry)

    from lfa.corpus import load_corpus
    train, held_out = load_corpus(corpus, tokenizer, max_length=64, val_fraction=0.2, seed=0,
                                  keep_short_whole=True)
    assert rebuilt == {"train_chunks": len(train), "val_chunks": len(held_out)}
    assert rebuilt["train_chunks"] > 0 and rebuilt["val_chunks"] > 0

    # ...and a recorded entry short-circuits it, which is what future runs will take.
    recorded = {**entry, "n_train_chunks": 7, "n_val_chunks": 2}
    assert run_recipe.companion_chunk_counts(recorded) == {"train_chunks": 7, "val_chunks": 2}
