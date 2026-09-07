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
fifteen-point curves. Then two SANITY checks -- the research instrument's domain and WikiText-2
numbers, one draw each -- which say the run produced a domain-adapted model but cannot be the
criterion, because the anchor is drawn from independent RNG streams on the two sides and two full
runs are two draws of a stochastic objective.

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
@pytest.mark.gpu
def test_the_recipe_lands_where_the_research_code_lands(tmp_path):
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
        assert check["ok"], (
            f"{check['name']}: measured {check['measured']}, outside {check['tolerance']}. Do not "
            f"widen the tolerance -- compare {out / 'workspace'}'s run config and per-epoch "
            f"losses against {results['reference']['run']} before concluding anything."
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
    assert expected["domain_ppl"] == {"tol_rel": 0.02}
    assert expected["seed_drift"] == {"tol_abs_pct": 1.0}
    assert set(expected) == {"_comment", "optimizer_steps", "corpus_counts", "content_curve",
                             "held_out_curve", "domain_ppl", "seed_drift"}


def test_the_tolerance_file_says_which_check_is_the_criterion():
    """The reasoning is the durable part: the next reader must not repeat it from scratch."""
    comment = " ".join(json.loads(run_recipe.EXPECTED_FILE.read_text())["_comment"]).lower()
    assert "independent rng streams" in comment          # the real dominant divergence term
    assert "not the criterion" in comment                # what the perplexities are for
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
    return [check["name"] for check in checks if not check["ok"]]


def test_a_run_that_matches_the_reference_passes_every_check():
    assert _failed(_checks()) == []
    assert [check["kind"] for check in _checks()] == [
        "frame", "criterion", "criterion", "criterion", "criterion", "sanity", "sanity"]


def test_the_two_instrument_checks_are_labelled_as_sanity_checks_not_the_criterion():
    """Whoever reads the output must not mistake the coarse check for the evidence."""
    sanity = [check["name"] for check in _checks() if check["kind"] == "sanity"]
    assert len(sanity) == 2
    assert all(name.startswith("SANITY (one draw, not the criterion)") for name in sanity)
    assert "SANITY" in run_recipe.format_checks(_checks())


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


# --- the sanity checks ------------------------------------------------------------------------

@pytest.mark.parametrize("domain, seed, failing", [
    # 3 % over the reference's domain perplexity: outside the 2 % tolerance.
    (9.17, 16.61, "SANITY (one draw, not the criterion): domain direct-QA perplexity"),
    # +1.5 pp of WikiText-2 drift (16.61 -> 16.88 against base 18.18): outside 1.0 pp.
    (8.90, 16.88, "SANITY (one draw, not the criterion): WikiText-2 drift vs base 18.18"),
])
def test_a_run_that_drifts_on_an_instrument_fails_that_instrument_only(domain, seed, failing):
    assert _failed(_checks(domain=domain, seed=seed)) == [failing]


def test_two_runs_of_different_configurations_are_not_evidence_about_either():
    """A frame difference fails on its own, whatever everything else came out as."""
    frame = [{"field": "lora_rank", "reference": 32, "companion": 16}]
    checks = _checks(frame=frame)
    assert checks[0]["ok"] is False and checks[0]["kind"] == "frame"
    assert _failed(checks) == ["the two runs are the same configuration"]
    assert "lora_rank" in run_recipe.format_checks(checks)


def test_a_missing_measurement_fails_rather_than_passing_quietly():
    failed = _failed(_checks(domain=None, seed=None))
    assert failed == ["SANITY (one draw, not the criterion): domain direct-QA perplexity",
                      "SANITY (one draw, not the criterion): WikiText-2 drift vs base 18.18"]


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


def test_a_reused_run_under_a_different_frame_is_refused(tmp_path):
    """The reuse path re-scores an existing run; it must first check it is the same run."""
    from lfa import Recipe

    recipe = dataclasses.replace(Recipe.load("qwen3-0.6b"), val_fraction=0.1)
    entry = {
        "corpus": str(tmp_path / "corpus"),
        "keep_short_whole": True,
        "val_fraction": 0.1,
        "recipe": dataclasses.asdict(recipe),
    }

    run_recipe._refuse_a_different_run(entry, recipe, tmp_path / "corpus", True)

    with pytest.raises(run_recipe.ReusedRunDiffers, match="keep_short_whole"):
        run_recipe._refuse_a_different_run(entry, recipe, tmp_path / "corpus", False)
    with pytest.raises(run_recipe.ReusedRunDiffers, match="corpus"):
        run_recipe._refuse_a_different_run(entry, recipe, tmp_path / "other", True)
    with pytest.raises(run_recipe.ReusedRunDiffers, match="recipe_digest"):
        run_recipe._refuse_a_different_run(
            entry, dataclasses.replace(recipe, lambda_qkv=20000.0), tmp_path / "corpus", True)


def test_a_missing_input_is_named_rather_than_guessed(tmp_path):
    with pytest.raises(run_recipe.MissingInput, match="the research code checkout"):
        run_recipe.resolve_inputs(tmp_path / "nowhere")
