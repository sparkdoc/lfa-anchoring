"""The equivalence test: this package trains what the research code trains.

One test here is opt-in (``pytest -m acceptance``) because it is over an hour of GPU time and it
reads gigabytes the companion does not distribute -- the corpus, the p(h) artifact, the held-out
Q&A set and the the research code reference run all live in the research checkout. Without them it skips,
naming what is missing. The rest of this file is cheap, runs in the default suite, and checks the
comparison logic itself: that the tolerance file says what it is supposed to say, that the
comparison accepts a matching run and rejects a drifting one, and that a missing input is named.

What the GPU test asserts is what ``run_recipe.py`` measures with the research code's own scorer, pointed
at BOTH checkpoints in turn:

* the two runs are the same configuration (their frames are compared field by field first),
* domain perplexity on the held-out chat-formatted Q&A set, within a relative tolerance,
* WikiText-2 drift over the full test split, within an absolute one, and
* the per-epoch content loss over the first five epochs.

The tolerances live in ``expected.json`` beside this file, which is the one place they are
described. They are not to be widened to accommodate a run: a number outside them is a finding.

Set ``LFA_ACCEPTANCE_OUT`` to a directory to keep (or reuse) the run: ``run_recipe.py`` will not
retrain a workspace that already carries a trained stage, nor rescore a checkpoint it has already
scored -- but it refuses to reuse either if the frame it finds is not the frame being asked for.
"""

from __future__ import annotations

import dataclasses
import json
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
            f"widen the tolerance -- compare {out / 'workspace'}'s run config and per-epoch losses "
            f"against {results['reference']['run']} before concluding anything."
        )


def test_the_tolerance_file_carries_exactly_the_agreed_tolerances():
    """The tolerances themselves: a guard against a loosening going unnoticed.

    ``expected.json`` holds tolerances and nothing else -- the reference's measured values are
    written out by the harness, not written down here -- so this is the whole of what could be
    quietly widened to rescue a run.
    """
    expected = json.loads(run_recipe.EXPECTED_FILE.read_text())
    assert expected["domain_ppl"] == {"tol_rel": 0.02}
    assert expected["seed_drift"] == {"tol_abs_pct": 1.0}
    assert expected["content_curve"] == {"epochs": 5, "tol_rel": 0.005}
    assert set(expected) == {"_comment", "domain_ppl", "seed_drift", "content_curve"}


def _checks(domain, seed, curve, reference_domain=8.90, reference_seed=16.61,
            reference_curve=(2.62, 2.55, 2.50, 2.46, 2.43), base=18.18, frame=()):
    return run_recipe.check_equivalence(
        companion={"domain_direct_ppl": domain, "seed_ppl": seed},
        reference={"domain_direct_ppl": reference_domain, "seed_ppl": reference_seed},
        base_seed_ppl=base,
        companion_curve=list(curve),
        reference_curve=list(reference_curve),
        frame=list(frame),
    )


MATCHING_CURVE = (2.62, 2.55, 2.50, 2.46, 2.43)


def test_a_run_that_matches_the_reference_passes_every_check():
    checks = _checks(8.9012, 16.6134, MATCHING_CURVE)
    assert [check["ok"] for check in checks] == [True, True, True, True]


@pytest.mark.parametrize("domain, seed, curve, failing", [
    # 3 % over the reference's domain perplexity: outside the 2 % tolerance.
    (9.17, 16.61, MATCHING_CURVE, "domain direct-QA perplexity vs the reference run"),
    # +1.5 pp of WikiText-2 drift (16.61 -> 16.88 against base 18.18): outside 1.0 pp.
    (8.90, 16.88, MATCHING_CURVE, "WikiText-2 drift vs the reference run (base 18.18)"),
    # epoch 3 is 1 % off the reference's: outside 0.5 %.
    (8.90, 16.61, (2.62, 2.55, 2.525, 2.46, 2.43),
     "content loss, epochs 1-5, vs the reference curve"),
    # a curve that stops before epoch 5 is not a pass by absence.
    (8.90, 16.61, (2.62, 2.55, 2.50), "content loss, epochs 1-5, vs the reference curve"),
])
def test_a_run_that_drifts_from_the_reference_fails_the_axis_it_drifts_on(domain, seed, curve,
                                                                         failing):
    failed = [check["name"] for check in _checks(domain, seed, curve) if not check["ok"]]
    assert failed == [failing]


def test_two_runs_of_different_configurations_are_not_evidence_about_either():
    """A frame difference fails on its own, whatever the perplexities came out as."""
    frame = [{"field": "lora_rank", "reference": 32, "companion": 16}]
    checks = _checks(8.9012, 16.6134, MATCHING_CURVE, frame=frame)
    assert checks[0]["ok"] is False
    assert [check["ok"] for check in checks[1:]] == [True, True, True]
    assert "lora_rank" in run_recipe.format_checks(checks)


def test_a_missing_measurement_fails_rather_than_passing_quietly():
    checks = _checks(None, None, MATCHING_CURVE)
    assert [check["ok"] for check in checks] == [True, False, False, True]


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
