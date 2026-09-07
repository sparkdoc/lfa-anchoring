"""The acceptance test: the shipped LFA recipe lands on the paper's published operating point.

This is opt-in (``pytest -m acceptance``) because it is roughly an hour of GPU time on one
RTX 3090 and it reads gigabytes that the companion does not distribute -- the Chalmers corpus,
the p(h) artifact and the held-out Q&A set all live in the research checkout. Without them the
test skips, naming what is missing.

What it asserts is what ``run_recipe.py`` measures with the research code's own scorer, because that is
the instrument the published numbers came off:

* domain perplexity on the held-out chat-formatted Q&A set (the paper's 8.76 at seed 42, 8.83 at
  seed 1337), and
* WikiText-2 drift over the full test split against the base model's 18.18 (the paper's -10.0 %).

The bands live in ``expected.json`` beside this file, together with what each tolerance covers.
They are not to be widened to accommodate a run: a number outside them is a finding.

One frame setting is deliberate and is passed explicitly rather than taken from the recipe.
``keep_short_whole=False`` reproduces the research loader, under which a document shorter than
the epoch's random chunk offset is dropped from that epoch. The bundled recipe ships ``True``,
which is the better default; but it is a *frame* field -- realized Q&A exposure differs about
threefold between the two settings, and lambda is coupled to corpus composition -- so a run under
one is not comparable with a number measured under the other. The published points were measured
under ``False``, so the acceptance run uses ``False``.

Set ``LFA_ACCEPTANCE_OUT`` to a directory to keep (or reuse) the run: ``run_recipe.py`` will not
retrain a workspace that already carries a trained stage, nor rescore a checkpoint it has already
scored, so a re-run against an existing directory only re-checks the band.
"""

from __future__ import annotations

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
def test_shipped_recipe_reproduces_the_published_operating_point(tmp_path):
    research = os.environ.get("LFA_RESEARCH_ROOT") or str(run_recipe.DEFAULT_RESEARCH_ROOT)
    try:
        run_recipe.resolve_inputs(research)
    except run_recipe.MissingInput as missing:
        pytest.skip(str(missing))

    out = Path(os.environ.get("LFA_ACCEPTANCE_OUT") or (tmp_path / "acceptance"))
    assert run_recipe.main(["--the research code", research, "--out", str(out)]) == 0

    results = json.loads((out / "results.json").read_text())
    # Reported whatever the verdict, so a failure carries its numbers rather than just a name.
    print(run_recipe.format_checks(results["checks"]))

    for check in results["checks"]:
        low, high = check["band"]
        assert check["ok"], (
            f"{check['name']}: measured {check['measured']}, outside the published band "
            f"{low:.4f}..{high:.4f} (reference {check['reference']}). Do not widen the band -- "
            f"compare {out / 'workspace'}'s run config and per-epoch losses against the research "
            "run gmm_r32_lam100000_f0.13 before concluding anything."
        )


@pytest.mark.acceptance
def test_expected_band_is_the_published_point():
    """The band file itself: a guard against an accidental loosening going unnoticed.

    It asserts the *published* values and the tolerances the controller set, so widening a
    tolerance to rescue a run fails here as well as in review.
    """
    expected = json.loads(run_recipe.EXPECTED_FILE.read_text())
    assert expected["domain_ppl"] == {"ref": 8.76, "ref_seed2": 8.83, "tol_rel": 0.02}
    assert expected["seed_ppl"] == {"ref": 16.36, "base": 18.18, "drift_pct_ref": -10.0,
                                    "tol_abs_pct": 1.5}


@pytest.mark.acceptance
@pytest.mark.parametrize(
    "domain_ppl, seed_ppl, domain_ok, seed_ok",
    [
        (8.7625, 16.3585, True, True),      # the published seed-42 point
        (8.8293, 16.2389, True, True),      # the published seed-1337 point
        (8.77, 16.36, True, True),          # the mu fast path's expected drift
        (9.80, 17.14, False, False),        # e20: the judge-optimal dose, not this one
        (12.32, 15.85, False, False),       # e5: under-dosed
    ],
)
def test_band_accepts_the_published_points_and_rejects_neighbouring_doses(
    domain_ppl, seed_ppl, domain_ok, seed_ok
):
    """The band is tight enough to be a test: adjacent doses of the same run fall outside it."""
    domain, seed = run_recipe.check_band(domain_ppl, seed_ppl)
    assert domain["ok"] is domain_ok
    assert seed["ok"] is seed_ok


@pytest.mark.acceptance
def test_a_missing_input_is_named_rather_than_guessed(tmp_path):
    with pytest.raises(run_recipe.MissingInput, match="the research code checkout"):
        run_recipe.resolve_inputs(tmp_path / "nowhere")
