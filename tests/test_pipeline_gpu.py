"""The whole pipeline on the card, at a trial frame: build, prepare with the supplement, train,
evaluate, fuse; then a second workspace that must reuse the stored artifact. Once per model:

    pytest tests/test_pipeline_gpu.py -m gpu -q -k 0.6B
    pytest tests/test_pipeline_gpu.py -m gpu -q -k 1.7B
"""
import dataclasses

import pytest

from lfa.cli import main
from lfa.recipe import Recipe

pytestmark = pytest.mark.gpu


def _recipe_for(model: str, tmp_path) -> tuple[list[str], str]:
    """The ``--recipe`` arguments the model's commands take, and the calibration note `train`
    must log for it.

    Qwen3-0.6B runs on its bundled recipe, whose lambda is calibrated against a self-generated
    artifact at a recorded frame; the trial build is off that frame, so `train` says
    "different frame". No recipe is bundled for Qwen3-1.7B, so it runs on a copy of the 0.6B one
    naming 1.7B and marked ``calibrated_artifact: uncalibrated`` (docs/adding-a-model.md), and
    `train` says instead that lambda was calibrated against another artifact.
    """
    if model == "Qwen/Qwen3-0.6B":
        return [], "different frame"
    copy = dataclasses.replace(Recipe.load("qwen3-0.6b"), name="qwen3-1.7b-trial",
                               model_id=model, calibrated_artifact="uncalibrated")
    path = copy.save(tmp_path / "qwen3-1.7b-trial.yaml")
    return ["--recipe", str(path)], "calibrated against 'uncalibrated'"


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"], ids=["0.6B", "1.7B"])
def test_the_pipeline_end_to_end_at_a_trial_frame(model, tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("LFA_ARTIFACT_STORE", str(tmp_path / "store"))
    recipe_args, calibration_note = _recipe_for(model, tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    for i in range(12):
        (src / f"doc{i}.txt").write_text(
            f"Document {i}. " + "The lighthouse keeper logged the tides and the weather. " * 60)
    ws = tmp_path / "ws"

    assert main(["init", str(ws), "--model", model, "--artifact", "self-generated",
                 "--n-raw", "60", "--max-new-tokens", "128", *recipe_args]) == 0
    assert main(["prepare-domain", str(src), "--out", str(tmp_path / "corpus"),
                 "--supplement", "--model", model, *recipe_args]) == 0
    assert list((tmp_path / "corpus.supplement").glob("*/supplement.jsonl"))
    with caplog.at_level("INFO"):
        assert main(["train", "--workspace", str(ws), "--corpus", str(tmp_path / "corpus"),
                     "--epochs", "1"]) == 0
    assert any("Supplement reused" in r.message for r in caplog.records)
    assert any(calibration_note in r.message for r in caplog.records)   # the trial-frame note
    assert main(["evaluate", "--workspace", str(ws), "--n-windows", "5"]) == 0
    assert main(["fuse", "--workspace", str(ws)]) == 0

    caplog.clear()
    with caplog.at_level("INFO"):
        assert main(["init", str(tmp_path / "ws2"), "--model", model, "--artifact",
                     "self-generated", "--n-raw", "60", "--max-new-tokens", "128",
                     *recipe_args]) == 0
    assert any("Reused the self-generated artifact" in r.message for r in caplog.records)
