"""The whole pipeline on the card, at a trial frame: build, prepare with the supplement, train,
evaluate, fuse; then a second workspace that must reuse the stored artifact. Once per model:

    pytest tests/test_pipeline_gpu.py -m gpu -q -k 0.6B
    pytest tests/test_pipeline_gpu.py -m gpu -q -k 1.7B

A model with no bundled recipe yet runs on an uncalibrated copy of one: pass ``--recipe <path>``
to `init` and `prepare-domain`, the path written by

    dataclasses.replace(Recipe.load("qwen3-0.6b"), name="<model>-trial", model_id=<model>,
                        calibrated_artifact="uncalibrated").save(<path>)

and expect `train` to say "calibrated against 'uncalibrated'" instead of "different frame".
"""
import json

import pytest

from lfa.cli import main
from lfa.recipe import Recipe

pytestmark = pytest.mark.gpu

#: Each model runs on its bundled recipe, adopted by `--model`. Both recipes are calibrated
#: against the model's own self-generated artifact at the recorded frame, and the trial build is
#: off that frame, so `train` says "different frame".
TRIAL_FRAME_NOTE = "different frame"


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"], ids=["0.6B", "1.7B"])
def test_the_pipeline_end_to_end_at_a_trial_frame(model, tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("LFA_ARTIFACT_STORE", str(tmp_path / "store"))
    src = tmp_path / "src"
    src.mkdir()
    for i in range(12):
        (src / f"doc{i}.txt").write_text(
            f"Document {i}. " + "The lighthouse keeper logged the tides and the weather. " * 60)
    ws = tmp_path / "ws"

    assert main(["init", str(ws), "--model", model, "--artifact", "self-generated",
                 "--n-raw", "60", "--max-new-tokens", "128"]) == 0
    assert json.loads((ws / "workspace.json").read_text())["recipe"] == Recipe.bundled_for(model)
    assert main(["prepare-domain", str(src), "--out", str(tmp_path / "corpus"),
                 "--supplement", "--model", model]) == 0
    assert list((tmp_path / "corpus.supplement").glob("*/supplement.jsonl"))
    with caplog.at_level("INFO"):
        assert main(["train", "--workspace", str(ws), "--corpus", str(tmp_path / "corpus"),
                     "--epochs", "1"]) == 0
    assert any("Supplement reused" in r.message for r in caplog.records)
    assert any(TRIAL_FRAME_NOTE in r.message for r in caplog.records)   # the trial-frame note
    assert main(["evaluate", "--workspace", str(ws), "--n-windows", "5"]) == 0
    assert main(["fuse", "--workspace", str(ws)]) == 0

    caplog.clear()
    with caplog.at_level("INFO"):
        assert main(["init", str(tmp_path / "ws2"), "--model", model, "--artifact",
                     "self-generated", "--n-raw", "60", "--max-new-tokens", "128"]) == 0
    assert any("Reused the self-generated artifact" in r.message for r in caplog.records)
