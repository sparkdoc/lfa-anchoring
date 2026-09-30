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
