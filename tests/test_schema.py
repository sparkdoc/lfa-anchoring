import pytest, torch
from lfa.artifact.schema import load_artifact, save_artifact, validate_against_model, ArtifactModelMismatch, parse_site_key
from lfa.adapters import get_adapter

def test_roundtrip_quantized(tiny_artifact, tmp_path):
    params, _ = tiny_artifact
    p = save_artifact(params, tmp_path / "a.pt", quantize=True)
    back = load_artifact(p)
    assert torch.allclose(back["1_pre_mlp"]["pca_components"].float(), params["1_pre_mlp"]["pca_components"].float(), atol=0.03)
    assert back["__meta__"]["model_id"] == "tiny"

def test_parse_keys():
    assert parse_site_key("12_pre_mlp") == (12, "pre_mlp") and parse_site_key("__meta__") is None

def test_validate_mismatch(tiny_artifact, tiny_model):
    params, _ = tiny_artifact; model, _ = tiny_model
    validate_against_model(params, model, get_adapter(model), model_id="tiny")           # ok
    bad = dict(params); bad["0_pre_mlp"] = dict(params["0_pre_mlp"], mean=torch.zeros(64))
    with pytest.raises(ArtifactModelMismatch, match="64"):
        validate_against_model(bad, model, get_adapter(model), model_id="tiny")
    with pytest.raises(ArtifactModelMismatch, match="other"):
        validate_against_model(params, model, get_adapter(model), model_id="other")
