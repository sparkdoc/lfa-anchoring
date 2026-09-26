import pytest, torch
from lfa.artifact.schema import load_artifact, make_meta, save_artifact, validate_against_model, ArtifactModelMismatch, parse_site_key
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


def test_make_meta_says_who_built_the_statistics_and_who_wrote_the_block():
    """`built_with` is provenance for the STATISTICS, so a meta block added to an artifact this
    package did not build must be able to say so (RELEASING.md step 1 does). `lfa_version` records
    who wrote the block either way, so the two never have to answer the same question."""
    from lfa import __version__

    default = make_meta("m", 8, 2, ["pre_mlp"], 100)
    assert default["built_with"] == "lfa-anchoring"

    added = make_meta("m", 8, 2, ["pre_mlp"], 100,
                      built_with="research code; meta block added by lfa-anchoring")
    assert added["built_with"] == "research code; meta block added by lfa-anchoring"
    assert added["lfa_version"] == default["lfa_version"] == __version__


def test_meta_carries_provenance_when_given_and_none_otherwise():
    from lfa.artifact.schema import SELF_GENERATED, make_meta

    plain = make_meta("m", 32, 2, ["pre_qkv"], 10)
    assert plain["provenance"] is None and plain["corpus_sha256"] is None

    selfgen = make_meta("m", 32, 2, ["pre_qkv"], 10, provenance=SELF_GENERATED,
                        corpus_sha256="ab" * 32)
    assert selfgen["provenance"] == "self-generated"
    assert selfgen["corpus_sha256"] == "ab" * 32
