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


def test_make_meta_says_this_package_built_the_statistics():
    """`built_with` is what an artifact is accepted on, so it is not a parameter: every meta block
    this package writes says it built the statistics. `lfa_version` records which version."""
    from lfa import __version__

    meta = make_meta("m", 8, 2, ["pre_mlp"], 100)
    assert meta["built_with"] == "lfa-anchoring"
    assert meta["lfa_version"] == __version__
    with pytest.raises(TypeError):
        make_meta("m", 8, 2, ["pre_mlp"], 100, built_with="research code")


def test_meta_carries_provenance_when_given_and_none_otherwise():
    from lfa.artifact.schema import SELF_GENERATED, make_meta

    plain = make_meta("m", 32, 2, ["pre_qkv"], 10)
    assert plain["provenance"] is None and plain["corpus_sha256"] is None

    selfgen = make_meta("m", 32, 2, ["pre_qkv"], 10, provenance=SELF_GENERATED,
                        corpus_sha256="ab" * 32)
    assert selfgen["provenance"] == "self-generated"
    assert selfgen["corpus_sha256"] == "ab" * 32


def test_meta_records_the_layer_group_size():
    from lfa.artifact.schema import make_meta

    plain = make_meta("m", 32, 2, ["pre_qkv"], 10)
    assert "layer_group_size" in plain and plain["layer_group_size"] is None
    assert make_meta("m", 32, 2, ["pre_qkv"], 10, layer_group_size=7)["layer_group_size"] == 7


def test_make_meta_records_the_self_generated_frame():
    meta = make_meta("m", 8, 2, ["pre_qkv"], 10, provenance="self-generated",
                     selfgen_frame={"n_raw": 60})
    assert meta["selfgen_frame"] == {"n_raw": 60}
    assert "selfgen_frame" not in make_meta("m", 8, 2, ["pre_qkv"], 10)


# ------------------------------------------------------------------ only artifacts built here
# The package accepts only p(h) artifacts it built itself: every file it writes carries a meta
# block saying `built_with: lfa-anchoring` and diagonal mixture heads, so either missing is a
# file built somewhere else, refused with a sentence that ends in what to do instead.

def _without_meta(params):
    return {key: value for key, value in params.items() if key != "__meta__"}


def _built_with(params, who):
    return dict(params, __meta__=dict(params["__meta__"], built_with=who))


def _with_full_covariance(params):
    entry = dict(params["1_pre_mlp"], gmm_covariance_type="full")
    return dict(params, **{"1_pre_mlp": entry})


@pytest.mark.parametrize("make", [_without_meta, lambda p: _built_with(p, "research code"),
                                  _with_full_covariance],
                         ids=["no-meta", "foreign-built_with", "full-covariance"])
def test_load_artifact_refuses_a_file_this_package_did_not_build(tiny_artifact, tmp_path, make):
    params, _ = tiny_artifact
    path = tmp_path / "elsewhere.pt"
    torch.save(make(params), path)
    with pytest.raises(ValueError, match=r"not built by lfa-anchoring.*lfa build-artifact`\.$"):
        load_artifact(path)


@pytest.mark.parametrize("make", [_without_meta, lambda p: _built_with(p, "research code"),
                                  _with_full_covariance],
                         ids=["no-meta", "foreign-built_with", "full-covariance"])
def test_validate_against_model_refuses_an_artifact_this_package_did_not_build(
        tiny_artifact, tiny_model, make):
    """No early return for a meta-less artifact: a missing meta is an error here too."""
    params, _ = tiny_artifact; model, _ = tiny_model
    with pytest.raises(ValueError, match="not built by lfa-anchoring"):
        validate_against_model(make(params), model, get_adapter(model))


def test_the_refusal_names_who_built_a_foreign_file(tiny_artifact, tmp_path):
    params, _ = tiny_artifact
    path = tmp_path / "elsewhere.pt"
    torch.save(_built_with(params, "research code"), path)
    with pytest.raises(ValueError, match="'research code'"):
        load_artifact(path)


# ------------------------------------------------------------------ the artifact format version
# `format_version` names the stored layout. An artifact is read by every copy and every release of
# this package that reads its format; a file with no `format_version` predates the field and is
# format 1, which is the layout every release has written.

def test_make_meta_records_the_artifact_format():
    from lfa.artifact import ARTIFACT_FORMAT as exported
    from lfa.artifact.schema import ARTIFACT_FORMAT, SUPPORTED_ARTIFACT_FORMATS

    assert exported == ARTIFACT_FORMAT == 1
    assert ARTIFACT_FORMAT in SUPPORTED_ARTIFACT_FORMATS
    assert make_meta("m", 8, 2, ["pre_mlp"], 100)["format_version"] == ARTIFACT_FORMAT


def _with_format(params, value):
    return dict(params, __meta__=dict(params["__meta__"], format_version=value))


def test_a_meta_with_no_format_version_is_format_1_and_loads(tiny_artifact, tmp_path):
    params, _ = tiny_artifact
    meta = {key: value for key, value in params["__meta__"].items() if key != "format_version"}
    path = tmp_path / "earlier.pt"
    torch.save(dict(params, __meta__=meta), path)
    assert "format_version" not in load_artifact(path)["__meta__"]    # read as is, not rewritten


_NEXT_FORMAT = object()     # ARTIFACT_FORMAT + 1, resolved inside the test


@pytest.mark.parametrize("value, shown", [(_NEXT_FORMAT, None), ("1", "'1'"), (1.0, "1.0"),
                                          (True, "True"), (None, "None")],
                         ids=["next-integer", "string", "float", "bool", "none"])
def test_an_unknown_format_is_refused_naming_both_formats(tiny_artifact, tmp_path, value, shown):
    """A later layout, or a value that is not an integer at all, is refused rather than misread:
    one sentence naming both formats and ending in what to do."""
    from lfa.artifact.schema import ARTIFACT_FORMAT, ForeignArtifact

    params, _ = tiny_artifact
    if value is _NEXT_FORMAT:
        value = ARTIFACT_FORMAT + 1
        shown = str(value)
    path = tmp_path / "later.pt"
    torch.save(_with_format(params, value), path)
    with pytest.raises(ForeignArtifact) as refusal:
        load_artifact(path)
    message = str(refusal.value)
    assert f"uses artifact format {shown}," in message
    assert f"reads format {ARTIFACT_FORMAT}:" in message
    assert "--artifact self-generated --rebuild" in message and "lfa build-artifact" in message
    assert message.endswith("or use the lfa-anchoring release it was built with.")
    assert "\n" not in message


def test_lfa_version_is_a_record_and_never_checked(tiny_artifact, tmp_path):
    params, _ = tiny_artifact
    path = tmp_path / "other_release.pt"
    torch.save(dict(params, __meta__=dict(params["__meta__"], lfa_version="0.0.1")), path)
    assert load_artifact(path)["__meta__"]["lfa_version"] == "0.0.1"
