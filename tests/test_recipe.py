"""The recipe layer: the shipped operating point, its stage-2 lift, and its coupling warnings.

A recipe is a *joint* operating point, so what these tests check is that the bundled Qwen3-0.6B
file still carries exactly the shipped point, that turning it into a `TrainConfig` preserves the
two settings a reader is most likely to lose (the short-document rule and the held-out split),
and that
a run which departs from the calibrated rank or artifact is told that lambda no longer means what
it meant.
"""

import dataclasses

import pytest
import yaml

from lfa import Recipe
from lfa.recipe import BUNDLED_DIR, RECORDED_SELF_GENERATED_FRAME, SELF_GENERATED_REFERENCE


SHIPPED = dict(
    name="qwen3-0.6b", model_id="Qwen/Qwen3-0.6B", artifact="self-generated",
    lora_rank=32, lora_alpha=64, freeze_embed=True, full_weight=False,
    lambda_qkv=100000.0, lambda_mlp=100000.0, mu=0.05,
    anchor_end_ratio=0.1, anchor_schedule="cosine", n_anchor_samples=16,
    epochs=15, checkpoint_mode="rolling", checkpoint_every=5,
    learning_rate=3e-4, lr_schedule="cosine", batch_size=6, gradient_accumulation_steps=1,
    warmup_steps=50, weight_decay=0.01, sequence_length=512, seed=42, keep_short_whole=True,
    val_fraction=0.1,
    stage2_lambda_multiplier=3.0, calibrated_rank=32,
    calibrated_artifact="self-generated",
    self_generated_frame={"n_raw": 2500, "n_chat": 0, "max_new_tokens": 2048,
                          "max_samples": 600000, "gmm_k": 32, "pca_variance": 0.95},
    supplement_fraction=0.13,
)


def test_bundled_recipe_is_the_published_operating_point():
    recipe = Recipe.load("qwen3-0.6b")
    assert dataclasses.asdict(recipe) == SHIPPED


#: The Qwen3-1.7B point: the 0.6B recipe's frame with this model's own measured lambda.
SHIPPED_1_7B = dict(SHIPPED, name="qwen3-1.7b", model_id="Qwen/Qwen3-1.7B",
                    lambda_qkv=1000000.0, lambda_mlp=1000000.0)


def test_the_qwen3_1_7b_recipe_is_its_calibrated_operating_point():
    """lambda 1e6 and the 3x stage-2 multiplier were measured on this model; every other value
    is the 0.6B recipe's frame."""
    assert dataclasses.asdict(Recipe.load("qwen3-1.7b")) == SHIPPED_1_7B


@pytest.mark.parametrize("model_id,name", [("Qwen/Qwen3-0.6B", "qwen3-0.6b"),
                                           ("Qwen/Qwen3-1.7B", "qwen3-1.7b")])
def test_each_bundled_model_has_its_recipe(model_id, name):
    recipe = Recipe.load(name)
    assert Recipe.bundled_for(model_id) == name and recipe.model_id == model_id
    assert recipe.calibrated_artifact == SELF_GENERATED_REFERENCE
    assert recipe.self_generated_frame == RECORDED_SELF_GENERATED_FRAME
    assert recipe.calibrated_rank == recipe.lora_rank == 32
    assert recipe.lambda_qkv == recipe.lambda_mlp > 0


def test_the_qwen3_1_7b_header_says_how_its_lambda_was_chosen_and_that_it_does_not_port():
    comments = "\n".join(line for line in (BUNDLED_DIR / "qwen3-1.7b.yaml").read_text().splitlines()
                         if line.lstrip().startswith("#"))
    for phrase in ("Qwen3-1.7B", "Apache-2.0", "1,000,000", "held-out Darwin", "WikiText-2",
                   "one seed", "RTX 3090", "50,000", "does not port"):
        assert phrase in comments, f"the 1.7B recipe's comment block never mentions {phrase!r}"


def test_bundled_recipe_is_found_from_the_package_not_the_working_directory(tmp_path, monkeypatch):
    """`load` resolves against the installed package, so a wheel and a `cd` both work."""
    monkeypatch.chdir(tmp_path)
    assert Recipe.load("qwen3-0.6b").lambda_qkv == 100000
    assert BUNDLED_DIR.is_dir() and (BUNDLED_DIR / "qwen3-0.6b.yaml").is_file()


def test_unknown_bundled_name_names_what_is_available():
    with pytest.raises(FileNotFoundError, match="qwen3-0.6b"):
        Recipe.load("qwen4-70b")


@pytest.mark.parametrize("name", ["qwen3-0.6b", "qwen3-1.7b"])
def test_yaml_documents_the_couplings_a_reader_has_to_know(name):
    """The file is read by humans before it is read by the loader; the guidance is the point."""
    text = (BUNDLED_DIR / f"{name}.yaml").read_text()
    comments = "\n".join(line for line in text.splitlines() if line.lstrip().startswith("#")).lower()
    for phrase in ("rank", "artifact", "corpus composition", "full-weight", "re-tune"):
        assert phrase in comments, f"the recipe's comment block never mentions {phrase!r}"


def test_the_recipe_carries_no_reproduce_the_old_stream_switch():
    """The loader has one chunking. A recipe field that reproduced a superseded one would be a
    compatibility shim in a file of tuned values, and the tuned values would then depend on it."""
    fields = {field.name for field in dataclasses.fields(Recipe)}
    assert "rotate_offset" not in fields
    assert "rotate_offset" not in (BUNDLED_DIR / "qwen3-0.6b.yaml").read_text()


def test_yaml_says_what_keep_short_whole_chooses_between():
    """Two runs under different settings of it see different text; the file has to say so, and
    has to say it as a property of short documents rather than as a compatibility note."""
    comments = "\n".join(line for line in (BUNDLED_DIR / "qwen3-0.6b.yaml").read_text().splitlines()
                         if line.lstrip().startswith("#")).lower()
    assert "keep_short_whole" in comments
    assert "whole" in comments and "fragment" in comments


# ==============================================================================================
# to_train_config
# ==============================================================================================

def test_stage_one_config_carries_the_recipe_verbatim(tmp_path):
    config = Recipe.load("qwen3-0.6b").to_train_config(1, tmp_path / "stats.pt")

    assert config.lambda_qkv == 100000 and config.lambda_mlp == 100000
    assert config.model_id == "Qwen/Qwen3-0.6B"
    assert config.artifact_path == str(tmp_path / "stats.pt")
    assert (config.mu, config.n_anchor_samples) == (0.05, 16)
    assert (config.anchor_end_ratio, config.anchor_schedule) == (0.1, "cosine")
    assert (config.lora_rank, config.lora_alpha) == (32, 64)
    assert config.use_lora is True and config.full_weight is False and config.freeze_embed is True
    assert (config.learning_rate, config.batch_size, config.gradient_accumulation_steps) == (3e-4, 6, 1)
    assert (config.warmup_steps, config.weight_decay, config.sequence_length, config.seed) == (50, 0.01, 512, 42)


def test_the_schedule_is_a_cosine_over_the_epochs_actually_trained(tmp_path):
    """One knob, and it is laid over `epochs`: there is no horizon that outlives the run."""
    config = Recipe.load("qwen3-0.6b").to_train_config(1, tmp_path / "stats.pt")
    assert config.num_epochs == 15
    assert config.lr_schedule == "cosine"


def test_checkpointing_leaves_a_run_resumable(tmp_path):
    """`rolling` writes `latest_model` + `training_state.pt`, which is what a resume needs."""
    config = Recipe.load("qwen3-0.6b").to_train_config(1, tmp_path / "stats.pt")
    assert config.checkpoint_mode == "rolling"
    assert config.checkpoint_every == 5


def test_stage_two_multiplies_lambda(tmp_path):
    config = Recipe.load("qwen3-0.6b").to_train_config(2, tmp_path / "stats.pt")
    assert config.lambda_qkv == 300000
    assert config.lambda_mlp == 300000


def test_the_stage_two_lift_is_a_level_not_a_compounding_factor(tmp_path):
    """Stage 3 anchors like stage 2: the multiplier says "a later stage", not "per stage"."""
    recipe = Recipe.load("qwen3-0.6b")
    assert recipe.to_train_config(3, tmp_path / "s.pt").lambda_qkv == 300000


def test_stage_below_one_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="stage"):
        Recipe.load("qwen3-0.6b").to_train_config(0, tmp_path / "s.pt")


def test_keep_short_whole_defaults_to_the_recipe_and_can_be_overridden(tmp_path):
    """The override exists so a comparison against the historical research loader can be run."""
    recipe = Recipe.load("qwen3-0.6b")
    assert recipe.to_train_config(1, tmp_path / "s.pt").keep_short_whole is True
    assert recipe.to_train_config(1, tmp_path / "s.pt", keep_short_whole=False).keep_short_whole is False


def test_full_weight_recipe_turns_lora_off(tmp_path):
    recipe = dataclasses.replace(Recipe.load("qwen3-0.6b"), full_weight=True)
    config = recipe.to_train_config(1, tmp_path / "s.pt")
    assert config.full_weight is True and config.use_lora is False


# ==============================================================================================
# warnings
# ==============================================================================================

def _selfgen_meta(model_id, **frame):
    return {"provenance": "self-generated", "model_id": model_id,
            "selfgen_frame": {**RECORDED_SELF_GENERATED_FRAME, "seed": 42, **frame}}


def test_no_warnings_at_the_calibrated_point():
    assert Recipe.load("qwen3-0.6b").warnings(32, "self-generated:abc",
                                              _selfgen_meta("Qwen/Qwen3-0.6B")) == []


def test_a_different_rank_warns_that_lambda_must_be_retuned():
    [warning] = Recipe.load("qwen3-0.6b").warnings(16, "self-generated:abc",
                                                   _selfgen_meta("Qwen/Qwen3-0.6B"))
    assert "rank" in warning and "16" in warning and "32" in warning
    assert "re-tune" in warning.lower()


def test_the_rank_warning_quotes_qwen3_0_6b_s_measured_range_only_for_qwen3_0_6b():
    """The rank-16 range was measured on Qwen3-0.6B; another model's recipe is told to
    re-calibrate, not handed a range that was never measured for it."""
    [small] = Recipe.load("qwen3-0.6b").warnings(16, "self-generated:abc",
                                                 _selfgen_meta("Qwen/Qwen3-0.6B"))
    assert "Qwen3-0.6B" in small and "2e4-5e4" in small
    [large] = Recipe.load("qwen3-1.7b").warnings(16, "self-generated:abc",
                                                 _selfgen_meta("Qwen/Qwen3-1.7B"))
    assert "calibrated at rank 32" in large and "re-tune" in large.lower()
    assert "model-integration-cookbook.md" in large and "§5" in large
    assert "2e4" not in large and "Qwen3-0.6B" not in large


def test_the_rank_warning_quotes_both_lambdas_when_they_differ():
    recipe = dataclasses.replace(Recipe.load("qwen3-0.6b"), lambda_mlp=50000.0)
    [warning] = recipe.warnings(16, "self-generated:abc", _selfgen_meta("Qwen/Qwen3-0.6B"))
    assert "100000" in warning and "50000" in warning


def test_a_different_artifact_warns_on_its_own():
    [warning] = Recipe.load("qwen3-0.6b").warnings(32, "qwen3-0.6b-diagonal")
    assert "artifact" in warning and "qwen3-0.6b-diagonal" in warning


def test_both_couplings_warn_together():
    warnings = Recipe.load("qwen3-0.6b").warnings(8, "qwen3-0.6b-diagonal")
    assert len(warnings) == 2


def test_a_full_weight_recipe_adds_its_own_note():
    recipe = dataclasses.replace(Recipe.load("qwen3-0.6b"), full_weight=True)
    warnings = recipe.warnings(32, "self-generated:abc", _selfgen_meta("Qwen/Qwen3-0.6B"))
    assert len(warnings) == 1
    assert "full-weight" in warnings[0].lower() and "unvalidated" in warnings[0]


def test_the_full_weight_note_quotes_qwen3_0_6b_s_range_only_for_qwen3_0_6b():
    small = dataclasses.replace(Recipe.load("qwen3-0.6b"), full_weight=True)
    [note] = small.warnings(32, "self-generated:abc", _selfgen_meta("Qwen/Qwen3-0.6B"))
    assert "Qwen3-0.6B" in note and "50,000-100,000" in note
    large = dataclasses.replace(Recipe.load("qwen3-1.7b"), full_weight=True)
    [note] = large.warnings(32, "self-generated:abc", _selfgen_meta("Qwen/Qwen3-1.7B"))
    assert "unvalidated" in note and "LoRA" in note
    assert "model-integration-cookbook.md" in note and "§5" in note
    assert "50,000" not in note and "Qwen3-0.6B" not in note


# ==============================================================================================
# save / load
# ==============================================================================================

def test_save_then_load_roundtrips(tmp_path):
    recipe = dataclasses.replace(Recipe.load("qwen3-0.6b"), name="probe", lora_rank=16,
                                 lambda_qkv=20000.0, lambda_mlp=20000.0)
    path = tmp_path / "probe.yaml"
    recipe.save(path)
    assert Recipe.load(path) == recipe


def test_a_saved_recipe_is_plain_readable_yaml(tmp_path):
    path = tmp_path / "probe.yaml"
    Recipe.load("qwen3-0.6b").save(path)
    loaded = yaml.safe_load(path.read_text())
    assert loaded["lambda_qkv"] == 100000 and loaded["lr_schedule"] == "cosine"


def test_an_unknown_field_is_refused_rather_than_ignored(tmp_path):
    path = tmp_path / "probe.yaml"
    path.write_text(yaml.safe_dump({**SHIPPED, "lambda_qvk": 1.0}))
    with pytest.raises(ValueError, match="lambda_qvk"):
        Recipe.load(path)


# ==============================================================================================
# validation
# ==============================================================================================

def _write(tmp_path, **overrides):
    path = tmp_path / "probe.yaml"
    path.write_text(yaml.safe_dump({**SHIPPED, **overrides}))
    return path


def test_a_missing_required_field_is_named(tmp_path):
    path = tmp_path / "probe.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in SHIPPED.items() if k not in ("name", "artifact")}))
    with pytest.raises(ValueError, match="missing required recipe field.*artifact, name"):
        Recipe.load(path)


def test_epochs_must_be_at_least_one(tmp_path):
    with pytest.raises(ValueError, match="epochs"):
        Recipe.load(_write(tmp_path, epochs=0))


def test_an_unknown_lr_schedule_is_refused_at_load(tmp_path):
    with pytest.raises(ValueError, match="lr_schedule"):
        Recipe.load(_write(tmp_path, lr_schedule="one-cycle"))


def test_a_constant_schedule_is_allowed(tmp_path):
    recipe = Recipe.load(_write(tmp_path, lr_schedule="constant"))
    assert recipe.to_train_config(1, tmp_path / "s.pt").lr_schedule == "constant"


def test_the_stage_two_multiplier_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="stage2_lambda_multiplier"):
        Recipe.load(_write(tmp_path, stage2_lambda_multiplier=0))


def test_lora_rank_must_be_at_least_one(tmp_path):
    with pytest.raises(ValueError, match="lora_rank"):
        Recipe.load(_write(tmp_path, lora_rank=0))


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_val_fraction_must_leave_something_to_train_on(tmp_path, bad):
    with pytest.raises(ValueError, match="val_fraction"):
        Recipe.load(_write(tmp_path, val_fraction=bad))


def test_the_shipped_point_holds_a_tenth_of_the_documents_out(tmp_path):
    """A tenth of the corpus is never trained on, so the domain perplexity a stage reports is a
    held-out measurement; a recipe that trained on everything would report a fit under the same
    name."""
    recipe = Recipe.load("qwen3-0.6b")
    assert recipe.val_fraction == 0.1
    assert yaml.safe_load((BUNDLED_DIR / "qwen3-0.6b.yaml").read_text())["val_fraction"] == 0.1
    assert recipe.to_train_config(1, tmp_path / "s.pt").val_fraction == 0.1


def _recipe(**overrides):
    """The bundled Qwen3 point with fields overridden (it is calibrated by declaration)."""
    return dataclasses.replace(Recipe.load("qwen3-0.6b"), **overrides)


def test_supplement_fraction_is_validated_and_defaults_to_the_measured_frame():
    assert _recipe().supplement_fraction == 0.13
    with pytest.raises(ValueError, match="supplement_fraction"):
        _recipe(supplement_fraction=1.0)


# ==============================================================================================
# the self-generated calibration
# ==============================================================================================

def test_the_bundled_recipe_is_calibrated_against_the_self_generated_artifact():
    recipe = Recipe.load("qwen3-0.6b")
    assert recipe.artifact == SELF_GENERATED_REFERENCE
    assert recipe.calibrated_artifact == SELF_GENERATED_REFERENCE
    assert recipe.self_generated_frame == RECORDED_SELF_GENERATED_FRAME


@pytest.mark.parametrize("name", ["qwen3-0.6b", "qwen3-1.7b"])
def test_a_self_generated_artifact_at_the_recorded_frame_is_silent(name):
    recipe = Recipe.load(name)
    meta = _selfgen_meta(recipe.model_id)
    assert recipe.warnings(recipe.calibrated_rank, "self-generated:abc", meta) == []


@pytest.mark.parametrize("name", ["qwen3-0.6b", "qwen3-1.7b"])
def test_a_trial_frame_names_the_fields_that_differ(name):
    recipe = Recipe.load(name)
    meta = _selfgen_meta(recipe.model_id, n_raw=60, max_new_tokens=128)
    notes = recipe.warnings(recipe.calibrated_rank, "self-generated:abc", meta)
    assert len(notes) == 1
    assert "n_raw 60 (calibrated at 2500)" in notes[0]
    assert "max_new_tokens 128 (calibrated at 2048)" in notes[0]


def test_a_self_generated_meta_with_no_frame_is_off_frame_in_every_field():
    """Every self-generated build records its frame, so a meta without one is not a separate
    case: it is told the frame differs, field by field, like any other off-frame build."""
    recipe = Recipe.load("qwen3-0.6b")
    meta = {"provenance": "self-generated", "model_id": recipe.model_id}
    notes = recipe.warnings(recipe.calibrated_rank, "self-generated:abc", meta)
    assert len(notes) == 1
    assert "built at a different frame" in notes[0]
    for key, value in recipe.self_generated_frame.items():
        assert f"{key} None (calibrated at {value!r})" in notes[0]


def test_another_models_self_generated_artifact_is_a_mismatch():
    recipe = Recipe.load("qwen3-0.6b")
    notes = recipe.warnings(32, "x", _selfgen_meta("other/model"))
    assert "'other/model'" in notes[0]


def test_a_real_text_artifact_against_a_self_generated_calibration_is_noted():
    recipe = Recipe.load("qwen3-0.6b")
    notes = recipe.warnings(32, "/data/mine.pt", {"model_id": recipe.model_id})
    assert len(notes) == 1 and "fitted on the model's own text" in notes[0]
    assert "/data/mine.pt" in notes[0]


def test_a_recipe_calibrated_against_a_named_artifact_still_compares_ids(tmp_path):
    recipe = Recipe(name="mine", model_id="m", artifact="/a.pt", calibrated_artifact="/a.pt")
    assert recipe.warnings(32, "/a.pt") == []
    assert "'/a.pt'" in recipe.warnings(32, "/b.pt")[0]


def test_bundled_for_finds_the_recipe_that_names_the_model():
    assert Recipe.bundled_for("Qwen/Qwen3-0.6B") == "qwen3-0.6b"
    assert Recipe.bundled_for("Qwen/Qwen3-1.7B") == "qwen3-1.7b"
    assert Recipe.bundled_for("nobody/nothing") is None


def test_the_recorded_frame_agrees_with_selfgen_options():
    from lfa.selfgen.artifact_corpus import SelfGenOptions
    frame = SelfGenOptions().artifact_frame()
    assert {k: frame[k] for k in RECORDED_SELF_GENERATED_FRAME} == RECORDED_SELF_GENERATED_FRAME


@pytest.mark.parametrize("bad", [["n_raw"], {"n_rows": 60}])
def test_a_self_generated_frame_that_is_not_a_frame_is_refused(bad):
    with pytest.raises(ValueError, match="self_generated_frame"):
        Recipe(name="r", model_id="m", artifact="self-generated", self_generated_frame=bad)
