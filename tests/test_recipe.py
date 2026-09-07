"""The recipe layer: the shipped operating point, its stage-2 lift, and its coupling warnings.

A recipe is a *joint* operating point, so what these tests check is that the bundled Qwen3-0.6B
file still carries exactly the published point, that turning it into a `TrainConfig` preserves the
two settings a reader is most likely to lose (the 100-epoch schedule horizon behind a 15-epoch
dose, and the checkpointing that makes that dose recoverable), and that a run which departs from
the calibrated rank or artifact is told that lambda no longer means what it meant.
"""

import dataclasses

import pytest
import yaml

from lfa import Recipe
from lfa.recipe import BUNDLED_DIR


SHIPPED = dict(
    name="qwen3-0.6b", model_id="Qwen/Qwen3-0.6B", artifact="qwen3-0.6b-gmm1543k-int8",
    lora_rank=32, lora_alpha=64, freeze_embed=True, full_weight=False,
    lambda_qkv=100000.0, lambda_mlp=100000.0, mu=0.05, mu_end_ratio=1.0,
    anchor_end_ratio=0.1, anchor_schedule="cosine", n_anchor_samples=16,
    epochs=15, schedule_horizon_epochs=100, checkpoint_mode="all", checkpoint_every=5,
    learning_rate=3e-4, batch_size=6, gradient_accumulation_steps=1,
    warmup_steps=50, weight_decay=0.01, sequence_length=512, seed=42, keep_short_whole=True,
    stage2_lambda_multiplier=3.0, calibrated_rank=32,
    calibrated_artifact="qwen3-0.6b-gmm1543k-int8",
)


def test_bundled_recipe_is_the_published_operating_point():
    recipe = Recipe.load("qwen3-0.6b")
    assert dataclasses.asdict(recipe) == SHIPPED


def test_bundled_recipe_is_found_from_the_package_not_the_working_directory(tmp_path, monkeypatch):
    """`load` resolves against the installed package, so a wheel and a `cd` both work."""
    monkeypatch.chdir(tmp_path)
    assert Recipe.load("qwen3-0.6b").lambda_qkv == 100000
    assert BUNDLED_DIR.is_dir() and (BUNDLED_DIR / "qwen3-0.6b.yaml").is_file()


def test_unknown_bundled_name_names_what_is_available():
    with pytest.raises(FileNotFoundError, match="qwen3-0.6b"):
        Recipe.load("qwen4-70b")


def test_yaml_documents_the_couplings_a_reader_has_to_know():
    """The file is read by humans before it is read by the loader; the guidance is the point."""
    text = (BUNDLED_DIR / "qwen3-0.6b.yaml").read_text()
    comments = "\n".join(line for line in text.splitlines() if line.lstrip().startswith("#")).lower()
    for phrase in ("rank", "artifact", "full-weight", "perplexity-optimal", "100-epoch"):
        assert phrase in comments, f"the recipe's comment block never mentions {phrase!r}"


# ==============================================================================================
# to_train_config
# ==============================================================================================

def test_stage_one_config_carries_the_recipe_verbatim(tmp_path):
    config = Recipe.load("qwen3-0.6b").to_train_config(1, tmp_path / "stats.pt")

    assert config.lambda_qkv == 100000 and config.lambda_mlp == 100000
    assert config.model_id == "Qwen/Qwen3-0.6B"
    assert config.artifact_path == str(tmp_path / "stats.pt")
    assert (config.mu, config.mu_end_ratio, config.n_anchor_samples) == (0.05, 1.0, 16)
    assert (config.anchor_end_ratio, config.anchor_schedule) == (0.1, "cosine")
    assert (config.lora_rank, config.lora_alpha) == (32, 64)
    assert config.use_lora is True and config.full_weight is False and config.freeze_embed is True
    assert (config.learning_rate, config.batch_size, config.gradient_accumulation_steps) == (3e-4, 6, 1)
    assert (config.warmup_steps, config.weight_decay, config.sequence_length, config.seed) == (50, 0.01, 512, 42)


def test_the_fifteen_epoch_dose_keeps_its_hundred_epoch_schedule_horizon(tmp_path):
    """e15 is the dose of a 100-epoch run: a 15-epoch horizon would decay the LR to its floor."""
    config = Recipe.load("qwen3-0.6b").to_train_config(1, tmp_path / "stats.pt")
    assert config.num_epochs == 15
    assert config.schedule_horizon_epochs == 100


def test_checkpointing_keeps_every_dose_reachable(tmp_path):
    config = Recipe.load("qwen3-0.6b").to_train_config(1, tmp_path / "stats.pt")
    assert config.checkpoint_mode == "all"
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
    """The override exists so a comparison against the research code's historical loader can be run."""
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

def test_no_warnings_at_the_calibrated_point():
    assert Recipe.load("qwen3-0.6b").warnings(32, "qwen3-0.6b-gmm1543k-int8") == []


def test_a_different_rank_warns_that_lambda_must_be_retuned():
    [warning] = Recipe.load("qwen3-0.6b").warnings(16, "qwen3-0.6b-gmm1543k-int8")
    assert "rank" in warning and "16" in warning and "32" in warning
    assert "re-tune" in warning.lower()


def test_a_different_artifact_warns_on_its_own():
    [warning] = Recipe.load("qwen3-0.6b").warnings(32, "qwen3-0.6b-diagonal")
    assert "artifact" in warning and "qwen3-0.6b-diagonal" in warning


def test_both_couplings_warn_together():
    warnings = Recipe.load("qwen3-0.6b").warnings(8, "qwen3-0.6b-diagonal")
    assert len(warnings) == 2


def test_a_full_weight_recipe_adds_its_own_note():
    recipe = dataclasses.replace(Recipe.load("qwen3-0.6b"), full_weight=True)
    warnings = recipe.warnings(32, "qwen3-0.6b-gmm1543k-int8")
    assert len(warnings) == 1
    assert "full-weight" in warnings[0].lower() and "unvalidated" in warnings[0]


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
    assert loaded["lambda_qkv"] == 100000 and loaded["schedule_horizon_epochs"] == 100


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


def test_epochs_must_be_at_least_one(tmp_path):
    with pytest.raises(ValueError, match="epochs"):
        Recipe.load(_write(tmp_path, epochs=0))


def test_the_schedule_horizon_may_not_be_shorter_than_the_dose(tmp_path):
    with pytest.raises(ValueError, match="15.*10|10.*15"):
        Recipe.load(_write(tmp_path, epochs=15, schedule_horizon_epochs=10))


def test_an_absent_schedule_horizon_is_allowed(tmp_path):
    """`None` means "lay the schedule over the epochs actually trained"."""
    recipe = Recipe.load(_write(tmp_path, schedule_horizon_epochs=None))
    assert recipe.to_train_config(1, tmp_path / "s.pt").schedule_horizon_epochs is None


def test_the_stage_two_multiplier_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="stage2_lambda_multiplier"):
        Recipe.load(_write(tmp_path, stage2_lambda_multiplier=0))


def test_lora_rank_must_be_at_least_one(tmp_path):
    with pytest.raises(ValueError, match="lora_rank"):
        Recipe.load(_write(tmp_path, lora_rank=0))
