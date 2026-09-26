"""The Workspace: one directory that carries a model, its p(h) artifact, and its history.

Everything runs on the CPU `tiny_model` with a one-epoch, rank-2 recipe, so a "stage" here is
seconds of training over eight short documents. What is being tested is the state machine and the
bookkeeping -- which model each stage starts from, which artifact version it anchors against, what
lands in `history.json`, and the guards that keep a chain in order -- not convergence.

The registry is monkeypatched with a "tiny" entry whose checksum is the fixture artifact's real
digest, and the shipped downloader is replaced by a copy, so `Workspace.init` exercises the real
fetch path offline.
"""

import copy
import dataclasses
import json
from pathlib import Path

import pytest
import torch
import yaml
from transformers import AutoModelForCausalLM

import lfa.workspace as workspace_module
from lfa import Recipe, Workspace
from lfa.artifact.fetch import ARTIFACTS, sha256_file
from lfa.artifact.schema import load_artifact
from lfa.corpus import load_corpus
from lfa.evaluate import domain_perplexity
from lfa.models import ShardingRefused, load_teacher, load_tokenizer
from lfa.sampler import Sampler
from lfa.train import EMBED_ANCHOR_DISABLED_NOTICE
from lfa.workspace import LOADER_FRAME_NOTICE, StageOrderError

from conftest import make_corpus, tiny_recipe

NEED, K_DOMAIN = 400, 2


# ------------------------------------------------------------------------------------ fixtures
#
# `base_dir`, `corpus_a`, `corpus_b`, `registry` and the `tiny_recipe` helper live in
# `tests/conftest.py`: `test_cli.py` drives the same state machine through the command line and
# needs exactly the same setup.

def new_workspace(path, base_dir, **overrides) -> Workspace:
    return Workspace.init(path, str(base_dir), artifact="tiny", **overrides)


@pytest.fixture(scope="module")
def flow(tmp_path_factory, registry, base_dir, corpus_a, corpus_b):
    """A whole two-domain chain, run once: init -> train A -> extend -> train B."""
    root = tmp_path_factory.mktemp("workspace")
    ws = new_workspace(root, base_dir)
    first = ws.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")
    extended = ws.extend(need=NEED, k_domain=K_DOMAIN, device="cpu")
    second = ws.train(corpus_b, recipe=tiny_recipe(base_dir), device="cpu")
    return ws, first, extended, second


# ---------------------------------------------------------------------------------------- init

def test_init_writes_the_state_the_history_and_the_first_artifact(tmp_path, registry, base_dir):
    ws = new_workspace(tmp_path, base_dir)

    assert (tmp_path / "workspace.json").is_file()
    assert json.loads((tmp_path / "history.json").read_text()) == []
    assert (tmp_path / "artifacts" / "v1.pt").is_file()
    assert sha256_file(tmp_path / "artifacts" / "v1.pt") == registry["sha256"]

    assert ws.state["model_id"] == str(base_dir)
    assert ws.state["base_model"] == str(base_dir)
    assert ws.state["current_model"] == str(base_dir)
    assert ws.state["current_artifact"] == str(tmp_path / "artifacts" / "v1.pt")
    assert ws.state["artifact_version"] == 1
    assert ws.state["stage"] == 0
    assert ws.state["last_stage_adapter"] is None


def test_init_accepts_a_local_artifact_file_and_copies_it_in(tmp_path, tiny_artifact, base_dir):
    _, path = tiny_artifact
    ws = Workspace.init(tmp_path, str(base_dir), artifact=str(path))

    assert ws.state["current_artifact"] == str(tmp_path / "artifacts" / "v1.pt")
    assert sha256_file(tmp_path / "artifacts" / "v1.pt") == sha256_file(path)
    assert ws.state["artifact_id"] is None          # not from the registry: no n_samples to read


def test_a_local_file_does_not_shadow_a_published_artifact_id(tmp_path, registry, base_dir,
                                                              monkeypatch):
    """`--artifact tiny` means the published artifact, whatever happens to be named `tiny` here."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tiny").write_bytes(b"not the published artifact")

    ws = new_workspace(tmp_path / "ws", base_dir)

    assert ws.state["artifact_id"] == "tiny"
    assert sha256_file(ws.state["current_artifact"]) == registry["sha256"]


def test_a_local_copy_can_be_recorded_as_the_published_artifact_it_is(tmp_path, registry,
                                                                     base_dir, tiny_artifact):
    """`artifact_id` is how a file fetched out of band keeps its provenance.

    Without it the workspace records a path, and `Recipe.warnings` then reports that lambda was
    calibrated against a different artifact than the one being used -- when it is the same one.
    """
    _, artifact_path = tiny_artifact

    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact=str(artifact_path),
                        fetch=False, artifact_id="tiny")

    assert ws.state["artifact_id"] == "tiny"
    # ...which is what the recipe's calibration is read against, so it warns about nothing.
    assert ws._artifact_id() == "tiny"
    assert tiny_recipe(base_dir).warnings(2, ws._artifact_id()) == []


def test_an_artifact_id_that_is_not_published_is_refused(tmp_path, base_dir, tiny_artifact):
    """It is a provenance claim, so a claim about an artifact nobody publishes is a mistake."""
    _, artifact_path = tiny_artifact
    with pytest.raises(ValueError, match="not a published artifact id"):
        Workspace.init(tmp_path / "ws", str(base_dir), artifact=str(artifact_path),
                       fetch=False, artifact_id="no-such-artifact")


def test_an_artifact_id_that_contradicts_the_artifact_is_refused(tmp_path, registry, base_dir):
    """Both name a published artifact, and they disagree: that is a mistake, not a preference."""
    with pytest.raises(ValueError, match="names a different one"):
        Workspace.init(tmp_path / "ws", str(base_dir), artifact="tiny",
                       artifact_id="qwen3-0.6b-gmm1543k-int8")

    # The same id twice is not a contradiction, so it is accepted.
    ws = Workspace.init(tmp_path / "ws2", str(base_dir), artifact="tiny", artifact_id="tiny")
    assert ws.state["artifact_id"] == "tiny"


def test_an_artifact_that_is_neither_an_id_nor_a_path_says_both(tmp_path, base_dir):
    with pytest.raises(ValueError, match="qwen3-0.6b-diagonal"):
        Workspace.init(tmp_path, str(base_dir), artifact="no-such-artifact")


def test_init_refuses_to_overwrite_an_existing_workspace(tmp_path, registry, base_dir):
    new_workspace(tmp_path, base_dir)
    with pytest.raises(FileExistsError, match="open"):
        new_workspace(tmp_path, base_dir)


def test_an_artifact_that_cannot_be_fetched_leaves_no_workspace_behind(tmp_path, base_dir,
                                                                       monkeypatch):
    """Today's most likely refusal of all: the README's flagship command, before the assets exist.

    The argument checks were hoisted above the `mkdir`s in the last round, but the *fetch* still
    happened after them, so this one refusal left `<path>/artifacts/` behind -- a directory that
    reads as a half-made workspace.
    """
    import lfa.workspace as workspace_module
    from lfa.artifact.fetch import ArtifactNotPublished

    def unpublished(artifact_id, dest_dir):
        raise ArtifactNotPublished(f"Artifact {artifact_id!r} has no published release asset yet")

    monkeypatch.setattr(workspace_module, "fetch_artifact", unpublished)
    workspace = tmp_path / "flagship"

    with pytest.raises(ArtifactNotPublished):
        Workspace.init(workspace, str(base_dir), artifact="qwen3-0.6b-gmm1543k-int8")

    assert not workspace.exists()


def test_a_workspace_that_was_already_there_survives_a_failed_init(tmp_path, base_dir,
                                                                   monkeypatch):
    """The cleanup removes only what this call made, and only while it is still empty."""
    import lfa.workspace as workspace_module
    from lfa.artifact.fetch import ArtifactNotPublished

    def unpublished(artifact_id, dest_dir):
        raise ArtifactNotPublished("not published")

    monkeypatch.setattr(workspace_module, "fetch_artifact", unpublished)
    existing = tmp_path / "already_here"
    (existing / "artifacts").mkdir(parents=True)
    (existing / "notes.txt").write_text("mine")

    with pytest.raises(ArtifactNotPublished):
        Workspace.init(existing, str(base_dir), artifact="qwen3-0.6b-gmm1543k-int8")

    assert (existing / "notes.txt").read_text() == "mine"
    assert (existing / "artifacts").is_dir()


def test_a_refused_init_leaves_nothing_behind(tmp_path, base_dir, tiny_artifact):
    """A refusal must not leave a directory that looks like a half-made workspace.

    `init` used to create `<path>/` and `<path>/artifacts/` before it had checked its arguments,
    so both refusals below left debris -- against the standard `fetch_artifact` is held to, which
    downloads to `<name>.part` precisely so an interruption leaves nothing artifact-shaped.
    """
    unknown_id = tmp_path / "by_id"
    with pytest.raises(ValueError, match="published artifact id"):
        Workspace.init(unknown_id, str(base_dir), artifact="not-an-artifact")
    assert not unknown_id.exists()

    missing_file = tmp_path / "by_path"
    with pytest.raises(ValueError):
        Workspace.init(missing_file, str(base_dir), artifact=str(tmp_path / "nope.pt"))
    assert not missing_file.exists()

    _, path = tiny_artifact
    conflicting = tmp_path / "conflict"
    with pytest.raises(ValueError, match="artifact_id"):
        Workspace.init(conflicting, str(base_dir), artifact=str(path), artifact_id="not-published")
    assert not conflicting.exists()

    # ...and a good one still creates exactly what it should.
    good = Workspace.init(tmp_path / "good", str(base_dir), artifact=str(path))
    assert (good.path / "artifacts" / "v1.pt").is_file()


def test_re_initialising_a_workspace_names_a_command_a_cli_user_can_run(tmp_path, registry,
                                                                       base_dir):
    """The refusal used to offer `Workspace.open(path)` -- a Python call -- to a CLI user."""
    new_workspace(tmp_path, base_dir)
    with pytest.raises(FileExistsError) as refusal:
        new_workspace(tmp_path, base_dir)

    assert "lfa train --workspace" in str(refusal.value)
    assert "Workspace.open(path)" in str(refusal.value)   # still there, for a Python caller


def test_a_corpus_path_that_does_not_exist_says_what_to_do(tmp_path, registry, base_dir):
    ws = new_workspace(tmp_path, base_dir)
    with pytest.raises(FileNotFoundError, match="prepare-domain") as refusal:
        ws.train(tmp_path / "typo_domain", recipe=tiny_recipe(base_dir), device="cpu")
    assert "Corpus path not found" in str(refusal.value)


def test_open_reads_back_what_init_wrote(tmp_path, registry, base_dir):
    created = new_workspace(tmp_path, base_dir)
    reopened = Workspace.open(tmp_path)

    assert reopened.state == created.state
    assert reopened.history == []


def test_open_says_so_when_there_is_no_workspace(tmp_path):
    with pytest.raises(FileNotFoundError, match="lfa init"):
        Workspace.open(tmp_path)


def test_init_without_fetching_leaves_no_artifact_and_says_what_to_run(tmp_path, registry,
                                                                        base_dir, corpus_a,
                                                                        caplog):
    with caplog.at_level("WARNING", logger="lfa.workspace"):
        ws = new_workspace(tmp_path, base_dir, fetch=False)

    assert ws.state["current_artifact"] is None
    assert not (tmp_path / "artifacts" / "v1.pt").exists()
    assert any("lfa fetch-artifact" in record.message for record in caplog.records)

    with pytest.raises(RuntimeError, match="fetch-artifact"):
        ws.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")


def test_init_adopts_a_bundled_recipe_that_names_the_same_model(tmp_path, tiny_artifact):
    """A workspace over Qwen3-0.6B finds the shipped recipe by its model_id, not by name."""
    _, path = tiny_artifact
    ws = Workspace.init(tmp_path, "Qwen/Qwen3-0.6B", artifact=str(path))
    assert ws.state["recipe"] == "qwen3-0.6b"


def _as_shipped(params: dict) -> dict:
    """The fixture artifact reshaped like the published `qwen3-0.6b-gmm1543k-int8` file.

    Verified against that file on 2026-09-07: 84 keys, all of them sites, **no `n_samples` on any
    block and no `__meta__` block at all**. Every other artifact in this suite carries both, so
    without this the shipped shape is exercised only in pieces and never end to end -- which is
    how the documented route came to be one that trains for an hour and then cannot `extend`.
    """
    stripped = copy.deepcopy(params)
    stripped.pop("__meta__", None)
    for key in list(stripped):
        stripped[key].pop("n_samples", None)
    return stripped


def test_the_documented_local_artifact_route_trains_extends_and_warns_about_nothing(
        tmp_path, registry, base_dir, corpus_a, tiny_artifact, caplog):
    """`--artifact <file> --artifact-id <published id>`: what README, quickstart and both
    examples tell a reader to do while the release assets do not exist yet.

    Both halves are load-bearing, and only the second is visible before an hour of GPU time has
    been spent: the id is what `Recipe.warnings` reads the calibration against (without it every
    stage warns that lambda was calibrated elsewhere, against the very file it was calibrated
    on), and it is what supplies the base sample count that `extend` needs, since the shipped
    artifact carries none.
    """
    base, _ = tiny_artifact
    shipped = tmp_path / "distribution_stats.pt"
    torch.save(_as_shipped(base), shipped)

    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact=str(shipped),
                        artifact_id="tiny", fetch=False)
    with caplog.at_level("WARNING", logger="lfa.workspace"):
        ws.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")

    assert [r.message for r in caplog.records if r.name == "lfa.workspace"
            and r.levelname == "WARNING"] == []
    extended = ws.extend(need=NEED, k_domain=K_DOMAIN, device="cpu")
    assert load_artifact(extended)["1_pre_mlp"]["n_samples"] == registry["n_samples_total"] + NEED


def test_the_same_route_without_the_artifact_id_warns_falsely_and_then_cannot_extend(
        tmp_path, registry, base_dir, corpus_a, tiny_artifact, caplog):
    """Why the id is in the documentation and not only in `docs/recipes.md`.

    This is the workspace a reader got from the four documented lines before 2026-09-07: a false
    calibration warning on every stage, and a refusal at the first `extend` -- after the stage has
    already trained.
    """
    base, _ = tiny_artifact
    shipped = tmp_path / "distribution_stats.pt"
    torch.save(_as_shipped(base), shipped)

    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact=str(shipped), fetch=False)
    with caplog.at_level("WARNING", logger="lfa.workspace"):
        ws.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")

    assert any("calibrated against" in r.message for r in caplog.records
               if r.name == "lfa.workspace")
    with pytest.raises(ValueError, match="base_n"):
        ws.extend(need=NEED, k_domain=K_DOMAIN, device="cpu")


# --------------------------------------------------------------------------------------- train

def test_the_first_stage_records_its_recipe_and_the_lambda_it_applied(flow, corpus_a, base_dir):
    ws, first, _, _ = flow
    recipe = tiny_recipe(base_dir)

    assert first["stage"] == 1
    assert first["corpus"] == str(corpus_a)
    assert first["recipe"] == dataclasses.asdict(recipe)
    assert first["lambda_applied"] == recipe.lambda_qkv        # stage 1: no multiplier
    assert first["artifact_version"] == 1
    assert first["output_dir"] == str(ws.path / "runs" / "stage1")
    assert first["lfa_version"] == workspace_module._lfa_version()
    assert first["timestamp"].startswith("20")                 # ISO-8601
    assert (first["device"], first["dtype"]) == ("cpu", "float32")   # fp32 on CPU, bf16 on a card
    assert (ws.path / "runs" / "stage1" / "final_model" / "adapter_model.safetensors").is_file()


def test_the_second_stage_starts_from_the_fused_model_and_the_extended_artifact(flow, corpus_b,
                                                                                base_dir):
    ws, _, _, second = flow
    recipe = tiny_recipe(base_dir)

    assert second["stage"] == 2
    assert second["corpus"] == str(corpus_b)
    assert second["lambda_applied"] == recipe.stage2_lambda_multiplier * recipe.lambda_qkv
    assert second["artifact_version"] == 2
    assert second["base_model"] == str(ws.path / "models" / "stage1_fused")
    config = json.loads((ws.path / "runs" / "stage2" / "config.json").read_text())
    assert config["lambda_qkv"] == 30.0


def test_the_history_file_carries_both_stages_in_order(flow):
    ws, first, _, second = flow
    history = json.loads((ws.path / "history.json").read_text())

    assert len(history) == 2
    assert [entry["stage"] for entry in history] == [1, 2]
    assert history[0]["corpus"] == first["corpus"] and history[1]["corpus"] == second["corpus"]
    assert ws.history == history
    assert Workspace.open(ws.path).history == history


def test_training_builds_the_embedding_lookup_before_the_run(tmp_path, registry, base_dir,
                                                             corpus_a, monkeypatch, caplog):
    """The shipped artifact carries no lookup table, so the workspace must rebuild it from the
    teacher: without that, L_embed is silently dropped (train only warns)."""
    built = []

    class RecordingSampler(Sampler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(workspace_module, "Sampler", RecordingSampler)
    ws = new_workspace(tmp_path, base_dir)

    with caplog.at_level("WARNING", logger="lfa.train"):
        ws.train(corpus_a, recipe=tiny_recipe(base_dir, freeze_embed=False), device="cpu")

    assert len(built) == 1 and built[0].has_embedding_lookup()
    assert EMBED_ANCHOR_DISABLED_NOTICE not in caplog.text


def test_training_surfaces_the_recipes_off_calibration_warnings(tmp_path, registry, base_dir,
                                                                corpus_a, caplog):
    ws = new_workspace(tmp_path, base_dir)
    recipe = tiny_recipe(base_dir, calibrated_rank=32)

    with caplog.at_level("WARNING", logger="lfa.workspace"):
        ws.train(corpus_a, recipe=recipe, device="cpu")

    warned = [r.message for r in caplog.records if r.name == "lfa.workspace"
              and r.levelname == "WARNING"]
    assert warned == recipe.warnings(2, "tiny")
    assert "calibrated at rank 32" in warned[0]


def test_training_states_the_loader_frame_when_short_documents_are_kept(tmp_path, registry,
                                                                       base_dir, corpus_a,
                                                                       caplog):
    ws = new_workspace(tmp_path, base_dir)

    with caplog.at_level("INFO", logger="lfa.workspace"):
        ws.train(corpus_a, recipe=tiny_recipe(base_dir, keep_short_whole=True), device="cpu")

    assert any(r.message == LOADER_FRAME_NOTICE for r in caplog.records)
    # It has to say what the setting does, not merely that a setting exists: it changes the
    # training stream and is invisible in every number the run goes on to report.
    assert "trained whole in every epoch" in LOADER_FRAME_NOTICE
    assert "keep_short_whole=False" in LOADER_FRAME_NOTICE


def test_the_notice_is_silent_when_short_documents_are_cut_instead(tmp_path, registry, base_dir,
                                                                   corpus_a, caplog):
    ws = new_workspace(tmp_path, base_dir)

    with caplog.at_level("INFO", logger="lfa.workspace"):
        ws.train(corpus_a, recipe=tiny_recipe(base_dir, keep_short_whole=False), device="cpu")

    assert not any(r.message == LOADER_FRAME_NOTICE for r in caplog.records)


def test_epochs_overrides_the_recipes_epoch_count(tmp_path, registry, base_dir, corpus_a):
    ws = new_workspace(tmp_path, base_dir)
    entry = ws.train(corpus_a, recipe=tiny_recipe(base_dir),
                     epochs=2, device="cpu")

    assert entry["recipe"]["epochs"] == 2
    history = json.loads((ws.path / "runs" / "stage1" / "training_history.json").read_text())
    assert [record["epoch"] for record in history] == [1, 2]


def test_keep_short_whole_can_be_overridden_per_run(tmp_path, registry, base_dir, corpus_a,
                                                    caplog):
    """The loader frame is a per-run override: two runs under different settings see different
    text, so it has to be reachable without editing the recipe."""
    ws = new_workspace(tmp_path, base_dir)
    with caplog.at_level("INFO", logger="lfa.workspace"):
        entry = ws.train(corpus_a, recipe=tiny_recipe(base_dir, keep_short_whole=True),
                         keep_short_whole=False, device="cpu")

    config = json.loads((Path(entry["output_dir"]) / "config.json").read_text())
    assert config["keep_short_whole"] is False
    assert entry["keep_short_whole"] is False
    assert entry["recipe"]["keep_short_whole"] is True            # the recipe is left as it is
    assert not any(record.message == LOADER_FRAME_NOTICE for record in caplog.records)


def test_a_run_records_no_chunk_offset_setting_because_there_is_none(tmp_path, registry,
                                                                      base_dir, corpus_a):
    """The chunk offset rotates the boundaries, full stop: there is no setting for it, so a run
    has nothing to record about it and `Workspace.train` has no argument for it."""
    ws = new_workspace(tmp_path, base_dir)
    entry = ws.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")

    config = json.loads((Path(entry["output_dir"]) / "config.json").read_text())
    assert "rotate_offset" not in config and "rotate_offset" not in entry
    with pytest.raises(TypeError):
        ws.train(corpus_a, recipe=tiny_recipe(base_dir), rotate_offset=False, device="cpu")


def test_full_weight_can_be_overridden_per_run(tmp_path, registry, base_dir, corpus_a):
    ws = new_workspace(tmp_path, base_dir)
    entry = ws.train(corpus_a, recipe=tiny_recipe(base_dir), full_weight=True, device="cpu")

    config = json.loads((Path(entry["output_dir"]) / "config.json").read_text())
    assert (config["full_weight"], config["use_lora"]) == (True, False)
    assert entry["full_weight"] is True
    # The override is folded into the recipe before the recipe is recorded, which is what makes a
    # later reconstruction of this stage -- the unanchored control -- train the way this run did.
    assert entry["recipe"]["full_weight"] is True
    assert not (Path(entry["adapter"]) / "adapter_config.json").exists()


def test_train_refuses_a_device_that_would_shard_the_model(tmp_path, registry, base_dir,
                                                           corpus_a):
    ws = new_workspace(tmp_path, base_dir)
    with pytest.raises(ShardingRefused):
        ws.train(corpus_a, recipe=tiny_recipe(base_dir), device="auto")


def test_train_without_a_recipe_anywhere_says_what_to_pass(tmp_path, registry, base_dir,
                                                           corpus_a):
    ws = new_workspace(tmp_path, base_dir)
    with pytest.raises(ValueError, match="recipe"):
        ws.train(corpus_a, device="cpu")


# --------------------------------------------------------------------------- the stage order

def test_a_new_corpus_before_extending_names_the_command_to_run(tmp_path, registry, base_dir,
                                                                corpus_a, corpus_b):
    ws = new_workspace(tmp_path, base_dir)
    ws.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")

    with pytest.raises(StageOrderError, match="lfa extend"):
        ws.train(corpus_b, recipe=tiny_recipe(base_dir), device="cpu")

    assert issubclass(StageOrderError, RuntimeError)
    assert ws.state["stage"] == 1


def test_the_same_corpus_again_is_the_same_stage_not_the_next_one(tmp_path, registry, base_dir,
                                                                  corpus_a):
    """More epochs on the domain already being learned need no extension -- and no lambda bump."""
    ws = new_workspace(tmp_path, base_dir)
    first = ws.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")
    again = ws.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")

    assert again["stage"] == 1
    assert again["lambda_applied"] == 10.0
    assert ws.state["stage"] == 1
    assert len(ws.history) == 2

    # A repeat is a second run, not an overwrite: both runs stay readable on disk.
    assert Path(first["output_dir"]).name == "stage1"
    assert Path(again["output_dir"]).name == "stage1_run2"
    for entry in (first, again):
        assert (Path(entry["output_dir"]) / "training_history.json").is_file()
        assert (Path(entry["adapter"]) / "adapter_model.safetensors").is_file()


def test_a_resumed_run_keeps_training_the_adapter_it_saved(tmp_path, registry, base_dir,
                                                           corpus_a):
    """A resume must re-attach the saved adapter, not wrap a fresh one over the same base.

    The probe is a resume with nothing left to do (`epochs` equal to the epochs already run):
    the epoch loop runs zero times and the model is saved as it stands. Re-attached, that is the
    adapter that was already there; wrapped fresh, it is an untrained one whose B is all zeros --
    and it would have overwritten the checkpoint it resumed from.
    """
    ws = new_workspace(tmp_path, base_dir)
    entry = ws.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")
    trained = _adapter_weights(entry["adapter"])
    assert any(tensor.abs().sum() > 0 for name, tensor in trained.items() if "lora_B" in name)

    resumed = ws.train(corpus_a, recipe=tiny_recipe(base_dir), resume=True, device="cpu")

    assert resumed["output_dir"] == entry["output_dir"]          # the same run, continuing
    after = _adapter_weights(resumed["adapter"])
    assert set(after) == set(trained)
    for name, tensor in trained.items():
        assert torch.equal(after[name], tensor), name


def test_a_resumed_run_continues_the_epoch_count(tmp_path, registry, base_dir, corpus_a):
    ws = new_workspace(tmp_path, base_dir)
    ws.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")
    resumed = ws.train(corpus_a, recipe=tiny_recipe(base_dir),
                       epochs=2, resume=True, device="cpu")

    history = json.loads((Path(resumed["output_dir"]) / "training_history.json").read_text())
    assert [record["epoch"] for record in history] == [1, 2]
    assert len(ws.history) == 2 and ws.state["stage"] == 1


def _adapter_weights(adapter_dir):
    import safetensors.torch

    return safetensors.torch.load_file(Path(adapter_dir) / "adapter_model.safetensors")


def test_an_explicit_run_name_will_not_write_over_the_run_already_there(tmp_path, registry,
                                                                        base_dir, corpus_a):
    """The only silent data-loss path the workspace had.

    `_run_name` keeps the DEFAULT names apart (a repeat becomes `stage1_run2`), but an explicit
    name -- a chain spec's `name:`, or the same `output_name` twice -- went through as given and
    the trainer writes with `exist_ok=True`. Two history entries then pointed at one directory:
    the first run's config, curve and checkpoint gone, and `evaluate`/`extend`/`fuse` silently
    resolving the first entry to the second run's model.
    """
    ws = new_workspace(tmp_path, base_dir)
    first = ws.train(corpus_a, recipe=tiny_recipe(base_dir), output_name="dup", device="cpu")
    curve = (Path(first["output_dir"]) / "training_history.json").read_text()

    with pytest.raises(FileExistsError, match="dup") as refusal:
        ws.train(corpus_a, recipe=tiny_recipe(base_dir), output_name="dup", device="cpu")

    # Every remedy the message names has to be one that works: this run saved a training state,
    # so resume does; deleting the directory always does.
    assert "delete that directory" in str(refusal.value)
    assert "resume=True to continue" in str(refusal.value)

    assert len(ws.history) == 1                              # nothing was appended
    assert (Path(first["output_dir"]) / "training_history.json").read_text() == curve
    assert [p.name for p in (ws.path / "runs").iterdir()] == ["dup"]

    # ...and a resume of that same run is still allowed: it continues the run rather than
    # discarding it, which is the one case where writing into an occupied directory is the point.
    resumed = ws.train(corpus_a, recipe=tiny_recipe(base_dir), output_name="dup", epochs=2,
                       resume=True, device="cpu")
    assert resumed["output_dir"] == first["output_dir"] and len(ws.history) == 2


def test_a_start_interrupted_before_its_first_epoch_can_simply_be_run_again(tmp_path, registry,
                                                                             base_dir, corpus_a):
    """`lfa.train.train` writes `config.json` before epoch 1, so a run killed in its first epoch
    leaves a config and nothing else. That is not a run to protect: there is no curve, no
    checkpoint and no optimizer state, `resume=True` would raise `FileNotFoundError` on the
    missing `training_state.pt`, and refusing would leave the user unable to run the command
    again.
    """
    ws = new_workspace(tmp_path, base_dir)
    run_dir = ws.path / "runs" / "attempt"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text('{"num_epochs": 1}')

    entry = ws.train(corpus_a, recipe=tiny_recipe(base_dir), output_name="attempt", device="cpu")

    assert entry["output_dir"] == str(run_dir)
    assert (run_dir / "training_history.json").is_file()
    assert (run_dir / "final_model").is_dir()
    assert len(ws.history) == 1


def test_a_half_written_run_with_no_state_is_refused_without_offering_resume(tmp_path, registry,
                                                                            base_dir, corpus_a):
    """A directory that got as far as a checkpoint but has no `training_state.pt` is protected --
    and the message must not send the user to a resume that cannot work."""
    ws = new_workspace(tmp_path, base_dir)
    run_dir = ws.path / "runs" / "partial"
    (run_dir / "final_model").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="final_model") as refusal:
        ws.train(corpus_a, recipe=tiny_recipe(base_dir), output_name="partial", device="cpu")

    assert "delete that directory" in str(refusal.value)
    assert "resume=True cannot help" in str(refusal.value)


def test_a_chain_spec_that_names_two_domains_alike_is_refused_before_anything_trains(
        tmp_path, registry, base_dir, corpus_a, corpus_b):
    """The same collision, caught where it costs nothing rather than one stage in."""
    recipe_path = tiny_recipe(base_dir).save(tmp_path / "tiny_recipe.yaml")
    ws = new_workspace(tmp_path / "ws", base_dir, recipe=str(recipe_path))
    spec = tmp_path / "domains.yaml"
    spec.write_text(yaml.safe_dump({"domains": [{"name": "dup", "corpus": str(corpus_a)},
                                                {"name": "dup", "corpus": str(corpus_b)}]}))

    with pytest.raises(ValueError, match="dup"):
        ws.chain(spec, device="cpu", need=NEED, k_domain=K_DOMAIN)

    assert ws.history == [] and not (ws.path / "runs").exists()


def test_a_stage_records_which_implementation_trained_it(flow):
    """Which CODE produced a run, not merely which release.

    The port-verification harness (`docs/verification.md`, and it lives with the research code)
    may re-score a run it kept from an earlier session; without this field a kept run and a changed
    objective look exactly alike, and its criterion then certifies an implementation that never
    executed. A version string cannot do it: two commits of one version share it.
    """
    _, first, _, second = flow
    identity = workspace_module.code_identity()

    for entry in (first, second):
        assert entry["implementation"]["code_digest"] == identity["code_digest"]
        assert len(entry["implementation"]["code_digest"]) == 16
    # The revision is provenance for a human and may legitimately be absent (an installed wheel).
    assert set(first["implementation"]) == {"code_digest", "git_revision"}


def test_the_code_digest_moves_when_the_package_source_moves(tmp_path):
    """It is a fingerprint of the sources, not of the version: any edit has to change it."""
    import shutil

    from lfa.workspace import source_digest

    package = Path(workspace_module.__file__).resolve().parent
    copied = tmp_path / "lfa"
    shutil.copytree(package, copied, ignore=shutil.ignore_patterns("__pycache__"))

    assert source_digest(copied) == source_digest(package)

    (copied / "losses.py").write_text((copied / "losses.py").read_text() + "\n# an edit\n")
    assert source_digest(copied) != source_digest(package)

    # A recipe is part of the implementation too: it is what the run is an instance of.
    shutil.copytree(package, tmp_path / "lfa2", ignore=shutil.ignore_patterns("__pycache__"))
    recipe = tmp_path / "lfa2" / "recipes" / "qwen3-0.6b.yaml"
    recipe.write_text(recipe.read_text().replace("lora_rank: 32", "lora_rank: 16"))
    assert source_digest(tmp_path / "lfa2") != source_digest(package)


def test_extending_with_nothing_to_extend_says_so(tmp_path, registry, base_dir):
    ws = new_workspace(tmp_path, base_dir)
    with pytest.raises(StageOrderError, match="train"):
        ws.extend(need=NEED, k_domain=K_DOMAIN, device="cpu")


# -------------------------------------------------------------------------------------- extend

def test_extend_fuses_the_stage_and_writes_the_next_artifact_version(flow):
    ws, _, extended, _ = flow

    assert extended == ws.path / "artifacts" / "v2.pt"
    assert (ws.path / "models" / "stage1_fused" / "config.json").is_file()
    assert ws.state["artifact_version"] == 2
    assert ws.state["current_artifact"] == str(extended)
    assert ws.state["current_model"] == str(ws.path / "models" / "stage1_fused")


def test_the_extended_artifact_unions_the_domain_into_the_base(flow, tiny_artifact):
    base, _ = tiny_artifact
    _, _, extended, _ = flow
    entry = load_artifact(extended)["1_pre_mlp"]

    assert entry["gmm_n_components"] == base["1_pre_mlp"]["gmm_n_components"] + K_DOMAIN
    assert entry["n_samples"] == 1000 + NEED
    assert torch.isclose(entry["gmm_weights"].sum(), torch.tensor(1.0), atol=1e-4)


def test_extend_collects_under_the_frame_the_stage_trained_under(tmp_path, registry, base_dir,
                                                                corpus_a, monkeypatch):
    """The new components describe the training stream the model saw, so the collection uses the
    stage's own loader frame rather than the loader's default."""
    seen = {}
    real = workspace_module.extend_artifact

    def recording(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(workspace_module, "extend_artifact", recording)
    ws = new_workspace(tmp_path, base_dir)
    ws.train(corpus_a, recipe=tiny_recipe(base_dir, keep_short_whole=True),
             keep_short_whole=False, device="cpu")

    ws.extend(need=NEED, k_domain=K_DOMAIN, device="cpu")

    assert seen["keep_short_whole"] is False
    assert seen["seq_len"] == 64


def test_extend_reads_the_base_count_from_the_registry_when_the_artifact_carries_none(
        tmp_path, registry, base_dir, corpus_a, tiny_artifact):
    """The shipped artifact predates the per-block count; its n_samples_total lives in the
    registry entry, and the mixture's weighting is wrong without it."""
    base, _ = tiny_artifact
    ws = new_workspace(tmp_path, base_dir)
    _strip_counts(ws.state["current_artifact"], base)
    ws.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")

    out = ws.extend(need=NEED, k_domain=K_DOMAIN, device="cpu")
    assert load_artifact(out)["1_pre_mlp"]["n_samples"] == registry["n_samples_total"] + NEED


def test_extend_names_base_n_when_nothing_supplies_the_count(tmp_path, base_dir, corpus_a,
                                                             tiny_artifact):
    base, path = tiny_artifact
    ws = Workspace.init(tmp_path, str(base_dir), artifact=str(path))
    _strip_counts(ws.state["current_artifact"], base)
    ws.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")

    with pytest.raises(ValueError, match="base_n"):
        ws.extend(need=NEED, k_domain=K_DOMAIN, device="cpu")


def test_the_base_count_predicate_reads_the_mixture_sites_only(tmp_path, registry, base_dir,
                                                               tiny_artifact):
    """A site with no mixture is not part of the merge, so its missing count is not a missing
    count: reading every site instead would send the extension to the registry for a number the
    artifact already has, and weight the new domain against the wrong pool."""
    base, _ = tiny_artifact
    ws = new_workspace(tmp_path, base_dir)
    params = copy.deepcopy(base)
    params["0_pre_qkv"] = {"mean": torch.zeros(32), "std": torch.ones(32)}   # no mixture, no count
    # No meta total either, so the per-block counts on the mixture sites are the only answer.
    params["__meta__"] = {k: v for k, v in params["__meta__"].items() if k != "n_samples_total"}
    torch.save(params, ws.state["current_artifact"])

    assert ws._resolve_base_n() is None                  # the mixture sites carry their own


def _strip_counts(artifact_path, base):
    """Rewrite an artifact as the shipped one is shaped: no per-block counts, no meta total."""
    params = copy.deepcopy(base)
    for key in list(params):
        if key[0].isdigit():
            params[key].pop("n_samples", None)
    params["__meta__"] = {k: v for k, v in params["__meta__"].items() if k != "n_samples_total"}
    torch.save(params, artifact_path)


# ------------------------------------------------------------------------------------ evaluate

def test_evaluate_reads_the_stage_against_the_model_it_started_from(flow):
    ws, _, _, second = flow
    result = ws.evaluate(n_windows=None, device="cpu")

    for column in ("before", "after"):
        assert result[column]["general"] is None                  # skipped: n_windows=None
        assert 0 < result[column]["domain"] < float("inf")
    assert result["unanchored"] is None
    assert "domain" in result["table"]
    assert "not measured" in result["table"]

    # "before" is the model this stage started from, scored on this stage's corpus -- not the
    # workspace's current model and not the base the chain began at.
    assert result["before"]["domain"] == _domain_ppl(second["base_model"], second["corpus"])
    assert result["after"]["domain"] != result["before"]["domain"]

    recorded = json.loads((ws.path / "history.json").read_text())[-1]["perplexity"]
    assert recorded["after"]["domain"] == result["after"]["domain"]


def _domain_ppl(model_path, corpus_path):
    """Domain perplexity of a checkpoint, computed without going through the workspace."""
    tokenizer = load_tokenizer(model_path)
    heldout, _ = load_corpus(corpus_path, tokenizer, max_length=64, val_fraction=0.0, seed=0)
    model = load_teacher(model_path, device="cpu", dtype=torch.float32)
    return domain_perplexity(model, tokenizer, heldout, device="cpu")


def test_evaluate_measures_the_general_axis_when_it_can(flow, monkeypatch):
    ws, _, _, _ = flow
    seen = []

    def fake_general(model, tokenizer, **kwargs):
        seen.append(kwargs)
        return 42.0

    monkeypatch.setattr(workspace_module, "wikitext2_perplexity", fake_general)
    result = ws.evaluate(n_windows=8, device="cpu")

    assert result["before"]["general"] == result["after"]["general"] == 42.0
    assert [kwargs["n_windows"] for kwargs in seen] == [8, 8]
    assert "WikiText-2" in result["table"]


def test_evaluate_says_so_when_the_general_split_cannot_be_fetched(flow, monkeypatch, caplog):
    ws, _, _, _ = flow

    def unavailable(*args, **kwargs):
        raise OSError("offline: wikitext could not be downloaded")

    monkeypatch.setattr(workspace_module, "wikitext2_perplexity", unavailable)
    with caplog.at_level("WARNING", logger="lfa.workspace"):
        result = ws.evaluate(n_windows=8, device="cpu")

    assert result["before"]["general"] is None and result["after"]["general"] is None
    assert any("wikitext" in r.message.lower() for r in caplog.records)


def test_the_table_survives_the_general_axis_failing_for_one_model_only(flow, monkeypatch):
    """WikiText-2 can be scored for one model and fail for the next -- an intermittent Hub.

    The table used to be selected on `before` alone, so the successful first column sent a `None`
    second column into `perplexity_table`, which formatted it as a number: `TypeError`, from a
    package whose quickstart says a traceback is a bug in it.
    """
    ws, _, _, _ = flow
    answers = [42.0, OSError("offline: wikitext could not be downloaded")]

    def flaky(*args, **kwargs):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(workspace_module, "wikitext2_perplexity", flaky)
    result = ws.evaluate(n_windows=8, device="cpu")

    assert result["before"]["general"] == 42.0 and result["after"]["general"] is None
    assert "not measured (after)" in result["table"]
    assert "domain" in result["table"]


def test_evaluate_can_rerun_the_stage_with_the_anchor_switched_off(tmp_path, registry, base_dir,
                                                                   corpus_a):
    ws = new_workspace(tmp_path, base_dir)
    ws.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")

    result = ws.evaluate(compare_unanchored=True, n_windows=None, device="cpu")

    assert 0 < result["unanchored"]["domain"] < float("inf")
    assert (ws.path / "runs" / "stage1_unanchored" / "final_model").is_dir()
    config = json.loads((ws.path / "runs" / "stage1_unanchored" / "config.json").read_text())
    assert (config["lambda_qkv"], config["lambda_mlp"], config["mu"]) == (0.0, 0.0, 0.0)
    assert "unanchored" in result["table"]


def test_evaluate_before_any_stage_says_what_is_missing(tmp_path, registry, base_dir):
    ws = new_workspace(tmp_path, base_dir)
    with pytest.raises(RuntimeError, match="train"):
        ws.evaluate(n_windows=None, device="cpu")


# ------------------------------------------------------------------- the held-out split

def test_a_stage_records_the_documents_it_held_out(tmp_path, registry, base_dir, corpus_a):
    """The shipped recipe holds a tenth of the documents out; the split is a property of the run,
    so the history has to say how it fell or `evaluate` cannot rebuild it."""
    ws = new_workspace(tmp_path, base_dir)
    entry = ws.train(corpus_a, recipe=tiny_recipe(base_dir, val_fraction=0.5), device="cpu")

    assert entry["val_fraction"] == 0.5
    assert (entry["n_train_docs"], entry["n_val_docs"]) == (4, 4)          # eight documents, halved
    assert json.loads((ws.path / "runs" / "stage1" / "config.json").read_text())["val_fraction"] == 0.5


def test_the_held_out_split_is_scored_after_every_epoch_of_the_stage(tmp_path, registry, base_dir,
                                                                    corpus_a):
    """The split the workspace builds goes to the trainer, not only to `evaluate`: the run's own
    history carries the held-out loss per epoch, which is what says a run has begun to over-fit."""
    ws = new_workspace(tmp_path, base_dir)
    entry = ws.train(corpus_a, recipe=tiny_recipe(base_dir, val_fraction=0.5, epochs=2),
                     device="cpu")

    history = json.loads((Path(entry["output_dir"]) / "training_history.json").read_text())
    assert len(history) == 2
    assert all(record["val_loss"] > 0 and record["val_perplexity"] > 1 for record in history)


def test_a_stage_that_holds_nothing_out_has_no_validation_curve(tmp_path, registry, base_dir,
                                                                corpus_a):
    ws = new_workspace(tmp_path, base_dir)
    entry = ws.train(corpus_a, recipe=tiny_recipe(base_dir, val_fraction=0.0), device="cpu")

    history = json.loads((Path(entry["output_dir"]) / "training_history.json").read_text())
    assert all("val_loss" not in record for record in history)


def test_evaluate_scores_the_half_the_stage_never_trained_on(tmp_path, registry, base_dir,
                                                             corpus_a, monkeypatch):
    """Domain perplexity of a model on text it was just trained on is a fit. With a hold-out
    recorded, `evaluate` rebuilds that split and scores the other half instead."""
    ws = new_workspace(tmp_path, base_dir)
    recipe = tiny_recipe(base_dir, val_fraction=0.5)
    ws.train(corpus_a, recipe=recipe, device="cpu")

    tokenizer = load_tokenizer(str(base_dir))
    trained_on, held_out = load_corpus(corpus_a, tokenizer, max_length=recipe.sequence_length,
                                       val_fraction=0.5, seed=recipe.seed,
                                       keep_short_whole=recipe.keep_short_whole)
    assert len(held_out) and len(trained_on)

    scored = []
    real = workspace_module.domain_perplexity
    monkeypatch.setattr(workspace_module, "domain_perplexity",
                        lambda model, tok, corpus, **kw: scored.append(
                            [ex["input_ids"].tolist() for ex in corpus]) or real(model, tok,
                                                                                corpus, **kw))
    ws.evaluate(n_windows=None, device="cpu")

    expected = [ex["input_ids"].tolist() for ex in held_out]
    assert scored == [expected, expected]                       # the "before" and "after" columns
    assert expected != [ex["input_ids"].tolist() for ex in trained_on]


def test_evaluate_falls_back_to_the_training_corpus_when_nothing_was_held_out(
        tmp_path, registry, base_dir, corpus_a, caplog):
    """A stage trained on everything has no held-out split, and the number is then a fit -- which
    the log says, rather than the caller having to remember the recipe."""
    ws = new_workspace(tmp_path, base_dir)
    ws.train(corpus_a, recipe=tiny_recipe(base_dir, val_fraction=0.0), device="cpu")

    with caplog.at_level("INFO", logger="lfa.workspace"):
        result = ws.evaluate(n_windows=None, device="cpu")

    assert 0 < result["after"]["domain"] < float("inf")
    assert any("not a held-out measurement" in r.getMessage() for r in caplog.records)


def test_a_named_corpus_is_scored_whole(tmp_path, registry, base_dir, corpus_a, corpus_b,
                                        monkeypatch):
    """Text the caller names IS the held-out text; splitting it again would score a fraction of
    what was asked for."""
    ws = new_workspace(tmp_path, base_dir)
    recipe = tiny_recipe(base_dir, val_fraction=0.5)
    ws.train(corpus_a, recipe=recipe, device="cpu")

    tokenizer = load_tokenizer(str(base_dir))
    whole, _ = load_corpus(corpus_b, tokenizer, max_length=recipe.sequence_length,
                           val_fraction=0.0, seed=recipe.seed,
                           keep_short_whole=recipe.keep_short_whole)

    scored = []
    monkeypatch.setattr(workspace_module, "domain_perplexity",
                        lambda model, tok, corpus, **kw: scored.append(len(corpus)) or 1.0)
    ws.evaluate(corpus_b, n_windows=None, device="cpu")

    assert scored == [len(whole), len(whole)]


# ---------------------------------------------------------------------------------------- fuse

def test_fuse_exports_a_plain_model_that_loads_on_its_own(flow):
    ws, _, _, _ = flow
    out = ws.fuse()

    assert out == ws.path / "models" / "stage2_fused_export"
    assert not (out / "adapter_config.json").exists()
    model = AutoModelForCausalLM.from_pretrained(out)
    assert model.config.num_hidden_layers == 2
    assert (out / "tokenizer.json").is_file()


def test_fuse_writes_where_it_is_told(tmp_path, registry, base_dir, corpus_a):
    ws = new_workspace(tmp_path, base_dir)
    ws.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")

    out = ws.fuse(out_dir=tmp_path / "export")
    assert out == tmp_path / "export" and (out / "config.json").is_file()


# --------------------------------------------------------------------- the full-weight stage

def test_a_full_weight_stage_carries_no_adapter_through_fuse_and_evaluate(tmp_path, registry,
                                                                          base_dir, corpus_a):
    """Full weight has nothing to merge: the checkpoint IS the model, everywhere it is read."""
    ws = new_workspace(tmp_path, base_dir)
    entry = ws.train(corpus_a, recipe=tiny_recipe(base_dir, full_weight=True), device="cpu")

    assert not (Path(entry["adapter"]) / "adapter_config.json").exists()
    assert (Path(entry["adapter"]) / "model.safetensors").is_file()

    result = ws.evaluate(compare_unanchored=True, n_windows=None, device="cpu")
    assert 0 < result["after"]["domain"] < float("inf")

    # The control answers "this run without the anchor", so it trains the way this run did.
    control = json.loads((ws.path / "runs" / "stage1_unanchored" / "config.json").read_text())
    assert (control["full_weight"], control["use_lora"]) == (True, False)
    assert (control["lambda_qkv"], control["mu"]) == (0.0, 0.0)
    assert 0 < result["unanchored"]["domain"] < float("inf")

    out = ws.fuse()
    assert (out / "config.json").is_file() and (out / "tokenizer.json").is_file()
    assert not (out / "adapter_config.json").exists()


# --------------------------------------------------------------------------------------- chain

def test_chain_runs_every_domain_and_extends_between_them(tmp_path, registry, base_dir, corpus_a,
                                                          corpus_b):
    ws = _chain_workspace(tmp_path, base_dir)
    spec = tmp_path / "domains.yaml"
    spec.write_text(yaml.safe_dump({
        "domains": [{"name": "alpha", "corpus": str(corpus_a)},
                    {"name": "beta", "corpus": str(corpus_b), "epochs": 1}],
    }))

    entries = ws.chain(spec, device="cpu", need=NEED, k_domain=K_DOMAIN)

    assert [entry["stage"] for entry in entries] == [1, 2]
    assert [Path(entry["output_dir"]).name for entry in entries] == ["alpha", "beta"]
    assert entries[1]["lambda_applied"] == 30.0            # stage 2 anchors harder
    assert ws.state["artifact_version"] == 2
    assert ws.state["stage"] == 2
    assert len(ws.history) == 2


def test_chain_resolves_a_relative_corpus_against_the_spec_file(tmp_path, registry, base_dir,
                                                                corpus_a):
    ws = _chain_workspace(tmp_path, base_dir)
    spec_dir = tmp_path / "spec_dir"
    make_corpus(spec_dir / "alpha", "consciousness")
    spec = spec_dir / "domains.yaml"
    spec.write_text(yaml.safe_dump({"domains": [{"name": "alpha", "corpus": "alpha"}]}))

    entries = ws.chain(spec, device="cpu")

    assert entries[0]["corpus"] == str(spec_dir / "alpha")
    assert ws.state["artifact_version"] == 1               # one domain: nothing to extend between


def test_a_spec_that_asks_not_to_extend_between_domains_is_rejected(tmp_path, registry,
                                                                    base_dir, corpus_a):
    """Folding each domain in IS the chain; the second domain has nowhere to start otherwise."""
    ws = _chain_workspace(tmp_path, base_dir)
    spec = tmp_path / "flat.yaml"
    spec.write_text(yaml.safe_dump({"domains": [{"name": "alpha", "corpus": str(corpus_a)}],
                                    "extend_between": False}))

    with pytest.raises(ValueError, match="extend_between"):
        ws.chain(spec, device="cpu")


@pytest.mark.parametrize("position", [1, 2, 3])
@pytest.mark.parametrize("broken, error, expected", [
    ({"extend_between": False}, ValueError, "always folds each domain"),
    ({"banana": 7}, ValueError, "does not have"),
    ({"corpus": None}, ValueError, "has no 'corpus'"),
    ({"corpus": "not_a_corpus_that_exists"}, FileNotFoundError, "not there"),
])
def test_every_domain_is_validated_before_the_first_one_trains(tmp_path, registry, base_dir,
                                                               corpus_a, corpus_b, position,
                                                               broken, error, expected):
    """The half of this that the first fix missed: the checks ran inside the training loop.

    A typo on domain 2 was therefore reported *after* domain 1 had trained and been folded in --
    at the shipped recipe, about two hours of GPU time before the sentence appeared. The
    duplicate-name check twenty lines above had already been hoisted for exactly that reason;
    this test is the rest of the spec it was written to.
    """
    ws = _chain_workspace(tmp_path, base_dir)
    domains = [{"name": f"d{i}", "corpus": str(corpus_a if i % 2 else corpus_b)}
               for i in range(1, 4)]
    domains[position - 1].update(broken)
    if broken.get("corpus") is None and "corpus" in broken:
        domains[position - 1].pop("corpus")
    spec = tmp_path / "domains.yaml"
    spec.write_text(yaml.safe_dump({"domains": domains}))

    with pytest.raises(error, match=expected) as refusal:
        ws.chain(spec, device="cpu", need=NEED, k_domain=K_DOMAIN)

    assert f"domain {position}" in str(refusal.value)
    assert str(spec) in str(refusal.value)
    # Nothing trained: not the broken domain, and not the good ones before it either.
    assert ws.history == [] and not (ws.path / "runs").exists()
    assert ws.state["stage"] == 0


@pytest.mark.parametrize("extra, expected", [
    ({"extend_between": False}, "always folds each domain"),
    ({"banana": 7}, "does not have"),
])
def test_a_domain_carrying_a_field_a_domain_does_not_have_is_refused(tmp_path, registry, base_dir,
                                                                    corpus_a, extra, expected):
    """The rule a recipe file already lives under, applied to a chain spec's domain entries.

    `extend_between` is refused at the spec's TOP level with an explanation; per-domain -- the
    more natural place to put a field about what happens between this domain and the next -- it
    was accepted in silence and the chain started a full stage at the shipped recipe. An
    invented key was likewise never looked at.
    """
    ws = _chain_workspace(tmp_path, base_dir)
    spec = tmp_path / "domains.yaml"
    spec.write_text(yaml.safe_dump({"domains": [{"name": "alpha", "corpus": str(corpus_a),
                                                 **extra}]}))

    with pytest.raises(ValueError, match=expected) as refusal:
        ws.chain(spec, device="cpu", need=NEED, k_domain=K_DOMAIN)

    assert str(spec) in str(refusal.value)
    assert "corpus, epochs, name" in str(refusal.value)    # what a domain may carry
    assert ws.history == [] and not (ws.path / "runs").exists()


def test_a_spec_with_no_domains_is_rejected(tmp_path, registry, base_dir):
    ws = _chain_workspace(tmp_path, base_dir)
    spec = tmp_path / "empty.yaml"
    spec.write_text(yaml.safe_dump({"domains": []}))

    with pytest.raises(ValueError, match="domains"):
        ws.chain(spec, device="cpu")


def _chain_workspace(tmp_path, base_dir) -> Workspace:
    """A workspace whose default recipe is the tiny one, so a spec need not name it."""
    recipe_path = tiny_recipe(base_dir).save(tmp_path / "tiny_recipe.yaml")
    return new_workspace(tmp_path / "ws", base_dir, recipe=str(recipe_path))


# --------------------------------------------------------------------- self-generated init
SELF_GENERATED = "self-generated"


def test_init_self_generated_builds_into_v1_and_records_provenance(tmp_path, base_dir,
                                                                    tiny_artifact, monkeypatch):
    import lfa.workspace as ws_module
    _, fixture = tiny_artifact

    def fake_build(model_id, out_path, options, **kwargs):
        Path(out_path).write_bytes(fixture.read_bytes())
        Path(out_path).with_suffix(".corpus.jsonl").write_text('{"text": "x"}\n')
        Path(str(Path(out_path).with_suffix(".corpus.jsonl")) + ".manifest.json").write_text(
            '{"corpus_sha256": "%s"}' % ("e" * 64))
        return Path(out_path)
    monkeypatch.setattr(ws_module, "build_artifact_self_generated", fake_build)

    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact=SELF_GENERATED)

    assert (tmp_path / "ws" / "artifacts" / "v1.pt").is_file()
    assert (tmp_path / "ws" / "artifacts" / "v1.corpus.jsonl").is_file()
    assert ws.state["artifact_id"] == "self-generated:" + "e" * 12
    assert ws.state["artifact_provenance"] == SELF_GENERATED


def test_init_self_generated_rolls_back_when_the_build_raises(tmp_path, base_dir, monkeypatch):
    import lfa.workspace as ws_module

    def failing(model_id, out_path, options, **kwargs):
        raise RuntimeError("no card")
    monkeypatch.setattr(ws_module, "build_artifact_self_generated", failing)

    with pytest.raises(RuntimeError, match="no card"):
        Workspace.init(tmp_path / "ws", str(base_dir), artifact=SELF_GENERATED)
    assert not (tmp_path / "ws").exists()


def test_init_self_generated_rollback_removes_the_partial_files_the_build_left(
        tmp_path, base_dir, monkeypatch):
    """A build that dies mid-fit has already written its corpus; `rmdir` alone would keep it."""
    import lfa.workspace as ws_module

    def dies_after_the_corpus(model_id, out_path, options, **kwargs):
        corpus = Path(out_path).with_suffix(".corpus.jsonl")
        corpus.write_text('{"text": "x"}\n')
        Path(str(corpus) + ".manifest.json").write_text('{"corpus_sha256": "%s"}' % ("e" * 64))
        raise MemoryError("host RAM")
    monkeypatch.setattr(ws_module, "build_artifact_self_generated", dies_after_the_corpus)

    with pytest.raises(MemoryError, match="host RAM"):
        Workspace.init(tmp_path / "ws", str(base_dir), artifact=SELF_GENERATED)
    assert not (tmp_path / "ws").exists()


def test_init_self_generated_refuses_an_artifact_id_before_creating_anything(tmp_path, base_dir):
    with pytest.raises(ValueError, match="corpus hash"):
        Workspace.init(tmp_path / "ws", str(base_dir), artifact=SELF_GENERATED,
                       artifact_id="tiny")
    assert not (tmp_path / "ws").exists()


def test_every_other_init_records_no_provenance(tmp_path, base_dir, tiny_artifact):
    _, fixture = tiny_artifact
    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact=str(fixture))
    assert ws.state["artifact_provenance"] is None
    assert Workspace.open(tmp_path / "ws").state["artifact_provenance"] is None


def test_train_reads_a_self_generated_artifact_by_its_provenance_not_its_id(
        tmp_path, base_dir, corpus_a, tiny_artifact, monkeypatch, caplog):
    """The recipe is calibrated against "tiny"; the self-generated id is not that. Without the
    artifact's meta the run would warn about an artifact mismatch; with it, the note is the
    calibrate-lambda one for this model's own text."""
    import lfa.workspace as ws_module
    from lfa.artifact.schema import META_KEY

    params, _ = tiny_artifact

    def fake_build(model_id, out_path, options, **kwargs):
        built = copy.deepcopy(params)
        built[META_KEY] = dict(built[META_KEY], model_id=model_id, provenance=SELF_GENERATED,
                               corpus_sha256="e" * 64)
        torch.save(built, out_path)
        corpus = Path(out_path).with_suffix(".corpus.jsonl")
        corpus.write_text('{"text": "x"}\n')
        Path(str(corpus) + ".manifest.json").write_text('{"corpus_sha256": "%s"}' % ("e" * 64))
        return Path(out_path)
    monkeypatch.setattr(ws_module, "build_artifact_self_generated", fake_build)

    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact=SELF_GENERATED)
    assert ws._artifact_meta()["provenance"] == SELF_GENERATED
    with caplog.at_level("WARNING", logger="lfa.workspace"):
        ws.train(corpus_a, recipe=tiny_recipe(base_dir), device="cpu")

    warned = [r.message for r in caplog.records if r.name == "lfa.workspace"
              and r.levelname == "WARNING"]
    assert any("fitted on the model's own text" in note for note in warned)
    assert not any("calibrated against" in note for note in warned)


def test_artifact_meta_is_empty_without_an_artifact(tmp_path, registry, base_dir):
    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact="tiny", fetch=False)
    assert ws._artifact_meta() == {}


def test_init_self_generated_rollback_keeps_files_it_did_not_write(tmp_path, base_dir,
                                                                    monkeypatch):
    """The rollback removes what this init made -- not a v1 file that was already there."""
    import lfa.workspace as ws_module

    def failing(model_id, out_path, options, **kwargs):
        Path(out_path).with_suffix(".corpus.jsonl").write_text('{"text": "x"}\n')
        raise RuntimeError("no card")
    monkeypatch.setattr(ws_module, "build_artifact_self_generated", failing)
    kept = tmp_path / "ws" / "artifacts" / "v1.pt"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"not mine")

    with pytest.raises(RuntimeError, match="no card"):
        Workspace.init(tmp_path / "ws", str(base_dir), artifact=SELF_GENERATED)
    assert kept.read_bytes() == b"not mine"
    assert not (tmp_path / "ws" / "artifacts" / "v1.corpus.jsonl").exists()
