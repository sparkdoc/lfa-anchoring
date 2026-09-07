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
from lfa.workspace import StageOrderError

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


def test_an_artifact_that_is_neither_an_id_nor_a_path_says_both(tmp_path, base_dir):
    with pytest.raises(ValueError, match="qwen3-0.6b-diagonal"):
        Workspace.init(tmp_path, str(base_dir), artifact="no-such-artifact")


def test_init_refuses_to_overwrite_an_existing_workspace(tmp_path, registry, base_dir):
    new_workspace(tmp_path, base_dir)
    with pytest.raises(FileExistsError, match="open"):
        new_workspace(tmp_path, base_dir)


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

    assert any("loader frame" in r.message and "not comparable" in r.message
               for r in caplog.records)


def test_the_paper_loader_frame_says_nothing(tmp_path, registry, base_dir, corpus_a, caplog):
    ws = new_workspace(tmp_path, base_dir)

    with caplog.at_level("INFO", logger="lfa.workspace"):
        ws.train(corpus_a, recipe=tiny_recipe(base_dir, keep_short_whole=False), device="cpu")

    assert not any("loader frame" in r.message for r in caplog.records)


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
    assert not any("loader frame" in record.message for record in caplog.records)


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
