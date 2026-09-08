"""The training loop: objective composition, history, checkpoints, resume, guards.

Everything here runs on the CPU tiny model, so a "run" is two epochs over eight short
documents. The point is behaviour -- what lands in `training_history.json`, that a resumed run
really learns, that a switched-off anchor is switched off -- not convergence.
"""

import copy
import json
import logging
import shutil
import math
import warnings
from pathlib import Path

import pytest
import safetensors.torch
import torch

from lfa.adapters import get_adapter
from lfa.corpus import ChunkedCorpus, make_dataloader
from lfa.models import apply_lora
from lfa.sampler import Sampler
from lfa.train import (
    EMBED_ANCHOR_DISABLED_NOTICE,
    OVERTRAINING_RATIO,
    ResumeSourceHasNoAdapter,
    TrainConfig,
    _build_scheduler,
    held_out_turned_around,
    train,
    train_epoch,
    train_step,
)

FULL_WEIGHT_NOTICE = (
    "Full-weight anchoring is unvalidated on this model in the LFA paper (LoRA is the validated "
    "path); calibrate λ in 50,000–100,000 and check held-out domain perplexity, not only "
    "general-text perplexity."
)


def make_config(**overrides) -> TrainConfig:
    """The small-and-stable run every test here starts from."""
    kwargs = dict(
        lambda_qkv=10.0, lambda_mlp=10.0, mu=0.05,
        n_anchor_samples=4,
        learning_rate=1e-2, warmup_steps=1, logging_steps=1,
        batch_size=2, gradient_accumulation_steps=1, num_epochs=2, sequence_length=64,
        use_lora=True, lora_rank=2, lora_alpha=4,
    )
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


@pytest.fixture
def setup(tiny_model, tiny_texts, tiny_artifact, tmp_path):
    """(teacher, student_factory, dataset, sampler, adapter) for a CPU run.

    The base config is written to disk and named on every copy, because that is how a real run
    is set up (the student comes from a checkpoint) and because PEFT reads the base model's
    config when it saves an adapter.
    """
    model, tokenizer = tiny_model
    _, artifact_path = tiny_artifact
    base_dir = tmp_path / "base"
    model.config.save_pretrained(base_dir)

    teacher = copy.deepcopy(model).eval()
    for param in teacher.parameters():
        param.requires_grad = False
    adapter = get_adapter(teacher)
    dataset = ChunkedCorpus(tiny_texts[:8], tokenizer, max_length=64)
    sampler = Sampler(artifact_path, device="cpu", seed=0)

    def fresh_student():
        student = copy.deepcopy(model)
        student.config._name_or_path = str(base_dir)
        student.name_or_path = str(base_dir)
        return student

    return teacher, fresh_student, dataset, sampler, adapter


def _one_batch(dataset):
    """One collated micro-batch, without going through a DataLoader."""
    return next(iter(make_dataloader(dataset, 2, False, 42, 0)))


# --- the objective ---------------------------------------------------------------------------

def test_history_records_two_epochs_and_content_falls_from_the_baseline(setup, tmp_path):
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config()
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)

    state = train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    history = json.loads((tmp_path / "training_history.json").read_text())
    assert [entry["epoch"] for entry in history] == [1, 2]
    assert all(entry["loss_anchor"] >= 0 for entry in history)
    assert state.baseline is not None
    assert history[-1]["loss_content"] < state.baseline["loss_content"]
    assert (tmp_path / "final_model" / "adapter_model.safetensors").exists()
    assert (tmp_path / "training_state.pt").exists()
    assert json.loads((tmp_path / "config.json").read_text())["lambda_qkv"] == 10.0


def test_anchor_and_mu_off_leaves_the_content_loss_alone(setup, tmp_path):
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config(lambda_qkv=0.0, lambda_mlp=0.0, mu=0.0)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)

    _, metrics = train_step(student, teacher, _one_batch(dataset), sampler, adapter, config, None)
    assert metrics.loss_anchor == 0.0
    assert metrics.loss_mu == 0.0
    assert metrics.loss_total == pytest.approx(metrics.loss_content)

    state = train(teacher, student, dataset, sampler, adapter, config, tmp_path)
    history = json.loads((tmp_path / "training_history.json").read_text())
    assert all(entry["loss_anchor"] == 0.0 for entry in history)
    assert all(entry["loss_total"] == pytest.approx(entry["loss_content"]) for entry in history)
    assert state.baseline["loss_anchor"] == 0.0


def test_anchor_is_positive_once_the_student_has_moved(setup, tmp_path):
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config()
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)
    for name, param in student.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                param.normal_(0, 0.1)

    _, metrics = train_step(student, teacher, _one_batch(dataset), sampler, adapter, config, None)
    assert metrics.loss_anchor > 0 and metrics.loss_qkv > 0 and metrics.loss_mlp > 0
    assert metrics.loss_mu > 0
    assert metrics.loss_total == pytest.approx(
        metrics.loss_content + metrics.loss_anchor + metrics.loss_mu
    )


# --- checkpoints and resume ------------------------------------------------------------------

def test_periodic_checkpoints_are_written_and_final_model_is_what_ships(setup, tmp_path):
    """There is no `best_model`: the run ships `final_model`, and nothing selects on train loss."""
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config(checkpoint_mode="all", checkpoint_every=2)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)

    train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    assert (tmp_path / "checkpoint_epoch_2").is_dir()
    assert not (tmp_path / "checkpoint_epoch_1").exists()
    assert (tmp_path / "final_model").is_dir()
    assert not (tmp_path / "best_model").exists()


def _apply_repo_warning_filters() -> list[str]:
    """Apply this repo's `pyproject.toml` warning filters to the current `catch_warnings` scope.

    Read from the file and parsed with pytest's own parser, so a test that wants the project's
    real strictness gets exactly it -- `error` first, then the two torch pin-memory deprecations
    that are exempted by message. Returns the raw specs, for a test that wants to assert on them.
    """
    import tomllib

    from _pytest.config import parse_warning_filter

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    specs = tomllib.loads(pyproject.read_text())["tool"]["pytest"]["ini_options"]["filterwarnings"]
    for spec in specs:
        warnings.filterwarnings(*parse_warning_filter(spec, escape=False))
    return specs


def test_the_repo_turns_warnings_into_errors():
    """The premise of the test below: a warning is a failure here, not a line of noise."""
    assert _apply_repo_warning_filters()[0] == "error"


def test_a_run_over_a_hub_id_base_saves_offline_without_warning(setup, tmp_path, monkeypatch):
    """A checkpoint save must not warn, because a warning is an error in this suite.

    `PeftModel.save_pretrained` resolves `save_embedding_layers="auto"` by asking the Hub whether
    the base model's config exists. With a Hub-id base and no network -- exactly the case the
    acceptance harness creates, since it sets `HF_HUB_OFFLINE=1` itself -- that check cannot
    answer and PEFT warns. Under this repo's `filterwarnings = ["error", ...]` that warning aborts
    training at the first checkpoint, so this is a two-second stand-in for the two-hour run that
    would otherwise be the only thing to catch it.

    The repo's own `filterwarnings` list is read out of `pyproject.toml` and applied around the
    call, rather than trusting how pytest happened to be invoked -- so the guard holds under `-p
    no:cacheprovider`, a different `-c`, or a bare `pytest tests/test_train.py`, and it stays in
    step with the config instead of duplicating it.
    """
    teacher, fresh_student, dataset, sampler, adapter = setup
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    config = make_config(num_epochs=1, checkpoint_mode="all", checkpoint_every=1)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)
    # What a run over `Qwen/Qwen3-0.6B` records, and what the Hub lookup then fails on.
    student.peft_config["default"].base_model_name_or_path = "Qwen/Qwen3-0.6B"

    with warnings.catch_warnings():
        _apply_repo_warning_filters()
        train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    assert (tmp_path / "final_model" / "adapter_model.safetensors").exists()
    assert (tmp_path / "checkpoint_epoch_1" / "adapter_model.safetensors").exists()


@pytest.mark.parametrize("freeze_embed", [True, False])
def test_deciding_save_embedding_layers_ourselves_saves_the_same_tensors(freeze_embed, setup,
                                                                        tmp_path):
    """The fix must be silent, not lossy: same adapter, with and without PEFT's "auto".

    The unfrozen case is the one that could have gone wrong -- the embedding is saved there, but
    through `modules_to_save`, which this flag does not govern.
    """
    from lfa.train import _save_student

    _, fresh_student, _, _, adapter = setup
    config = make_config(freeze_embed=freeze_embed)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank,
                         alpha=config.lora_alpha, freeze_embed=freeze_embed)
    student.peft_config["default"].base_model_name_or_path = "Qwen/Qwen3-0.6B"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # the "auto" path warns; that is the point
        student.save_pretrained(tmp_path / "auto")
    _save_student(student, tmp_path / "ours")

    auto = safetensors.torch.load_file(tmp_path / "auto" / "adapter_model.safetensors")
    ours = safetensors.torch.load_file(tmp_path / "ours" / "adapter_model.safetensors")
    assert set(auto) == set(ours) and auto
    assert all(torch.equal(auto[key], ours[key]) for key in auto)
    assert any("embed" in key for key in ours) is (not freeze_embed)


def test_resume_reattaches_the_adapter_and_keeps_learning(setup, tmp_path):
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config()
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)
    train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    # A resume from a *fresh process* has only the base model and the saved adapter.
    extended = make_config(num_epochs=3)
    state = train(teacher, fresh_student(), dataset, sampler, adapter, extended, tmp_path,
                  resume=True)

    assert state.epoch == 3
    history = json.loads((tmp_path / "training_history.json").read_text())
    assert [entry["epoch"] for entry in history] == [1, 2, 3]
    assert history[-1]["grad_norm"] > 0
    assert history[-1]["global_step"] > history[-2]["global_step"]

    # The re-attached model is a NEW object, so the run has to hand it back: the caller's own
    # `student` variable still points at the bare base model.
    from peft import get_peft_model_state_dict
    trainable = [name for name, p in state.model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name in trainable), trainable
    saved = safetensors.torch.load_file(tmp_path / "final_model" / "adapter_model.safetensors")
    live = get_peft_model_state_dict(state.model)
    assert set(saved) == set(live) and saved
    assert all(torch.equal(saved[key], live[key].detach().cpu()) for key in saved)


def test_resume_without_a_prior_run_says_so(setup, tmp_path):
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config()
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)
    with pytest.raises(FileNotFoundError, match="Cannot resume"):
        train(teacher, student, dataset, sampler, adapter, config, tmp_path, resume=True)


# --- the held-out pass -----------------------------------------------------------------------

def test_the_held_out_split_is_scored_after_every_epoch(setup, tiny_model, tmp_path, tiny_texts):
    """Loss and perplexity per epoch, from the documents the run never trains on."""
    teacher, fresh_student, dataset, sampler, adapter = setup
    _, tokenizer = tiny_model
    holdout = ChunkedCorpus(tiny_texts[8:], tokenizer, max_length=64)
    config = make_config()
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)

    state = train(teacher, student, dataset, sampler, adapter, config, tmp_path,
                  tokenizer=tokenizer, val_dataset=holdout)

    history = json.loads((tmp_path / "training_history.json").read_text())
    assert len(history) == 2
    for record in history:
        assert record["val_loss"] > 0
        assert record["val_perplexity"] == pytest.approx(math.exp(record["val_loss"]), rel=1e-6)
        assert record["val_tokens"] == history[0]["val_tokens"] > 0    # the same text every epoch
    assert state.history == history


# --------------------------------------------------- the run that trained past its optimum
#
# The shipped recipe's epoch count was tuned on ~1,700 documents. On the small corpus a first
# user actually brings, the held-out perplexity turns around early -- one measured run bottomed
# at epoch 2 (6.67) and ended at epoch 15 (16.88), and `final_model` is the last epoch by
# design. The trainer printed all fifteen numbers and drew no conclusion from them.

def _curve(values):
    return [{"epoch": i + 1, "val_perplexity": value} for i, value in enumerate(values)]


@pytest.mark.parametrize("values, expected", [
    # The measured over-training run: minimum at epoch 2, ending 2.5x worse.
    ([6.91, 6.67, 6.78, 6.84, 7.53, 14.46, 16.88], (2, 6.67, 16.88)),
    ([9.0, 8.0, 7.0, 6.0], None),                        # still falling: nothing to say
    ([9.0, 8.0, 7.0, 7.0 * OVERTRAINING_RATIO * 1.01], (3, 7.0, 7.0 * OVERTRAINING_RATIO * 1.01)),
    ([9.0, 8.0, 7.0, 7.2], None),                        # an ordinary wobble is not a finding
    # Two points are enough: the shortest runs are exactly where a small corpus turns early.
    ([6.6, 305.9], (1, 6.6, 305.9)),
    ([9.0, 8.0], None),                                  # two points, still falling
    ([], None),                                          # val_fraction=0: no curve at all
])
def test_the_turnaround_is_reported_only_when_it_is_one(values, expected):
    assert held_out_turned_around(_curve(values)) == expected


def test_a_run_shorter_than_its_own_warmup_says_the_corpus_is_too_small(setup, tmp_path,
                                                                        caplog):
    """Three documents, two optimizer steps, a much worse model, and not a word about it.

    Said in the recipe's own terms rather than as a corpus-size rule of thumb: a run with fewer
    optimizer steps than its warmup never reaches the learning rate the operating point was tuned
    at, so whatever comes out is not that operating point.
    """
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config(num_epochs=1, warmup_steps=500)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)

    with caplog.at_level(logging.WARNING, logger="lfa.train"):
        train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    warned = [record.getMessage() for record in caplog.records
              if "very small for this recipe" in record.getMessage()]
    assert len(warned) == 1
    assert "against a warmup of 500" in warned[0]
    assert "More documents is the fix" in warned[0]


def test_an_ordinary_run_is_not_called_too_small(setup, tmp_path, caplog):
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config(num_epochs=1, warmup_steps=1)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)

    with caplog.at_level(logging.WARNING, logger="lfa.train"):
        train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    assert not [r for r in caplog.records if "very small for this recipe" in r.getMessage()]


def test_a_history_without_validation_numbers_says_nothing(tiny_texts):
    assert held_out_turned_around([{"epoch": 1, "loss_total": 2.0}] * 5) is None


def test_a_run_that_trained_past_its_optimum_says_so_and_names_the_dose(setup, tmp_path,
                                                                        monkeypatch, caplog):
    """The wiring: the trainer reads its own curve at the end and turns it into one sentence."""
    import lfa.train as train_module

    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config(num_epochs=1)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)
    monkeypatch.setattr(train_module, "held_out_turned_around", lambda history: (2, 6.67, 16.88))

    with caplog.at_level(logging.WARNING, logger="lfa.train"):
        train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    warned = [record.getMessage() for record in caplog.records
              if record.levelname == "WARNING" and "past its own optimum" in record.getMessage()]
    assert len(warned) == 1
    assert "epoch 2 (6.670)" in warned[0] and "16.880" in warned[0]
    assert "--epochs 2" in warned[0]                      # the dose to re-run at
    assert "no best checkpoint is kept" in warned[0]      # why final_model is not that model


def test_a_run_that_is_still_improving_says_nothing_about_its_dose(setup, tiny_model, tmp_path,
                                                                   tiny_texts, caplog):
    teacher, fresh_student, dataset, sampler, adapter = setup
    _, tokenizer = tiny_model
    holdout = ChunkedCorpus(tiny_texts[8:], tokenizer, max_length=64)
    config = make_config()
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)

    with caplog.at_level(logging.WARNING, logger="lfa.train"):
        train(teacher, student, dataset, sampler, adapter, config, tmp_path,
              tokenizer=tokenizer, val_dataset=holdout)

    assert not [r for r in caplog.records if "past its own optimum" in r.getMessage()]


def test_without_a_held_out_split_the_history_carries_no_validation_columns(setup, tmp_path):
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config(num_epochs=1)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)

    train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    [record] = json.loads((tmp_path / "training_history.json").read_text())
    assert "val_loss" not in record and "val_perplexity" not in record


@pytest.mark.parametrize("freeze_embed", [True, False])
def test_the_held_out_pass_leaves_the_training_stream_untouched(freeze_embed, setup, tiny_model,
                                                                tiny_artifact, tmp_path,
                                                                tiny_texts):
    """The proof that validation is free: the same run with and without it trains identically.

    A held-out pass that drew from the training RNG, or left the model in `eval()`, or stepped
    anything, would move the losses -- so the two runs are compared as exact floats, not as
    approximations, and their adapters must come out byte for byte the same.

    `freeze_embed=False` is the case that actually bites, and it is why this is parametrized.
    Iterating a `DataLoader` draws a base seed from torch's global CPU RNG, and the embedding
    anchor is the one block that used to draw from that same stream (`lfa.losses.embed_anchor_loss`
    picks token ids by corpus frequency). With the term live, a validation pass therefore shifted
    every later embed draw -- invisibly, because the shipped recipe freezes the embedding and
    switches the term off. The draw now goes through the sampler's private generator, so the
    claim holds in both configurations rather than only in the shipped one.
    """
    teacher, fresh_student, dataset, _, adapter = setup
    _, tokenizer = tiny_model
    _, artifact_path = tiny_artifact
    holdout = ChunkedCorpus(tiny_texts[8:], tokenizer, max_length=64)

    def run(directory, val_dataset):
        config = make_config(freeze_embed=freeze_embed)
        dataset.rechunk(0, config.seed)
        torch.manual_seed(0)
        # A FRESH sampler per run: it carries its own generator, so reusing one object would
        # make the second run differ for a reason that has nothing to do with validation.
        sampler = Sampler(artifact_path, device="cpu", seed=0)
        if not freeze_embed:            # the term needs the layer-0 table to be live at all
            sampler.build_embedding_lookup_from_model(teacher, adapter)
        student = apply_lora(fresh_student(), adapter, rank=config.lora_rank,
                             alpha=config.lora_alpha, freeze_embed=freeze_embed)
        return train(teacher, student, dataset, sampler, adapter, config, directory,
                     tokenizer=tokenizer, val_dataset=val_dataset)

    without = run(tmp_path / "without", None)
    with_val = run(tmp_path / "with", holdout)

    # The embed term must actually be running in the unfrozen case, or this proves nothing there.
    assert (without.history[-1]["loss_embed"] > 0) is (not freeze_embed)

    for a, b in zip(without.history, with_val.history):
        for key in ("loss_total", "loss_content", "loss_anchor", "loss_embed", "loss_mu",
                    "grad_norm", "learning_rate", "global_step"):
            assert a[key] == b[key], key
    plain = safetensors.torch.load_file(tmp_path / "without" / "final_model"
                                        / "adapter_model.safetensors")
    validated = safetensors.torch.load_file(tmp_path / "with" / "final_model"
                                            / "adapter_model.safetensors")
    assert set(plain) == set(validated)
    assert all(torch.equal(plain[key], validated[key]) for key in plain)


def test_the_embedding_anchor_draws_from_the_sampler_not_the_global_stream(setup, tiny_artifact,
                                                                           tiny_model):
    """The mechanism behind the test above, isolated: a seeded sampler owns the token draw.

    Two calls with equally-seeded samplers agree even though the global RNG has been moved between
    them, and an unseeded sampler still follows the global stream -- which is what keeps the
    unseeded path call-for-call identical to the reference implementation.
    """
    from lfa.losses import embed_anchor_loss

    teacher, fresh_student, _, _, adapter = setup
    _, artifact_path = tiny_artifact
    student = fresh_student()
    with torch.no_grad():
        student.get_input_embeddings().weight.add_(0.05)

    def embed_loss(seed, disturb):
        sampler = Sampler(artifact_path, device="cpu", seed=seed)
        sampler.build_embedding_lookup_from_model(teacher, adapter)
        torch.manual_seed(11)
        if disturb:
            torch.randint(0, 100, (1,))          # what iterating a DataLoader costs
        loss = embed_anchor_loss(teacher, student, sampler, adapter, n_samples=8)
        return float(loss.detach())

    assert embed_loss(seed=0, disturb=False) == embed_loss(seed=0, disturb=True)
    assert embed_loss(seed=None, disturb=False) != embed_loss(seed=None, disturb=True)


# --- guards ----------------------------------------------------------------------------------

def test_full_weight_warns_that_it_is_unvalidated(setup, tmp_path, caplog):
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config(full_weight=True, use_lora=False, num_epochs=1)

    with caplog.at_level("WARNING", logger="lfa.train"):
        train(teacher, fresh_student(), dataset, sampler, adapter, config, tmp_path)

    assert sum(record.message == FULL_WEIGHT_NOTICE for record in caplog.records) == 1


def test_a_frozen_student_is_reported_not_silently_trained(setup, tmp_path, caplog):
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config(num_epochs=1, use_lora=False)
    student = fresh_student()
    for param in student.parameters():
        param.requires_grad = False
    # The symptom this guard exists for: gradient checkpointing makes the embedding output
    # require grad, so the loss is differentiable and the run proceeds happily while no
    # PARAMETER receives a gradient. That is what a frozen (is_trainable=False) adapter does.
    student.enable_input_require_grads()

    with caplog.at_level("WARNING", logger="lfa.train"):
        train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    frozen = [r for r in caplog.records if "adapter frozen" in r.message]
    assert len(frozen) == 1, [r.message for r in caplog.records]
    history = json.loads((tmp_path / "training_history.json").read_text())
    assert history[-1]["grad_norm"] == 0.0


def test_rolling_checkpoints_overwrite_and_leave_a_resumable_state(setup, tmp_path):
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config(checkpoint_mode="rolling", checkpoint_every=1)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)

    train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    assert (tmp_path / "latest_model").is_dir()
    assert not any(p.name.startswith("checkpoint_epoch_") for p in tmp_path.iterdir())


def test_an_empty_epoch_is_skipped_rather_than_dividing_by_zero(setup, tiny_model, caplog):
    teacher, fresh_student, _, sampler, adapter = setup
    _, tokenizer = tiny_model
    config = make_config()
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)
    empty = ChunkedCorpus([], tokenizer, max_length=64)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)

    with caplog.at_level("WARNING", logger="lfa.train"):
        metrics, global_step = train_epoch(
            student, teacher, make_dataloader(empty, 2, False, 42, 0), sampler, adapter,
            config, optimizer, global_step=7,
        )

    assert (metrics.num_steps, global_step) == (0, 7)
    assert any("Empty training epoch" in record.message for record in caplog.records)


def test_accumulation_steps_over_an_odd_number_of_batches(setup):
    """ga=2 halves the optimizer steps, and the trailing odd micro-batch still takes one."""
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config(gradient_accumulation_steps=2, num_epochs=1)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)
    dataloader = make_dataloader(dataset, config.batch_size, False, config.seed, 0)
    n_batches = len(dataloader)
    assert n_batches % 2 == 1, "this test needs an odd batch count to exercise the last window"
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)

    metrics, global_step = train_epoch(student, teacher, dataloader, sampler, adapter, config,
                                       optimizer)

    assert metrics.num_steps == math.ceil(n_batches / 2) == global_step


def test_resume_inside_the_warmup_replays_the_schedule(setup, tmp_path):
    """A mid-warmup resume must replay the schedule, not compound the lr the optimizer saved.

    torch's schedulers are chainable -- a step multiplies the group's current lr -- so restoring
    the optimizer (which overwrites that lr with a mid-warmup value) and then fast-forwarding
    compounds the restored value. The interrupted run must land on the same learning rate as the
    uninterrupted one.
    """
    teacher, fresh_student, dataset, sampler, adapter = setup
    # The warmup is set to two and a half epochs so that the epoch the comparison reads (the
    # second) is strictly inside it. That is where the bug lives: at the warmup's end SequentialLR
    # re-derives the next phase's rate from `base_lrs`, which wipes a compounded value and hides
    # it.
    dataset.rechunk(0, make_config().seed)
    steps_per_epoch = math.ceil(len(dataset) / 6)
    knobs = dict(warmup_steps=int(2.5 * steps_per_epoch), batch_size=6)

    def run(directory, epochs, resume=False):
        config = make_config(num_epochs=epochs, **knobs)
        dataset.rechunk(0, config.seed)          # same chunking => same steps_per_epoch
        student = fresh_student() if resume else apply_lora(
            fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)
        return train(teacher, student, dataset, sampler, adapter, config, directory,
                     resume=resume)

    uninterrupted = run(tmp_path / "one_go", 3)
    run(tmp_path / "interrupted", 1)
    resumed = run(tmp_path / "interrupted", 3, resume=True)

    assert resumed.history[-1]["global_step"] == uninterrupted.history[-1]["global_step"]
    # End of epoch 2: the first epoch the resume is responsible for, and still inside the warmup.
    assert resumed.history[1]["learning_rate"] == pytest.approx(
        uninterrupted.history[1]["learning_rate"], rel=1e-9)
    # ...and it is still climbing through the warmup, not sitting at the peak.
    assert 0 < resumed.history[1]["learning_rate"] < make_config().learning_rate


def _research_schedule(optimizer, learning_rate, warmup_steps, total_steps, lr_floor=0.0):
    """the research code's scheduler, transcribed, at ``cosine_fraction=1.0, plateau_end_factor=1.0``.

    From ``src/lra_training.py`` (the phase-boundary block that builds ``warmup → plateau →
    cosine``): with ``cosine_fraction=1.0`` the plateau is empty and the schedule is the
    ``warmup → cosine`` branch. Kept here as an independent copy rather than imported, so that
    :func:`lfa.train._build_scheduler` is checked against the research formula itself and not
    against another call into its own code.
    """
    warmup_steps = min(warmup_steps, total_steps)
    cosine_steps = total_steps - warmup_steps
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                               total_iters=warmup_steps)
    if cosine_steps <= 0:
        constant = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0,
                                                       total_iters=total_steps - warmup_steps)
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, constant],
                                                     milestones=[warmup_steps])
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps,
                                                       eta_min=lr_floor * learning_rate)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, decay],
                                                 milestones=[warmup_steps])


@pytest.mark.parametrize("warmup_steps, steps_per_epoch, num_epochs, lr_floor", [
    (50, 603, 15, 0.0),        # the shipped point: warmup(50) → cosine(8995)
    (1, 7, 2, 0.0),            # a tiny run, where the warmup is most of it
    (50, 3, 5, 0.0),           # warmup longer than the whole run
    (10, 20, 3, 0.05),         # a non-zero floor
])
def test_the_cosine_schedule_matches_researchs_scheduler_step_for_step(
    warmup_steps, steps_per_epoch, num_epochs, lr_floor
):
    """``lr_schedule="cosine"`` is the research code's schedule at ``cosine_fraction=1.0``, exactly.

    The learning rate at a given step is the single largest lever on where a run lands (a wrong
    schedule cost 0.9 of domain perplexity once), so the two formulas are compared at *every*
    step rather than at the end.
    """
    config = make_config(learning_rate=3e-4, warmup_steps=warmup_steps, num_epochs=num_epochs,
                         lr_schedule="cosine", lr_floor=lr_floor)
    total_steps = steps_per_epoch * num_epochs

    ours_param = torch.nn.Parameter(torch.zeros(1))
    theirs_param = torch.nn.Parameter(torch.zeros(1))
    ours_opt = torch.optim.AdamW([ours_param], lr=config.learning_rate)
    theirs_opt = torch.optim.AdamW([theirs_param], lr=config.learning_rate)

    ours = _build_scheduler(ours_opt, config, total_steps, logging.getLogger("test"))
    theirs = _research_schedule(theirs_opt, config.learning_rate, warmup_steps, total_steps,
                                 lr_floor)

    with warnings.catch_warnings():                # stepping a scheduler with no optimizer step
        warnings.filterwarnings("ignore", message=r".*before `optimizer\.step\(\)`.*")
        for step in range(total_steps + 5):    # past the end too: both must stay well-defined
            assert ours_opt.param_groups[0]["lr"] == pytest.approx(
                theirs_opt.param_groups[0]["lr"], rel=1e-12, abs=1e-15), f"step {step}"
            ours.step()
            theirs.step()


def test_a_constant_schedule_holds_the_peak_after_the_warmup():
    """``lr_schedule="constant"``: the research code's no-cosine branch -- warmup, then flat."""
    config = make_config(learning_rate=3e-4, warmup_steps=5, lr_schedule="constant")
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=config.learning_rate)
    scheduler = _build_scheduler(optimizer, config, 40, logging.getLogger("test"))

    seen = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*before `optimizer\.step\(\)`.*")
        for _ in range(40):
            seen.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

    assert seen[0] == pytest.approx(3e-5)                      # start_factor 0.1
    assert all(lr == pytest.approx(3e-4) for lr in seen[5:])   # flat from the warmup's end


def test_an_unknown_lr_schedule_is_refused_at_construction():
    with pytest.raises(ValueError, match="lr_schedule"):
        make_config(lr_schedule="linear-warmdown")


def test_a_missing_embedding_lookup_is_warned_about_once(setup, tmp_path, caplog):
    """L_embed is off whenever the artifact ships no layer-0 table -- say so, once."""
    teacher, fresh_student, dataset, sampler, adapter = setup
    assert sampler.has_embedding_lookup() is False
    config = make_config(num_epochs=1, freeze_embed=False)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)

    with caplog.at_level("WARNING", logger="lfa.train"):
        state = train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    assert sum(record.message == EMBED_ANCHOR_DISABLED_NOTICE
               for record in caplog.records) == 1
    assert state.history[-1]["loss_embed"] == 0.0        # the term really is absent


def test_a_rebuilt_embedding_lookup_is_not_warned_about(setup, tmp_path, caplog):
    """With the table rebuilt from the teacher, L_embed is live and prices a moved embedding."""
    teacher, fresh_student, dataset, sampler, adapter = setup
    sampler.build_embedding_lookup_from_model(teacher, adapter)
    config = make_config(num_epochs=1, freeze_embed=False)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)
    with torch.no_grad():
        student.get_input_embeddings().weight.add_(0.05)

    with caplog.at_level("WARNING", logger="lfa.train"):
        state = train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    assert EMBED_ANCHOR_DISABLED_NOTICE not in [r.message for r in caplog.records]
    assert state.history[-1]["loss_embed"] > 0


def test_a_deliberately_frozen_embedding_is_not_warned_about(setup, tmp_path, caplog):
    """freeze_embed switches L_embed off on purpose; that is not the missing-table case."""
    teacher, fresh_student, dataset, sampler, adapter = setup
    assert sampler.has_embedding_lookup() is False
    config = make_config(num_epochs=1, freeze_embed=True)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)

    with caplog.at_level("WARNING", logger="lfa.train"):
        train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    assert EMBED_ANCHOR_DISABLED_NOTICE not in [r.message for r in caplog.records]


def test_a_lora_resume_whose_checkpoint_saved_no_adapter_is_refused(setup, tmp_path):
    """There is nothing to resume, and every later symptom is indirect.

    Left to run, this used to reach `optimizer.load_state_dict`, which raises about a parameter
    group of the wrong size -- true only because the saved moments happen to be there to
    disagree, and only after the run has been set up. `assert_trainable_params` cannot catch it
    either: `load_student` hands back a model whose every parameter is trainable.
    """
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config()
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)
    train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    shutil.rmtree(tmp_path / "final_model")               # a run whose adapter is not there

    with pytest.raises(ResumeSourceHasNoAdapter, match="no adapter"):
        train(teacher, fresh_student(), dataset, sampler, adapter, config, tmp_path, resume=True)
