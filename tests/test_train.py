"""The training loop: objective composition, history, checkpoints, resume, guards.

Everything here runs on the CPU tiny model, so a "run" is two epochs over eight short
documents. The point is behaviour -- what lands in `training_history.json`, that a resumed run
really learns, that a switched-off anchor is switched off -- not convergence.
"""

import copy
import json
import shutil
import math

import pytest
import safetensors.torch
import torch

from lfa.adapters import get_adapter
from lfa.corpus import ChunkedCorpus, make_dataloader
from lfa.models import apply_lora
from lfa.sampler import Sampler
from lfa.train import (
    EMBED_ANCHOR_DISABLED_NOTICE,
    ResumeSourceHasNoAdapter,
    TrainConfig,
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

def test_checkpoints_and_best_model_are_written(setup, tmp_path):
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config(checkpoint_mode="all", checkpoint_every=2)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)

    train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    assert (tmp_path / "checkpoint_epoch_2").is_dir()
    assert not (tmp_path / "checkpoint_epoch_1").exists()
    assert (tmp_path / "best_model").is_dir()


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
    # A horizon longer than the run keeps both runs strictly inside the warmup window, which is
    # where the bug lives (the next phase resets the lr from base_lrs and hides it).
    knobs = dict(warmup_steps=50, schedule_horizon_epochs=10, batch_size=6)

    def run(directory, epochs, resume=False):
        config = make_config(num_epochs=epochs, **knobs)
        dataset.rechunk(0, config.seed)          # same chunking => same steps_per_epoch
        student = fresh_student() if resume else apply_lora(
            fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)
        return train(teacher, student, dataset, sampler, adapter, config, directory,
                     resume=resume)

    uninterrupted = run(tmp_path / "one_go", 2)
    run(tmp_path / "interrupted", 1)
    resumed = run(tmp_path / "interrupted", 2, resume=True)

    assert resumed.history[-1]["global_step"] == uninterrupted.history[-1]["global_step"]
    assert resumed.history[-1]["learning_rate"] == pytest.approx(
        uninterrupted.history[-1]["learning_rate"], rel=1e-9)
    # ...and it is still climbing through the warmup, not sitting at the peak.
    assert 0 < resumed.history[-1]["learning_rate"] < make_config().learning_rate


def test_schedule_horizon_keeps_the_learning_rate_high_past_num_epochs(setup, tmp_path):
    """The recipe trains 15 epochs of a 100-epoch cosine; the horizon field is what allows that."""
    teacher, fresh_student, dataset, sampler, adapter = setup

    def final_lr(directory, **overrides):
        config = make_config(num_epochs=2, warmup_steps=1, **overrides)
        dataset.rechunk(0, config.seed)
        student = apply_lora(fresh_student(), adapter, rank=config.lora_rank,
                             alpha=config.lora_alpha)
        state = train(teacher, student, dataset, sampler, adapter, config, directory)
        return state.history[-1]["learning_rate"]

    short = final_lr(tmp_path / "short")
    long_horizon = final_lr(tmp_path / "long", schedule_horizon_epochs=20)

    assert long_horizon > 10 * short
    assert json.loads((tmp_path / "long" / "config.json").read_text())[
        "schedule_horizon_epochs"] == 20


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
