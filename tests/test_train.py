"""The training loop: objective composition, history, checkpoints, resume, guards.

Everything here runs on the CPU tiny model, so a "run" is two epochs over eight short
documents. The point is behaviour -- what lands in `training_history.json`, that a resumed run
really learns, that a switched-off anchor is switched off -- not convergence.
"""

import copy
import json

import pytest
import torch

from lfa.adapters import get_adapter
from lfa.corpus import ChunkedCorpus, make_dataloader
from lfa.models import apply_lora
from lfa.sampler import Sampler
from lfa.train import TrainConfig, train, train_epoch, train_step

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
