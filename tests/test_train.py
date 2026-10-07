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
    OPTIONAL_RERUN_GAP,
    HeldOutSummary,
    ResumeSourceHasNoAdapter,
    TrainConfig,
    _build_scheduler,
    _log_held_out_summary,
    held_out_summary,
    train,
    train_epoch,
    train_step,
)

FULL_WEIGHT_NOTICE = (
    "Full-weight anchoring is unvalidated: every measured λ is for LoRA. Re-calibrate λ for full "
    "weight (docs/model-integration-cookbook.md §5) and check held-out domain perplexity, not "
    "only general-text perplexity."
)


def make_config(**overrides) -> TrainConfig:
    """The small-and-stable run every test here starts from."""
    kwargs = dict(
        model_id="tiny",
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
    real strictness gets exactly it -- `error` first, then the exemptions `pyproject.toml`
    explains (deprecation-class warnings attributed to modules outside `lfa` and `tests`, and
    torch's NVML report). Returns the raw specs, for a test that wants to assert on them.
    """
    import tomllib

    from _pytest.config import parse_warning_filter

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    specs = tomllib.loads(pyproject.read_text())["tool"]["pytest"]["ini_options"]["filterwarnings"]
    for spec in specs:
        warnings.filterwarnings(*parse_warning_filter(spec, escape=False))
    return specs


def test_a_config_names_its_model():
    """No model is assumed: a run's config has to say which model it trains."""
    with pytest.raises(TypeError, match="model_id"):
        TrainConfig()
    assert make_config().model_id == "tiny"


def test_the_repo_turns_warnings_into_errors():
    """The premise of the test below: a warning is a failure here, not a line of noise."""
    assert _apply_repo_warning_filters()[0] == "error"


def test_a_run_over_a_hub_id_base_saves_offline_without_warning(setup, tmp_path, monkeypatch):
    """A checkpoint save must not warn, because a warning is an error in this suite.

    `PeftModel.save_pretrained` resolves `save_embedding_layers="auto"` by asking the Hub whether
    the base model's config exists. With a Hub-id base and no network -- exactly the case the
    port-verification harness creates, since it sets `HF_HUB_OFFLINE=1` itself -- that check cannot
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
# How many epochs a corpus carries depends on its size and on lambda. On a small corpus at a
# weak enough anchor the held-out perplexity turns around early -- one measured run at
# lambda = 100,000 bottomed at epoch 2 (6.67) and ended at epoch 15 (16.88), and `final_model` is
# the last epoch by design. The trainer printed all fifteen numbers and drew no conclusion from them.

def _curve(values):
    return [{"epoch": i + 1, "val_perplexity": value} for i, value in enumerate(values)]


@pytest.mark.parametrize("values, expected", [
    # The measured over-training run: minimum at epoch 2, ending 2.5x worse.
    ([6.91, 6.67, 6.78, 6.84, 7.53, 14.46, 16.88], ("turned", 2, 6.67, 7, 16.88)),
    # The measured Wells run: lowest 27.02 at epoch 8, ending 6.1 % above it -- under the warning
    # threshold, and still a curve that turned.
    ([30.0, 29.0, 28.5, 28.0, 27.6, 27.3, 27.1, 27.02, 27.2, 27.5, 27.8, 28.0, 28.3, 28.5, 28.67],
     ("turned", 8, 27.02, 15, 28.67)),
    ([9.0, 8.0, 7.0, 6.0], ("still_falling", 4, 6.0, 4, 6.0)),
    ([9.0, 8.0, 7.0, 7.2], ("turned", 3, 7.0, 4, 7.2)),   # a wobble still turned; the log grades it
    ([6.6, 305.9], ("turned", 1, 6.6, 2, 305.9)),        # two points are enough to turn
    ([9.0, 8.0], ("still_falling", 2, 8.0, 2, 8.0)),
    ([7.5], ("still_falling", 1, 7.5, 1, 7.5)),          # one point: its minimum is its last epoch
    ([9.0, 7.0, 7.0], ("turned", 2, 7.0, 3, 7.0)),       # a tie: the earliest epoch is the minimum
    ([], ("no_curve", None, None, None, None)),          # val_fraction=0: no curve at all
])
def test_the_held_out_summary_reads_the_curve(values, expected):
    summary = held_out_summary(_curve(values))
    assert (summary.verdict, summary.best_epoch, summary.best,
            summary.final_epoch, summary.final) == expected
    if summary.verdict == "no_curve":
        assert summary.gap is None
    else:
        assert summary.gap == pytest.approx(summary.final / summary.best - 1.0)


def test_the_wells_gap_is_six_point_one_percent():
    """The trial run that the old 1.10 threshold left silent: +6.1 %."""
    summary = held_out_summary(_curve([27.5, 27.02, 28.67]))
    assert summary.gap == pytest.approx(28.67 / 27.02 - 1.0)
    assert round(summary.gap * 100, 1) == 6.1


def test_epochs_without_a_held_out_number_are_skipped():
    history = [{"epoch": 1, "val_perplexity": 9.0},
               {"epoch": 2, "loss_total": 2.0},                       # missing
               {"epoch": 3, "val_perplexity": None},                  # None
               {"epoch": 4, "val_perplexity": float("nan")},          # NaN
               {"epoch": 5, "val_perplexity": 8.0}]
    assert held_out_summary(history) == HeldOutSummary(
        verdict="still_falling", best_epoch=5, best=8.0, final_epoch=5, final=8.0, gap=0.0)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan"), 0.0, -1.0])
def test_an_unusable_value_mid_curve_is_never_the_minimum_or_the_end(bad):
    """A bad epoch the run recovered from is skipped: it is neither the minimum nor the end."""
    summary = held_out_summary(_curve([9.0, 8.0, bad, 8.4]))
    assert (summary.verdict, summary.best_epoch, summary.final_epoch) == ("turned", 2, 4)
    assert summary.gap == pytest.approx(0.05)
    text = json.dumps(summary.to_dict(), allow_nan=False)        # raises on NaN or Infinity
    assert json.loads(text)["final"] == 8.4


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), float("-inf"), 0.0])
def test_an_unusable_last_value_is_a_divergence_not_a_shorter_run(bad):
    """The last epoch diverged: the summary must not quietly end at the last finite epoch (which
    would read "of 4" and "more epochs may lower it") -- it says the run diverged at epoch 5."""
    summary = held_out_summary(_curve([9.0, 8.0, 8.2, 8.4, bad]))
    assert summary == HeldOutSummary(verdict="diverged", best_epoch=2, best=8.0, final_epoch=5,
                                     final=None, gap=None)
    assert json.loads(json.dumps(summary.to_dict(), allow_nan=False))["final"] is None


def test_a_last_epoch_without_a_held_out_number_is_not_a_divergence():
    history = _curve([9.0, 8.0]) + [{"epoch": 3, "loss_total": 2.0}]
    assert held_out_summary(history).verdict == "still_falling"


def test_a_curve_of_nothing_but_unusable_values_diverged_with_no_minimum():
    summary = held_out_summary(_curve([float("inf"), float("nan"), float("inf")]))
    assert summary == HeldOutSummary(verdict="diverged", final_epoch=3)
    json.dumps(summary.to_dict(), allow_nan=False)


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_a_diverged_run_warns_not_to_ship_it(caplog, bad):
    said = _said(caplog, [9.0, 8.0, 8.2, 8.4, bad])
    assert said[0] == ("INFO", "Held-out perplexity: lowest 8.000 at epoch 2 of 5; final not "
                               "finite.")
    [(level, message)] = said[1:]
    assert level == "WARNING"
    assert "last epoch (5) was not finite" in message and "diverged" in message
    assert "should not be shipped" in message
    assert "--epochs 2" in message
    assert "stronger anchor or a lower learning rate" in message
    assert "more epochs" not in message


def test_a_run_that_was_never_finite_warns_without_a_dose(caplog):
    said = _said(caplog, [float("inf"), float("inf")])
    assert said[0] == ("INFO", "Held-out perplexity: no finite value at any epoch up to epoch 2.")
    [(level, message)] = said[1:]
    assert level == "WARNING" and "should not be shipped" in message
    assert "--epochs" not in message


def test_the_summary_is_a_plain_record_for_the_history():
    summary = held_out_summary(_curve([9.0, 8.0, 8.4]))
    record = summary.to_dict()
    assert record == {"verdict": "turned", "best_epoch": 2, "best": 8.0, "final_epoch": 3,
                      "final": 8.4, "gap": pytest.approx(0.05)}
    assert json.loads(json.dumps(record))["verdict"] == "turned"
    assert held_out_summary([]).to_dict() == {"verdict": "no_curve", "best_epoch": None,
                                              "best": None, "final_epoch": None, "final": None,
                                              "gap": None}


# ----------------------------------------------- what the end of a run says, verdict by verdict

def _said(caplog, values):
    """``[(levelname, message)]`` logged for the held-out curve ``values``."""
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="lfa.test.held_out"):
        _log_held_out_summary(held_out_summary(_curve(values)),
                              logging.getLogger("lfa.test.held_out"))
    return [(r.levelname, r.getMessage()) for r in caplog.records]


def test_a_turned_curve_past_the_threshold_warns_and_names_the_dose(caplog):
    said = _said(caplog, [6.91, 6.67, 6.78, 16.88])
    assert said[0] == ("INFO", "Held-out perplexity: lowest 6.670 at epoch 2 of 4; final 16.880 "
                               "(+153.1 % over the lowest).")
    [(level, warning)] = said[1:]
    assert level == "WARNING"
    assert "past its own optimum" in warning
    assert "epoch 2 (6.670)" in warning and "16.880, epoch 4" in warning
    assert "--epochs 2" in warning                        # the dose to re-run at
    assert "no best checkpoint is kept" in warning        # why final_model is not that model


def test_the_wells_run_is_no_longer_silent(caplog):
    """+6.1 %: under the warning threshold, and a re-run at its minimum was better on both axes."""
    said = _said(caplog, [27.5, 27.02, 28.67])
    assert said[0] == ("INFO", "Held-out perplexity: lowest 27.020 at epoch 2 of 3; final 28.670 "
                               "(+6.1 % over the lowest).")
    [(level, message)] = said[1:]
    assert level == "INFO"
    assert "turned at epoch 2" in message
    # Likely, not certain: the one measured re-run at a minimum is one seed on one domain.
    assert "--epochs 2 is likely to ship a better model on this domain" in message
    assert "usually" not in message
    assert "no best checkpoint is kept" in message
    assert "optional" not in message                     # 6.1 % is above the 1 % cut
    assert not any(ch.isdigit() for ch in message.replace("--epochs 2", "").replace("epoch 2", ""))


@pytest.mark.parametrize("ratio, level", [
    (OVERTRAINING_RATIO, "WARNING"),                     # at the threshold: warned
    (OVERTRAINING_RATIO * 1.0001, "WARNING"),
    (OVERTRAINING_RATIO * 0.9999, "INFO"),               # just under it: advised, not warned
])
def test_the_warning_threshold_boundary(caplog, ratio, level):
    said = _said(caplog, [8.0, 7.0, 7.0 * ratio])
    assert [lvl for lvl, _ in said] == ["INFO", level]
    assert "--epochs 2" in said[1][1]


@pytest.mark.parametrize("final, optional", [
    (7.0, True),                                         # back to the minimum: +0.0 %
    (7.0 * 1.009, True),                                 # +0.9 %
    (7.0 * (1 + OPTIONAL_RERUN_GAP) * 0.9996, False),    # +0.9996 % prints as +1.0 %: not "under"
    (7.0 * 1.02, False),
])
def test_a_rise_under_one_percent_calls_the_re_run_optional(caplog, final, optional):
    said = _said(caplog, [8.0, 7.0, final])
    [(level, message)] = said[1:]
    assert level == "INFO"
    assert "turned at epoch 2" in message
    if optional:
        # Only the turn, the size of the rise, and that the re-run is optional: the 1 % cut is a
        # judgement, so the message claims no spread and promises no better model.
        assert message == ("The held-out curve turned at epoch 2 but ended under 1 % above its "
                           "lowest, so a re-run with --epochs 2 is optional.")
    else:
        assert "optional" not in message
        assert "is likely to ship a better model" in message
    assert "spread" not in message
    assert "usually" not in message and "the better model" not in message


@pytest.mark.parametrize("gap, text", [
    (0.0, "+0.0 %"), (0.061, "+6.1 %"), (0.0004, "+0.0 %"), (1.531, "+153.1 %"),
])
def test_the_gap_formatting(gap, text):
    from lfa.train import _format_gap
    assert _format_gap(gap) == text


def test_a_still_falling_curve_says_more_epochs_may_help(caplog):
    said = _said(caplog, [9.0, 8.0, 7.0])
    assert said[0] == ("INFO", "Held-out perplexity: lowest 7.000 at epoch 3 of 3; final 7.000 "
                               "(+0.0 % over the lowest).")
    [(level, message)] = said[1:]
    assert level == "INFO"
    assert "had not turned by the last epoch" in message
    assert "more epochs may lower it further" in message and "--epochs" in message


# A re-run at an earlier run's turn: the trial's 15-epoch run turned at epoch 10 and advised
# `--epochs 10`; that re-run ended at its own lowest and was told to raise --epochs again, back
# toward 15. With the earlier run named, the verdict is to stop and compare the two finals.

def _rerun_said(final, earlier_final, run="stage1", epoch=10):
    from lfa.train import EarlierTurn, held_out_messages

    summary = HeldOutSummary("still_falling", epoch, final, epoch, final, 0.0)
    return held_out_messages(summary, EarlierTurn(run=run, epoch=epoch, final=earlier_final))


def test_a_rerun_at_the_turn_that_did_not_turn_is_told_to_stop_and_choose_on_both_axes():
    """The trial's numbers: stage1_run2 at --epochs 10 ended at 13.839, stage1 at 14.021. The
    held-out finals are read, then called one axis: the choice is put on both, against the
    run-to-run spread, and both exports are named, since either run can be the keeper."""
    (_, curve), (level, advice) = _rerun_said(13.839, 14.021)
    assert curve == "lowest 13.839 at epoch 10 of 10; final 13.839 (+0.0 % over the lowest)."
    assert level == logging.INFO
    assert advice == (
        "This run is the re-run at stage1's turn (epoch 10), and not turning is what such a "
        "re-run shows, not a sign that more epochs would help: stop here. On the held-out curve "
        "this run ended at 13.839 and stage1 at 14.021, so this run is 1.3 % lower. That is one "
        "axis: keep the run that is not worse on either axis by more than the run-to-run spread "
        "measured on Qwen3-1.7B, three repeats at one seed (about 0.5 % on WikiText-2 and 0.15 % "
        "on the held-out domain; docs/tuning.md, 'When to "
        "stop'), setting this run's `lfa evaluate` table beside stage1's (the one evaluate "
        "printed for it, or `lfa evaluate --run stage1`). When each run is better on one axis, "
        "docs/tuning.md (step 2) says what to weigh. `lfa fuse` exports this run, the stage's "
        "latest, and `lfa fuse --run stage1` exports stage1.")
    assert "raise" not in advice and "larger --epochs" not in advice
    # One colon to a sentence (the review's O4).
    assert all(sentence.count(":") <= 1 for sentence in advice.split(". "))


def test_a_rerun_that_ended_above_the_earlier_runs_final_names_the_earlier_runs_export():
    [_, (_, advice)] = _rerun_said(14.5, 14.021, run="stage1_run2", epoch=8)
    assert "re-run at stage1_run2's turn (epoch 8)" in advice
    assert ("this run ended at 14.500 and stage1_run2 at 14.021, so stage1_run2 is 3.4 % lower."
            in advice)
    assert "`lfa evaluate --run stage1_run2`" in advice
    assert advice.endswith("`lfa fuse --run stage1_run2` exports stage1_run2.")


def test_two_finals_within_the_run_to_run_spread_are_called_the_same():
    """Under 0.15 % apart (docs/tuning.md 'When to stop') is noise on that axis, and the verdict
    says so; WikiText-2 is still to be read."""
    from lfa.train import GENERAL_SPREAD, HELD_OUT_SPREAD

    assert (HELD_OUT_SPREAD, GENERAL_SPREAD) == (0.0015, 0.005)
    [_, (_, advice)] = _rerun_said(14.030, 14.021)
    assert ("this run ended at 14.030 and stage1 at 14.021, 0.06 % apart, within the run-to-run "
            "spread of about 0.15 % (measured on Qwen3-1.7B). That is one axis:") in advice
    [_, (_, beyond)] = _rerun_said(14.050, 14.021)                     # 0.21 %: not noise
    assert "this run ended at 14.050 and stage1 at 14.021, so stage1 is 0.2 % lower." in beyond


def test_a_rerun_at_the_lowest_epoch_of_a_run_that_diverged_keeps_this_run():
    """A diverged run advises --epochs <its lowest> too (the review's O1). Its final is not
    finite and it is not to be shipped, so there is nothing to compare: keep the re-run."""
    from lfa.train import EarlierTurn, held_out_messages

    summary = HeldOutSummary("still_falling", 4, 9.5, 4, 9.5, 0.0)
    [_, (level, advice)] = held_out_messages(summary, EarlierTurn("stage1", 4, None))
    assert level == logging.INFO
    assert advice == (
        "This run is the re-run at stage1's lowest epoch (epoch 4), before it diverged, and not "
        "turning is what such a re-run shows, not a sign that more epochs would help: stop here. "
        "The earlier run, stage1, diverged and should not be shipped, so keep this run.")


@pytest.mark.parametrize("summary", [
    HeldOutSummary("turned", 6, 13.435, 10, 13.906, 13.906 / 13.435 - 1),
    HeldOutSummary("turned", 5, 13.517, 6, 13.526, 13.526 / 13.517 - 1),
    HeldOutSummary("diverged", 2, 8.0, 5),
    HeldOutSummary("no_curve"),
], ids=["turned-likely", "turned-optional", "diverged", "no-curve"])
def test_every_other_verdict_ignores_an_earlier_turn(summary):
    from lfa.train import EarlierTurn, held_out_messages

    earlier = EarlierTurn(run="stage1", epoch=summary.final_epoch or 1, final=14.021)
    assert held_out_messages(summary, earlier) == held_out_messages(summary)


def test_the_trainer_logs_the_rerun_verdict_it_is_handed(caplog):
    """`train` passes `rerun_of` to the end-of-run lines; without it they stay generic."""
    from lfa.train import EarlierTurn

    summary = held_out_summary(_curve([15.0, 14.0, 13.839]))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="lfa.test.held_out"):
        _log_held_out_summary(summary, logging.getLogger("lfa.test.held_out"),
                              EarlierTurn(run="stage1", epoch=3, final=14.021))
    said = [r.getMessage() for r in caplog.records]
    assert said[0].startswith("Held-out perplexity: lowest 13.839 at epoch 3 of 3;")
    assert said[1].startswith("This run is the re-run at stage1's turn (epoch 3)")


def test_a_one_point_curve_is_read_as_not_yet_turned(caplog):
    said = _said(caplog, [7.5])
    assert said[0][1].startswith("Held-out perplexity: lowest 7.500 at epoch 1 of 1;")
    assert "had not turned" in said[1][1]
    assert all(level == "INFO" for level, _ in said)


def test_no_curve_says_there_was_nothing_to_choose_the_dose_by(caplog):
    [(level, message)] = _said(caplog, [])
    assert level == "INFO"
    assert message.startswith("Held-out perplexity: none was measured")
    assert "nothing to choose its dose by" in message
    assert "val_fraction above 0" in message
    from lfa.prepare_domain import SUGGESTED_SPLIT_CHARS
    assert f"lfa prepare-domain --split-chars {SUGGESTED_SPLIT_CHARS}`" in message


def test_a_workspace_stage_records_its_held_out_summary(tmp_path, base_dir, corpus_a,
                                                         tiny_artifact):
    """The verdict outlives the log: the stage's history entry carries the same reading."""
    from conftest import tiny_recipe
    from lfa import Workspace

    ws = Workspace.init(tmp_path / "ws", str(base_dir), artifact=str(tiny_artifact[1]))
    entry = ws.train(corpus_a, recipe=tiny_recipe(base_dir, val_fraction=0.5, epochs=2),
                     device="cpu")

    history = json.loads((Path(entry["output_dir"]) / "training_history.json").read_text())
    assert entry["held_out"] == held_out_summary(history).to_dict()
    assert entry["held_out"]["verdict"] in ("still_falling", "turned")
    assert entry["held_out"]["final_epoch"] == 2
    assert json.loads((ws.path / "history.json").read_text())[-1]["held_out"] == entry["held_out"]


def test_a_resumed_run_reads_the_whole_restored_curve(setup, tiny_model, tmp_path, tiny_texts,
                                                       caplog):
    """A resume restores the history, so the end-of-run reading spans every epoch, the ones
    before the interruption included -- not only the epochs this process trained."""
    teacher, fresh_student, dataset, sampler, adapter = setup
    _, tokenizer = tiny_model
    holdout = ChunkedCorpus(tiny_texts[8:], tokenizer, max_length=64)
    config = make_config()
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)
    train(teacher, student, dataset, sampler, adapter, config, tmp_path,
          tokenizer=tokenizer, val_dataset=holdout)

    with caplog.at_level(logging.INFO, logger="lfa.train"):
        state = train(teacher, fresh_student(), dataset, sampler, adapter,
                      make_config(num_epochs=3), tmp_path, resume=True,
                      tokenizer=tokenizer, val_dataset=holdout)

    assert [record["epoch"] for record in state.history] == [1, 2, 3]
    summary = held_out_summary(state.history)
    assert summary.final_epoch == 3
    [line] = [r.getMessage() for r in caplog.records
              if r.getMessage().startswith("Held-out perplexity:")]
    assert f"at epoch {summary.best_epoch} of 3;" in line
    assert f"lowest {min(r['val_perplexity'] for r in state.history):.3f}" in line


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


def test_the_corpus_shape_warnings_are_logged_before_the_first_step(setup, tmp_path, caplog):
    """The wiring: whatever the corpus says about its own shape, the run says out loud, against
    THIS run's batch size, accumulation and epoch count -- and before any of them are spent."""
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config(num_epochs=15)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)
    expected = dataset.shape_warnings(batch_size=config.batch_size, epochs=config.num_epochs)
    assert expected, "the eight-document fixture corpus is meant to be a shape worth warning about"

    with caplog.at_level(logging.INFO, logger="lfa.train"):     # INFO: the epoch lines too
        train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    logged = [r.getMessage() for r in caplog.records
              if r.levelname == "WARNING" and r.getMessage().startswith("Corpus shape:")]
    assert logged == expected
    # Before the run, not after it: the first epoch line comes later in the same capture.
    first_epoch = next(i for i, r in enumerate(caplog.records) if "Epoch 1/" in r.getMessage())
    assert all(i < first_epoch for i, r in enumerate(caplog.records)
               if r.getMessage().startswith("Corpus shape:"))


def test_a_dataset_that_is_not_a_chunked_corpus_is_not_diagnosed(setup, tmp_path):
    """A caller wiring LFA into their own framework may hand the trainer something else. The
    diagnostic must not be the thing that refuses it."""
    from lfa.train import _corpus_shape_warnings

    _, _, dataset, _, _ = setup
    assert _corpus_shape_warnings(list(dataset), make_config()) == []


def test_an_ordinary_run_is_not_called_too_small(setup, tmp_path, caplog):
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config(num_epochs=1, warmup_steps=1)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)

    with caplog.at_level(logging.WARNING, logger="lfa.train"):
        train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    assert not [r for r in caplog.records if "very small for this recipe" in r.getMessage()]


def test_a_history_without_validation_numbers_has_no_curve(tiny_texts):
    assert held_out_summary([{"epoch": 1, "loss_total": 2.0}] * 5).verdict == "no_curve"


def test_a_run_that_trained_past_its_optimum_says_so_and_names_the_dose(setup, tmp_path,
                                                                        monkeypatch, caplog):
    """The wiring: the trainer reads its own curve at the end and turns it into one sentence."""
    import lfa.train as train_module

    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config(num_epochs=1)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)
    monkeypatch.setattr(train_module, "held_out_summary", lambda history: HeldOutSummary(
        verdict="turned", best_epoch=2, best=6.67, final_epoch=15, final=16.88,
        gap=16.88 / 6.67 - 1))

    with caplog.at_level(logging.INFO, logger="lfa.train"):
        train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    assert [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("Held-out perplexity:")] == [
        "Held-out perplexity: lowest 6.670 at epoch 2 of 15; final 16.880 (+153.1 % over the "
        "lowest)."]
    warned = [record.getMessage() for record in caplog.records
              if record.levelname == "WARNING" and "past its own optimum" in record.getMessage()]
    assert len(warned) == 1
    assert "epoch 2 (6.670)" in warned[0] and "16.880" in warned[0]
    assert "--epochs 2" in warned[0]                      # the dose to re-run at
    assert "no best checkpoint is kept" in warned[0]      # why final_model is not that model


def test_a_run_with_a_held_out_split_always_reports_its_curve(setup, tiny_model, tmp_path,
                                                              tiny_texts, caplog):
    teacher, fresh_student, dataset, sampler, adapter = setup
    _, tokenizer = tiny_model
    holdout = ChunkedCorpus(tiny_texts[8:], tokenizer, max_length=64)
    config = make_config()
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)

    with caplog.at_level(logging.INFO, logger="lfa.train"):
        state = train(teacher, student, dataset, sampler, adapter, config, tmp_path,
                      tokenizer=tokenizer, val_dataset=holdout)

    assert not [r for r in caplog.records if "past its own optimum" in r.getMessage()]
    # Whatever the curve did, the run said what it was, once, from its own history.
    summary = held_out_summary(state.history)
    assert summary.verdict in ("still_falling", "turned")
    [line] = [r.getMessage() for r in caplog.records
              if r.getMessage().startswith("Held-out perplexity:")]
    assert line == (f"Held-out perplexity: lowest {summary.best:.3f} at epoch "
                    f"{summary.best_epoch} of 2; final {summary.final:.3f} "
                    f"({summary.gap * 100:+.1f} % over the lowest).")


def test_without_a_held_out_split_the_history_carries_no_validation_columns(setup, tmp_path,
                                                                            caplog):
    teacher, fresh_student, dataset, sampler, adapter = setup
    config = make_config(num_epochs=1)
    student = apply_lora(fresh_student(), adapter, rank=config.lora_rank, alpha=config.lora_alpha)

    with caplog.at_level(logging.INFO, logger="lfa.train"):
        train(teacher, student, dataset, sampler, adapter, config, tmp_path)

    [record] = json.loads((tmp_path / "training_history.json").read_text())
    assert "val_loss" not in record and "val_perplexity" not in record
    # ...and the end of the run says that there was no curve to choose the dose by.
    assert [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("Held-out perplexity:")
            and "nothing to choose its dose by" in r.getMessage()]


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


def test_the_full_weight_notice_quotes_no_models_lambda_range():
    """The trainer does not know which model's calibration applies, so it names none: a range
    measured on one model would be read as guidance for every other. `Recipe.warnings` quotes
    Qwen3-0.6B's range for that model's recipe."""
    from lfa.train import FULL_WEIGHT_NOTICE as notice
    assert notice == FULL_WEIGHT_NOTICE
    assert "50,000" not in notice and "100,000" not in notice


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
    """The research scheduler, transcribed, at ``cosine_fraction=1.0, plateau_end_factor=1.0``.

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
def test_the_cosine_schedule_matches_the_research_scheduler_step_for_step(
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
