"""The `lfa` command line: the same Workspace and pipeline calls, reached through argv.

The CLI owns no behaviour of its own, so what is tested here is the front door and nothing
behind it: that every subcommand parses and renders its help, that a flag reaches the value it
names (`--full-weight` is checked in the run's own `config.json`, since it is a mode setting a
silent default would change without saying so), that a library
refusal comes out as one line and exit 2 rather than a traceback, and that `--workspace` really
does default to the working directory.

Everything runs on the CPU `tiny_model` with the one-epoch rank-2 recipe from `conftest.py`, so a
"stage" here is seconds of training over eight short documents.
"""

import json
import shlex
from pathlib import Path

import pytest
import torch
import yaml

from conftest import tiny_recipe
from lfa.cli import main
from lfa.seed_corpus import prepare_seed_corpus
from lfa.workspace import Workspace

#: Every subcommand the CLI publishes; the help and the parse of each one is asserted below.
SUBCOMMANDS = [
    "init", "train", "extend", "evaluate", "fuse", "chain",
    "build-artifact", "prepare-seed-corpus", "prepare-domain", "list-artifacts",
    "prepare-supplement", "regenerate-artifact", "probe-artifact",
]


# ------------------------------------------------------------------------------------ fixtures

#: The fixture artifact's path, set once per module: what `--artifact PATH` is given.
TINY_ARTIFACT_PATH: str = ""


@pytest.fixture(scope="module", autouse=True)
def _tiny_artifact_path(tiny_artifact):
    global TINY_ARTIFACT_PATH
    TINY_ARTIFACT_PATH = str(tiny_artifact[1])


@pytest.fixture(scope="module")
def recipe_path(tmp_path_factory, base_dir):
    """The tiny recipe on disk -- what `--recipe PATH` is given."""
    return tiny_recipe(base_dir).save(tmp_path_factory.mktemp("recipes") / "tiny.yaml")


@pytest.fixture(scope="module")
def trained(tmp_path_factory, base_dir, corpus_a, recipe_path):
    """A workspace taken through `lfa init` and one `lfa train`, entirely through argv."""
    workspace = tmp_path_factory.mktemp("cli") / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH]) == 0
    assert main(["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
                 "--recipe", str(recipe_path), "--device", "cpu"]) == 0
    return workspace


def error_lines(capsys) -> list[str]:
    """The CLI's own error lines on stderr -- and never a traceback."""
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    return [line for line in captured.err.splitlines() if line.startswith("lfa: ")]


# ---------------------------------------------------------------------------------------- help

def test_the_top_level_help_exits_zero_and_names_every_subcommand(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for subcommand in SUBCOMMANDS:
        assert subcommand in out


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_every_subcommand_renders_its_own_help(subcommand, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([subcommand, "--help"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("usage:")


# ------------------------------------------------------------------------------ list-artifacts

def test_list_artifacts_lists_the_store(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("lfa.cli.list_store", lambda: [
        {"path": tmp_path / "e", "model_id": "Qwen/Qwen3-0.6B",
         "frame": {"n_raw": 2500, "max_new_tokens": 2048, "gmm_k": 32},
         "built_at": "2026-09-28", "size_mb": 108, "state": "built"}])
    assert main(["list-artifacts"]) == 0
    out = capsys.readouterr().out
    assert "Qwen/Qwen3-0.6B" in out and "2500 documents x 2048 tokens" in out and "108 MB" in out
    assert "  built 2026-09-28  " in out and "built  built" not in out      # "built" said once


def test_list_artifacts_says_which_entries_were_published(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("lfa.cli.list_store", lambda: [
        {"path": tmp_path / "e", "model_id": "Qwen/Qwen3-0.6B",
         "frame": {"n_raw": 2500, "max_new_tokens": 2048, "gmm_k": 32},
         "built_at": "2026-10-04 09:00", "size_mb": 108, "state": "published"}])
    assert main(["list-artifacts"]) == 0
    out = capsys.readouterr().out
    assert "  published, downloaded 2026-10-04 09:00  " in out and "built" not in out


def test_list_artifacts_shows_an_unfinished_entry_by_its_state(tmp_path, monkeypatch, capsys):
    """An entry with no artifact yet has no build date: its state stands alone."""
    frame = {"n_raw": 2500, "max_new_tokens": 2048, "gmm_k": 32}
    monkeypatch.setattr("lfa.cli.list_store", lambda: [
        {"path": tmp_path / "a", "model_id": "m", "frame": frame, "built_at": None,
         "size_mb": 3, "state": "in progress: 400/2500 documents"},
        {"path": tmp_path / "b", "model_id": "m", "frame": frame, "built_at": None,
         "size_mb": 9, "state": "corpus complete, not fitted"}])
    assert main(["list-artifacts"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "  in progress: 400/2500 documents  3 MB  " in lines[0]
    assert "  corpus complete, not fitted  9 MB  " in lines[1]
    assert not any("built" in line for line in lines)


def test_list_artifacts_on_an_empty_store_says_how_to_fill_it(monkeypatch, capsys):
    monkeypatch.setattr("lfa.cli.list_store", lambda: [])
    assert main(["list-artifacts"]) == 0
    assert "--artifact self-generated" in capsys.readouterr().out


def test_list_artifacts_reads_the_store_the_environment_names(tmp_path, monkeypatch, capsys):
    """Unpatched: `$LFA_ARTIFACT_STORE` pointing at nothing is an empty store, not an error."""
    monkeypatch.setenv("LFA_ARTIFACT_STORE", str(tmp_path / "no_store_here"))
    assert main(["list-artifacts"]) == 0
    assert "No self-generated artifacts" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------- init

def test_init_needs_an_artifact_and_its_help_names_self_generated(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["init", "ws", "--model", "m"])
    assert exit_info.value.code == 2
    assert "--artifact" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["init", "--help"])
    assert "self-generated" in capsys.readouterr().out


def test_init_help_says_what_happens_on_a_store_miss(capsys):
    """A miss fetches a published artifact for this exact model and frame, else builds; and
    --rebuild always builds here."""
    with pytest.raises(SystemExit):
        main(["init", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "published artifact" in out and "verified" in out
    assert "never downloads" in out


def test_there_is_no_subcommand_that_downloads_an_artifact(capsys):
    with pytest.raises(SystemExit):
        main(["fetch-artifact", "x"])
    assert "invalid choice" in capsys.readouterr().err


# --------------------------------------------------------------------------------- init, train

def test_init_then_train_leaves_one_history_entry(trained, corpus_a):
    assert (trained / "workspace.json").is_file()
    assert (trained / "artifacts" / "v1.pt").is_file()

    history = json.loads((trained / "history.json").read_text())
    assert len(history) == 1
    assert history[0]["stage"] == 1
    assert history[0]["corpus"] == str(corpus_a)
    assert Path(history[0]["adapter"], "adapter_config.json").is_file()


def test_train_reports_the_run_directory_it_wrote(tmp_path, base_dir, corpus_a,
                                                  recipe_path, capsys):
    workspace = tmp_path / "ws"
    main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH])
    capsys.readouterr()

    assert main(["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
                 "--recipe", str(recipe_path), "--device", "cpu"]) == 0

    assert str(workspace / "runs" / "stage1") in capsys.readouterr().out


def test_the_recipes_loader_frame_reaches_the_runs_config(tmp_path, base_dir, corpus_a,
                                                         recipe_path):
    """There is no flag for it: `keep_short_whole` is the recipe's, and the run records it."""
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH]) == 0

    assert main(["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
                 "--recipe", str(recipe_path), "--device", "cpu"]) == 0

    config = json.loads((workspace / "runs" / "stage1" / "config.json").read_text())
    assert config["keep_short_whole"] is yaml.safe_load(
        recipe_path.read_text())["keep_short_whole"]


def test_full_weight_reaches_the_runs_config(tmp_path, base_dir, corpus_a, recipe_path):
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH]) == 0

    assert main(["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
                 "--recipe", str(recipe_path), "--device", "cpu", "--full-weight"]) == 0

    config = json.loads((workspace / "runs" / "stage1" / "config.json").read_text())
    assert (config["full_weight"], config["use_lora"]) == (True, False)


# -------------------------------------------------------------------------------------- errors

def test_the_cli_reports_its_version(capsys):
    """The first line of any bug report, and it did not exist."""
    from lfa import __version__

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"lfa-anchoring {__version__}"


def test_init_says_the_next_command_rather_than_repeating_the_library_line(tmp_path,
                                                                            base_dir, capsys):
    """`Workspace.init` logs that it created the workspace and the CLI configures logging, so
    printing the same sentence here showed the very first line the package emits twice."""
    assert main(["init", str(tmp_path / "ws"), "--model", str(base_dir),
                 "--artifact", TINY_ARTIFACT_PATH]) == 0

    printed = capsys.readouterr().out.strip().splitlines()
    assert len(printed) == 2
    # The documents and their supplement come first (the user trial went straight to train).
    assert printed[0].startswith(f"Next: lfa prepare-domain <your files> --out <dir> "
                                 f"--supplement --model {base_dir}")
    assert "--split-chars 3500" in printed[0] and "--recipe" not in printed[0]
    assert printed[1] == f"Then: lfa train --workspace {tmp_path / 'ws'} --corpus <dir>"


def test_init_names_a_recipe_the_supplement_would_not_find_by_itself(tmp_path, base_dir,
                                                                     recipe_path, capsys):
    assert main(["init", str(tmp_path / "ws"), "--model", str(base_dir),
                 "--artifact", TINY_ARTIFACT_PATH, "--recipe", str(recipe_path)]) == 0
    assert f"--model {base_dir} --recipe {recipe_path}" in capsys.readouterr().out


def test_an_interrupted_command_says_how_to_continue_rather_than_printing_a_traceback(
        trained, corpus_a, monkeypatch, capsys):
    """Ctrl-C is how anyone stops an hour-long run, and a bare `KeyboardInterrupt` traceback
    reads as a crash -- which the documented rule would then call a bug in this package."""
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(Workspace, "train", interrupted)

    assert main(["train", "--workspace", str(trained), "--corpus", str(corpus_a),
                 "--device", "cpu"]) == 130               # the shell's convention for SIGINT

    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "--resume" in captured.err and "interrupted" in captured.err
    assert "`lfa init`" in captured.err                   # a self-generated build resumes too
    assert "download" in captured.err and "starts again" in captured.err


def test_chain_passes_the_extension_knobs_it_advertises(tmp_path, base_dir, monkeypatch):
    """`--need` is the knob docs/faq.md tells a memory-constrained user to turn down, and a
    chain runs an extension between every pair of domains."""
    seen = {}
    monkeypatch.setattr(Workspace, "chain",
                        lambda self, spec, **kwargs: seen.update(kwargs) or [])
    spec = tmp_path / "domains.yaml"
    spec.write_text("domains: []\n")
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH]) == 0

    assert main(["chain", str(spec), "--workspace", str(workspace),
                 "--need", "1234", "--k-domain", "3"]) == 0

    assert (seen["need"], seen["k_domain"]) == (1234, 3)


def test_a_second_corpus_without_an_extend_exits_two_with_one_line(trained, corpus_b,
                                                                   recipe_path, capsys):
    code = main(["train", "--workspace", str(trained), "--corpus", str(corpus_b),
                 "--recipe", str(recipe_path), "--device", "cpu"])

    assert code == 2
    lines = error_lines(capsys)
    assert len(lines) == 1
    assert "lfa extend" in lines[0]


def test_a_sharding_request_without_the_flag_exits_two_with_one_line(trained, corpus_a,
                                                                     recipe_path, capsys):
    code = main(["train", "--workspace", str(trained), "--corpus", str(corpus_a),
                 "--recipe", str(recipe_path), "--device", "auto"])

    assert code == 2
    lines = error_lines(capsys)
    assert len(lines) == 1
    assert "allow-sharding" in lines[0] or "allow_sharding" in lines[0]


def test_an_artifact_that_is_neither_self_generated_nor_a_file_exits_two_with_one_line(
        tmp_path, base_dir, capsys):
    code = main(["init", str(tmp_path / "ws"), "--model", str(base_dir),
                 "--artifact", "no-such-artifact"])

    assert code == 2
    lines = error_lines(capsys)
    assert len(lines) == 1
    assert "--artifact self-generated" in lines[0]


def test_an_artifact_this_package_did_not_build_exits_two_with_one_line(tmp_path, base_dir,
                                                                       tiny_artifact, capsys):
    params, _ = tiny_artifact
    foreign = tmp_path / "foreign.pt"
    torch.save({key: value for key, value in params.items() if key != "__meta__"}, foreign)
    code = main(["init", str(tmp_path / "ws"), "--model", str(base_dir),
                 "--artifact", str(foreign)])

    assert code == 2
    lines = error_lines(capsys)
    assert len(lines) == 1
    assert "not built by lfa-anchoring" in lines[0] and "lfa build-artifact" in lines[0]
    assert not (tmp_path / "ws").exists()


def test_an_artifact_in_a_later_format_exits_two_with_one_line(tmp_path, base_dir, tiny_artifact,
                                                              capsys):
    from lfa.artifact.schema import ARTIFACT_FORMAT

    params, _ = tiny_artifact
    later = tmp_path / "later.pt"
    torch.save(dict(params, __meta__=dict(params["__meta__"], format_version=ARTIFACT_FORMAT + 1)),
               later)
    code = main(["init", str(tmp_path / "ws"), "--model", str(base_dir), "--artifact", str(later)])

    assert code == 2
    lines = error_lines(capsys)
    assert len(lines) == 1
    assert f"uses artifact format {ARTIFACT_FORMAT + 1}," in lines[0]
    assert "lfa build-artifact" in lines[0]
    assert not (tmp_path / "ws").exists()


@pytest.mark.parametrize("subcommand", ["fuse", "evaluate"])
def test_reading_a_workspace_with_no_trained_stage_exits_two_with_one_line(subcommand, tmp_path,
                                                                          base_dir, capsys):
    """`lfa fuse` (or `evaluate`) right after `init`: a first-session mistake, not an exotic one.

    It used to raise a bare `RuntimeError`, which is not in `USER_FACING_ERRORS`, so the message --
    which already ends in the command to run instead -- arrived as the last line of a traceback.
    """
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH]) == 0
    capsys.readouterr()

    code = main([subcommand, "--workspace", str(workspace)])

    assert code == 2
    lines = error_lines(capsys)                  # asserts there is no traceback
    assert len(lines) == 1
    assert "lfa train" in lines[0]


def test_a_seed_corpus_source_that_cannot_be_loaded_exits_two_with_one_line(tmp_path, capsys,
                                                                            monkeypatch):
    """No network is the ordinary case for `prepare-seed-corpus`, and it is not a bug."""
    def unreachable(path, *, split, cache_dir=None):
        raise OSError("We couldn't connect to https://huggingface.co")

    monkeypatch.setattr("lfa.cli.prepare_seed_corpus",
                        lambda *args, **kwargs: prepare_seed_corpus(*args, **kwargs,
                                                                    loader=unreachable))

    code = main(["prepare-seed-corpus", "--out", str(tmp_path / "seed.jsonl")])

    assert code == 2
    lines = error_lines(capsys)
    assert len(lines) == 1
    assert "RedPajama" in lines[0]


def test_evaluate_keeps_the_domain_axis_when_the_hub_is_unreachable(trained, monkeypatch, capsys):
    """An offline machine still gets the domain number: the general axis is skipped, not fatal.

    `Workspace._general` catches the `DatasetUnavailable` that `lfa.evaluate` now raises instead
    of a `datasets` traceback, so the command still exits 0 and prints the table with the general
    row replaced by a line that says it was not measured.
    """
    import datasets

    def unreachable(*args, **kwargs):
        raise ConnectionError("Couldn't reach https://huggingface.co")

    monkeypatch.setattr(datasets, "load_dataset", unreachable)
    capsys.readouterr()

    code = main(["evaluate", "--workspace", str(trained), "--device", "cpu"])

    assert code == 0
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "domain" in captured.out
    assert "not measured" in captured.out


@pytest.mark.parametrize("broken", ["recipe", "chain spec"])
def test_a_yaml_file_that_does_not_parse_exits_two_with_one_line(broken, tmp_path,
                                                                 base_dir, corpus_a, capsys):
    """A typo in a YAML file is a user's mistake, and it names the file like every other one.

    The semantic failures already did (a non-mapping, an unknown field, a missing `domains`); only
    the parse error escaped, as `yaml.parser.ParserError`.
    """
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH]) == 0
    broken_file = tmp_path / f"{broken.split()[0]}.yaml"
    broken_file.write_text("domains: [\n  - name: alpha")          # unterminated flow sequence
    capsys.readouterr()

    if broken == "recipe":
        argv = ["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
                "--recipe", str(broken_file), "--device", "cpu"]
    else:
        argv = ["chain", str(broken_file), "--workspace", str(workspace), "--device", "cpu"]

    code = main(argv)

    assert code == 2
    lines = error_lines(capsys)                  # asserts there is no traceback
    assert len(lines) == 1
    assert broken_file.name in lines[0] and "not valid YAML" in lines[0]


def test_a_directory_that_is_not_a_workspace_exits_two_with_one_line(tmp_path, corpus_a,
                                                                     recipe_path, capsys):
    code = main(["train", "--workspace", str(tmp_path), "--corpus", str(corpus_a),
                 "--recipe", str(recipe_path), "--device", "cpu"])

    assert code == 2
    lines = error_lines(capsys)
    assert len(lines) == 1
    assert "lfa init" in lines[0]                # the message ends in the command to run instead


def test_initialising_over_an_existing_workspace_exits_two_with_one_line(tmp_path,
                                                                         base_dir, capsys):
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH]) == 0

    code = main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH])

    assert code == 2
    lines = error_lines(capsys)
    assert len(lines) == 1
    assert "already an LFA workspace" in lines[0]


# ------------------------------------------------------------------------------ extend, chain

def test_extend_folds_the_stage_into_a_second_artifact_version(tmp_path, base_dir,
                                                               corpus_a, recipe_path):
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH]) == 0
    assert main(["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
                 "--recipe", str(recipe_path), "--device", "cpu"]) == 0

    assert main(["extend", "--workspace", str(workspace), "--need", "400",
                 "--k-domain", "2", "--device", "cpu"]) == 0

    assert (workspace / "artifacts" / "v2.pt").is_file()
    assert (workspace / "models" / "stage1_fused" / "config.json").is_file()
    state = json.loads((workspace / "workspace.json").read_text())
    assert state["artifact_version"] == 2
    assert state["pending_extend"] is False


def test_chain_trains_every_domain_in_the_spec(tmp_path, base_dir, corpus_a, corpus_b,
                                               recipe_path):
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH,
                 "--recipe", str(recipe_path)]) == 0
    spec = tmp_path / "domains.yaml"
    spec.write_text(yaml.safe_dump({"domains": [{"name": "alpha", "corpus": str(corpus_a)},
                                                {"name": "beta", "corpus": str(corpus_b)}]}))

    assert main(["chain", str(spec), "--workspace", str(workspace), "--device", "cpu"]) == 0

    history = json.loads((workspace / "history.json").read_text())
    assert len(history) == 2
    assert [entry["stage"] for entry in history] == [1, 2]
    # The chain folded the first domain in before starting the second: that is what a chain is.
    assert history[1]["artifact_version"] == 2


# ------------------------------------------------------------------------------------ evaluate

def test_evaluate_prints_the_table_with_the_general_axis_skipped(trained, capsys):
    assert main(["evaluate", "--workspace", str(trained), "--n-windows", "none",
                 "--device", "cpu"]) == 0

    out = capsys.readouterr().out
    assert "domain" in out
    assert "not measured" in out                 # `--n-windows none` skips the general axis


# ------------------------------------------------------- fuse, and the --workspace cwd default

def test_fuse_defaults_the_workspace_to_the_working_directory(trained, tmp_path, monkeypatch):
    out = tmp_path / "exported"
    monkeypatch.chdir(trained)

    assert main(["fuse", "--out", str(out)]) == 0

    assert (out / "config.json").is_file()
    assert not (out / "adapter_config.json").exists()      # merged, not an adapter


def test_a_second_fuse_after_a_rerun_says_it_replaced_the_first_export(
        tmp_path, base_dir, corpus_a, recipe_path, caplog):
    """The trial's second `fuse` overwrote the first export without a word: it now names the run
    the default directory holds and says the earlier export was replaced. The first fuse into an
    empty directory says nothing of the kind."""
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH]) == 0
    train = ["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
             "--recipe", str(recipe_path), "--device", "cpu"]
    assert main(train) == 0
    export = workspace / "models" / "stage1_fused_export"

    with caplog.at_level("INFO", logger="lfa.workspace"):
        assert main(["fuse", "--workspace", str(workspace)]) == 0
    assert (export / "config.json").is_file()
    assert not [r for r in caplog.records if "replaced" in r.getMessage()]

    assert main(train) == 0                                             # the re-run: stage1_run2
    caplog.clear()
    with caplog.at_level("INFO", logger="lfa.workspace"):
        assert main(["fuse", "--workspace", str(workspace)]) == 0
    [line] = [r.getMessage() for r in caplog.records if "replaced" in r.getMessage()]
    assert line == (f"{export} held an earlier export, which this one replaced: it now holds "
                    f"stage 1's run {workspace / 'runs' / 'stage1_run2'}, the stage's latest.")


def test_fuse_and_evaluate_run_reach_the_earlier_run_from_the_command_line(
        tmp_path, base_dir, corpus_a, recipe_path, caplog, capsys):
    """`--run stage1` while stage1_run2 is the latest: fuse merges stage1's adapter into the
    stage's default directory, says the export it replaced now holds a run that is not the
    latest, and evaluate reads stage1. An unknown name and a missing run are one-line refusals
    naming the flag."""
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH]) == 0
    train = ["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
             "--recipe", str(recipe_path), "--device", "cpu"]
    assert main(train) == 0
    assert main(train) == 0                                             # the re-run: stage1_run2
    run1, run2 = workspace / "runs" / "stage1", workspace / "runs" / "stage1_run2"
    export = workspace / "models" / "stage1_fused_export"

    assert main(["fuse", "--workspace", str(workspace)]) == 0             # the latest, first
    caplog.clear()
    with caplog.at_level("INFO"):
        assert main(["fuse", "--workspace", str(workspace), "--run", "stage1"]) == 0
    assert any(r.getMessage() == f"Fusing adapter {run1 / 'final_model'} into {base_dir}"
               for r in caplog.records)
    [line] = [r.getMessage() for r in caplog.records if "replaced" in r.getMessage()]
    assert line == (f"{export} held an earlier export, which this one replaced: it now holds "
                    f"stage 1's run {run1}, not the stage's latest, {run2}.")

    caplog.clear()
    with caplog.at_level("INFO", logger="lfa.workspace"):
        assert main(["evaluate", "--workspace", str(workspace), "--run", "stage1",
                     "--n-windows", "none", "--device", "cpu"]) == 0
    assert any(r.getMessage().startswith(f"The run read: {run1} ") for r in caplog.records)

    capsys.readouterr()
    assert main(["fuse", "--workspace", str(workspace), "--run", "stage2"]) == 2
    assert error_lines(capsys) == [
        "lfa: Stage 1 has no run named 'stage2': its runs are stage1, stage1_run2."]


def test_a_run_of_an_earlier_stage_is_refused_naming_the_flag(tmp_path, base_dir, corpus_a,
                                                               corpus_b, recipe_path, capsys):
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH]) == 0
    for corpus in (corpus_a, corpus_b):
        assert main(["train", "--workspace", str(workspace), "--corpus", str(corpus),
                     "--recipe", str(recipe_path), "--device", "cpu"]) == 0
        if corpus == corpus_a:
            assert main(["extend", "--workspace", str(workspace), "--need", "400",
                         "--k-domain", "2", "--device", "cpu"]) == 0
    capsys.readouterr()

    assert main(["fuse", "--workspace", str(workspace), "--run", "stage1"]) == 2
    assert error_lines(capsys) == [
        "lfa: stage1 is a run of stage 1, and --run names a run of the latest stage, 2: stage2. "
        "Stage 1 was folded into the chain when it was extended, and the stages after it build "
        "on that."]


def test_an_override_note_on_the_command_line_names_the_flag_alone(tmp_path, base_dir, corpus_a,
                                                                    recipe_path, caplog):
    """The trial read "lambda 400000 set by --lambda (lambda_= from Python)" as if both were
    used. On the CLI the note names the flag; Python's spelling is not mentioned."""
    from lfa.recipe import SPEAKS_TO_CLI

    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH]) == 0
    with caplog.at_level("WARNING", logger="lfa.workspace"):
        assert main(["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
                     "--recipe", str(recipe_path), "--device", "cpu",
                     "--lambda", "25", "--mu", "0.5"]) == 0
    notes = [r.getMessage() for r in caplog.records if " set by " in r.getMessage()]
    assert notes[0].startswith("lambda 25 set by --lambda; the recipe calibrated 10.")
    assert notes[1] == "mu 0.5 set by --mu; the recipe's is 0.05."
    assert not any("Python" in note or "lambda_=" in note or "mu=" in note for note in notes)
    assert SPEAKS_TO_CLI.get() is False                     # and the CLI's context is left behind


def test_build_artifact_refuses_both_a_corpus_and_self_generated(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["build-artifact", "--model", "m", "--out", "x.pt", "--corpus", "c.jsonl",
              "--self-generated"])
    assert exit_info.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_build_artifact_needs_one_of_them(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["build-artifact", "--model", "m", "--out", "x.pt"])
    assert exit_info.value.code == 2


def test_build_artifact_self_generated_takes_the_recorded_frame(monkeypatch, capsys):
    import lfa.cli as cli

    seen = {}

    def fake(model_id, out_path, options, *, quantize, seed):
        seen.update(options=options, quantize=quantize, seed=seed)
        return out_path

    monkeypatch.setattr(cli, "build_artifact_self_generated", fake)
    assert main(["build-artifact", "--model", "m", "--out", "x.pt", "--self-generated"]) == 0
    options = seen["options"]
    assert options.max_samples == 600_000 and options.gmm_k == 32
    assert options.n_raw == 2500 and options.n_chat == 0 and options.seed == 42  # no chat share
    assert options.layer_group_size is None                # chosen from host RAM by the library

    assert main(["build-artifact", "--model", "m", "--out", "x.pt", "--self-generated",
                 "--layer-group-size", "4", "--max-samples", "1000"]) == 0
    assert seen["options"].layer_group_size == 4 and seen["options"].max_samples == 1000


def test_build_artifact_over_a_corpus_keeps_its_own_sample_count(monkeypatch, capsys):
    import lfa.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "build_artifact",
                        lambda model_id, corpus, out, **kwargs: seen.update(kwargs) or out)
    assert main(["build-artifact", "--model", "m", "--out", "x.pt", "--corpus", "c.jsonl"]) == 0
    assert seen["max_samples"] == 1_500_000


def _fake_store_entry(tmp_path, fixture, seen, monkeypatch):
    """Patch the workspace's `obtain_self_generated` with a finished entry, recording its call."""
    import lfa.workspace as ws_module
    entry = tmp_path / "entry"
    entry.mkdir()
    (entry / "artifact.pt").write_bytes(fixture.read_bytes())
    (entry / "corpus.jsonl").write_text('{"text": "x"}\n')
    manifest = {"corpus_sha256": "f" * 64}
    (entry / "corpus.jsonl.manifest.json").write_text(json.dumps(manifest))

    def fake_obtain(model_id, options, *, rebuild=False, **_):
        seen.update(n_raw=options.n_raw, rebuild=rebuild)
        return entry / "artifact.pt", manifest
    monkeypatch.setattr(ws_module, "obtain_self_generated", fake_obtain)


def test_init_self_generated_is_routed_to_the_store(tmp_path, base_dir, tiny_artifact,
                                                     monkeypatch, capsys):
    seen = {}
    _fake_store_entry(tmp_path, tiny_artifact[1], seen, monkeypatch)

    assert main(["init", str(tmp_path / "ws"), "--model", str(base_dir),
                 "--artifact", "self-generated", "--n-raw", "7"]) == 0
    assert seen == {"n_raw": 7, "rebuild": False}
    state = json.loads((tmp_path / "ws" / "workspace.json").read_text())
    assert state["artifact_id"] == "self-generated:" + "f" * 12


def test_init_rebuild_reaches_the_store(tmp_path, base_dir, tiny_artifact, monkeypatch, capsys):
    seen = {}
    _fake_store_entry(tmp_path, tiny_artifact[1], seen, monkeypatch)

    assert main(["init", str(tmp_path / "ws"), "--model", str(base_dir),
                 "--artifact", "self-generated", "--rebuild"]) == 0
    assert seen["rebuild"] is True


def test_a_self_generated_build_already_running_exits_two_with_one_line(tmp_path, base_dir,
                                                                        monkeypatch, capsys):
    """`StoreLocked` is a `RuntimeError`, so it is reported as a line only because it is named."""
    import lfa.workspace as ws_module
    from lfa.artifact.store import StoreLocked

    def locked(model_id, options, **_):
        raise StoreLocked("the entry is being built by process 4242 (lock /x/.lock). Wait.")
    monkeypatch.setattr(ws_module, "obtain_self_generated", locked)

    code = main(["init", str(tmp_path / "ws"), "--model", str(base_dir),
                 "--artifact", "self-generated"])
    assert code == 2
    lines = error_lines(capsys)
    assert len(lines) == 1 and "4242" in lines[0]
    assert not (tmp_path / "ws").exists()


def test_a_failed_published_download_exits_two_with_one_line(tmp_path, base_dir, monkeypatch,
                                                            capsys):
    import lfa.workspace as ws_module
    from lfa.artifact.published import PublishedArtifactUnavailable

    def refused(model_id, options, **_):
        raise PublishedArtifactUnavailable(
            "Downloading https://example.org/a.pt failed: HTTP 404. Pass --rebuild.")
    monkeypatch.setattr(ws_module, "obtain_self_generated", refused)

    code = main(["init", str(tmp_path / "ws"), "--model", str(base_dir),
                 "--artifact", "self-generated"])
    assert code == 2
    lines = error_lines(capsys)
    assert len(lines) == 1 and "https://example.org/a.pt" in lines[0]
    assert not (tmp_path / "ws").exists()


def test_train_no_supplement_and_supplement_file_are_exclusive(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["train", "--corpus", "c", "--no-supplement", "--supplement", "s.jsonl"])
    assert exit_info.value.code == 2


# ------------------------------------------------------------------ the supplement with the data

def test_prepare_domain_with_a_supplement_needs_a_model(tmp_path, capsys):
    src = tmp_path / "src.txt"
    src.write_text("word " * 400)
    assert main(["prepare-domain", str(src), "--out", str(tmp_path / "out"),
                 "--supplement"]) == 2
    assert "--model" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def two_sources(tmp_path) -> Path:
    """A directory of two documents: the fewest a held-out split can be taken from, so the
    supplement is not refused for the corpus's size."""
    src = tmp_path / "src"
    src.mkdir()
    for name in ("a.txt", "b.txt"):              # distinct texts: the same one twice is refused
        (src / name).write_text(f"{name} " + "word " * 400)
    return src


def test_prepare_domain_with_a_supplement_writes_both(tmp_path, monkeypatch):
    src = two_sources(tmp_path)
    seen = {}
    monkeypatch.setattr("lfa.cli.prepare_supplement",
                        lambda corpus, model_id, **kw: seen.update(corpus=corpus, model=model_id,
                                                                   **kw) or tmp_path / "s.jsonl")
    assert main(["prepare-domain", str(src), "--out", str(tmp_path / "out"), "--supplement",
                 "--model", "Qwen/Qwen3-0.6B", "--domain-description", "old books"]) == 0
    assert list((tmp_path / "out").glob("*.txt"))
    assert seen["model"] == "Qwen/Qwen3-0.6B" and seen["domain_description"] == "old books"


def test_prepare_supplement_with_a_model_needs_no_workspace(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("lfa.cli.prepare_supplement",
                        lambda corpus, model_id, **kw: seen.update(model=model_id) or tmp_path)
    assert main(["prepare-supplement", "--corpus", str(tmp_path), "--model", "m"]) == 0
    assert seen == {"model": "m"}


def test_prepare_supplement_without_a_model_is_the_workspace_route_here(tmp_path, monkeypatch):
    """No flag at all: the workspace in the current directory writes."""
    seen = {}

    class _Opened:
        def prepare_supplement(self, corpus, **kw):
            seen.update(corpus=corpus)
            return tmp_path / "ws-supplement.jsonl"

    monkeypatch.setattr("lfa.cli.Workspace.open",
                        lambda path: seen.update(workspace=path) or _Opened())
    monkeypatch.setattr("lfa.cli.prepare_supplement",
                        lambda *a, **kw: pytest.fail("the workspace-free route was taken"))
    assert main(["prepare-supplement", "--corpus", "c"]) == 0
    assert seen == {"workspace": ".", "corpus": "c"}


@pytest.mark.parametrize("workspace", [".", "ws"])
def test_prepare_supplement_refuses_a_workspace_and_a_model_together(tmp_path, monkeypatch,
                                                                     capsys, workspace):
    """Explicit `--workspace .` is refused too: explicitness is not read off the value."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lfa.cli.prepare_supplement",
                        lambda *a, **kw: pytest.fail("nothing may be written"))
    monkeypatch.setattr("lfa.cli.Workspace.open",
                        lambda path: pytest.fail("no workspace may be opened"))
    assert main(["prepare-supplement", "--corpus", str(tmp_path / "c"),
                 "--workspace", workspace, "--model", "m"]) == 2
    lines = error_lines(capsys)
    assert len(lines) == 1 and "--workspace" in lines[0] and "--model" in lines[0]
    assert list(tmp_path.iterdir()) == []


def test_prepare_domain_with_a_supplement_and_no_recipe_writes_nothing(tmp_path, monkeypatch,
                                                                       capsys):
    src = tmp_path / "src.txt"
    src.write_text("word " * 400)
    monkeypatch.setattr("lfa.cli.prepare_supplement",
                        lambda *a, **kw: pytest.fail("nothing may be written"))
    assert main(["prepare-domain", str(src), "--out", str(tmp_path / "out"), "--supplement",
                 "--model", "nobody/nothing"]) == 2
    lines = error_lines(capsys)
    assert len(lines) == 1 and "--recipe" in lines[0]
    assert not (tmp_path / "out").exists()


def test_prepare_domain_passes_the_resolved_recipe_on(tmp_path, monkeypatch):
    from lfa import Recipe

    src = two_sources(tmp_path)
    seen = {}
    monkeypatch.setattr("lfa.cli.prepare_supplement",
                        lambda corpus, model_id, **kw: seen.update(kw) or tmp_path / "s.jsonl")
    assert main(["prepare-domain", str(src), "--out", str(tmp_path / "out"), "--supplement",
                 "--model", "Qwen/Qwen3-0.6B"]) == 0
    assert isinstance(seen["recipe"], Recipe) and seen["recipe"].name == "qwen3-0.6b"


# ------------------------------------------------------- one long file, and --split-chars

def a_book(tmp_path) -> Path:
    """One file of forty 1,000-character paragraphs: a book, as far as the loader can tell."""
    src = tmp_path / "book.txt"
    src.write_text("\n\n".join((f"Paragraph {i}. " + "word " * 300)[:1000]
                                for i in range(40)))
    return src


def test_one_file_with_a_supplement_is_refused_before_the_writer_is_reached(tmp_path,
                                                                            monkeypatch, capsys):
    """One book is one document: the trainer could hold nothing out of it, so the supplement --
    minutes of generation -- is not written, and neither is the corpus a re-run would double."""
    monkeypatch.setattr("lfa.cli.prepare_supplement",
                        lambda *a, **kw: pytest.fail("the supplement writer was reached"))
    monkeypatch.setattr("lfa.supplements.write_supplement",
                        lambda *a, **kw: pytest.fail("the supplement writer was reached"))
    out = tmp_path / "out"
    assert main(["prepare-domain", str(a_book(tmp_path)), "--out", str(out), "--supplement",
                 "--model", "Qwen/Qwen3-0.6B"]) == 2
    lines = error_lines(capsys)
    assert len(lines) == 1
    assert "1 document(s)" in lines[0] and "--split-chars 3500" in lines[0]
    assert "Nothing was read or written" in lines[0]
    assert not out.exists()


def test_the_up_arrow_split_rerun_into_the_same_out_is_refused(tmp_path, monkeypatch, capsys,
                                                               caplog):
    """The warning path, then the fix it names re-run into the same --out: the unsplit book is
    already there, so the split pieces and their supplement would sit beside it and every held-out
    piece would be trained on verbatim. Refused, naming the file, with nothing added."""
    out, book = tmp_path / "out", a_book(tmp_path)
    with caplog.at_level("WARNING", logger="lfa.prepare_domain"):
        assert main(["prepare-domain", str(book), "--out", str(out)]) == 0
    assert any("--split-chars 3500" in r.getMessage() for r in caplog.records)

    monkeypatch.setattr("lfa.cli.prepare_supplement",
                        lambda *a, **kw: pytest.fail("the supplement writer was reached"))
    capsys.readouterr()
    assert main(["prepare-domain", str(book), "--out", str(out), "--split-chars", "3500",
                 "--supplement", "--model", "Qwen/Qwen3-0.6B"]) == 2
    lines = error_lines(capsys)
    assert len(lines) == 1
    assert f"100% of its sentence text (sentences of 60 characters or more) is already in " \
           f"{out}; the largest shares are in {out / 'book.txt'} (100%)" in lines[0]
    assert "fresh --out directory" in lines[0]
    assert "delete the earlier preparation of this text" in lines[0]
    assert [p.name for p in out.iterdir()] == ["book.txt"]


def test_the_same_book_split_writes_the_corpus_and_the_supplement(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("lfa.cli.prepare_supplement",
                        lambda corpus, model_id, **kw: seen.update(corpus=corpus) or "s.jsonl")
    out = tmp_path / "out"
    assert main(["prepare-domain", str(a_book(tmp_path)), "--out", str(out), "--supplement",
                 "--model", "Qwen/Qwen3-0.6B", "--split-chars", "3500"]) == 0
    assert sorted(p.name for p in out.iterdir())[:2] == ["book-0001.txt", "book-0002.txt"]
    assert len(list(out.iterdir())) == 10                  # 40 x 1,000 characters at 3,500
    assert seen["corpus"] == str(out)


def test_one_file_without_a_supplement_is_written_and_warned_about(tmp_path, caplog):
    out = tmp_path / "out"
    with caplog.at_level("WARNING", logger="lfa.prepare_domain"):
        assert main(["prepare-domain", str(a_book(tmp_path)), "--out", str(out)]) == 0
    assert [p.name for p in out.iterdir()] == ["book.txt"]
    assert any("--split-chars 3500" in r.getMessage() for r in caplog.records)


def test_prepare_domain_help_names_split_chars_and_its_size(capsys):
    with pytest.raises(SystemExit):
        main(["prepare-domain", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--split-chars N" in help_text and "3500" in help_text
    assert "a book, a report" in help_text


# ------------------------------------------------------------------------ --lambda and --mu

def test_lambda_and_mu_reach_the_runs_config_and_its_history_entry(tmp_path, base_dir,
                                                                   corpus_a, recipe_path):
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir),
                 "--artifact", TINY_ARTIFACT_PATH]) == 0

    assert main(["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
                 "--recipe", str(recipe_path), "--device", "cpu",
                 "--lambda", "2.5e1", "--mu", "0"]) == 0

    config = json.loads((workspace / "runs" / "stage1" / "config.json").read_text())
    assert (config["lambda_qkv"], config["lambda_mlp"], config["mu"]) == (25.0, 25.0, 0.0)
    [entry] = json.loads((workspace / "history.json").read_text())
    assert (entry["lambda_applied"], entry["mu_applied"]) == (25.0, 0.0)
    assert (entry["lambda_override"], entry["mu_override"]) == (25.0, 0.0)


@pytest.mark.parametrize("flag, value", [("--lambda", "-1"), ("--lambda", "nan"),
                                         ("--mu", "inf"), ("--mu", "lots")])
def test_a_lambda_or_mu_that_cannot_weight_a_loss_is_a_usage_error(flag, value, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["train", "--corpus", "x", flag, value])
    assert exit_info.value.code == 2
    assert f"argument {flag}" in capsys.readouterr().err


def test_a_second_run_of_a_stage_says_which_run_later_commands_read(tmp_path, base_dir,
                                                                    corpus_a, recipe_path,
                                                                    capsys):
    workspace = tmp_path / "ws"
    main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH])
    train = ["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
             "--recipe", str(recipe_path), "--device", "cpu"]
    assert main(train) == 0
    assert "latest" not in capsys.readouterr().out

    assert main(train) == 0

    out = capsys.readouterr().out
    assert f"Stage 1 written to {workspace / 'runs' / 'stage1_run2'}" in out
    assert ("This is the latest of stage 1's runs: `lfa evaluate`, `lfa fuse`, `lfa extend` and "
            f"`lfa regenerate-artifact` read it, not {workspace / 'runs' / 'stage1'} (`lfa "
            "evaluate` and `lfa fuse` read an earlier run when `--run` names it).") in out


def test_the_commands_the_control_note_prints_run_as_written(tmp_path, base_dir, corpus_a,
                                                             recipe_path, monkeypatch, capsys):
    """`evaluate --compare-unanchored` prints three commands when the control turned; each one
    is run here exactly as printed, and the third reads a control at its own best epoch."""
    real = Workspace._run_training

    def control_turns(self, config, *args, anchored, **kwargs):
        training, counts = real(self, config, *args, anchored=anchored, **kwargs)
        if not anchored and config.mu == 0:
            training.history = [{"epoch": 1, "val_perplexity": 25.59},
                                {"epoch": 2, "val_perplexity": 433.78}]
        return training, counts

    workspace = tmp_path / "ws"
    main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH])
    main(["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
          "--recipe", str(recipe_path), "--device", "cpu"])
    monkeypatch.setattr(Workspace, "_run_training", control_turns)
    capsys.readouterr()

    assert main(["evaluate", "--workspace", str(workspace), "--compare-unanchored",
                 "--n-windows", "none", "--device", "cpu"]) == 0
    monkeypatch.setattr(Workspace, "_run_training", real)

    out = capsys.readouterr().out
    assert "part of the unanchored column's gap is dose" in " ".join(out.split())
    commands = [line.strip() for line in out.splitlines() if line.startswith("  lfa ")]
    assert [command.split()[1] for command in commands] == ["init", "train", "evaluate"]
    for command in commands:
        argv = shlex.split(command)
        assert argv[0] == "lfa"
        assert main(argv[1:]) == 0, command

    fresh = Path(shlex.split(commands[0])[2])
    [entry] = json.loads((fresh / "history.json").read_text())
    assert (entry["lambda_applied"], entry["lambda_mlp_applied"], entry["mu_applied"]) == (0, 0, 0)
    assert entry["epochs"] == 1 and entry["corpus"] == str(corpus_a)
    stage = json.loads((workspace / "history.json").read_text())[0]
    assert {**entry["recipe"], "epochs": 0} == {**stage["recipe"], "epochs": 0}
    assert "perplexity" in entry                             # the third command scored it


def test_prepare_supplement_on_one_document_exits_two_before_the_writer(tmp_path, base_dir,
                                                                       monkeypatch, capsys):
    import lfa.supplements as supplements_module

    def reached(*args, **kwargs):
        raise AssertionError("the supplement writer was reached")

    monkeypatch.setattr(supplements_module, "checkpoint_sha256", reached)
    monkeypatch.setattr(supplements_module, "write_supplement", reached)
    book = tmp_path / "wells"
    book.mkdir()
    (book / "book.txt").write_text("One long book. " * 500)
    recipe = tiny_recipe(base_dir, val_fraction=0.1).save(tmp_path / "held.yaml")

    assert main(["prepare-supplement", "--model", str(base_dir), "--corpus", str(book),
                 "--recipe", str(recipe), "--device", "cpu"]) == 2
    [line] = error_lines(capsys)
    assert "holds 1 document(s)" in line and "--split-chars 3500" in line


def test_a_resume_at_another_lambda_exits_two_with_one_line(tmp_path, base_dir, corpus_a,
                                                            recipe_path, capsys):
    workspace = tmp_path / "ws"
    main(["init", str(workspace), "--model", str(base_dir), "--artifact", TINY_ARTIFACT_PATH])
    train = ["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
             "--recipe", str(recipe_path), "--device", "cpu"]
    assert main(train) == 0
    capsys.readouterr()

    assert main(train + ["--resume", "--epochs", "2", "--lambda", "0"]) == 2
    [line] = error_lines(capsys)
    assert "started at lambda 10.0: resume with --lambda 10.0 or without --lambda" in line
