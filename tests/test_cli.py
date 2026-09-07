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
from pathlib import Path

import pytest
import yaml

from conftest import tiny_recipe
from lfa.cli import main

#: Every subcommand the CLI publishes; the help and the parse of each one is asserted below.
SUBCOMMANDS = [
    "init", "fetch-artifact", "train", "extend", "evaluate", "fuse", "chain",
    "build-artifact", "prepare-seed-corpus", "prepare-domain", "list-artifacts",
]


# ------------------------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def recipe_path(tmp_path_factory, base_dir):
    """The tiny recipe on disk -- what `--recipe PATH` is given."""
    return tiny_recipe(base_dir).save(tmp_path_factory.mktemp("recipes") / "tiny.yaml")


@pytest.fixture(scope="module")
def trained(tmp_path_factory, registry, base_dir, corpus_a, recipe_path):
    """A workspace taken through `lfa init` and one `lfa train`, entirely through argv."""
    workspace = tmp_path_factory.mktemp("cli") / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", "tiny"]) == 0
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

def test_list_artifacts_prints_the_shipped_artifact_id(capsys):
    assert main(["list-artifacts"]) == 0

    out = capsys.readouterr().out
    assert "qwen3-0.6b-gmm1543k-int8" in out
    assert "qwen3-0.6b-diagonal" in out


# ------------------------------------------------------------------------------ fetch-artifact

def test_fetch_artifact_writes_the_artifact_into_the_destination(tmp_path, registry, capsys):
    dest = tmp_path / "artifacts"

    assert main(["fetch-artifact", "tiny", "--dest", str(dest)]) == 0

    assert (dest / "tiny.pt").is_file()
    assert str(dest / "tiny.pt") in capsys.readouterr().out


# --------------------------------------------------------------------------------- init, train

def test_init_then_train_leaves_one_history_entry(trained, corpus_a):
    assert (trained / "workspace.json").is_file()
    assert (trained / "artifacts" / "v1.pt").is_file()

    history = json.loads((trained / "history.json").read_text())
    assert len(history) == 1
    assert history[0]["stage"] == 1
    assert history[0]["corpus"] == str(corpus_a)
    assert Path(history[0]["adapter"], "adapter_config.json").is_file()


def test_train_reports_the_run_directory_it_wrote(tmp_path, registry, base_dir, corpus_a,
                                                  recipe_path, capsys):
    workspace = tmp_path / "ws"
    main(["init", str(workspace), "--model", str(base_dir), "--artifact", "tiny"])
    capsys.readouterr()

    assert main(["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
                 "--recipe", str(recipe_path), "--device", "cpu"]) == 0

    assert str(workspace / "runs" / "stage1") in capsys.readouterr().out


def test_the_recipes_loader_frame_reaches_the_runs_config(tmp_path, registry, base_dir, corpus_a,
                                                         recipe_path):
    """There is no flag for it: `keep_short_whole` is the recipe's, and the run records it."""
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", "tiny"]) == 0

    assert main(["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
                 "--recipe", str(recipe_path), "--device", "cpu"]) == 0

    config = json.loads((workspace / "runs" / "stage1" / "config.json").read_text())
    assert config["keep_short_whole"] is yaml.safe_load(
        recipe_path.read_text())["keep_short_whole"]


def test_a_local_artifact_can_be_recorded_as_the_published_one_it_copies(tmp_path, registry,
                                                                        base_dir, tiny_artifact):
    """`--artifact-id`: the file was fetched out of band, but the recipe's calibration still
    reads against the registry id rather than against a path."""
    _, artifact_path = tiny_artifact
    workspace = tmp_path / "ws"

    assert main(["init", str(workspace), "--model", str(base_dir),
                 "--artifact", str(artifact_path), "--artifact-id", "tiny"]) == 0

    assert json.loads((workspace / "workspace.json").read_text())["artifact_id"] == "tiny"


def test_an_unpublished_artifact_id_is_refused(tmp_path, registry, base_dir, tiny_artifact):
    _, artifact_path = tiny_artifact
    assert main(["init", str(tmp_path / "ws"), "--model", str(base_dir),
                 "--artifact", str(artifact_path), "--artifact-id", "not-published"]) == 2


def test_full_weight_reaches_the_runs_config(tmp_path, registry, base_dir, corpus_a, recipe_path):
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", "tiny"]) == 0

    assert main(["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
                 "--recipe", str(recipe_path), "--device", "cpu", "--full-weight"]) == 0

    config = json.loads((workspace / "runs" / "stage1" / "config.json").read_text())
    assert (config["full_weight"], config["use_lora"]) == (True, False)


# -------------------------------------------------------------------------------------- errors

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


def test_an_unknown_artifact_id_exits_two_with_one_line(tmp_path, base_dir, capsys):
    code = main(["init", str(tmp_path / "ws"), "--model", str(base_dir),
                 "--artifact", "no-such-artifact"])

    assert code == 2
    assert len(error_lines(capsys)) == 1


def test_a_directory_that_is_not_a_workspace_exits_two_with_one_line(tmp_path, corpus_a,
                                                                     recipe_path, capsys):
    code = main(["train", "--workspace", str(tmp_path), "--corpus", str(corpus_a),
                 "--recipe", str(recipe_path), "--device", "cpu"])

    assert code == 2
    lines = error_lines(capsys)
    assert len(lines) == 1
    assert "lfa init" in lines[0]                # the message ends in the command to run instead


def test_initialising_over_an_existing_workspace_exits_two_with_one_line(tmp_path, registry,
                                                                         base_dir, capsys):
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", "tiny"]) == 0

    code = main(["init", str(workspace), "--model", str(base_dir), "--artifact", "tiny"])

    assert code == 2
    lines = error_lines(capsys)
    assert len(lines) == 1
    assert "already an LFA workspace" in lines[0]


# ------------------------------------------------------------------------------ extend, chain

def test_extend_folds_the_stage_into_a_second_artifact_version(tmp_path, registry, base_dir,
                                                               corpus_a, recipe_path):
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", "tiny"]) == 0
    assert main(["train", "--workspace", str(workspace), "--corpus", str(corpus_a),
                 "--recipe", str(recipe_path), "--device", "cpu"]) == 0

    assert main(["extend", "--workspace", str(workspace), "--need", "400",
                 "--k-domain", "2", "--device", "cpu"]) == 0

    assert (workspace / "artifacts" / "v2.pt").is_file()
    assert (workspace / "models" / "stage1_fused" / "config.json").is_file()
    state = json.loads((workspace / "workspace.json").read_text())
    assert state["artifact_version"] == 2
    assert state["pending_extend"] is False


def test_chain_trains_every_domain_in_the_spec(tmp_path, registry, base_dir, corpus_a, corpus_b,
                                               recipe_path):
    workspace = tmp_path / "ws"
    assert main(["init", str(workspace), "--model", str(base_dir), "--artifact", "tiny",
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
