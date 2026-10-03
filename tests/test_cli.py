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
    "prepare-supplement", "regenerate-artifact",
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
    assert len(printed) == 1
    assert printed[0].startswith("Next: lfa train --workspace")


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


def test_prepare_domain_with_a_supplement_writes_both(tmp_path, monkeypatch):
    src = tmp_path / "src.txt"
    src.write_text("word " * 400)
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

    src = tmp_path / "src.txt"
    src.write_text("word " * 400)
    seen = {}
    monkeypatch.setattr("lfa.cli.prepare_supplement",
                        lambda corpus, model_id, **kw: seen.update(kw) or tmp_path / "s.jsonl")
    assert main(["prepare-domain", str(src), "--out", str(tmp_path / "out"), "--supplement",
                 "--model", "Qwen/Qwen3-0.6B"]) == 0
    assert isinstance(seen["recipe"], Recipe) and seen["recipe"].name == "qwen3-0.6b"
