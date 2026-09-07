"""The shipped examples and the CLI's help, run as programs.

Documentation rots in a way library code does not: an example that stops working still *reads*
correctly, and the reader who finds out is the one trying to use it. So both example scripts are
run here as subprocesses, on the tiny fixture model and on the CPU, and the assertions are the
ones a reader would make -- the run exits 0, the workspace carries the history it should, and the
chain wrote one entry per domain.

The examples are run through ``sys.executable``, with the repository root on ``PYTHONPATH``, so
these pass in a checkout that has not been installed as well as in one that has. The ``--help``
cases prefer the installed ``lfa`` console script and fall back to the module, so the entry point
is covered wherever it exists.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import make_corpus, tiny_recipe

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"

#: Every subcommand the CLI defines. Kept as a literal list rather than read off the parser: the
#: point is to notice a subcommand that stopped parsing, and a list generated from the parser
#: would follow it into the change.
SUBCOMMANDS = ["init", "fetch-artifact", "list-artifacts", "train", "extend", "evaluate", "fuse",
               "chain", "build-artifact", "prepare-seed-corpus", "prepare-domain"]


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command from the repository root with ``lfa`` importable, capturing both streams."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), environment.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return subprocess.run(command, cwd=REPO_ROOT, env=environment, capture_output=True,
                          text=True, **kwargs)


def script(name: str, *arguments: str) -> subprocess.CompletedProcess:
    return run([sys.executable, str(EXAMPLES / name), *arguments])


def assert_succeeded(result: subprocess.CompletedProcess) -> None:
    """Fail with the child's own output: a bare exit status says nothing about what broke."""
    assert result.returncode == 0, (
        f"exit {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n"
        f"{result.stderr}")


@pytest.fixture(scope="module")
def recipe_file(tmp_path_factory, base_dir):
    """The tiny operating point as a YAML file, for the examples' ``--recipe``.

    The examples default to the bundled recipe that names the model they are given, and the
    fixture model is not one; a real run of either example on Qwen3-0.6B needs no ``--recipe``.
    """
    return tiny_recipe(base_dir).save(tmp_path_factory.mktemp("recipe") / "tiny.yaml")


# ==============================================================================================
# examples/quickstart.py
# ==============================================================================================

def test_the_quickstart_example_trains_evaluates_and_exports(tmp_path, base_dir, tiny_artifact,
                                                             recipe_file):
    _, artifact_path = tiny_artifact
    corpus = make_corpus(tmp_path / "domain", "consciousness")
    workspace = tmp_path / "ws"

    result = script("quickstart.py",
                    "--model", str(base_dir), "--artifact", str(artifact_path),
                    "--corpus", str(corpus), "--out", str(workspace),
                    "--recipe", str(recipe_file), "--epochs", "1",
                    "--device", "cpu", "--n-windows", "none")

    assert_succeeded(result)
    history = json.loads((workspace / "history.json").read_text())
    assert len(history) == 1
    assert history[0]["stage"] == 1
    assert history[0]["perplexity"]["after"]["domain"] > 0     # the evaluate step ran
    assert (workspace / "models" / "stage1_fused_export").is_dir()   # the fuse step ran
    assert "domain" in result.stdout                                  # the table was printed


def test_the_quickstart_example_refuses_a_workspace_that_already_exists(tmp_path, base_dir,
                                                                       tiny_artifact,
                                                                       recipe_file):
    """The one refusal a reader is most likely to meet: a re-run over a finished workspace."""
    _, artifact_path = tiny_artifact
    corpus = make_corpus(tmp_path / "domain", "consciousness")
    workspace = tmp_path / "ws"
    common = ["--model", str(base_dir), "--artifact", str(artifact_path),
              "--corpus", str(corpus), "--out", str(workspace),
              "--recipe", str(recipe_file), "--epochs", "1",
              "--device", "cpu", "--n-windows", "none"]

    assert_succeeded(script("quickstart.py", *common))
    again = script("quickstart.py", *common)

    assert again.returncode != 0
    assert "already an LFA workspace" in again.stderr


# ==============================================================================================
# examples/chain_three_domains.py
# ==============================================================================================

def test_the_chain_example_trains_every_domain_in_the_spec(tmp_path, base_dir, tiny_artifact,
                                                           recipe_file):
    _, artifact_path = tiny_artifact
    out = tmp_path / "chain"

    result = script("chain_three_domains.py",
                    "--model", str(base_dir), "--artifact", str(artifact_path),
                    "--out", str(out), "--recipe", str(recipe_file),
                    "--epochs", "1", "--documents", "4", "--sentences", "8",
                    "--device", "cpu")

    assert_succeeded(result)
    history = json.loads((out / "workspace" / "history.json").read_text())
    assert len(history) == 3
    assert [entry["stage"] for entry in history] == [1, 2, 3]
    # Each stage started from the model the previous one was folded into, and anchored against
    # the artifact that stage was merged into.
    assert [entry["artifact_version"] for entry in history] == [1, 2, 3]
    assert history[0]["base_model"] == str(base_dir)
    assert history[1]["base_model"] != history[0]["base_model"]
    # Stage 1 uses the recipe's lambda; every later stage multiplies it (3x by default).
    assert history[1]["lambda_applied"] == history[0]["lambda_applied"] * 3
    assert history[2]["lambda_applied"] == history[1]["lambda_applied"]


def test_the_chain_example_generates_a_corpus_per_domain_and_leaves_the_spec_alone(
        tmp_path, base_dir, tiny_artifact, recipe_file):
    """The generated corpora go under --out, and the shipped spec is not written to."""
    _, artifact_path = tiny_artifact
    out = tmp_path / "chain"
    shipped_spec = EXAMPLES / "domains.yaml"
    before = shipped_spec.read_bytes()

    assert_succeeded(script("chain_three_domains.py",
                            "--model", str(base_dir), "--artifact", str(artifact_path),
                            "--out", str(out), "--recipe", str(recipe_file),
                            "--epochs", "1", "--documents", "4", "--sentences", "8",
                            "--device", "cpu"))

    assert shipped_spec.read_bytes() == before
    assert not (EXAMPLES / "corpora").exists()
    generated = sorted(path.name for path in (out / "corpora").iterdir())
    assert generated == ["archaeology", "cybersecurity", "philosophy"]
    assert len(list((out / "corpora" / "philosophy").glob("*.txt"))) == 4


# ==============================================================================================
# The command line's own help
# ==============================================================================================

@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_every_subcommand_has_help(subcommand):
    """``lfa <sub> --help`` parses and prints a usage line, for every subcommand there is."""
    console_script = Path(sys.executable).parent / "lfa"
    command = ([str(console_script)] if console_script.is_file()
               else [sys.executable, "-c", "from lfa.cli import main; raise SystemExit(main())"])

    result = run([*command, subcommand, "--help"])

    assert_succeeded(result)
    assert result.stdout.startswith("usage:")
    assert subcommand in result.stdout
