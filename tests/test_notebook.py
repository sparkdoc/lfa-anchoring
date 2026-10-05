"""The demonstration notebooks, kept from rotting.

A notebook rots differently from library code: an `ImportError` in cell 12 does not stop the
prose reading correctly, and the reader who finds out is the one who set aside an afternoon for
it. There are two notebooks and two layers of checking.

`examples/two_domain_walkthrough.ipynb` is the **how-to**: build or reuse the self-generated
artifact, two domains one after the other, an unanchored control beside each stage, a plain
checkpoint at the end. `examples/what_the_anchor_does.ipynb` is **optional** and continues from
the workspace the walkthrough leaves on disk: each control re-run at its own best number of epochs, and the fixed
generation probes. The split means a name can be used in one notebook and defined only in the
other, which is what `test_no_cell_uses_a_name_no_cell_defines` is for.

The **default suite** parses both notebooks and checks the things a broken edit breaks first --
that they are valid `nbformat`, that every code cell compiles, that no cell reads a name nothing
binds, that the walkthrough still says its settings are demo scale, that it hands the reader on
to the companion and the companion points back, and that the documents which point a reader at
them still do. None of that needs a GPU, a network or a minute.

The **`notebook` marker** actually executes them, with nothing stubbed: it builds or reuses the
self-generated artifact, downloads the two books, trains, and writes a checkpoint. That is the only
thing that can say the walkthrough still works, and it costs what the walkthrough costs. When
they were recorded (2026-10-05, one RTX 3090, the self-generated artifact at the recorded frame
already in the store, the supplement on) the walkthrough took 35.4 minutes and the companion 19.3
(two more training runs and 54 generations); add roughly 1.4 GB of downloads on a cold cache,
and on a cold store the published artifact's download (126 MB with its corpus). The companion's
test executes the walkthrough again before itself, so the tier took 1 h 29 min (5362.85 s) on
that card. The default `addopts` deselects it, like `gpu` and `slow`::

    pytest tests/test_notebook.py -m notebook -q

Both run in a `tmp_path`, because the notebooks write their workspace into the working directory,
and the companion must run in the *same* directory as the walkthrough. Set `LFA_ARTIFACT` to an
artifact file to skip the build.
"""

from __future__ import annotations

import ast
import builtins
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WALKTHROUGH = REPO_ROOT / "examples" / "two_domain_walkthrough.ipynb"
COMPANION = REPO_ROOT / "examples" / "what_the_anchor_does.ipynb"
NOTEBOOKS = [WALKTHROUGH, COMPANION]

#: Files that send a reader to the walkthrough. A rename that misses one of these leaves a dead
#: link in the two places a reader actually starts from.
LINKING_DOCUMENTS = [REPO_ROOT / "README.md", REPO_ROOT / "docs" / "quickstart.md"]


def read_notebook(path=WALKTHROUGH):
    nbformat = pytest.importorskip("nbformat", reason="nbformat is in the [dev] extra")
    return nbformat.read(str(path), as_version=4)


def markdown_of(path):
    return "\n".join(cell.source for cell in read_notebook(path).cells
                     if cell.cell_type == "markdown")


def code_of(path):
    return "\n".join(cell.source for cell in read_notebook(path).cells
                     if cell.cell_type == "code")


# ----------------------------------------------------------------------------------------------
# Default suite: cheap structural checks
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_the_notebook_is_there_and_parses(path):
    notebook = read_notebook(path)
    assert notebook.cells, f"{path.name} has no cells"
    assert notebook.metadata.get("kernelspec", {}).get("name") == "python3", (
        f"{path.name} does not name a python3 kernel, so `jupyter` and `nbclient` cannot pick "
        "an interpreter for it"
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_cell_compiles(path):
    """A syntax error anywhere is a notebook that dies partway with the prose still reading fine.

    Compiling is not executing -- a `NameError` still gets through -- but it is the failure that
    a careless edit produces, and it is free to check.
    """
    broken = []
    for index, cell in enumerate(read_notebook(path).cells):
        if cell.cell_type != "code":
            continue
        try:
            compile(cell.source, f"<{path.name} cell {index}>", "exec")
        except SyntaxError as error:
            broken.append(f"cell {index}: {error}")
    assert not broken, "; ".join(broken)


def _free_names(source: str) -> set[str]:
    """Names the code reads that nothing in it binds.

    Deliberately coarse: it asks only whether a name is bound *somewhere*, not whether it is
    bound before it is used. That is enough for the failure this pair of notebooks can produce --
    a cell that moved to the other notebook taking its definition with it.
    """
    tree = ast.parse(source)
    bound, loaded = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (bound if isinstance(node.ctx, (ast.Store, ast.Del)) else loaded).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    return loaded - bound - set(dir(builtins))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_cell_uses_a_name_no_cell_defines(path):
    """The failure mode of splitting one notebook into two.

    `dose_seconds` was defined in section 8b and summed in section 10's wall-clock total; the
    probe panels read `DOSE_A`, `FAIR1`, `STAGE1_FUSED` and the five-model `table`. Moving cells
    between the two notebooks without carrying those definitions produces a `NameError` several
    minutes into a run, which is exactly the kind of break a reader finds and a reviewer does not.
    """
    missing = sorted(_free_names(code_of(path)))
    assert not missing, f"{path.name} reads names nothing in it binds: {missing}"


def test_the_walkthrough_says_it_is_a_demo_scale_rather_than_the_recipe():
    """The one claim the notebook must never lose in an edit.

    Its epochs are a quarter of the recipe's, so a reader who takes its settings for the
    recommended ones has been misled by this repository. The markdown has to keep saying so.
    """
    markdown = markdown_of(WALKTHROUGH)
    assert "demo settings" in markdown
    assert "shipped recipe" in markdown


def test_the_walkthrough_hands_the_reader_on_to_the_companion():
    """The second notebook is optional, so nothing else would notice if the link went."""
    assert COMPANION.name in markdown_of(WALKTHROUGH), (
        f"{WALKTHROUGH.name} no longer points at {COMPANION.name}, which is the only place a "
        "reader is told the second notebook exists"
    )


def test_the_companion_says_which_notebook_it_continues_and_fails_clearly_without_it():
    """It runs on the walkthrough's workspace, so it has to name it -- twice.

    Once in the prose, so a reader opening it first knows what to run; and once in the setup
    cell, so someone who runs it anyway gets a sentence naming the walkthrough instead of a
    `NameError` twenty lines later.
    """
    assert WALKTHROUGH.name in markdown_of(COMPANION)
    setup = next(cell for cell in read_notebook(COMPANION).cells if cell.cell_type == "code")
    assert "raise SystemExit" in setup.source and WALKTHROUGH.name in setup.source, (
        "the companion's setup cell must stop with a message naming the walkthrough when the "
        "lfa_demo/ workspace is not there"
    )


def test_the_walkthrough_no_longer_advertises_the_sections_that_moved():
    """Sections 8b, 11 and 12 live in the companion now.

    A stale pointer here sends a reader looking for a section that is not in the file, which is
    how the merged notebook's cross-references were found to have rotted in the first place.
    """
    markdown = markdown_of(WALKTHROUGH)
    for stale in ("8b", "section 11", "section 12", "Section 11", "Section 12"):
        assert stale not in markdown, (
            f"{WALKTHROUGH.name} still refers to {stale!r}, which moved to {COMPANION.name}"
        )


def test_the_documents_that_point_at_it_still_do():
    for document in LINKING_DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        assert WALKTHROUGH.name in text, (
            f"{document.relative_to(REPO_ROOT)} no longer mentions {WALKTHROUGH.name}"
        )


def test_the_readme_points_at_both_notebooks():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert COMPANION.name in readme, (
        f"README.md does not mention {COMPANION.name}; a notebook nothing links to is a notebook "
        "nobody opens"
    )


# ----------------------------------------------------------------------------------------------
# The `notebook` marker: run the things
# ----------------------------------------------------------------------------------------------

def _execute(path, working_directory):
    nbformat = pytest.importorskip("nbformat", reason="nbformat is in the [dev] extra")
    nbclient = pytest.importorskip("nbclient", reason="nbclient is in the [dev] extra")
    pytest.importorskip("ipykernel", reason="ipykernel is in the [dev] extra")

    notebook = nbformat.read(str(path), as_version=4)
    client = nbclient.NotebookClient(
        notebook, timeout=5400, kernel_name="python3",
        resources={"metadata": {"path": str(working_directory)}},
    )
    client.execute()
    return notebook


def _printed(notebook):
    return "\n".join(
        output.get("text", "")
        for cell in notebook.cells if cell.cell_type == "code"
        for output in cell.get("outputs", []) if output.output_type == "stream"
    )


@pytest.mark.notebook
def test_the_walkthrough_runs_end_to_end(tmp_path):
    """Execute every cell in a scratch directory, then check it produced what it promises.

    The assertions are the ones a reader would make: two stages in the history, an unanchored
    control beside each of them, an extended artifact, a plain fused checkpoint, and a final
    table naming all five models. An execution error raises out of `execute()` with the offending
    cell attached, so nothing here has to guess at what went wrong.
    """
    notebook = _execute(WALKTHROUGH, tmp_path)

    workspace = tmp_path / "lfa_demo" / "workspace"
    history = json.loads((workspace / "history.json").read_text())
    assert [entry["stage"] for entry in history] == [1, 2]
    assert history[1]["artifact_version"] == 2, "stage 2 did not anchor against the extension"

    for expected in ("runs/stage1/final_model", "runs/stage1_unanchored/final_model",
                     "runs/stage2/final_model", "runs/stage2_unanchored/final_model",
                     "models/stage1_fused", "artifacts/v2.pt"):
        assert (workspace / expected).exists(), f"{expected} was not produced"
    # `fuse` exports to `models/stage{N}_fused_export`, which is a DIFFERENT directory from the
    # `models/stage{N}_fused` that `extend` leaves behind: the notebook fuses stage 2, which was
    # never extended, so only the export exists for it.
    exported = workspace / "models" / "stage2_fused_export"
    assert (exported / "config.json").is_file() and (exported / "model.safetensors").is_file(), (
        f"`fuse` did not write a loadable checkpoint to {exported}"
    )
    assert not (exported / "adapter_config.json").exists(), (
        "the export still carries a PEFT adapter; it is supposed to be a plain checkpoint"
    )

    printed = _printed(notebook)
    for label in ("base Qwen3-0.6B", "after A — anchored", "after A — NO anchor",
                  "after A+B — anchored", "after A+B — NO anchor on B"):
        assert label in printed, f"the five-model table is missing {label!r}"


@pytest.mark.notebook
def test_the_companion_runs_on_the_walkthroughs_workspace(tmp_path):
    """The walkthrough, then the companion, in one working directory.

    This is the only check that the split holds at runtime: the companion's setup cell has to
    rebuild every name the moved cells use, and the two dose-matched controls have to train in
    workspaces of their own so that the chain the walkthrough built is left alone. Without those
    controls the pair reports only the 4-epoch ones, which overstate the anchor by however much of
    the gap is dose -- a substantive regression, not a cosmetic one.
    """
    _execute(WALKTHROUGH, tmp_path)
    notebook = _execute(COMPANION, tmp_path)

    for dose in ("workspace_dose_a", "workspace_dose_b"):
        assert (tmp_path / "lfa_demo" / dose / "runs" / "stage1" / "final_model").exists(), (
            f"the dose-matched control {dose} was not trained"
        )

    printed = _printed(notebook)
    assert "each control's own best dose" in printed, "section 8b printed no dose line"
    for probe in ("What did Darwin mean by natural selection?",
                  "What is the capital of Japan, and roughly how many people live there?"):
        assert probe in printed, f"the probe panels are missing {probe!r}"
