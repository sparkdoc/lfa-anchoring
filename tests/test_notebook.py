"""The demonstration notebook, kept from rotting.

A notebook rots differently from library code: an `ImportError` in cell 12 does not stop the
prose reading correctly, and the reader who finds out is the one who set aside an afternoon for
it. So there are two layers here.

The **default suite** parses the committed notebook and checks the things a broken edit breaks
first -- that it is valid `nbformat`, that every code cell compiles, that it names a kernel, and
that the documents which point a reader at it still do. None of that needs a GPU, a network or a
minute.

The **`notebook` marker** actually executes it, top to bottom, with nothing stubbed: it fetches
the artifact, downloads the two books, trains four times and writes a checkpoint. That is the
only thing that can say the walkthrough still works, and it costs what the walkthrough costs --
about 21 minutes on one RTX 3090, plus roughly 1.4 GB of downloads on a cold cache. The default
`addopts` deselects it, like `gpu` and `slow`::

    pytest tests/test_notebook.py -m notebook -q

It runs in a `tmp_path`, because the notebook writes its workspace into the working directory.
Set `LFA_ARTIFACT` to a local copy of the p(h) artifact to have the notebook verify that file's
checksum instead of downloading it again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "examples" / "two_domain_walkthrough.ipynb"

#: Files that send a reader to the notebook. A rename that misses one of these leaves a dead link
#: in the two places a reader actually starts from.
LINKING_DOCUMENTS = [REPO_ROOT / "README.md", REPO_ROOT / "docs" / "quickstart.md"]


def read_notebook():
    nbformat = pytest.importorskip("nbformat", reason="nbformat is in the [dev] extra")
    return nbformat.read(str(NOTEBOOK), as_version=4)


# ----------------------------------------------------------------------------------------------
# Default suite: cheap structural checks
# ----------------------------------------------------------------------------------------------

def test_the_notebook_is_there_and_parses():
    notebook = read_notebook()
    assert notebook.cells, "the notebook has no cells"
    assert notebook.metadata.get("kernelspec", {}).get("name") == "python3", (
        "the notebook does not name a python3 kernel, so `jupyter` and `nbclient` cannot pick "
        "an interpreter for it"
    )


def test_every_code_cell_compiles():
    """A syntax error anywhere is a notebook that dies partway with the prose still reading fine.

    Compiling is not executing -- a `NameError` still gets through -- but it is the failure that
    a careless edit produces, and it is free to check.
    """
    broken = []
    for index, cell in enumerate(read_notebook().cells):
        if cell.cell_type != "code":
            continue
        try:
            compile(cell.source, f"<cell {index}>", "exec")
        except SyntaxError as error:
            broken.append(f"cell {index}: {error}")
    assert not broken, "; ".join(broken)


def test_the_notebook_says_it_is_a_demo_scale_rather_than_the_recipe():
    """The one claim the notebook must never lose in an edit.

    Its corpora are a tenth of what the operating point was tuned on and its epochs a third, so
    a reader who takes its settings for the recommended ones has been misled by this repository.
    The markdown has to keep saying so.
    """
    markdown = "\n".join(cell.source for cell in read_notebook().cells
                         if cell.cell_type == "markdown")
    assert "demo settings" in markdown
    assert "shipped recipe" in markdown


def test_the_documents_that_point_at_it_still_do():
    for document in LINKING_DOCUMENTS:
        assert NOTEBOOK.name in document.read_text(encoding="utf-8"), (
            f"{document.relative_to(REPO_ROOT)} no longer mentions {NOTEBOOK.name}"
        )


# ----------------------------------------------------------------------------------------------
# The `notebook` marker: run the thing
# ----------------------------------------------------------------------------------------------

@pytest.mark.notebook
def test_the_walkthrough_runs_end_to_end(tmp_path):
    """Execute every cell in a scratch directory, then check it produced what it promises.

    The assertions are the ones a reader would make: two stages in the history, an unanchored
    control beside each of them, the two dose-matched controls section 8b adds, an extended
    artifact, a plain fused checkpoint, and a final table naming all five models. An execution
    error raises out of `execute()` with the offending cell attached, so nothing here has to
    guess at what went wrong.
    """
    nbformat = pytest.importorskip("nbformat", reason="nbformat is in the [dev] extra")
    nbclient = pytest.importorskip("nbclient", reason="nbclient is in the [dev] extra")
    pytest.importorskip("ipykernel", reason="ipykernel is in the [dev] extra")

    notebook = nbformat.read(str(NOTEBOOK), as_version=4)
    client = nbclient.NotebookClient(
        notebook, timeout=5400, kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()

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

    # Section 8b: the two controls re-run at their own best dose, each in its own workspace so
    # the chain above is untouched. Without these the notebook reports only the 5-epoch controls,
    # which overstate the anchor -- so their absence is a substantive regression, not a cosmetic
    # one.
    for dose in ("workspace_dose_a", "workspace_dose_b"):
        assert (tmp_path / "lfa_demo" / dose / "runs" / "stage1" / "final_model").exists(), (
            f"the dose-matched control {dose} was not trained"
        )

    printed = "\n".join(
        output.get("text", "")
        for cell in notebook.cells if cell.cell_type == "code"
        for output in cell.get("outputs", []) if output.output_type == "stream"
    )
    for label in ("base Qwen3-0.6B", "after A — anchored", "after A — NO anchor",
                  "after A+B — anchored", "after A+B — NO anchor on B"):
        assert label in printed, f"the five-model table is missing {label!r}"
