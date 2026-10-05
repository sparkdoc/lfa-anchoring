"""Shipped text is written for a user of this package, not for a reader of the research record.

The research repository, its claim ids and its retired artifact are not things a user can open,
so no shipped file names them. `docs/verification.md` is a dated record of the port, and is
exempt. A notebook is scanned cell by
cell: each cell's source, which a user runs and reads, and the outputs recorded under each code
cell, which a user reads without running anything.
"""
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where shipped text lives. Tasks that sweep more of the tree extend this.
SCANNED_ROOTS = ["lfa", "README.md", "RELEASING.md", "docs", "examples", "pyproject.toml",
                 "MANIFEST.in"]

EXEMPT = {"docs/verification.md"}
EXEMPT_DIRS = {"__pycache__", ".worktrees"}
SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".ipynb", ".in"}

#: Built by parts so this file does not match itself.
FORBIDDEN = [
    ("a research claim id", re.compile(r"\bC1[0-9]\b")),
    ("the research record", re.compile(r"\bthe LFA " + r"record\b", re.IGNORECASE)),
    ("the research repository", re.compile("mr-" + "fusion")),
    ("the retired published artifact", re.compile("gmm" + "1543k")),
    ("the retired --artifact-id flag", re.compile("--artifact" + "-id")),
]


def _shipped_files():
    for root in SCANNED_ROOTS:
        base = REPO_ROOT / root
        paths = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in paths:
            relative = path.relative_to(REPO_ROOT).as_posix()
            if (not path.is_file() or path.suffix not in SUFFIXES or relative in EXEMPT
                    or any(relative == d or relative.startswith(d + "/") or f"/{d}/" in relative
                           for d in EXEMPT_DIRS)):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if path.suffix == ".ipynb":
                for index, cell in enumerate(json.loads(text)["cells"]):
                    yield f"{relative} cell {index}", "".join(cell["source"])
                    for number, output in enumerate(cell.get("outputs", [])):
                        printed = output.get("text") or output.get("data", {}).get("text/plain")
                        if printed:
                            yield f"{relative} cell {index} output {number}", "".join(printed)
            else:
                yield relative, text


def test_no_shipped_file_speaks_in_the_research_records_terms():
    hits = []
    for relative, text in _shipped_files():
        for number, line in enumerate(text.splitlines(), 1):
            for label, pattern in FORBIDDEN:
                if pattern.search(line):
                    hits.append(f"{relative}:{number}: {label}: {line.strip()[:100]}")
    assert not hits, "\n".join(hits)


def test_a_notebooks_recorded_outputs_are_scanned_as_well_as_its_sources():
    # The stored artifact's size, printed by the walkthrough's artifact cell: it is in a recorded
    # output and in no cell's source.
    needle = "110.0 MB"
    raw = (REPO_ROOT / "examples" / "two_domain_walkthrough.ipynb").read_text(encoding="utf-8")
    assert needle in raw, "the recorded output this test reads has changed; pick another string"

    scanned = dict(_shipped_files())
    cells = {key: text for key, text in scanned.items()
             if key.startswith("examples/two_domain_walkthrough.ipynb cell ")}
    sources = [text for key, text in cells.items() if " output " not in key]
    outputs = [text for key, text in cells.items() if " output " in key]
    assert any("Workspace.init" in text for text in sources), "the cell sources are not scanned"
    assert not any(needle in text for text in sources)
    assert any(needle in text for text in outputs), "the recorded outputs are not scanned"


def test_the_scan_sees_a_planted_hit(tmp_path, monkeypatch):
    planted = REPO_ROOT / "lfa" / "_planted_for_the_wording_test.py"
    planted.write_text("# see C12 in mr-" + "fusion\n")
    try:
        hits = [rel for rel, text in _shipped_files() if "C12" in text]
        assert "lfa/_planted_for_the_wording_test.py" in hits
    finally:
        planted.unlink()
