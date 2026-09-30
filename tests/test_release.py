"""The checks that must pass before anything is published, and only then.

Marked `release` and deselected from the default suite, because these assert the *end* state of
the release procedure: today the repository deliberately carries the placeholder they refuse.
`RELEASING.md` runs them at the step that fills it in, so the gate is part of the procedure
rather than something a person has to remember.

    pytest -m release -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Built by parts so this file is not itself a hit when the scan runs over the tree.
PLACEHOLDER = "<" + "org" + ">"

#: Where the placeholder is allowed to survive: the release procedure that describes filling it
#: in, and this file. Everything else that carries it would ship dead.
ALLOWED = {"RELEASING.md", "tests/test_release.py"}

#: Extensions worth scanning: what ships (the package, the docs, the metadata), not fixtures.
SCANNED = {".py", ".md", ".toml", ".cfg", ".yaml", ".yml", ".txt", ".in"}

SKIPPED_DIRS = {".git", ".venv", "venv", "build", "dist", "__pycache__", ".pytest_cache",
                "_runs", "fixtures", ".worktrees"}


def scan_for(needle: str, root: Path = REPO_ROOT) -> dict[str, int]:
    """``{relative path: occurrences}`` for every scanned file under ``root`` holding ``needle``."""
    root = Path(root)
    found: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in SCANNED:
            continue
        if set(path.relative_to(root).parts) & SKIPPED_DIRS:
            continue
        relative = path.relative_to(root).as_posix()
        if relative in ALLOWED:
            continue
        count = path.read_text(encoding="utf-8", errors="replace").count(needle)
        if count:
            found[relative] = count
    return found


@pytest.mark.release
def test_no_release_placeholder_survives_anywhere_that_ships():
    """`<org>` in a published link is a dead link, in the README and the metadata.

    `[project.urls]` and the README's absolute links are what a PyPI visitor follows. Both carry
    the same placeholder, and `RELEASING.md` replaces them in one edit -- this is what says it
    happened.
    """
    remaining = scan_for(PLACEHOLDER)
    assert not remaining, (
        "the release placeholder is still in "
        + ", ".join(f"{path} ({count}x)" for path, count in remaining.items())
        + ". RELEASING.md replaces it in pyproject.toml and README.md; until it is replaced, "
          "every published link is dead."
    )


def test_the_scan_has_teeth(tmp_path):
    """The gate's own teeth, on a synthetic tree, in the DEFAULT suite.

    A scanner that quietly matched nothing would pass the release gate above for the wrong
    reason, on the day it mattered most -- and the gate itself cannot say so, because it is
    expected to fail until the release steps are done. So the mechanism is proved here instead:
    it sees a planted needle, respects the skip list and the allow list, and ignores a file type
    that does not ship.
    """
    (tmp_path / "README.md").write_text(f"see {PLACEHOLDER} and {PLACEHOLDER}\n")
    (tmp_path / "RELEASING.md").write_text(f"replace {PLACEHOLDER} here\n")   # allowed to hold it
    (tmp_path / "notes.rst").write_text(PLACEHOLDER)                          # not a shipped type
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "copy.md").write_text(PLACEHOLDER)                  # a skipped directory

    assert scan_for(PLACEHOLDER, tmp_path) == {"README.md": 2}
    assert scan_for("nothing like this", tmp_path) == {}
