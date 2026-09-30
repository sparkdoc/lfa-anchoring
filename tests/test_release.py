"""The scanner a release check uses to find a string anywhere that ships.

No check is marked ``release`` today: the one that was -- a placeholder in the published links,
filled in when the artifact registry was -- went with the registry. What stays is
:func:`scan_for` and the proof, in the default suite, that it can actually see: a scanner that
quietly matched nothing would pass any gate built on it for the wrong reason.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Extensions worth scanning: what ships (the package, the docs, the metadata), not fixtures.
SCANNED = {".py", ".md", ".toml", ".cfg", ".yaml", ".yml", ".txt", ".in"}

SKIPPED_DIRS = {".git", ".venv", "venv", "build", "dist", "__pycache__", ".pytest_cache",
                "_runs", "fixtures", ".worktrees"}


def scan_for(needle: str, root: Path = REPO_ROOT,
             allowed: frozenset[str] = frozenset()) -> dict[str, int]:
    """``{relative path: occurrences}`` for every scanned file under ``root`` holding ``needle``.

    ``allowed`` names files (relative to ``root``) that may hold it -- the document that
    describes what the check looks for, say -- and are left out of the result.
    """
    root = Path(root)
    found: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in SCANNED:
            continue
        if set(path.relative_to(root).parts) & SKIPPED_DIRS:
            continue
        relative = path.relative_to(root).as_posix()
        if relative in allowed:
            continue
        count = path.read_text(encoding="utf-8", errors="replace").count(needle)
        if count:
            found[relative] = count
    return found


def test_the_scan_has_teeth(tmp_path):
    """On a synthetic tree: it sees a planted needle, respects the skip list and the allow list,
    and ignores a file type that does not ship."""
    needle = "planted-" + "needle"                     # built by parts: this file is not a hit
    (tmp_path / "README.md").write_text(f"see {needle} and {needle}\n")
    (tmp_path / "RELEASING.md").write_text(f"describes {needle}\n")        # allowed to hold it
    (tmp_path / "notes.rst").write_text(needle)                            # not a shipped type
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "copy.md").write_text(needle)                    # a skipped directory
    (tmp_path / ".worktrees" / "branch").mkdir(parents=True)
    (tmp_path / ".worktrees" / "branch" / "README.md").write_text(needle)  # a leftover worktree

    assert scan_for(needle, tmp_path, allowed=frozenset({"RELEASING.md"})) == {"README.md": 2}
    assert scan_for(needle, tmp_path) == {"README.md": 2, "RELEASING.md": 1}
    assert scan_for("nothing like this", tmp_path) == {}
