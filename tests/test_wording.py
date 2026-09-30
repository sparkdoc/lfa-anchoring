"""Shipped text is written for a user of this package, not for a reader of the research record.

The research repository, its claim ids and its retired artifact are not things a user can open,
so no shipped file names them. `docs/verification.md` is a dated record of the port and
`docs/superpowers/` holds development documents; both are exempt.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where shipped text lives. Tasks that sweep more of the tree extend this.
SCANNED_ROOTS = ["lfa"]

EXEMPT = {"docs/verification.md"}
EXEMPT_DIRS = {"docs/superpowers", "__pycache__", ".worktrees"}
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
            yield relative, path.read_text(encoding="utf-8", errors="replace")


def test_no_shipped_file_speaks_in_the_research_records_terms():
    hits = []
    for relative, text in _shipped_files():
        for number, line in enumerate(text.splitlines(), 1):
            for label, pattern in FORBIDDEN:
                if pattern.search(line):
                    hits.append(f"{relative}:{number}: {label}: {line.strip()[:100]}")
    assert not hits, "\n".join(hits)


def test_the_scan_sees_a_planted_hit(tmp_path, monkeypatch):
    planted = REPO_ROOT / "lfa" / "_planted_for_the_wording_test.py"
    planted.write_text("# see C12 in mr-" + "fusion\n")
    try:
        hits = [rel for rel, text in _shipped_files() if "C12" in text]
        assert "lfa/_planted_for_the_wording_test.py" in hits
    finally:
        planted.unlink()
