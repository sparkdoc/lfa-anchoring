"""What a wheel ships, read from `pyproject.toml` rather than from a build.

`[tool.setuptools] packages` is an explicit list, so a subpackage added to `lfa/` without being
added there is silently left out of the wheel. An editable install hides that -- it imports from
the checkout -- and the first to find out is a user whose `import lfa` fails after `pip install`.
"""
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _packages() -> list[str]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return config["tool"]["setuptools"]["packages"]


def test_every_package_under_lfa_is_listed_for_the_wheel():
    on_disk = sorted(
        ".".join(init.parent.relative_to(REPO_ROOT).parts)
        for init in (REPO_ROOT / "lfa").rglob("__init__.py")
        if "__pycache__" not in init.parts
    )
    missing = [name for name in on_disk if name not in _packages()]
    assert not missing, f"pyproject.toml [tool.setuptools] packages leaves out {missing}"


def test_the_examples_are_still_installed_inside_the_package():
    assert "lfa.examples" in _packages()
