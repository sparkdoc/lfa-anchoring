"""Layerwise Function Anchoring (LFA)."""
__version__ = "0.1.0"
try:
    from .recipe import Recipe          # noqa: E402,F401  (Task 13)
    from .workspace import Workspace    # noqa: E402,F401  (Task 14)
except ImportError:  # pragma: no cover - Tasks 13/14 have not landed yet
    pass
