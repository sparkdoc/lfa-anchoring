"""Layerwise Function Anchoring (LFA)."""
__version__ = "0.1.0"

from .recipe import Recipe  # noqa: F401

try:
    from .workspace import Workspace  # noqa: F401
except ImportError:  # pragma: no cover - Task 14 has not landed yet
    pass
