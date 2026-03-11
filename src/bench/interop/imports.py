"""
Gated import helpers for optional third-party tools.

All third-party imports are try/except gated here.  Other interop modules
import availability flags from this module rather than importing directly.
"""

from __future__ import annotations

# ── Stim ────────────────────────────────────────────────────────────
try:
    import stim  # type: ignore[import-untyped]
    HAS_STIM = True
    STIM_VERSION: str | None = getattr(stim, "__version__", "unknown")
except ImportError:
    stim = None  # type: ignore[assignment]
    HAS_STIM = False
    STIM_VERSION = None

# ── PyMatching ──────────────────────────────────────────────────────
try:
    import pymatching  # type: ignore[import-untyped]
    HAS_PYMATCHING = True
    PYMATCHING_VERSION: str | None = getattr(pymatching, "__version__", "unknown")
except ImportError:
    pymatching = None  # type: ignore[assignment]
    HAS_PYMATCHING = False
    PYMATCHING_VERSION = None


def tool_versions() -> dict[str, str | None]:
    """Return a dict of optional tool names to their installed versions.

    Missing tools have ``None`` as their version.
    """
    return {
        "stim": STIM_VERSION,
        "pymatching": PYMATCHING_VERSION,
    }


def require_stim(operation: str = "this operation") -> None:
    """Raise ImportError if stim is not available."""
    if not HAS_STIM:
        raise ImportError(
            f"stim is required for {operation} but is not installed. "
            "Install it with: pip install stim"
        )


def require_pymatching(operation: str = "this operation") -> None:
    """Raise ImportError if pymatching is not available."""
    if not HAS_PYMATCHING:
        raise ImportError(
            f"pymatching is required for {operation} but is not installed. "
            "Install it with: pip install pymatching"
        )
