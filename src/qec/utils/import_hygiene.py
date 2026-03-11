"""
v9.4.1 — Import hygiene helper.

Ensures the repository root is on sys.path so that scripts run directly
(outside ``pip install -e .``) can resolve ``qec`` and ``bench`` imports.

Layer: utilities (shared).
No external dependencies. No side effects beyond sys.path mutation.
"""

from __future__ import annotations

import os
import sys


def ensure_repo_root_on_path() -> None:
    """Insert the repository root into ``sys.path`` if not already present."""
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    if root not in sys.path:
        sys.path.insert(0, root)
