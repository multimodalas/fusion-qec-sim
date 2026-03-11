"""
v10.0.0 — Artifact Metadata Stamp.

Generates reproducibility metadata for every discovery run, including
git commit, library versions, seed, and ISO 8601 timestamp.

Layer 3 — Utilities.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any

import numpy as np
import scipy


def _get_git_commit() -> str:
    """Return the current git commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def generate_run_metadata(seed: int) -> dict[str, Any]:
    """Generate reproducibility metadata for a discovery run.

    Parameters
    ----------
    seed : int
        The seed used for this run.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``repo_version`` : str — git commit hash
        - ``numpy_version`` : str — numpy version
        - ``scipy_version`` : str — scipy version
        - ``seed`` : int — the seed
        - ``timestamp`` : str — ISO 8601 timestamp
    """
    return {
        "repo_version": _get_git_commit(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
