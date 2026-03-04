"""
BP energy diagnostic — measures correlation between LLR field and beliefs.

Provides a simple, deterministic energy function for tracking BP
convergence behavior across iterations.
"""

from __future__ import annotations

import numpy as np


def bp_energy(llr: np.ndarray, beliefs: np.ndarray) -> float:
    """Simple BP energy diagnostic.

    Measures correlation between LLR field and current beliefs.
    Lower values indicate stronger agreement.

    Parameters
    ----------
    llr : np.ndarray
        LLR vector of length n.
    beliefs : np.ndarray
        Current variable beliefs (L_total) of length n.

    Returns
    -------
    float
        Energy value: -sum(llr * beliefs).
    """
    return float(-np.sum(
        np.asarray(llr, dtype=np.float64) * np.asarray(beliefs, dtype=np.float64)
    ))
