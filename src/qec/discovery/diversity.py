"""
v9.3.0 — Structural Diversity Regulator.

Deterministic diversity regulation for the discovery engine.
Computes structural signatures and penalties to prevent premature
convergence to a single graph family.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no hidden randomness, no global state, no input mutation.
"""

from __future__ import annotations

import numpy as np


def compute_structure_signature(H: np.ndarray) -> tuple[float, ...]:
    """Compute a deterministic structural signature for a parity-check matrix.

    Signature components:
      - mean variable degree
      - variance of variable degrees
      - mean check degree
      - variance of check degrees

    Parameters
    ----------
    H : np.ndarray
        Parity-check matrix (m x n).

    Returns
    -------
    tuple[float, ...]
        Structural signature tuple.
    """
    var_degrees = np.asarray(np.sum(H, axis=0)).ravel()
    chk_degrees = np.asarray(np.sum(H, axis=1)).ravel()

    signature = (
        float(np.mean(var_degrees)),
        float(np.var(var_degrees)),
        float(np.mean(chk_degrees)),
        float(np.var(chk_degrees)),
    )
    return signature


def compute_diversity_penalty(
    sig: tuple[float, ...],
    archive: list[tuple[float, ...]],
) -> float:
    """Compute diversity penalty for a candidate relative to an archive.

    Higher penalty for structures too similar to existing elites.
    Returns 0.0 when the archive is empty.

    Parameters
    ----------
    sig : tuple[float, ...]
        Structural signature of the candidate.
    archive : list[tuple[float, ...]]
        List of structural signatures from archive elites.

    Returns
    -------
    float
        Diversity penalty in [0, 1).
    """
    if not archive:
        return 0.0

    distances = [
        sum(abs(a - b) for a, b in zip(sig, s))
        for s in archive
    ]

    return 1.0 / (1.0 + min(distances))
