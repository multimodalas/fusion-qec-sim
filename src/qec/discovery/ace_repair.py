"""
v9.5.0 — ACE-Based Repair Operator.

Implements a lightweight Approximate Cycle Extrinsic (ACE) constraint
repair operator that discourages fragile local structures by rewiring
edges connected to low-degree variable nodes.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import numpy as np


def repair_graph_with_ace_constraint(H: np.ndarray) -> np.ndarray:
    """Repair fragile local structures using ACE constraints.

    Scans for edges where the connected variable node has degree < 2,
    indicating a fragile connection prone to trapping-set formation.
    Rewires such edges deterministically to an adjacent column position
    while preserving matrix shape and binary entries.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    np.ndarray
        Repaired parity-check matrix.
    """
    H_new = np.asarray(H, dtype=np.float64).copy()
    m, n = H_new.shape

    for i in range(m):
        for j in range(n):
            if H_new[i, j] == 1:
                degree = int(np.sum(H_new[:, j]))
                if degree < 2:
                    H_new[i, j] = 0.0
                    new_j = (j + 1) % n
                    H_new[i, new_j] = 1.0

    return H_new
