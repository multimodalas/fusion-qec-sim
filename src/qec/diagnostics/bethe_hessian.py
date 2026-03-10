"""
v6.0.0 — Bethe Hessian Spectral Diagnostics.

Computes the spectrum of the Bethe Hessian matrix for the Tanner graph
of a parity-check matrix.  The Bethe Hessian provides a spectral
characterization of community structure and phase transitions in
sparse graphs, and its negative eigenvalues indicate the presence of
detectable structure below the spectral threshold.

In the context of BP decoding, negative Bethe Hessian eigenvalues
may signal regimes where belief propagation is unable to converge
reliably due to structural degeneracies.

Operates purely on the parity-check matrix — does not run BP decoding.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def compute_bethe_hessian(
    parity_check_matrix: np.ndarray,
    r: float | None = None,
) -> dict[str, Any]:
    """Compute the Bethe Hessian spectrum for a Tanner graph.

    The Bethe Hessian is defined as:

        H_B = (r^2 - 1) I - r A + D

    where A is the Tanner graph adjacency matrix, D is the degree matrix,
    and r is a regularization parameter.

    If ``r`` is not provided, it defaults to sqrt(spectral_radius) of
    the non-backtracking matrix, approximated as sqrt(largest eigenvalue
    of A) for efficiency.

    Parameters
    ----------
    parity_check_matrix : np.ndarray
        Binary parity-check matrix H with shape (m, n).
    r : float or None
        Regularization parameter.  If None, defaults to
        sqrt(sqrt(largest_eigenvalue_of_A)), which approximates
        sqrt(non-backtracking spectral radius).

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:
        - ``bethe_eigenvalues``: list of eigenvalues (sorted ascending)
        - ``min_eigenvalue``: float, smallest eigenvalue
        - ``num_negative``: int, count of negative eigenvalues
        - ``r_used``: float, the regularization parameter used
    """
    H = np.asarray(parity_check_matrix, dtype=np.float64)
    m, n = H.shape

    # ── Build Tanner graph adjacency matrix ───────────────────────
    # Bipartite adjacency: A = [[0, H^T], [H, 0]]
    total = n + m
    top = np.concatenate([np.zeros((n, n), dtype=np.float64), H.T], axis=1)
    bottom = np.concatenate([H, np.zeros((m, m), dtype=np.float64)], axis=1)
    A = np.concatenate([top, bottom], axis=0)

    # ── Degree matrix ─────────────────────────────────────────────
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)

    # ── Determine r ───────────────────────────────────────────────
    if r is None:
        # Approximate: r = sqrt(spectral_radius_of_nb_matrix)
        # For bipartite graphs, the NB spectral radius is well
        # approximated by sqrt(largest_eigenvalue_of_A).
        eigvals_A = np.linalg.eigvalsh(A)
        max_eigval_A = float(np.max(np.abs(eigvals_A)))
        r_used = math.sqrt(max_eigval_A) if max_eigval_A > 0.0 else 1.0
    else:
        r_used = float(r)

    # ── Construct Bethe Hessian ───────────────────────────────────
    # H_B = (r^2 - 1) I - r A + D
    I = np.eye(total, dtype=np.float64)
    H_B = (r_used ** 2 - 1.0) * I - r_used * A + D

    # ── Compute spectrum ──────────────────────────────────────────
    eigvals = np.linalg.eigvalsh(H_B)
    # eigvalsh returns sorted ascending — keep that order.
    eigvals_sorted = np.sort(eigvals)

    min_eigenvalue = float(eigvals_sorted[0])
    num_negative = int(np.sum(eigvals_sorted < 0.0))

    bethe_eigenvalues = [float(v) for v in eigvals_sorted]

    return {
        "bethe_eigenvalues": bethe_eigenvalues,
        "min_eigenvalue": min_eigenvalue,
        "num_negative": num_negative,
        "r_used": r_used,
    }
