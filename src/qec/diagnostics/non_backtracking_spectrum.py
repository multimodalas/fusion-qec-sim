"""
v6.0.0 — Non-Backtracking (Hashimoto) Spectrum Diagnostics.

Computes eigenvalues of the non-backtracking matrix derived from the
Tanner graph of a parity-check matrix.  The non-backtracking matrix
captures the structure of message passing on the graph without
immediate reversal, making it a more faithful spectral proxy for
BP dynamics than the ordinary adjacency spectrum.

Operates purely on the parity-check matrix — does not run BP decoding.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_non_backtracking_spectrum(
    parity_check_matrix: np.ndarray,
) -> dict[str, Any]:
    """Compute eigenvalues of the non-backtracking (Hashimoto) matrix.

    The non-backtracking matrix B is defined on directed edges of the
    Tanner graph.  For directed edges (u -> v) and (v -> w), we have
    B_{(u->v), (v->w)} = 1 if w != u (no immediate backtrack).

    Parameters
    ----------
    parity_check_matrix : np.ndarray
        Binary parity-check matrix H with shape (m, n).

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:
        - ``nb_eigenvalues``: list of eigenvalue magnitudes (sorted
          descending by magnitude, real and imaginary parts as pairs)
        - ``spectral_radius``: float, largest eigenvalue magnitude
        - ``num_eigenvalues``: int, total number of eigenvalues
    """
    H = np.asarray(parity_check_matrix, dtype=np.float64)
    m, n = H.shape

    # ── Build Tanner graph bipartite adjacency ────────────────────
    # Variable nodes: 0..n-1, check nodes: n..n+m-1
    total_nodes = n + m

    # Build directed edge list from undirected Tanner edges.
    # Each undirected edge (u, v) produces two directed edges: u->v and v->u.
    directed_edges: list[tuple[int, int]] = []
    for ci in range(m):
        for vi in range(n):
            if H[ci, vi] != 0:
                u = vi          # variable node index
                w = n + ci      # check node index
                directed_edges.append((u, w))
                directed_edges.append((w, u))

    num_directed = len(directed_edges)

    if num_directed == 0:
        return {
            "nb_eigenvalues": [],
            "spectral_radius": 0.0,
            "num_eigenvalues": 0,
        }

    # ── Index directed edges ──────────────────────────────────────
    edge_to_idx: dict[tuple[int, int], int] = {}
    for idx, edge in enumerate(directed_edges):
        edge_to_idx[edge] = idx

    # ── Build adjacency list for outgoing edges from each node ────
    outgoing: dict[int, list[int]] = {}
    for idx, (u, v) in enumerate(directed_edges):
        if v not in outgoing:
            outgoing[v] = []
        outgoing[v].append(idx)

    # ── Construct non-backtracking matrix B ───────────────────────
    # B_{(u->v), (v->w)} = 1 if w != u
    B = np.zeros((num_directed, num_directed), dtype=np.float64)

    for idx_uv, (u, v) in enumerate(directed_edges):
        # For all directed edges (v -> w) where w != u
        for idx_vw in outgoing.get(v, []):
            _, w = directed_edges[idx_vw]
            if w != u:
                B[idx_uv, idx_vw] = 1.0

    # ── Compute eigenvalues ───────────────────────────────────────
    # Non-backtracking matrix is generally non-symmetric, so use eig.
    eigenvalues = np.linalg.eigvals(B)

    # Compute magnitudes.
    magnitudes = np.abs(eigenvalues)

    # Sort by magnitude descending, with deterministic tie-breaking
    # by real part descending, then imaginary part descending.
    sort_keys = np.lexsort((
        -eigenvalues.imag,
        -eigenvalues.real,
        -magnitudes,
    ))
    eigenvalues = eigenvalues[sort_keys]
    magnitudes = magnitudes[sort_keys]

    spectral_radius = float(magnitudes[0])

    # Convert eigenvalues to JSON-safe format: list of [real, imag] pairs.
    nb_eigenvalues: list[list[float]] = []
    for ev in eigenvalues:
        nb_eigenvalues.append([float(ev.real), float(ev.imag)])

    return {
        "nb_eigenvalues": nb_eigenvalues,
        "spectral_radius": spectral_radius,
        "num_eigenvalues": num_directed,
    }
