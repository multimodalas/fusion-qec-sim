"""
v7.6.1 — Non-Backtracking Spectral Diagnostics.

Computes spectral diagnostic metrics from the dominant eigenpair of
the non-backtracking operator on a Tanner graph:

- spectral radius
- dominant eigenvector
- inverse participation ratio (IPR)
- edge energy distribution
- eigenvector edge energy concentration (EEEC)
- spectral instability score (SIS)
- edge sensitivity ranking

Reuses the sparse NB operator from ``_spectral_utils``.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.qec.diagnostics._spectral_utils import (
    build_directed_edges,
    compute_ipr,
    compute_nb_dominant_eigenpair,
)


_ROUND = 12


# ── Internal: Tanner graph adapter ────────────────────────────────


class _TannerGraph:
    """Minimal graph adapter exposing .nodes() and .neighbors() for H."""

    def __init__(self, H: np.ndarray) -> None:
        H = np.asarray(H, dtype=np.float64)
        m, n = H.shape
        self._m = m
        self._n = n

        # Build adjacency: variable nodes 0..n-1, check nodes n..n+m-1
        self._adj: dict[int, list[int]] = {}
        for ci in range(m):
            for vi in range(n):
                if H[ci, vi] != 0:
                    cnode = n + ci
                    self._adj.setdefault(vi, []).append(cnode)
                    self._adj.setdefault(cnode, []).append(vi)

        # All nodes that participate in at least one edge
        self._nodes = sorted(self._adj.keys())

    def nodes(self) -> list[int]:
        return list(self._nodes)

    def neighbors(self, u: int) -> list[int]:
        return list(self._adj.get(u, []))


# ── Core: compute_nb_spectrum ─────────────────────────────────────


def compute_nb_spectrum(
    H: np.ndarray,
) -> dict[str, Any]:
    """Compute non-backtracking spectral diagnostics for a parity-check matrix.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``spectral_radius`` : float
        - ``eigenvector`` : np.ndarray (directed-edge indexed)
        - ``ipr`` : float — inverse participation ratio
        - ``edge_energy`` : np.ndarray — |v_e|^2 per directed edge
        - ``eeec`` : float — eigenvector edge energy concentration
        - ``sis`` : float — spectral instability score
    """
    H_arr = np.asarray(H, dtype=np.float64)

    graph = _TannerGraph(H_arr)

    # Compute dominant eigenpair using sparse operator
    spectral_radius, eigenvector, directed_edges = (
        compute_nb_dominant_eigenpair(graph)
    )

    # Normalize eigenvector
    norm = np.linalg.norm(eigenvector)
    if norm > 0:
        eigenvector = eigenvector / norm

    # IPR
    ipr = compute_ipr(eigenvector)

    # Edge energy: |v_e|^2 per directed edge
    edge_energy = np.abs(eigenvector) ** 2

    # EEEC: eigenvector edge energy concentration
    eeec = _compute_eeec(edge_energy)

    # SIS: spectral instability score
    sis = _compute_sis(spectral_radius, ipr, eeec)

    return {
        "spectral_radius": round(float(spectral_radius), _ROUND),
        "eigenvector": eigenvector,
        "ipr": round(float(ipr), _ROUND),
        "edge_energy": edge_energy,
        "eeec": round(float(eeec), _ROUND),
        "sis": round(float(sis), _ROUND),
    }


# ── EEEC ──────────────────────────────────────────────────────────


def _compute_eeec(edge_energy: np.ndarray) -> float:
    """Compute eigenvector edge energy concentration.

    EEEC = (sum of top-k edge energies) / (total edge energy)
    where k = ceil(sqrt(|E|)).

    Edge energy is normalized before concentration calculation.
    """
    num_edges = len(edge_energy)
    if num_edges == 0:
        return 0.0

    total = edge_energy.sum()
    if total == 0.0:
        return 0.0

    # Normalize distribution
    normalized = edge_energy / total

    k = math.ceil(math.sqrt(num_edges))

    # Deterministic sort: descending by value, ascending by index for ties
    indices = np.argsort(-normalized, kind="stable")
    top_k_energy = normalized[indices[:k]].sum()

    return float(top_k_energy)


# ── SIS ───────────────────────────────────────────────────────────


def _compute_sis(spectral_radius: float, ipr: float, eeec: float) -> float:
    """Compute spectral instability score.

    SIS = log1p(spectral_radius) * ipr * eeec
    """
    return float(np.log1p(spectral_radius) * ipr * eeec)


# ── Edge sensitivity ranking ─────────────────────────────────────


def compute_edge_sensitivity_ranking(
    H: np.ndarray,
) -> list[tuple[int, float]]:
    """Rank directed edges by spectral sensitivity.

    sensitivity(edge) = edge_energy[edge] = |v_e|^2

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    list[tuple[int, float]]
        Deterministic sorted list of (edge_index, sensitivity),
        sorted by sensitivity descending, then edge_index ascending.
    """
    result = compute_nb_spectrum(H)
    edge_energy = result["edge_energy"]

    ranking = []
    for i, energy in enumerate(edge_energy):
        ranking.append((i, round(float(energy), _ROUND)))

    # Sort: sensitivity descending, edge_index ascending for ties
    ranking.sort(key=lambda x: (-x[1], x[0]))

    return ranking
