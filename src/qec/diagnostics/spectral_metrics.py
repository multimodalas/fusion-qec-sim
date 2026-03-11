"""
v8.1.0 — Spectral Metrics Aggregator.

Aggregates all spectral stability diagnostics into a single dictionary.
Reuses existing NB spectrum computation where possible to avoid
redundant eigenpair calculations.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.diagnostics._spectral_utils import (
    compute_ipr,
    compute_nb_dominant_eigenpair,
)
from src.qec.diagnostics.bethe_hessian_margin import compute_bethe_hessian_margin
from src.qec.diagnostics.cycle_space_density import compute_cycle_space_density
from src.qec.diagnostics.spectral_nb import _TannerGraph, _compute_eeec, _compute_sis

_ROUND = 12


def compute_spectral_metrics(H: np.ndarray) -> dict[str, Any]:
    """Compute all spectral stability metrics for a parity-check matrix.

    Computes the dominant NB eigenpair once and derives all metrics
    from it, avoiding redundant spectral decompositions.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``spectral_radius`` : float
        - ``entropy`` : float
        - ``spectral_gap`` : float
        - ``bethe_margin`` : float
        - ``support_dimension`` : float
        - ``curvature`` : float
        - ``cycle_density`` : float
        - ``sis`` : float
    """
    H_arr = np.asarray(H, dtype=np.float64)
    graph = _TannerGraph(H_arr)

    # ── Single eigenpair computation ───────────────────────────────
    spectral_radius, eigenvector, directed_edges = (
        compute_nb_dominant_eigenpair(graph)
    )

    # Normalize and canonicalize sign
    norm = np.linalg.norm(eigenvector)
    if norm > 0:
        eigenvector = eigenvector / norm
    max_idx = int(np.argmax(np.abs(eigenvector)))
    if eigenvector[max_idx] < 0:
        eigenvector = -eigenvector

    # ── Edge energy distribution ───────────────────────────────────
    edge_energy = np.abs(eigenvector) ** 2
    total_energy = edge_energy.sum()

    # ── Spectral entropy ──────────────────────────────────────────
    if total_energy > 0:
        p = edge_energy / total_energy
        mask = p > 0
        entropy = float(-np.sum(p[mask] * np.log(p[mask])))
    else:
        entropy = 0.0

    # ── Effective support dimension ───────────────────────────────
    support_dimension = float(np.exp(entropy))

    # ── Spectral curvature ────────────────────────────────────────
    nonzero_energy = edge_energy[edge_energy > 0]
    if len(nonzero_energy) >= 2:
        log_energy = np.log(nonzero_energy)
        curvature = float(np.var(log_energy))
    else:
        curvature = 0.0

    # ── IPR, EEEC, SIS ───────────────────────────────────────────
    ipr = compute_ipr(eigenvector)
    eeec = _compute_eeec(edge_energy)
    sis = _compute_sis(spectral_radius, ipr, eeec)

    # ── Spectral gap (top-2 eigenvalues) ──────────────────────────
    from scipy.sparse.linalg import eigs
    from src.qec.diagnostics._spectral_utils import build_nb_operator

    op, _ = build_nb_operator(graph)
    n_edges = len(directed_edges)

    if n_edges >= 3:
        k = min(2, n_edges - 1)
        vals, _ = eigs(op, k=k, which="LM", tol=1e-6)
        magnitudes = np.sort(np.abs(vals))[::-1]
        spectral_gap = float(magnitudes[0] - magnitudes[1]) if len(magnitudes) >= 2 else 0.0
    else:
        spectral_gap = 0.0

    # ── Bethe Hessian margin ──────────────────────────────────────
    bethe_margin = compute_bethe_hessian_margin(H_arr)

    # ── Cycle space density ───────────────────────────────────────
    cycle_density = compute_cycle_space_density(H_arr)

    return {
        "spectral_radius": round(float(spectral_radius), _ROUND),
        "entropy": round(entropy, _ROUND),
        "spectral_gap": round(spectral_gap, _ROUND),
        "bethe_margin": round(bethe_margin, _ROUND),
        "support_dimension": round(support_dimension, _ROUND),
        "curvature": round(curvature, _ROUND),
        "cycle_density": round(cycle_density, _ROUND),
        "sis": round(sis, _ROUND),
    }
