"""
v5.4.0 — Tanner Spectral Fragility Diagnostics.

Analyzes the spectral structure and eigenmode localization of the
Tanner graph defined by a parity-check matrix.  Measures global
spectral connectivity, eigenmode localization, variable-node
concentration of fragile modes, and identifies which nodes dominate
the most localized spectral mode.

Operates purely on the parity-check matrix — does not run BP decoding.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_tanner_spectral_analysis(
    parity_check_matrix: np.ndarray,
    top_k_modes: int = 3,
    top_k_nodes: int = 10,
) -> dict[str, Any]:
    """Compute spectral fragility diagnostics for a Tanner graph.

    Parameters
    ----------
    parity_check_matrix : np.ndarray
        Binary parity-check matrix H with shape (m, n) where
        m = number of check nodes, n = number of variable nodes.
    top_k_modes : int
        Number of top eigenmodes to analyze for localization.
    top_k_nodes : int
        Number of top variable nodes to return for the most
        localized mode.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary of spectral diagnostics.
    """
    H = np.asarray(parity_check_matrix, dtype=np.float64)
    m, n = H.shape

    # ── Tanner graph construction ────────────────────────────────
    # Bipartite adjacency: A = [[0, H^T], [H, 0]]
    top = np.concatenate([np.zeros((n, n), dtype=np.float64), H.T], axis=1)
    bottom = np.concatenate([H, np.zeros((m, m), dtype=np.float64)], axis=1)
    A = np.concatenate([top, bottom], axis=0)

    num_edges = int(np.sum(H != 0))

    # ── Adjacency spectrum ───────────────────────────────────────
    eigvals, eigvecs = np.linalg.eigh(A)

    # Sort in descending order (deterministic — eigh returns ascending).
    sort_idx = np.argsort(-eigvals)
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    largest_eigenvalue = float(eigvals[0])
    second_largest = float(eigvals[1]) if len(eigvals) > 1 else 0.0
    adjacency_spectral_gap = largest_eigenvalue - second_largest

    # ── Laplacian spectrum ───────────────────────────────────────
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    L = D - A

    lap_eigvals = np.linalg.eigvalsh(L)
    # Sort ascending (eigvalsh already returns ascending, but be explicit).
    lap_eigvals = np.sort(lap_eigvals)

    laplacian_second_eigenvalue = float(lap_eigvals[1]) if len(lap_eigvals) > 1 else 0.0

    # ── Derived metric ───────────────────────────────────────────
    spectral_ratio = (
        laplacian_second_eigenvalue / largest_eigenvalue
        if largest_eigenvalue > 0.0
        else 0.0
    )

    # ── Eigenmode localization (IPR) ─────────────────────────────
    k = min(top_k_modes, eigvecs.shape[1])

    mode_iprs: list[float] = []
    variable_mode_iprs: list[float] = []

    for i in range(k):
        v = eigvecs[:, i]
        # Normalize (eigh returns normalized, but be safe).
        norm = np.linalg.norm(v)
        if norm > 0.0:
            v = v / norm
        ipr = float(np.sum(v ** 4))
        mode_iprs.append(ipr)

        # Variable-node portion: first n entries.
        v_var = v[:n]
        var_norm = np.linalg.norm(v_var)
        if var_norm > 0.0:
            v_var_normalized = v_var / var_norm
        else:
            v_var_normalized = v_var
        var_ipr = float(np.sum(v_var_normalized ** 4))
        variable_mode_iprs.append(var_ipr)

    max_mode_ipr = max(mode_iprs) if mode_iprs else 0.0
    max_variable_mode_ipr = max(variable_mode_iprs) if variable_mode_iprs else 0.0

    # Identify the most localized mode (by variable-node IPR).
    most_localized_mode_index = int(np.argmax(variable_mode_iprs)) if variable_mode_iprs else 0

    # ── Node localization mapping ────────────────────────────────
    v_localized = eigvecs[:, most_localized_mode_index]
    norm_loc = np.linalg.norm(v_localized)
    if norm_loc > 0.0:
        v_localized = v_localized / norm_loc

    v_variables = v_localized[:n]
    var_norm_loc = np.linalg.norm(v_variables)
    if var_norm_loc > 0.0:
        v_variables = v_variables / var_norm_loc
    weights = np.abs(v_variables)

    # Sort by descending weight (deterministic — use negative for stable sort).
    node_order = np.argsort(-weights)
    top_count = min(top_k_nodes, n)
    top_nodes = node_order[:top_count]

    localized_variable_nodes = [int(idx) for idx in top_nodes]
    localized_variable_weights = [float(weights[idx]) for idx in top_nodes]

    # Localized variable fraction: proportion of mass in top nodes.
    total_mass = float(np.sum(weights))
    if total_mass > 0.0:
        localized_variable_fraction = float(
            np.sum(weights[top_nodes]) / total_mass
        )
    else:
        localized_variable_fraction = 0.0

    return {
        "num_variable_nodes": n,
        "num_check_nodes": m,
        "num_edges": num_edges,
        "largest_eigenvalue": largest_eigenvalue,
        "adjacency_spectral_gap": adjacency_spectral_gap,
        "laplacian_second_eigenvalue": laplacian_second_eigenvalue,
        "spectral_ratio": spectral_ratio,
        "mode_iprs": mode_iprs,
        "variable_mode_iprs": variable_mode_iprs,
        "max_mode_ipr": max_mode_ipr,
        "max_variable_mode_ipr": max_variable_mode_ipr,
        "most_localized_mode_index": most_localized_mode_index,
        "localized_variable_nodes": localized_variable_nodes,
        "localized_variable_weights": localized_variable_weights,
        "localized_variable_fraction": localized_variable_fraction,
    }
