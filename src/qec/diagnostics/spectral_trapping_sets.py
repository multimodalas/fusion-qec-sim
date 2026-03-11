"""
v5.6.0 — Spectral Trapping-Set Diagnostics.

Identifies localized spectral structures in the Tanner graph that may
correspond to potential trapping sets or pseudocodeword-prone regions.
Localized spectral eigenvectors that concentrate mass on small subsets
of variable nodes indicate structural weaknesses that can cause BP
decoding failures.

Operates purely on pre-computed spectral eigenvectors — does not run
BP decoding.  Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_spectral_trapping_sets(
    spectral_modes: list[np.ndarray],
    variable_node_count: int,
) -> dict[str, Any]:
    """Identify localized spectral clusters in Tanner graph eigenvectors.

    For each spectral eigenvector, extracts variable-node components,
    computes per-node importance (absolute magnitude), and identifies
    nodes exceeding a deterministic threshold (mean + std) as members
    of a localized spectral cluster.

    Parameters
    ----------
    spectral_modes : list[np.ndarray]
        Eigenvectors from v5.4 Tanner spectral analysis.  Each vector
        contains components for all Tanner graph nodes (variable nodes
        followed by check nodes).
    variable_node_count : int
        Number of variable nodes in the Tanner graph.  The first
        ``variable_node_count`` components of each eigenvector
        correspond to variable nodes.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:
        - cluster_count: number of non-empty clusters found
        - clusters: list of cluster metadata dicts
        - largest_cluster_size: size of the largest cluster
        - mean_cluster_size: mean size across non-empty clusters

    Raises
    ------
    ValueError
        If ``spectral_modes`` is empty, ``variable_node_count`` is
        non-positive, or an eigenvector has fewer components than
        ``variable_node_count``.
    """
    if len(spectral_modes) == 0:
        raise ValueError("spectral_modes must not be empty")
    if variable_node_count <= 0:
        raise ValueError("variable_node_count must be positive")

    clusters: list[dict[str, Any]] = []

    for mode_idx, mode in enumerate(spectral_modes):
        v = np.asarray(mode, dtype=np.float64)

        if v.shape[0] < variable_node_count:
            raise ValueError(
                f"spectral_modes[{mode_idx}] has {v.shape[0]} components, "
                f"but variable_node_count is {variable_node_count}"
            )

        # Step 1: Extract variable-node components.
        v_var = v[:variable_node_count].copy()

        # Step 2: Normalize vector magnitude.
        norm = np.linalg.norm(v_var)
        if norm > 0.0:
            v_var = v_var / norm

        # Step 3: Compute node importance.
        importance = np.abs(v_var)

        # Step 4: Deterministic threshold.
        mean_imp = importance.mean()
        std_imp = importance.std()
        threshold = float(mean_imp) + float(std_imp)

        # Identify localized nodes (vectorized).
        indices = np.flatnonzero(importance > threshold)
        cluster_size = indices.size

        if cluster_size == 0:
            continue

        # Step 5–6: Record cluster metadata.
        cluster_importance = importance[indices]
        clusters.append({
            "mode_index": mode_idx,
            "cluster_size": cluster_size,
            "nodes": indices.astype(int).tolist(),
            "max_importance": float(cluster_importance.max()),
            "mean_importance": float(cluster_importance.mean()),
        })

    # Aggregate metrics.
    cluster_count = len(clusters)
    if cluster_count > 0:
        sizes = [c["cluster_size"] for c in clusters]
        largest_cluster_size = max(sizes)
        mean_cluster_size = float(np.mean(sizes))
    else:
        largest_cluster_size = 0
        mean_cluster_size = 0.0

    return {
        "cluster_count": cluster_count,
        "clusters": clusters,
        "largest_cluster_size": largest_cluster_size,
        "mean_cluster_size": mean_cluster_size,
    }
