"""
v10.0.0 — Spectral Fitness Metrics.

Deterministic metrics used by the fitness engine for evaluating
LDPC/QLDPC parity-check matrices.  All functions operate on dense
numpy arrays and return deterministic results.

Metrics:
  - NBT spectral radius (power iteration estimate)
  - Girth spectrum (BFS cycle detection)
  - ACE spectrum (approximate cycle extrinsic message degree)
  - Eigenvector IPR (inverse participation ratio)

Layer 3 — Fitness.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigs


_ROUND = 12


# -----------------------------------------------------------
# NBT Spectral Radius
# -----------------------------------------------------------


def compute_nbt_spectral_radius(H: np.ndarray) -> float:
    """Estimate non-backtracking spectral radius using power iteration.

    The non-backtracking spectral radius characterises expansion of the
    Tanner graph.  For a (d_v, d_c)-regular LDPC code, the Ramanujan
    bound gives lambda_max ~ 2*sqrt(d-1) where d is the average degree.
    Lower values indicate better expansion.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    float
        Estimated dominant eigenvalue magnitude of the NB operator.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape
    if m == 0 or n == 0:
        return 0.0

    # Build directed edge list: variable nodes 0..n-1, check nodes n..n+m-1
    directed_edges: list[tuple[int, int]] = []
    adj: dict[int, list[int]] = {}
    for ci in range(m):
        for vi in range(n):
            if H_arr[ci, vi] != 0:
                directed_edges.append((vi, n + ci))
                directed_edges.append((n + ci, vi))
                adj.setdefault(vi, []).append(n + ci)
                adj.setdefault(n + ci, []).append(vi)

    # Sort for determinism
    for node in adj:
        adj[node] = sorted(adj[node])
    directed_edges.sort()

    num_edges = len(directed_edges)
    if num_edges == 0:
        return 0.0

    edge_index = {e: i for i, e in enumerate(directed_edges)}

    # Power iteration on the NB operator
    x = np.ones(num_edges, dtype=np.float64)
    x /= np.linalg.norm(x)

    max_iters = 100
    for _ in range(max_iters):
        y = np.zeros(num_edges, dtype=np.float64)
        for i, (u, v) in enumerate(directed_edges):
            for w in adj.get(v, []):
                if w == u:
                    continue
                j = edge_index.get((v, w))
                if j is not None:
                    y[j] += x[i]
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-15:
            return 0.0
        x = y / norm_y

    # Rayleigh quotient estimate
    y = np.zeros(num_edges, dtype=np.float64)
    for i, (u, v) in enumerate(directed_edges):
        for w in adj.get(v, []):
            if w == u:
                continue
            j = edge_index.get((v, w))
            if j is not None:
                y[j] += x[i]

    rayleigh = float(np.dot(x, y))
    return round(abs(rayleigh), _ROUND)


# -----------------------------------------------------------
# Girth Spectrum
# -----------------------------------------------------------


def compute_girth_spectrum(H: np.ndarray) -> dict[str, Any]:
    """Compute the girth and short cycle counts of the Tanner graph.

    Uses BFS from each edge to detect the shortest cycle through that
    edge.  Returns the global girth and counts of cycles of length
    4, 6, and 8.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``girth`` : int — shortest cycle length (0 if acyclic)
        - ``cycle_counts`` : dict mapping {4: int, 6: int, 8: int}
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    if m == 0 or n == 0:
        return {"girth": 0, "cycle_counts": {4: 0, 6: 0, 8: 0}}

    # Build bipartite adjacency: variable nodes 0..n-1, check nodes n..n+m-1
    adj: dict[int, list[int]] = {}
    for vi in range(n):
        adj[vi] = []
    for ci in range(m):
        adj[n + ci] = []
    for ci in range(m):
        for vi in range(n):
            if H_arr[ci, vi] != 0:
                adj[vi].append(n + ci)
                adj[n + ci].append(vi)

    for node in adj:
        adj[node] = sorted(adj[node])

    total_nodes = n + m
    girth = float("inf")
    cycle_counts = {4: 0, 6: 0, 8: 0}

    # BFS from each node to find shortest cycles
    for start in range(total_nodes):
        if not adj.get(start, []):
            continue
        # BFS with parent tracking
        dist: dict[int, int] = {start: 0}
        parent: dict[int, int] = {start: -1}
        queue = [start]
        head = 0

        while head < len(queue):
            u = queue[head]
            head += 1
            d_u = dist[u]
            if d_u > 4:  # Only detect cycles up to length 8
                break
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = d_u + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v:
                    # Found a cycle
                    cycle_len = d_u + dist[v] + 1
                    if cycle_len < girth:
                        girth = cycle_len
                    if cycle_len in cycle_counts:
                        cycle_counts[cycle_len] += 1

    if girth == float("inf"):
        girth = 0

    # Each cycle is counted multiple times (from each node in the cycle)
    # Normalize: a cycle of length L is found from each of its L nodes
    for length in cycle_counts:
        if length > 0:
            cycle_counts[length] = cycle_counts[length] // length

    return {
        "girth": int(girth),
        "cycle_counts": cycle_counts,
    }


# -----------------------------------------------------------
# ACE Spectrum
# -----------------------------------------------------------


def compute_ace_spectrum(H: np.ndarray) -> np.ndarray:
    """Compute approximate cycle extrinsic message degree per variable node.

    ACE(v) measures the connectivity of the local neighbourhood of a
    variable node, excluding its own edges.  For each variable node,
    it computes the minimum extrinsic connectivity across its
    neighbouring check nodes.

    ACE(v) = min over checks c adjacent to v of:
        (degree(c) - 1) + sum over other variables w adjacent to c:
            (degree(w) - 1)
    divided by the number of neighbouring checks.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    np.ndarray
        ACE score per variable node, shape (n,).
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    if m == 0 or n == 0:
        return np.zeros(0, dtype=np.float64)

    # Precompute degrees
    var_deg = np.array([H_arr[:, vi].sum() for vi in range(n)], dtype=np.float64)
    check_deg = np.array([H_arr[ci].sum() for ci in range(m)], dtype=np.float64)

    ace = np.zeros(n, dtype=np.float64)

    for vi in range(n):
        checks = [ci for ci in range(m) if H_arr[ci, vi] != 0]
        if not checks:
            ace[vi] = 0.0
            continue

        min_ace = float("inf")
        for ci in checks:
            # Extrinsic connectivity of check ci w.r.t. variable vi
            extrinsic = check_deg[ci] - 1.0
            for vj in range(n):
                if vj != vi and H_arr[ci, vj] != 0:
                    extrinsic += var_deg[vj] - 1.0
            if extrinsic < min_ace:
                min_ace = extrinsic

        ace[vi] = round(min_ace / len(checks), _ROUND)

    return ace


# -----------------------------------------------------------
# Eigenvector IPR
# -----------------------------------------------------------


def estimate_eigenvector_ipr(H: np.ndarray) -> dict[str, float]:
    """Measure eigenvector localization via inverse participation ratio.

    Computes IPR for the top eigenvectors of H^T H.  Low IPR indicates
    good expansion (delocalized eigenvectors), while high IPR indicates
    localization (potential trapping sets).

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - ``mean_ipr`` : float — mean IPR across top eigenvectors
        - ``max_ipr`` : float — maximum IPR
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    if m == 0 or n == 0:
        return {"mean_ipr": 0.0, "max_ipr": 0.0}

    # Use sparse H^T H for eigenvector computation
    H_sparse = csr_matrix(H_arr)
    HtH = H_sparse.T @ H_sparse

    k = min(3, min(m, n) - 1)
    if k < 1:
        return {"mean_ipr": 0.0, "max_ipr": 0.0}

    try:
        vals, vecs = eigs(HtH.astype(np.float64), k=k, which="LM", tol=1e-6)
    except Exception:
        return {"mean_ipr": 0.0, "max_ipr": 0.0}

    iprs = []
    for i in range(vecs.shape[1]):
        v = np.abs(np.real(vecs[:, i]))
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        ipr = float(np.sum(v ** 4) / (np.sum(v ** 2) ** 2)) if np.sum(v ** 2) > 0 else 0.0
        iprs.append(ipr)

    if not iprs:
        return {"mean_ipr": 0.0, "max_ipr": 0.0}

    return {
        "mean_ipr": round(float(np.mean(iprs)), _ROUND),
        "max_ipr": round(float(np.max(iprs)), _ROUND),
    }
