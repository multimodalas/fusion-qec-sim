"""
v8.1.0 — Cycle Space Density Diagnostic.

Computes the cycle space density of the Tanner graph, defined as the
ratio of the cycle space dimension to the number of edges:

    density = (|E| - |V| + components) / |E|

where |E| is the number of undirected edges, |V| is the number of
nodes in the Tanner graph, and components is the number of connected
components.

High cycle density indicates many short cycles in the graph, which
contributes to BP instability through loopy message passing.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import numpy as np

_ROUND = 12


def compute_cycle_space_density(H: np.ndarray) -> float:
    """Compute cycle space density of the Tanner graph.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    float
        Cycle space density in [0, 1].
        Returns 0.0 for graphs with no edges.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    # Count undirected edges (nonzeros in H)
    num_edges = int(np.count_nonzero(H_arr))

    if num_edges == 0:
        return 0.0

    # Total nodes in bipartite Tanner graph
    num_nodes = n + m

    # Count connected components via union-find
    parent = list(range(num_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for ci in range(m):
        for vi in range(n):
            if H_arr[ci, vi] != 0:
                union(vi, n + ci)

    # Count components among nodes that participate in edges
    active_nodes = set()
    for ci in range(m):
        for vi in range(n):
            if H_arr[ci, vi] != 0:
                active_nodes.add(vi)
                active_nodes.add(n + ci)

    components = len({find(node) for node in active_nodes})

    # Cycle space dimension = |E| - |V_active| + components
    cycle_dim = num_edges - len(active_nodes) + components

    density = cycle_dim / num_edges

    return round(float(density), _ROUND)
