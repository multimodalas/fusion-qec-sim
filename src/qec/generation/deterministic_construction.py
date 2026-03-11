"""
v8.5.0 — Deterministic Cycle-Avoidant Tanner Graph Construction.

Builds Tanner graphs incrementally using a greedy heuristic that
penalizes short-cycle creation, neighborhood overlap, load imbalance,
and local curvature.

Layer 3 — Generation.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no hidden randomness, no global state, no input mutation.
"""

from __future__ import annotations

import numpy as np


def construct_deterministic_tanner_graph(
    spec: dict,
) -> np.ndarray:
    """Build a parity-check matrix using cycle-avoidant incremental placement.

    For each variable node, edges are placed to check nodes that minimize
    a composite score penalizing short cycles, neighborhood overlap,
    load imbalance, and local curvature.

    Parameters
    ----------
    spec : dict
        Construction specification with keys:

        - ``num_variables`` : int
        - ``num_checks`` : int
        - ``variable_degree`` : int
        - ``check_degree`` : int

    Returns
    -------
    np.ndarray
        Binary parity-check matrix, shape (num_checks, num_variables),
        dtype float64, values in {0, 1}.

    Raises
    ------
    ValueError
        If ``num_variables * variable_degree != num_checks * check_degree``.
    RuntimeError
        If the resulting matrix violates degree constraints.
    """
    n = spec["num_variables"]
    m = spec["num_checks"]
    dv = spec["variable_degree"]
    dc = spec["check_degree"]

    if n * dv != m * dc:
        raise ValueError(
            f"Degree constraint violated: {n} * {dv} != {m} * {dc}"
        )

    H = np.zeros((m, n), dtype=np.float64)
    chk_deg = np.zeros(m, dtype=int)
    # For each check, track which variables are connected
    chk_neighbors: list[list[int]] = [[] for _ in range(m)]

    # Optional construction-order variants for candidate diversity.
    # When absent, uses natural ordering (0, 1, ..., n-1).
    var_order = spec.get("_variable_order", list(range(n)))
    candidate_chk_order = spec.get("_check_order", list(range(m)))

    for v in var_order:
        edges_placed = 0
        while edges_placed < dv:
            best_score = None
            best_c = -1
            for c in candidate_chk_order:
                if H[c, v] != 0 or chk_deg[c] >= dc:
                    continue
                # -- short_cycle_penalty --
                # Checks already touching v (via other edges placed this round
                # or previously) that also share a variable with candidate c.
                cycle_penalty = 0
                for v2 in chk_neighbors[c]:
                    if v2 == v:
                        continue
                    # v2 is connected to c; count checks that connect
                    # both v and v2 (would create a 4-cycle).
                    for c2 in range(m):
                        if c2 != c and H[c2, v] != 0 and H[c2, v2] != 0:
                            cycle_penalty += 1

                # -- two_hop_overlap --
                # Number of variables reachable from v (through its current
                # checks) that are also neighbors of c.
                v_neighbors_set = set()
                for c2 in range(m):
                    if H[c2, v] != 0:
                        for v2 in chk_neighbors[c2]:
                            v_neighbors_set.add(v2)
                overlap = 0
                for v2 in chk_neighbors[c]:
                    if v2 in v_neighbors_set:
                        overlap += 1

                # -- check_load --
                load = chk_deg[c]

                # -- curvature_estimate --
                curvature = 0
                for v2 in chk_neighbors[c]:
                    curvature += int(H[:, v2].sum())

                score = (
                    12 * cycle_penalty
                    + 4 * overlap
                    + 2 * load
                    + 1 * curvature
                )

                if best_score is None or score < best_score or (
                    score == best_score and c < best_c
                ):
                    best_score = score
                    best_c = c

            if best_c < 0:
                break

            H[best_c, v] = 1.0
            chk_deg[best_c] += 1
            chk_neighbors[best_c].append(v)
            edges_placed += 1

    # -- Verify degree constraints --
    col_sums = H.sum(axis=0)
    row_sums = H.sum(axis=1)
    if not np.all(col_sums == dv):
        raise RuntimeError(
            f"Column degree violation: expected {dv}, got {col_sums.tolist()}"
        )
    if not np.all(row_sums == dc):
        raise RuntimeError(
            f"Row degree violation: expected {dc}, got {row_sums.tolist()}"
        )

    return H
