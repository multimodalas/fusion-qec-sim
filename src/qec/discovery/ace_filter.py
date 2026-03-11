"""
v9.0.0 — ACE-Gated Mutation Filter.

Prevents structural degradation during mutation using an Approximate
Cycle Extrinsic (ACE) metric.

ACE(cycle) ~ sum(deg(v) - 2) for nodes in cycle neighborhood.
Higher ACE is better.

Acceptance rule:
  if ace_delta < 0 and spectral score does not improve: reject
  else: accept

Tie-breaking:
  1. lower composite score
  2. higher ACE
  3. lower cycle pressure
  4. lower bad-edge score
  5. lexicographically smaller edge tuple

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_ROUND = 12


def compute_local_ace_score(
    H: np.ndarray,
    edges_subset: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Compute local ACE proxy near specified edges.

    ACE for an edge (ci, vi) is computed as the sum of (degree - 2)
    for all variable nodes in the 1-hop neighbourhood of ci and vi.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    edges_subset : list of (ci, vi) or None
        Edges to evaluate.  If None, evaluates all edges.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``ace_scores`` : dict mapping (ci, vi) -> ace_score
        - ``total_ace`` : float
        - ``mean_ace`` : float
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    # Precompute degrees
    col_degrees = H_arr.sum(axis=0)  # variable degrees
    row_degrees = H_arr.sum(axis=1)  # check degrees

    if edges_subset is None:
        edges_subset = [
            (ci, vi)
            for ci in range(m)
            for vi in range(n)
            if H_arr[ci, vi] != 0
        ]

    ace_scores: dict[tuple[int, int], float] = {}

    for ci, vi in edges_subset:
        if ci >= m or vi >= n:
            continue

        # Collect 1-hop variable neighbours of check ci
        var_neighbors = sorted(vj for vj in range(n) if H_arr[ci, vj] != 0)

        # Also include variables connected to checks that connect to vi
        check_neighbors = sorted(cj for cj in range(m) if H_arr[cj, vi] != 0)
        extended_vars = set(var_neighbors)
        for cj in check_neighbors:
            for vj in range(n):
                if H_arr[cj, vj] != 0:
                    extended_vars.add(vj)

        # ACE = sum(deg(v) - 2) for neighbourhood variables
        ace = sum(max(0.0, float(col_degrees[vj]) - 2.0) for vj in sorted(extended_vars))
        ace_scores[(ci, vi)] = round(ace, _ROUND)

    total = sum(ace_scores.values())
    mean = total / len(ace_scores) if ace_scores else 0.0

    return {
        "ace_scores": ace_scores,
        "total_ace": round(total, _ROUND),
        "mean_ace": round(mean, _ROUND),
    }


def ace_gate_mutation(
    H_before: np.ndarray,
    H_after: np.ndarray,
    *,
    composite_before: float,
    composite_after: float,
    cycle_pressure_after: float = 0.0,
    bad_edge_score_after: float = 0.0,
    mutated_edges: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Decide whether to accept or reject a mutation using ACE gating.

    Parameters
    ----------
    H_before : np.ndarray
        Parity-check matrix before mutation.
    H_after : np.ndarray
        Parity-check matrix after mutation.
    composite_before : float
        Composite score before mutation.
    composite_after : float
        Composite score after mutation.
    cycle_pressure_after : float
        Cycle pressure of H_after (for tie-breaking).
    bad_edge_score_after : float
        Bad-edge score of H_after (for tie-breaking).
    mutated_edges : list of (ci, vi) or None
        Edges that were modified.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``accept`` : bool
        - ``ace_before`` : float
        - ``ace_after`` : float
        - ``ace_delta`` : float
        - ``reason`` : str
    """
    ace_result_before = compute_local_ace_score(H_before, edges_subset=mutated_edges)
    ace_result_after = compute_local_ace_score(H_after, edges_subset=mutated_edges)

    ace_before = ace_result_before["total_ace"]
    ace_after = ace_result_after["total_ace"]
    ace_delta = round(ace_after - ace_before, _ROUND)

    # Acceptance logic
    if ace_delta < 0 and composite_after >= composite_before:
        # ACE degraded and spectral score did not improve
        accept = False
        reason = "ace_degraded_no_spectral_improvement"
    else:
        accept = True
        if ace_delta >= 0:
            reason = "ace_preserved_or_improved"
        else:
            reason = "ace_degraded_but_spectral_improved"

    return {
        "accept": accept,
        "ace_before": ace_before,
        "ace_after": ace_after,
        "ace_delta": ace_delta,
        "reason": reason,
    }


def ace_tiebreak_key(
    candidate: dict[str, Any],
) -> tuple:
    """Produce a deterministic tie-breaking sort key for ACE filtering.

    Order:
    1. lower composite score
    2. higher ACE
    3. lower cycle pressure
    4. lower bad-edge score
    5. lexicographically smaller candidate_id

    Parameters
    ----------
    candidate : dict[str, Any]
        Candidate with objectives and ace metadata.

    Returns
    -------
    tuple
        Sort key (lower is better).
    """
    obj = candidate.get("objectives", {})
    return (
        obj.get("composite_score", float("inf")),
        -candidate.get("ace_total", 0.0),
        candidate.get("cycle_pressure", float("inf")),
        candidate.get("bad_edge_score", float("inf")),
        candidate.get("candidate_id", ""),
    )
