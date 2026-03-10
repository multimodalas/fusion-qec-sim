"""
v6.2.0 — Spectral Trapping-Set Candidate Detection.

Identifies structural trapping-set candidates by counting node
participation across localized non-backtracking eigenmodes.

Nodes that appear in multiple localized modes are likely to belong
to fragile Tanner substructures such as trapping sets or absorbing
sets.  This is a structural probe for fragile subgraphs — it does
not claim perfect prediction.

Consumes localization results from compute_nb_localization_metrics()
(v6.1).  Does not recompute spectra.

Operates purely on localization outputs and the parity-check matrix.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_nb_trapping_candidates(
    parity_check_matrix: np.ndarray,
    localization_result: dict[str, Any],
    *,
    participation_threshold: int = 2,
) -> dict[str, Any]:
    """Identify spectral trapping-set candidates from localized NB modes.

    Counts how many localized non-backtracking eigenmodes each Tanner
    graph node participates in.  Nodes with participation count at or
    above *participation_threshold* are flagged as trapping-set
    candidates.  Connected components among candidate nodes in the
    Tanner graph form candidate clusters.

    Parameters
    ----------
    parity_check_matrix : np.ndarray
        Binary parity-check matrix H with shape (m, n).
        Used only for Tanner graph adjacency (cluster detection).
    localization_result : dict[str, Any]
        Output of ``compute_nb_localization_metrics()``.  Must contain
        ``localized_variable_nodes`` and ``localized_check_nodes``.
    participation_threshold : int
        Minimum number of localized modes a node must appear in to be
        flagged as a candidate.  Default 2.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``node_participation_counts``: dict mapping node label to
          participation count.  Variable nodes keyed as ``"v<i>"``,
          check nodes as ``"c<j>"`` (0-indexed).
        - ``candidate_variable_nodes``: sorted list of variable node
          indices that meet the participation threshold.
        - ``candidate_check_nodes``: sorted list of check node indices
          that meet the participation threshold.
        - ``candidate_clusters``: list of clusters, each a dict with
          ``"variable_nodes"`` and ``"check_nodes"`` (sorted lists).
          Clusters are sorted by size descending, then by smallest
          variable node index ascending for determinism.
        - ``max_node_participation``: int, maximum participation count
          across all nodes (0 if no localized modes).
        - ``num_candidate_nodes``: int, total number of candidate nodes
          (variable + check).
        - ``num_candidate_clusters``: int, number of connected
          components among candidate nodes.
        - ``participation_threshold``: int, threshold used.
    """
    H = np.asarray(parity_check_matrix, dtype=np.float64)
    m, n = H.shape

    loc_var_nodes = localization_result.get("localized_variable_nodes", [])
    loc_chk_nodes = localization_result.get("localized_check_nodes", [])

    # ── Count node participation across localized modes ──────────
    var_counts: dict[int, int] = {}
    chk_counts: dict[int, int] = {}

    for var_list in loc_var_nodes:
        for vi in var_list:
            var_counts[vi] = var_counts.get(vi, 0) + 1

    for chk_list in loc_chk_nodes:
        for ci in chk_list:
            chk_counts[ci] = chk_counts.get(ci, 0) + 1

    # ── Build serializable participation dict ────────────────────
    node_participation: dict[str, int] = {}
    for vi, cnt in sorted(var_counts.items()):
        node_participation[f"v{vi}"] = cnt
    for ci, cnt in sorted(chk_counts.items()):
        node_participation[f"c{ci}"] = cnt

    # ── Threshold candidates ─────────────────────────────────────
    cand_vars = sorted(vi for vi, c in var_counts.items()
                       if c >= participation_threshold)
    cand_chks = sorted(ci for ci, c in chk_counts.items()
                       if c >= participation_threshold)

    # ── Max participation ────────────────────────────────────────
    all_counts = list(var_counts.values()) + list(chk_counts.values())
    max_participation = max(all_counts) if all_counts else 0

    # ── Connected components among candidate nodes ───────────────
    # Build adjacency from H restricted to candidate nodes.
    cand_var_set = set(cand_vars)
    cand_chk_set = set(cand_chks)

    # Union-Find for connected components.
    # Nodes: variable nodes as ("v", i), check nodes as ("c", j).
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(x: tuple[str, int]) -> tuple[str, int]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[str, int], b: tuple[str, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Deterministic: smaller tuple becomes parent.
            if ra > rb:
                ra, rb = rb, ra
            parent[rb] = ra

    for vi in cand_vars:
        parent[("v", vi)] = ("v", vi)
    for ci in cand_chks:
        parent[("c", ci)] = ("c", ci)

    # Connect candidate variable and check nodes that share an edge.
    for ci in cand_chks:
        for vi in cand_vars:
            if H[ci, vi] != 0:
                union(("v", vi), ("c", ci))

    # Also connect candidate variable nodes that share a check node
    # (even if that check is not a candidate) and vice versa.
    # This captures the Tanner graph distance-2 connectivity.
    for ci in range(m):
        connected_cand_vars = [vi for vi in cand_vars if H[ci, vi] != 0]
        for k in range(1, len(connected_cand_vars)):
            union(("v", connected_cand_vars[0]),
                  ("v", connected_cand_vars[k]))

    for vi in range(n):
        connected_cand_chks = [ci for ci in cand_chks if H[ci, vi] != 0]
        for k in range(1, len(connected_cand_chks)):
            union(("c", connected_cand_chks[0]),
                  ("c", connected_cand_chks[k]))

    # Gather clusters.
    clusters_map: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for node in list(parent.keys()):
        root = find(node)
        if root not in clusters_map:
            clusters_map[root] = []
        clusters_map[root].append(node)

    clusters: list[dict[str, list[int]]] = []
    for members in clusters_map.values():
        cl_vars = sorted(i for t, i in members if t == "v")
        cl_chks = sorted(i for t, i in members if t == "c")
        clusters.append({
            "variable_nodes": cl_vars,
            "check_nodes": cl_chks,
        })

    # Deterministic sort: by total size descending, then by smallest
    # variable node (or check node if no variables) ascending.
    def _cluster_sort_key(cl: dict[str, list[int]]) -> tuple[int, int]:
        size = len(cl["variable_nodes"]) + len(cl["check_nodes"])
        first = (cl["variable_nodes"][0] if cl["variable_nodes"]
                 else cl["check_nodes"][0] if cl["check_nodes"]
                 else 0)
        return (-size, first)

    clusters.sort(key=_cluster_sort_key)

    return {
        "node_participation_counts": node_participation,
        "candidate_variable_nodes": cand_vars,
        "candidate_check_nodes": cand_chks,
        "candidate_clusters": clusters,
        "max_node_participation": max_participation,
        "num_candidate_nodes": len(cand_vars) + len(cand_chks),
        "num_candidate_clusters": len(clusters),
        "participation_threshold": participation_threshold,
    }
