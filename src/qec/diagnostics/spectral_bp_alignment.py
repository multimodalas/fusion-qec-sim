"""
v6.3.0 — Spectral–BP Attractor Alignment.

Measures whether spectral trapping-set candidates (from v6.2) align
with actual BP dynamical activity during decoding.  This is the first
explicit bridge between structural spectral diagnostics and observed
BP decoding trajectories.

Consumes:
  - v6.2 trapping-candidate outputs (candidate_variable_nodes,
    candidate_clusters)
  - Per-node BP activity scores derived from iteration-trace data
    (belief oscillation index)

Computes overlap-based alignment metrics between spectrally predicted
fragile nodes and dynamically active BP nodes.

This is an alignment diagnostic — not a proof of causality.

Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any


def compute_spectral_bp_alignment(
    trapping_candidate_result: dict[str, Any],
    bp_node_activity_scores: dict[int, float],
    *,
    activity_threshold_fraction: float = 0.1,
) -> dict[str, Any]:
    """Compute alignment between spectral candidates and BP-active nodes.

    Parameters
    ----------
    trapping_candidate_result : dict[str, Any]
        Output of ``compute_nb_trapping_candidates()`` (v6.2).
        Must contain ``candidate_variable_nodes`` and
        ``candidate_clusters``.
    bp_node_activity_scores : dict[int, float]
        Mapping from variable-node index to a non-negative BP activity
        score.  Higher values indicate greater dynamical activity.
        Typically derived from belief oscillation index (BOI).
    activity_threshold_fraction : float
        Fraction of max activity score used to threshold BP-active
        nodes.  A node is BP-active if its score is at least
        ``activity_threshold_fraction * max_score``.  Default 0.1.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``spectral_bp_alignment_score``: Jaccard index between
          candidate nodes and BP-active nodes.
        - ``candidate_node_overlap_fraction``: fraction of candidate
          nodes that are BP-active.
        - ``bp_node_overlap_fraction``: fraction of BP-active nodes
          that are candidates.
        - ``active_bp_nodes``: sorted list of BP-active node indices.
        - ``aligned_candidate_nodes``: sorted list of candidate nodes
          that are also BP-active.
        - ``num_aligned_candidate_nodes``: count of aligned nodes.
        - ``bp_node_activity_scores``: sorted list of
          ``[node_index, score]`` pairs for all scored nodes.
        - ``per_cluster_alignment_scores``: list of per-cluster
          alignment fractions (fraction of cluster variable nodes
          that are BP-active), one per candidate cluster.
        - ``top_aligned_clusters``: indices of clusters with
          alignment > 0, sorted by alignment descending.
        - ``max_cluster_alignment``: maximum per-cluster alignment.
        - ``activity_threshold_fraction``: threshold used.
    """
    cand_vars = trapping_candidate_result.get("candidate_variable_nodes", [])
    clusters = trapping_candidate_result.get("candidate_clusters", [])

    candidate_nodes = set(cand_vars)

    # ── Threshold BP-active nodes ──────────────────────────────────
    if bp_node_activity_scores:
        max_score = max(bp_node_activity_scores.values())
    else:
        max_score = 0.0

    threshold = activity_threshold_fraction * max_score
    bp_active_nodes = set()
    for node, score in bp_node_activity_scores.items():
        if max_score > 0.0 and score >= threshold:
            bp_active_nodes.add(node)

    # ── Global alignment metrics ───────────────────────────────────
    intersection = candidate_nodes & bp_active_nodes
    union = candidate_nodes | bp_active_nodes

    alignment_score = (
        len(intersection) / len(union) if union else 0.0
    )
    candidate_overlap = (
        len(intersection) / len(candidate_nodes) if candidate_nodes else 0.0
    )
    bp_overlap = (
        len(intersection) / len(bp_active_nodes) if bp_active_nodes else 0.0
    )

    # ── Per-cluster alignment ──────────────────────────────────────
    per_cluster_scores: list[float] = []
    for cluster in clusters:
        cluster_vars = set(cluster.get("variable_nodes", []))
        if cluster_vars:
            cl_alignment = len(cluster_vars & bp_active_nodes) / len(cluster_vars)
        else:
            cl_alignment = 0.0
        per_cluster_scores.append(round(cl_alignment, 12))

    # Top aligned clusters: indices with alignment > 0, sorted by
    # alignment descending then index ascending for determinism.
    indexed_scores = list(enumerate(per_cluster_scores))
    top_clusters = sorted(
        [i for i, s in indexed_scores if s > 0.0],
        key=lambda i: (-per_cluster_scores[i], i),
    )

    max_cluster_alignment = (
        max(per_cluster_scores) if per_cluster_scores else 0.0
    )

    # ── Serializable activity scores ───────────────────────────────
    sorted_activity = sorted(bp_node_activity_scores.items())
    activity_pairs = [[int(n), round(float(s), 12)] for n, s in sorted_activity]

    return {
        "spectral_bp_alignment_score": round(alignment_score, 12),
        "candidate_node_overlap_fraction": round(candidate_overlap, 12),
        "bp_node_overlap_fraction": round(bp_overlap, 12),
        "active_bp_nodes": sorted(bp_active_nodes),
        "aligned_candidate_nodes": sorted(intersection),
        "num_aligned_candidate_nodes": len(intersection),
        "bp_node_activity_scores": activity_pairs,
        "per_cluster_alignment_scores": per_cluster_scores,
        "top_aligned_clusters": top_clusters,
        "max_cluster_alignment": round(max_cluster_alignment, 12),
        "activity_threshold_fraction": activity_threshold_fraction,
    }
