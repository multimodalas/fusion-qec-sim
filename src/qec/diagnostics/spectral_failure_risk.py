"""
v6.4.0 — Spectral Failure Risk Scoring.

Computes a deterministic structural risk heuristic for candidate
clusters by combining spectral localization strength, repeated
participation in localized modes, and alignment with BP dynamical
activity.

This is a structural risk score — not a failure predictor.  It
identifies clusters most likely to influence decoder behavior based
on spectral and dynamical signals accumulated in v6.1–v6.3.

Consumes:
  - v6.1 localization metrics (IPR scores)
  - v6.2 trapping-candidate outputs (candidate_clusters,
    node_participation_counts)
  - v6.3 alignment outputs (per_cluster_alignment_scores)

Does not recompute spectra or localization.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any


def compute_spectral_failure_risk(
    localization_result: dict[str, Any],
    trapping_candidate_result: dict[str, Any],
    alignment_result: dict[str, Any],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute spectral failure risk scores for candidate clusters.

    Combines spectral localization (IPR), node participation counts,
    and BP alignment scores into a per-cluster risk score.

    Parameters
    ----------
    localization_result : dict[str, Any]
        Output of ``compute_nb_localization_metrics()`` (v6.1).
        Must contain ``ipr_scores`` and ``localized_modes``.
    trapping_candidate_result : dict[str, Any]
        Output of ``compute_nb_trapping_candidates()`` (v6.2).
        Must contain ``candidate_clusters`` and
        ``node_participation_counts``.
    alignment_result : dict[str, Any]
        Output of ``compute_spectral_bp_alignment()`` (v6.3).
        Must contain ``per_cluster_alignment_scores``.
    top_k : int
        Maximum number of top-risk clusters to return.  Default 5.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``node_risk_scores``: list of ``[node_index, score]`` pairs
          for all variable nodes with non-zero risk, sorted by node
          index.
        - ``cluster_risk_scores``: list of per-cluster risk scores,
          one per candidate cluster (same order as
          ``candidate_clusters``).
        - ``cluster_risk_ranking``: list of cluster indices sorted
          by risk descending.
        - ``max_cluster_risk``: float, maximum cluster risk score.
        - ``mean_cluster_risk``: float, mean cluster risk score.
        - ``top_risk_clusters``: list of up to *top_k* cluster
          indices with the highest risk scores (risk > 0).
        - ``num_high_risk_clusters``: int, count of clusters with
          risk > 0.
    """
    clusters = trapping_candidate_result.get("candidate_clusters", [])
    node_participation = trapping_candidate_result.get(
        "node_participation_counts", {},
    )
    per_cluster_alignment = alignment_result.get(
        "per_cluster_alignment_scores", [],
    )
    ipr_scores = localization_result.get("ipr_scores", [])
    localized_modes = localization_result.get("localized_modes", [])

    # ── Localization weight ──────────────────────────────────────
    # Normalized mean IPR of localized modes.
    if localized_modes and ipr_scores:
        localized_iprs = [ipr_scores[i] for i in localized_modes
                          if i < len(ipr_scores)]
        mean_localized_ipr = (
            sum(localized_iprs) / len(localized_iprs)
            if localized_iprs else 0.0
        )
    else:
        mean_localized_ipr = 0.0

    # Normalize: IPR values are already in [0, 1], so mean IPR
    # serves directly as the localization weight.
    localization_weight = mean_localized_ipr

    # ── Per-cluster risk scores ──────────────────────────────────
    cluster_risk_scores: list[float] = []

    for idx, cluster in enumerate(clusters):
        var_nodes = cluster.get("variable_nodes", [])

        # Participation weight: mean participation count for
        # variable nodes in this cluster.
        if var_nodes:
            participation_sum = 0.0
            for vi in var_nodes:
                key = f"v{vi}"
                participation_sum += float(
                    node_participation.get(key, 0),
                )
            participation_weight = participation_sum / len(var_nodes)
        else:
            participation_weight = 0.0

        # Alignment score for this cluster (from v6.3).
        if idx < len(per_cluster_alignment):
            alignment_score = per_cluster_alignment[idx]
        else:
            alignment_score = 0.0

        # Risk = participation_weight * alignment_score
        #       * localization_weight
        risk = participation_weight * alignment_score * localization_weight
        cluster_risk_scores.append(round(risk, 12))

    # ── Cluster ranking ──────────────────────────────────────────
    # Sort by risk descending, then by cluster index ascending
    # for determinism.
    cluster_risk_ranking = sorted(
        range(len(cluster_risk_scores)),
        key=lambda i: (-cluster_risk_scores[i], i),
    )

    max_cluster_risk = (
        max(cluster_risk_scores) if cluster_risk_scores else 0.0
    )
    mean_cluster_risk = (
        sum(cluster_risk_scores) / len(cluster_risk_scores)
        if cluster_risk_scores else 0.0
    )

    # Top risk clusters: up to top_k with risk > 0.
    top_risk = [
        i for i in cluster_risk_ranking
        if cluster_risk_scores[i] > 0.0
    ][:top_k]

    num_high_risk = sum(
        1 for s in cluster_risk_scores if s > 0.0
    )

    # ── Node-level risk scores ───────────────────────────────────
    # node_risk(v) = sum of cluster_risk for all clusters
    # containing v.
    node_risk_map: dict[int, float] = {}
    for idx, cluster in enumerate(clusters):
        risk = cluster_risk_scores[idx]
        if risk == 0.0:
            continue
        for vi in cluster.get("variable_nodes", []):
            node_risk_map[vi] = node_risk_map.get(vi, 0.0) + risk

    # Round and sort by node index.
    node_risk_pairs = [
        [vi, round(score, 12)]
        for vi, score in sorted(node_risk_map.items())
    ]

    return {
        "node_risk_scores": node_risk_pairs,
        "cluster_risk_scores": cluster_risk_scores,
        "cluster_risk_ranking": cluster_risk_ranking,
        "max_cluster_risk": round(max_cluster_risk, 12),
        "mean_cluster_risk": round(mean_cluster_risk, 12),
        "top_risk_clusters": top_risk,
        "num_high_risk_clusters": num_high_risk,
    }
