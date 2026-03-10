"""
v7.3.0 — Deterministic Spectral Graph Repair Loop.

Attempts to reduce predicted BP instability before decoding by testing
small deterministic graph rewrites on the Tanner graph.

Pipeline:
  baseline graph
  → spectral diagnostics
  → instability prediction
  → generate repair candidates
  → evaluate repaired candidates
  → choose best repair
  → optional baseline decode
  → optional repaired decode
  → compare outcomes

Consumes:
  - v6.0 NB spectral radius
  - v6.1 IPR localization
  - v6.2 trapping-set candidates
  - v6.4 spectral failure risk
  - v7.2 spectral instability score

Does not modify decoder internals.  Fully deterministic: no randomness,
no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.experiments.tanner_graph_repair import (
    _extract_edges,
    _build_edge_set,
    _build_adjacency,
    _get_cluster_edges,
    _get_boundary_edges,
    _generate_candidate_swaps,
    _apply_swap,
    _edges_to_H,
    _experimental_bp_flooding,
    _compute_syndrome,
)
from src.qec.experiments.spectral_instability_phase_map import (
    compute_spectral_instability_score,
)


# ── Core Feature 1 — Deterministic Repair Candidate Generation ───────


def generate_repair_candidates(
    H: np.ndarray,
    top_risk_clusters: list[int],
    candidate_clusters: list[dict[str, Any]],
    node_risk_scores: list[list],
    cluster_risk_scores: list[float],
    max_candidates: int = 10,
) -> dict[str, Any]:
    """Generate deterministic degree-preserving edge-swap repair candidates.

    Identifies the highest-risk cluster and generates edge swaps between
    cluster edges and boundary edges.  Candidates are ordered
    deterministically: cluster edges sorted, boundary edges sorted,
    outer product in that order.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    top_risk_clusters : list[int]
        Cluster indices sorted by risk descending (from v6.4).
    candidate_clusters : list[dict[str, Any]]
        Cluster definitions from v6.2, each with ``variable_nodes``
        and ``check_nodes``.
    node_risk_scores : list[list]
        Per-node risk scores ``[[node_idx, score], ...]`` from v6.4.
    cluster_risk_scores : list[float]
        Per-cluster risk scores from v6.4.
    max_candidates : int
        Maximum number of candidate swaps to generate.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``candidates``: list of swap descriptors
        - ``cluster_nodes``: sorted list of cluster node indices
        - ``cluster_risk_score``: float risk score for selected cluster
        - ``selected_cluster_index``: int index of selected cluster
    """
    m, n = H.shape

    if not top_risk_clusters or not candidate_clusters:
        return {
            "candidates": [],
            "cluster_nodes": [],
            "cluster_risk_score": 0.0,
            "selected_cluster_index": -1,
        }

    # Select highest-risk cluster.
    cluster_idx = top_risk_clusters[0]
    if cluster_idx >= len(candidate_clusters):
        return {
            "candidates": [],
            "cluster_nodes": [],
            "cluster_risk_score": 0.0,
            "selected_cluster_index": cluster_idx,
        }

    cluster = candidate_clusters[cluster_idx]
    cluster_var_nodes = sorted(cluster.get("variable_nodes", []))
    cluster_chk_nodes = sorted(cluster.get("check_nodes", []))

    if not cluster_var_nodes:
        return {
            "candidates": [],
            "cluster_nodes": [],
            "cluster_risk_score": (
                cluster_risk_scores[cluster_idx]
                if cluster_idx < len(cluster_risk_scores) else 0.0
            ),
            "selected_cluster_index": cluster_idx,
        }

    # Build cluster node set: variable nodes as-is, check nodes offset.
    cluster_nodes: set[int] = set(cluster_var_nodes)
    for ci in cluster_chk_nodes:
        cluster_nodes.add(n + ci)

    # Also include adjacent check nodes from H for variable-only clusters.
    for vi in cluster_var_nodes:
        for ci in range(m):
            if H[ci, vi] != 0:
                cluster_nodes.add(n + ci)

    edges = _extract_edges(H)
    edge_set = _build_edge_set(edges)
    adjacency = _build_adjacency(edges)

    cluster_edges = _get_cluster_edges(edges, cluster_nodes)
    boundary_edges = _get_boundary_edges(edges, cluster_nodes, adjacency)

    candidates = _generate_candidate_swaps(
        cluster_edges, boundary_edges, edge_set, n,
        max_candidates=max_candidates,
    )

    cluster_risk = (
        cluster_risk_scores[cluster_idx]
        if cluster_idx < len(cluster_risk_scores) else 0.0
    )

    return {
        "candidates": candidates,
        "cluster_nodes": sorted(cluster_nodes),
        "cluster_risk_score": round(float(cluster_risk), 12),
        "selected_cluster_index": cluster_idx,
    }


# ── Core Feature 2 — Predicted Instability Scoring ──────────────────


def score_repair_candidate(
    H: np.ndarray,
    candidate: dict[str, Any],
    nb_spectral_radius: float,
    spectral_instability_ratio: float,
    ipr_localization_score: float,
    cluster_risk_scores: list[float],
    avg_variable_degree: float,
    avg_check_degree: float,
    instability_score_before: float,
    predicted_instability_before: bool,
    *,
    instability_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score a repair candidate by recomputing instability prediction.

    For the candidate repaired graph, recomputes the spectral
    instability score using the same v7.2 scoring machinery, then
    compares with the baseline score.

    Parameters
    ----------
    H : np.ndarray
        Original parity-check matrix, shape (m, n).
    candidate : dict[str, Any]
        Swap descriptor with ``remove`` and ``add`` keys.
    nb_spectral_radius : float
        Baseline NB spectral radius.
    spectral_instability_ratio : float
        Baseline spectral instability ratio.
    ipr_localization_score : float
        Baseline IPR localization score.
    cluster_risk_scores : list[float]
        Baseline per-cluster risk scores.
    avg_variable_degree : float
        Average variable-node degree.
    avg_check_degree : float
        Average check-node degree.
    instability_score_before : float
        Baseline spectral instability score.
    predicted_instability_before : bool
        Baseline instability prediction.
    instability_threshold : float
        Threshold for instability prediction.

    Returns
    -------
    dict[str, Any]
        Per-candidate metrics dictionary.
    """
    m, n = H.shape

    # Apply swap to get repaired edge list and H.
    edges = _extract_edges(H)
    repaired_edges = _apply_swap(edges, candidate)
    H_repaired = _edges_to_H(repaired_edges, m, n)

    # Recompute NB spectral radius for repaired graph.
    from src.qec.diagnostics.non_backtracking_spectrum import (
        compute_non_backtracking_spectrum,
    )
    nb_result = compute_non_backtracking_spectrum(H_repaired)
    repaired_spectral_radius = nb_result.get("spectral_radius", 0.0)

    # Recompute instability ratio.
    repaired_avg_degree = (avg_variable_degree + avg_check_degree) / 2.0
    import math
    if repaired_avg_degree > 0.0:
        repaired_threshold = math.sqrt(repaired_avg_degree)
        repaired_instability_ratio = round(
            repaired_spectral_radius / repaired_threshold, 12,
        )
    else:
        repaired_instability_ratio = 0.0

    # Recompute spectral instability score using v7.2 machinery.
    repaired_score = compute_spectral_instability_score(
        nb_spectral_radius=repaired_spectral_radius,
        spectral_instability_ratio=repaired_instability_ratio,
        ipr_localization_score=ipr_localization_score,
        cluster_risk_scores=cluster_risk_scores,
        avg_variable_degree=avg_variable_degree,
        avg_check_degree=avg_check_degree,
    )

    predicted_instability_after = repaired_score > instability_threshold
    score_improvement = round(instability_score_before - repaired_score, 12)

    return {
        "swap": {
            "remove": [list(e) for e in candidate["remove"]],
            "add": candidate["add"],
            "description": candidate["description"],
        },
        "spectral_instability_score_before": round(instability_score_before, 12),
        "spectral_instability_score_after": round(repaired_score, 12),
        "predicted_instability_before": predicted_instability_before,
        "predicted_instability_after": predicted_instability_after,
        "score_improvement": score_improvement,
        "repaired_spectral_radius": round(repaired_spectral_radius, 12),
    }


# ── Core Feature 3 — Best Repair Selection ──────────────────────────


def select_best_repair(
    scored_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the best repair candidate deterministically.

    Primary objective: maximize score_improvement.

    Tie-breaking order:
      1. Largest instability score reduction (score_improvement desc)
      2. Lowest repaired instability score (score_after asc)
      3. Lexicographically smallest swap description (asc)

    Parameters
    ----------
    scored_candidates : list[dict[str, Any]]
        List of scored candidate metrics from
        :func:`score_repair_candidate`.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``best_swap``: swap descriptor or None
        - ``best_candidate_metrics``: full metrics for best or None
        - ``num_candidates_evaluated``: int
    """
    num_evaluated = len(scored_candidates)

    if not scored_candidates:
        return {
            "best_swap": None,
            "best_candidate_metrics": None,
            "num_candidates_evaluated": 0,
        }

    # Filter to candidates with positive improvement.
    improving = [c for c in scored_candidates if c["score_improvement"] > 0.0]

    if not improving:
        return {
            "best_swap": None,
            "best_candidate_metrics": None,
            "num_candidates_evaluated": num_evaluated,
        }

    # Deterministic sort: score_improvement desc, score_after asc,
    # description asc.
    improving.sort(key=lambda c: (
        -c["score_improvement"],
        c["spectral_instability_score_after"],
        c["swap"]["description"],
    ))

    best = improving[0]

    return {
        "best_swap": best["swap"],
        "best_candidate_metrics": best,
        "num_candidates_evaluated": num_evaluated,
    }


# ── Core Feature 4 — Spectral Repair Loop Experiment ────────────────


def run_spectral_graph_repair_loop(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    risk_result: dict[str, Any],
    *,
    nb_spectral_radius: float = 0.0,
    spectral_instability_ratio: float = 0.0,
    ipr_localization_score: float = 0.0,
    avg_variable_degree: float = 0.0,
    avg_check_degree: float = 0.0,
    instability_threshold: float = 0.5,
    max_candidates: int = 10,
    max_iters: int = 100,
    enable_decode_comparison: bool = True,
    candidate_clusters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the spectral graph repair loop experiment.

    Full pipeline:
      baseline graph
      → spectral diagnostics (reused from inputs)
      → instability prediction
      → generate repair candidates
      → evaluate repaired candidates
      → choose best repair
      → optional baseline decode
      → optional repaired decode
      → compare outcomes

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    llr : np.ndarray
        Per-variable log-likelihood ratios, length n.
    syndrome_vec : np.ndarray
        Binary syndrome vector, length m.
    risk_result : dict[str, Any]
        Output of ``compute_spectral_failure_risk()`` (v6.4).
    nb_spectral_radius : float
        NB spectral radius from v6.0 diagnostics.
    spectral_instability_ratio : float
        Spectral instability ratio from v6.8 predictor.
    ipr_localization_score : float
        IPR localization score from v6.1.
    avg_variable_degree : float
        Average variable-node degree.
    avg_check_degree : float
        Average check-node degree.
    instability_threshold : float
        Threshold for instability prediction.
    max_candidates : int
        Maximum number of candidate swaps.
    max_iters : int
        Maximum BP iterations for decode comparison.
    enable_decode_comparison : bool
        If True, run baseline and repaired decodes for comparison.
    candidate_clusters : list[dict[str, Any]] or None
        Cluster definitions from v6.2.  If None, extracted from
        risk_result (if available).

    Returns
    -------
    dict[str, Any]
        JSON-serializable artifact with repair loop results.
    """
    m, n = H.shape
    llr = np.asarray(llr, dtype=np.float64)
    syndrome_vec = np.asarray(syndrome_vec, dtype=np.uint8)

    node_risk_scores = risk_result.get("node_risk_scores", [])
    cluster_risk_scores_raw = risk_result.get("cluster_risk_scores", [])
    top_risk_clusters = risk_result.get("top_risk_clusters", [])

    # Resolve candidate clusters.
    if candidate_clusters is None:
        candidate_clusters = risk_result.get("candidate_clusters", [])

    # Compute baseline instability score.
    instability_score_before = compute_spectral_instability_score(
        nb_spectral_radius=nb_spectral_radius,
        spectral_instability_ratio=spectral_instability_ratio,
        ipr_localization_score=ipr_localization_score,
        cluster_risk_scores=cluster_risk_scores_raw,
        avg_variable_degree=avg_variable_degree,
        avg_check_degree=avg_check_degree,
    )
    predicted_instability_before = instability_score_before > instability_threshold

    # ── Step 1: Generate repair candidates ────────────────────────
    gen_result = generate_repair_candidates(
        H,
        top_risk_clusters=top_risk_clusters,
        candidate_clusters=candidate_clusters,
        node_risk_scores=node_risk_scores,
        cluster_risk_scores=cluster_risk_scores_raw,
        max_candidates=max_candidates,
    )

    candidates = gen_result["candidates"]
    cluster_nodes = gen_result["cluster_nodes"]
    cluster_risk_score = gen_result["cluster_risk_score"]

    # ── No candidates: return baseline-only result ────────────────
    if not candidates:
        return _no_repair_result(
            H, llr, syndrome_vec, max_iters,
            instability_score_before, predicted_instability_before,
            cluster_nodes, cluster_risk_score,
            enable_decode_comparison,
        )

    # ── Step 2: Score each candidate ──────────────────────────────
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics = score_repair_candidate(
            H, candidate,
            nb_spectral_radius=nb_spectral_radius,
            spectral_instability_ratio=spectral_instability_ratio,
            ipr_localization_score=ipr_localization_score,
            cluster_risk_scores=cluster_risk_scores_raw,
            avg_variable_degree=avg_variable_degree,
            avg_check_degree=avg_check_degree,
            instability_score_before=instability_score_before,
            predicted_instability_before=predicted_instability_before,
            instability_threshold=instability_threshold,
        )
        scored.append(metrics)

    # ── Step 3: Select best repair ────────────────────────────────
    selection = select_best_repair(scored)

    best_swap = selection["best_swap"]
    best_metrics = selection["best_candidate_metrics"]
    num_candidates_evaluated = selection["num_candidates_evaluated"]

    if best_swap is None:
        # No improving candidate found.
        return _no_repair_result(
            H, llr, syndrome_vec, max_iters,
            instability_score_before, predicted_instability_before,
            cluster_nodes, cluster_risk_score,
            enable_decode_comparison,
            num_candidates_evaluated=num_candidates_evaluated,
        )

    instability_score_after = best_metrics["spectral_instability_score_after"]
    predicted_instability_after = best_metrics["predicted_instability_after"]
    score_improvement = best_metrics["score_improvement"]

    # ── Step 4: Optional decode comparison ────────────────────────
    if enable_decode_comparison:
        # Baseline decode.
        baseline_correction, baseline_iters, baseline_residuals = (
            _experimental_bp_flooding(H, llr, syndrome_vec, max_iters)
        )
        baseline_success = bool(np.array_equal(
            _compute_syndrome(H, baseline_correction), syndrome_vec,
        ))

        # Repaired decode.
        edges = _extract_edges(H)
        repaired_edges = _apply_swap(edges, {
            "remove": best_swap["remove"],
            "add": best_swap["add"],
        })
        H_repaired = _edges_to_H(repaired_edges, m, n)

        repaired_correction, repaired_iters, repaired_residuals = (
            _experimental_bp_flooding(
                H_repaired, llr, syndrome_vec, max_iters,
            )
        )
        repaired_success = bool(np.array_equal(
            _compute_syndrome(H_repaired, repaired_correction),
            syndrome_vec,
        ))

        decode_success_before = baseline_success
        decode_success_after = repaired_success
        iterations_before = baseline_iters
        iterations_after = repaired_iters
        delta_success = int(repaired_success) - int(baseline_success)
        delta_iterations = repaired_iters - baseline_iters
    else:
        decode_success_before = False
        decode_success_after = False
        iterations_before = 0
        iterations_after = 0
        delta_success = 0
        delta_iterations = 0

    return {
        "repair_applied": True,
        "best_swap": {
            "remove": [list(e) for e in best_swap["remove"]],
            "add": best_swap["add"],
            "description": best_swap["description"],
        },
        "num_candidates_evaluated": num_candidates_evaluated,
        "spectral_instability_score_before": round(instability_score_before, 12),
        "spectral_instability_score_after": round(instability_score_after, 12),
        "score_improvement": round(score_improvement, 12),
        "predicted_instability_before": predicted_instability_before,
        "predicted_instability_after": predicted_instability_after,
        "decode_success_before": decode_success_before,
        "decode_success_after": decode_success_after,
        "delta_success": delta_success,
        "iterations_before": iterations_before,
        "iterations_after": iterations_after,
        "delta_iterations": delta_iterations,
        "cluster_nodes": cluster_nodes,
        "cluster_risk_score": round(cluster_risk_score, 12),
    }


def _no_repair_result(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    max_iters: int,
    instability_score_before: float,
    predicted_instability_before: bool,
    cluster_nodes: list[int],
    cluster_risk_score: float,
    enable_decode_comparison: bool,
    *,
    num_candidates_evaluated: int = 0,
) -> dict[str, Any]:
    """Build result when no valid repair improves the score."""
    if enable_decode_comparison:
        correction, iters, _ = _experimental_bp_flooding(
            H, llr, syndrome_vec, max_iters,
        )
        success = bool(np.array_equal(
            _compute_syndrome(H, correction), syndrome_vec,
        ))
        decode_success_before = success
        decode_success_after = success
        iterations_before = iters
        iterations_after = iters
    else:
        decode_success_before = False
        decode_success_after = False
        iterations_before = 0
        iterations_after = 0

    return {
        "repair_applied": False,
        "best_swap": None,
        "num_candidates_evaluated": num_candidates_evaluated,
        "spectral_instability_score_before": round(instability_score_before, 12),
        "spectral_instability_score_after": round(instability_score_before, 12),
        "score_improvement": 0.0,
        "predicted_instability_before": predicted_instability_before,
        "predicted_instability_after": predicted_instability_before,
        "decode_success_before": decode_success_before,
        "decode_success_after": decode_success_after,
        "delta_success": 0,
        "iterations_before": iterations_before,
        "iterations_after": iterations_after,
        "delta_iterations": 0,
        "cluster_nodes": cluster_nodes,
        "cluster_risk_score": round(cluster_risk_score, 12),
    }


# ── Phase Diagram Aggregation Metrics ────────────────────────────────


def compute_repair_loop_aggregate_metrics(
    trial_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate spectral graph repair loop results across trials.

    Parameters
    ----------
    trial_results : list[dict[str, Any]]
        List of per-trial results from
        :func:`run_spectral_graph_repair_loop`.

    Returns
    -------
    dict[str, Any]
        JSON-serializable aggregate metrics dictionary.
    """
    n = len(trial_results)
    if n == 0:
        return {
            "mean_repair_score_improvement": 0.0,
            "repair_activation_rate": 0.0,
            "repair_prediction_flip_rate": 0.0,
            "mean_repaired_instability_score": 0.0,
            "mean_delta_iterations_after_repair": 0.0,
            "mean_delta_success_after_repair": 0.0,
            "mean_candidates_evaluated": 0.0,
            "repair_decode_improvement_rate": 0.0,
            "num_trials": 0,
        }

    total_improvement = 0.0
    num_activated = 0
    num_prediction_flips = 0
    total_repaired_score = 0.0
    total_delta_iters = 0
    total_delta_success = 0
    total_candidates = 0
    num_decode_improvements = 0

    for trial in trial_results:
        total_improvement += trial.get("score_improvement", 0.0)
        if trial.get("repair_applied", False):
            num_activated += 1
        pred_before = trial.get("predicted_instability_before", False)
        pred_after = trial.get("predicted_instability_after", False)
        if pred_before != pred_after:
            num_prediction_flips += 1
        total_repaired_score += trial.get(
            "spectral_instability_score_after", 0.0,
        )
        total_delta_iters += trial.get("delta_iterations", 0)
        total_delta_success += trial.get("delta_success", 0)
        total_candidates += trial.get("num_candidates_evaluated", 0)
        if trial.get("delta_success", 0) > 0:
            num_decode_improvements += 1

    return {
        "mean_repair_score_improvement": round(total_improvement / n, 12),
        "repair_activation_rate": round(num_activated / n, 12),
        "repair_prediction_flip_rate": round(num_prediction_flips / n, 12),
        "mean_repaired_instability_score": round(total_repaired_score / n, 12),
        "mean_delta_iterations_after_repair": round(total_delta_iters / n, 12),
        "mean_delta_success_after_repair": round(total_delta_success / n, 12),
        "mean_candidates_evaluated": round(total_candidates / n, 12),
        "repair_decode_improvement_rate": round(num_decode_improvements / n, 12),
        "num_trials": n,
    }
