"""
v7.4.0 — Spectral Tanner Graph Design Rules.

Evaluates structural stability of a Tanner graph before decoding begins.
Consumes: v6.0 NB spectrum, v6.1 IPR, v6.4 cluster risk, v6.8 instability ratio.
Does not modify graph or decoder.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any

_W1_NB_RADIUS = 0.30
_W2_MAX_IPR = 0.20
_W3_MAX_CLUSTER_RISK = 0.20
_W4_INSTABILITY_RATIO = 0.15
_W5_SPECTRAL_GAP_PENALTY = 0.15

_DECIMAL_PLACES = 12

_DEFAULT_IPR_THRESHOLD = 0.3
_DEFAULT_CLUSTER_RISK_THRESHOLD = 0.5
_DEFAULT_SPECTRAL_GAP_THRESHOLD = 0.5


# ── Feature 1: Spectral Design Score ────────────────────────────────

def compute_spectral_design_score(
    normalized_nb_radius: float,
    max_ipr: float,
    max_cluster_risk: float,
    instability_ratio: float,
    spectral_gap_penalty: float,
    *,
    w1: float = _W1_NB_RADIUS,
    w2: float = _W2_MAX_IPR,
    w3: float = _W3_MAX_CLUSTER_RISK,
    w4: float = _W4_INSTABILITY_RATIO,
    w5: float = _W5_SPECTRAL_GAP_PENALTY,
) -> float:
    """Compute structural stability score in [0, 1].

    0 = stable graph, 1 = high instability risk.
    All inputs clamped to [0, 1].  Result rounded to 12 decimals.
    """
    score = (
        w1 * max(0.0, min(1.0, normalized_nb_radius))
        + w2 * max(0.0, min(1.0, max_ipr))
        + w3 * max(0.0, min(1.0, max_cluster_risk))
        + w4 * max(0.0, min(1.0, instability_ratio))
        + w5 * max(0.0, min(1.0, spectral_gap_penalty))
    )
    return round(max(0.0, min(1.0, score)), _DECIMAL_PLACES)


# ── Feature 2: Spectral Gap Diagnostic ──────────────────────────────

def compute_spectral_gap(
    nb_eigenvalues: list[list[float]],
) -> dict[str, float]:
    """Compute spectral gap from NB eigenvalues (sorted by magnitude desc).

    Returns spectral_gap = |λ1| - |λ2| and spectral_gap_ratio = |λ2|/|λ1|.
    Rounded to 12 decimals.
    """
    if len(nb_eigenvalues) < 2:
        return {"spectral_gap": 0.0, "spectral_gap_ratio": 0.0}

    mag1 = (nb_eigenvalues[0][0] ** 2 + nb_eigenvalues[0][1] ** 2) ** 0.5
    mag2 = (nb_eigenvalues[1][0] ** 2 + nb_eigenvalues[1][1] ** 2) ** 0.5
    gap = mag1 - mag2
    ratio = mag2 / mag1 if mag1 > 0.0 else 0.0

    return {
        "spectral_gap": round(gap, _DECIMAL_PLACES),
        "spectral_gap_ratio": round(ratio, _DECIMAL_PLACES),
    }


# ── Feature 3: Structural Risk Pattern Detection ────────────────────

def detect_structural_risk_patterns(
    max_ipr: float,
    cluster_risk_scores: list[float],
    spectral_gap: float,
    spectral_instability_ratio: float,
    localized_variable_nodes: list[int],
    top_risk_clusters: list[int],
    *,
    ipr_threshold: float = _DEFAULT_IPR_THRESHOLD,
    cluster_risk_threshold: float = _DEFAULT_CLUSTER_RISK_THRESHOLD,
    spectral_gap_threshold: float = _DEFAULT_SPECTRAL_GAP_THRESHOLD,
) -> dict[str, Any]:
    """Detect known BP failure motifs from spectral signals.

    Returns high_localization_nodes, high_risk_clusters (both sorted),
    spectral_gap_small, and structural_risk_detected.
    """
    high_localization = max_ipr >= ipr_threshold
    high_risk_clusters = sorted([
        i for i, s in enumerate(cluster_risk_scores)
        if s >= cluster_risk_threshold
    ])
    spectral_gap_small = spectral_gap < spectral_gap_threshold
    high_instability = spectral_instability_ratio > 1.0

    structural_risk_detected = (
        high_localization
        or len(high_risk_clusters) > 0
        or spectral_gap_small
        or high_instability
    )

    return {
        "high_localization_nodes": sorted(localized_variable_nodes),
        "high_risk_clusters": high_risk_clusters,
        "spectral_gap_small": spectral_gap_small,
        "structural_risk_detected": structural_risk_detected,
    }


# ── Feature 4: Graph Design Analysis Pipeline ───────────────────────

def run_spectral_graph_design_analysis(
    nb_spectrum_result: dict[str, Any],
    localization_result: dict[str, Any],
    risk_result: dict[str, Any],
    spectral_instability_ratio: float,
    avg_variable_degree: float,
    avg_check_degree: float,
) -> dict[str, Any]:
    """Run full spectral graph design analysis pipeline.

    Combines NB spectrum (v6.0), localization (v6.1), risk (v6.4), and
    instability ratio (v6.8) into a JSON-serializable design artifact.
    """
    # Extract upstream values.
    nb_spectral_radius = float(nb_spectrum_result.get("spectral_radius", 0.0))
    nb_eigenvalues = nb_spectrum_result.get("nb_eigenvalues", [])
    max_ipr = float(localization_result.get("max_ipr", 0.0))
    localized_var_nodes = list(
        localization_result.get("localized_variable_nodes", [])
    )
    cluster_risk_scores = list(risk_result.get("cluster_risk_scores", []))
    max_cluster_risk = float(risk_result.get("max_cluster_risk", 0.0))
    top_risk_clusters = list(risk_result.get("top_risk_clusters", []))

    # Compute spectral gap.
    gap_result = compute_spectral_gap(nb_eigenvalues)
    spectral_gap = gap_result["spectral_gap"]
    spectral_gap_ratio = gap_result["spectral_gap_ratio"]

    # Normalize components for design score.
    avg_degree = (avg_variable_degree + avg_check_degree) / 2.0
    if avg_degree > 0.0:
        normalized_nb_radius = nb_spectral_radius / (avg_degree + 1.0)
    else:
        normalized_nb_radius = 0.0
    normalized_nb_radius = max(0.0, min(1.0, normalized_nb_radius))

    instability_ratio_norm = max(
        0.0, min(1.0, spectral_instability_ratio / 2.0),
    )

    # Spectral gap penalty: small gap → high penalty.
    if spectral_gap >= _DEFAULT_SPECTRAL_GAP_THRESHOLD:
        spectral_gap_penalty = 0.0
    elif spectral_gap <= 0.0:
        spectral_gap_penalty = 1.0
    else:
        spectral_gap_penalty = 1.0 - spectral_gap / _DEFAULT_SPECTRAL_GAP_THRESHOLD

    # Compute design score.
    design_score = compute_spectral_design_score(
        normalized_nb_radius=normalized_nb_radius,
        max_ipr=max_ipr,
        max_cluster_risk=max_cluster_risk,
        instability_ratio=instability_ratio_norm,
        spectral_gap_penalty=spectral_gap_penalty,
    )

    # Detect structural risk patterns.
    risk_patterns = detect_structural_risk_patterns(
        max_ipr=max_ipr,
        cluster_risk_scores=cluster_risk_scores,
        spectral_gap=spectral_gap,
        spectral_instability_ratio=spectral_instability_ratio,
        localized_variable_nodes=localized_var_nodes,
        top_risk_clusters=top_risk_clusters,
    )

    return {
        "graph_design_score": design_score,
        "nb_spectral_radius": round(nb_spectral_radius, _DECIMAL_PLACES),
        "spectral_gap": spectral_gap,
        "spectral_gap_ratio": spectral_gap_ratio,
        "spectral_instability_ratio": round(
            spectral_instability_ratio, _DECIMAL_PLACES,
        ),
        "max_ipr": round(max_ipr, _DECIMAL_PLACES),
        "max_cluster_risk": round(max_cluster_risk, _DECIMAL_PLACES),
        "structural_risk_detected": risk_patterns["structural_risk_detected"],
        "high_localization_nodes": risk_patterns["high_localization_nodes"],
        "high_risk_clusters": risk_patterns["high_risk_clusters"],
    }
