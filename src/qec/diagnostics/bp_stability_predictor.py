"""
v6.8.0 — Spectral BP Stability Predictor.

Estimates BP decoding failure risk before BP decoding runs by combining
structural and spectral signals derived from the non-backtracking
(Hashimoto) spectrum of Tanner graphs.

Combines diagnostics from the v6.1–v6.7 spectral chain:
  - v6.1 NB localization (IPR, localized modes)
  - v6.2 spectral trapping-set candidates
  - v6.3 spectral–BP attractor alignment
  - v6.4 spectral failure risk scoring
  - v6.7 spectral graph optimization (spectral radius)

Inputs are pre-computed diagnostic outputs — this module does not
recompute spectra or localization.  Does not modify decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import math
from typing import Any


# ── Default weights ────────────────────────────────────────────────────

_W1_SPECTRAL = 0.5
_W2_LOCALIZATION = 0.25
_W3_CLUSTER_RISK = 0.15
_W4_CYCLE_DENSITY = 0.10

_FAILURE_RISK_THRESHOLD = 1.5
_INSTABILITY_RISK_THRESHOLD = 0.6


def compute_bp_stability_prediction(
    graph: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    w1: float = _W1_SPECTRAL,
    w2: float = _W2_LOCALIZATION,
    w3: float = _W3_CLUSTER_RISK,
    w4: float = _W4_CYCLE_DENSITY,
    failure_risk_threshold: float = _FAILURE_RISK_THRESHOLD,
    instability_risk_threshold: float = _INSTABILITY_RISK_THRESHOLD,
) -> dict[str, Any]:
    """Compute BP stability prediction from spectral diagnostics.

    Combines structural and spectral signals into a quantitative BP
    instability estimate.  Returns a JSON-serializable prediction
    dictionary.

    Parameters
    ----------
    graph : dict[str, Any]
        Tanner graph structure with keys:

        - ``num_variable_nodes`` : int
        - ``num_check_nodes`` : int
        - ``num_edges`` : int
        - ``avg_degree`` : float (average node degree)

        Optionally:

        - ``num_short_cycles`` : int (cycles of length <= 8)

    diagnostics : dict[str, Any]
        Combined outputs from existing spectral diagnostics:

        - ``spectral_radius`` : float (from v6.0/v6.7 NB spectrum)
        - ``nb_max_ipr`` : float (from v6.1 localization)
        - ``nb_num_localized_modes`` : int (from v6.1 localization)
        - ``max_cluster_risk`` : float (from v6.4 failure risk)

    w1 : float
        Weight for spectral instability ratio.  Default 0.5.
    w2 : float
        Weight for localization strength.  Default 0.25.
    w3 : float
        Weight for cluster risk signal.  Default 0.15.
    w4 : float
        Weight for cycle density.  Default 0.10.
    failure_risk_threshold : float
        Normalization threshold for bp_failure_risk.  Default 1.5.
    instability_risk_threshold : float
        Threshold for predicted_instability flag.  Default 0.6.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``bp_stability_score`` : float
        - ``bp_failure_risk`` : float (clamped to [0, 1])
        - ``predicted_instability`` : bool
        - ``spectral_radius`` : float
        - ``nb_instability_threshold`` : float
        - ``spectral_instability_ratio`` : float
        - ``localization_strength`` : float
        - ``cluster_risk_signal`` : float
        - ``cycle_density`` : float
    """
    # ── Extract graph structure ────────────────────────────────────
    num_variable_nodes = graph.get("num_variable_nodes", 0)
    num_check_nodes = graph.get("num_check_nodes", 0)
    num_edges = graph.get("num_edges", 0)
    avg_degree = graph.get("avg_degree", 0.0)
    num_short_cycles = graph.get("num_short_cycles", None)

    # ── Extract diagnostic signals ────────────────────────────────
    spectral_radius = float(diagnostics.get("spectral_radius", 0.0))
    nb_max_ipr = float(diagnostics.get("nb_max_ipr", 0.0))
    nb_num_localized_modes = int(diagnostics.get("nb_num_localized_modes", 0))
    max_cluster_risk = float(diagnostics.get("max_cluster_risk", 0.0))

    # ── 1. NB Spectral Instability ────────────────────────────────
    # Threshold: sqrt(avg_degree).  Ratio > 1 → likely BP instability.
    if avg_degree > 0.0:
        nb_instability_threshold = round(math.sqrt(avg_degree), 12)
    else:
        nb_instability_threshold = 0.0

    if nb_instability_threshold > 0.0:
        spectral_instability_ratio = round(
            spectral_radius / nb_instability_threshold, 12,
        )
    else:
        spectral_instability_ratio = 0.0

    # ── 2. Localization Strength ──────────────────────────────────
    localization_strength = round(
        nb_max_ipr * float(nb_num_localized_modes), 12,
    )

    # ── 3. Cluster Risk Signal ────────────────────────────────────
    cluster_risk_signal = round(max_cluster_risk, 12)

    # ── 4. Cycle Density ──────────────────────────────────────────
    if num_short_cycles is not None and num_variable_nodes > 0:
        cycle_density = round(
            float(num_short_cycles) / float(num_variable_nodes), 12,
        )
    else:
        # Approximate using clustering around high-risk nodes:
        # use spectral radius and degree as proxy.
        if num_variable_nodes > 0 and num_edges > 0:
            # Proxy: excess edges relative to tree-like structure.
            total_nodes = num_variable_nodes + num_check_nodes
            if total_nodes > 0:
                cycle_proxy = float(num_edges - total_nodes + 1)
                cycle_density = round(
                    max(0.0, cycle_proxy) / float(num_variable_nodes), 12,
                )
            else:
                cycle_density = 0.0
        else:
            cycle_density = 0.0

    # ── BP Stability Score ────────────────────────────────────────
    bp_stability_score = round(
        w1 * spectral_instability_ratio
        + w2 * localization_strength
        + w3 * cluster_risk_signal
        + w4 * cycle_density,
        12,
    )

    # ── Failure Risk ──────────────────────────────────────────────
    if failure_risk_threshold > 0.0:
        bp_failure_risk = round(
            min(1.0, bp_stability_score / failure_risk_threshold), 12,
        )
    else:
        bp_failure_risk = 0.0

    # ── Instability Flag ──────────────────────────────────────────
    predicted_instability = bp_failure_risk > instability_risk_threshold

    return {
        "bp_stability_score": bp_stability_score,
        "bp_failure_risk": bp_failure_risk,
        "predicted_instability": predicted_instability,
        "spectral_radius": round(spectral_radius, 12),
        "nb_instability_threshold": nb_instability_threshold,
        "spectral_instability_ratio": spectral_instability_ratio,
        "localization_strength": localization_strength,
        "cluster_risk_signal": cluster_risk_signal,
        "cycle_density": cycle_density,
    }
