"""
v7.0.0 — Spectral-Guided Decoder Control Framework.

Integrates spectral diagnostics and BP stability predictor outputs to
steer decoder behavior through an experimental control layer.

Consumes outputs from:
  - v6.4 spectral failure risk (node_risk_scores, cluster_risk_scores)
  - v6.8 BP stability predictor (bp_failure_risk, predicted_instability)
  - v6.9 prediction validation (accuracy metrics)

Control features:
  1. Predictor-guided scheduling — uses predicted instability to
     activate stabilization policies.
  2. Adaptive damping — deterministic per-node damping derived from
     normalized node risk scores.
  3. Decoder mode selection hook — control_mode selection based on
     bp_failure_risk threshold.

Does not modify decoder internals.  Fully deterministic: no randomness,
no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ── Control mode constants ────────────────────────────────────────────

CONTROL_MODE_STANDARD = "standard"
CONTROL_MODE_RISK_GUIDED_DAMPING = "risk_guided_damping"
CONTROL_MODE_RISK_GUIDED_SCHEDULE = "risk_guided_schedule"

# ── Default parameters ────────────────────────────────────────────────

_DEFAULT_BASE_DAMPING = 0.5
_DEFAULT_ALPHA = 0.25
_DEFAULT_DAMPING_MIN = 0.5
_DEFAULT_DAMPING_MAX = 0.9
_DEFAULT_RISK_THRESHOLD = 0.6
_DEFAULT_SCHEDULE_THRESHOLD_FRACTION = 0.5
_DEFAULT_MAX_ITERS = 100


# ── Helpers ───────────────────────────────────────────────────────────

def _compute_syndrome(H: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute binary syndrome s = H @ x (mod 2)."""
    return (H.astype(np.int32) @ x.astype(np.int32)) % 2


def _identify_high_risk_nodes(
    node_risk_scores: list[list[int | float]],
    *,
    threshold_fraction: float = _DEFAULT_SCHEDULE_THRESHOLD_FRACTION,
) -> list[int]:
    """Return node indices with risk >= threshold_fraction * max_risk.

    Parameters
    ----------
    node_risk_scores : list[list[int | float]]
        Each element is ``[node_index, score]``.
    threshold_fraction : float
        Fraction of max risk used as threshold.

    Returns
    -------
    list[int]
        Sorted list of high-risk node indices.
    """
    if not node_risk_scores:
        return []
    max_risk = max(pair[1] for pair in node_risk_scores)
    if max_risk <= 0.0:
        return []
    threshold = threshold_fraction * max_risk
    return sorted(
        int(pair[0]) for pair in node_risk_scores
        if pair[1] >= threshold
    )


def _compute_adaptive_damping(
    n: int,
    node_risk_scores: list[list[int | float]],
    *,
    base_damping: float = _DEFAULT_BASE_DAMPING,
    alpha: float = _DEFAULT_ALPHA,
    damping_min: float = _DEFAULT_DAMPING_MIN,
    damping_max: float = _DEFAULT_DAMPING_MAX,
) -> np.ndarray:
    """Compute deterministic per-node damping from risk scores.

    Formula:
        damping(node) = base_damping + alpha * normalized_node_risk

    Clamped to [damping_min, damping_max].

    Parameters
    ----------
    n : int
        Number of variable nodes.
    node_risk_scores : list[list[int | float]]
        Each element is ``[node_index, score]``.
    base_damping : float
        Base damping factor.
    alpha : float
        Scaling factor for normalized risk.
    damping_min : float
        Minimum allowed damping.
    damping_max : float
        Maximum allowed damping.

    Returns
    -------
    np.ndarray
        Per-node damping array of length n.
    """
    damping = np.full(n, base_damping, dtype=np.float64)

    if not node_risk_scores:
        return np.clip(damping, damping_min, damping_max)

    max_risk = max(pair[1] for pair in node_risk_scores)
    if max_risk <= 0.0:
        return np.clip(damping, damping_min, damping_max)

    for pair in node_risk_scores:
        node_idx = int(pair[0])
        score = float(pair[1])
        if node_idx < n:
            normalized = score / max_risk
            damping[node_idx] = base_damping + alpha * normalized

    return np.clip(damping, damping_min, damping_max)


def _select_control_mode(
    prediction: dict[str, Any],
    *,
    risk_threshold: float = _DEFAULT_RISK_THRESHOLD,
) -> str:
    """Select control mode based on predictor output.

    Parameters
    ----------
    prediction : dict[str, Any]
        BP stability predictor output with ``bp_failure_risk`` and
        ``predicted_instability``.
    risk_threshold : float
        Threshold for activating risk-guided modes.

    Returns
    -------
    str
        One of the CONTROL_MODE_* constants.
    """
    bp_failure_risk = float(prediction.get("bp_failure_risk", 0.0))
    predicted_instability = prediction.get("predicted_instability", False)

    if not predicted_instability:
        return CONTROL_MODE_STANDARD

    if bp_failure_risk > risk_threshold:
        return CONTROL_MODE_RISK_GUIDED_DAMPING

    return CONTROL_MODE_RISK_GUIDED_SCHEDULE


def _experimental_bp_flooding(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    max_iters: int,
    per_node_damping: np.ndarray,
) -> tuple[np.ndarray, int, list[float]]:
    """Minimal flooding BP with per-node damping for experiment.

    Self-contained experimental implementation — does not modify
    the existing decoder.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    llr : np.ndarray
        Per-variable log-likelihood ratios, length n.
    syndrome_vec : np.ndarray
        Binary syndrome vector, length m.
    max_iters : int
        Maximum BP iterations.
    per_node_damping : np.ndarray
        Per-variable damping factors, length n.

    Returns
    -------
    correction : np.ndarray
        Hard-decision binary vector, length n.
    iterations : int
        Number of iterations executed.
    residual_norms : list[float]
        Per-iteration residual norms.
    """
    H_f = H.astype(np.float64)
    m, n = H_f.shape
    llr = np.asarray(llr, dtype=np.float64).copy()

    s_sign = np.where(syndrome_vec.astype(np.float64) > 0.5, -1.0, 1.0)

    v2c = np.zeros((m, n), dtype=np.float64)
    for j in range(m):
        for i in range(n):
            if H_f[j, i] > 0.5:
                v2c[j, i] = llr[i]

    c2v = np.zeros((m, n), dtype=np.float64)
    residual_norms: list[float] = []

    for iteration in range(1, max_iters + 1):
        # Check-to-variable: tanh rule.
        for j in range(m):
            neighbors = [i for i in range(n) if H_f[j, i] > 0.5]
            if len(neighbors) < 2:
                for i in neighbors:
                    c2v[j, i] = 0.0
                continue
            for i in neighbors:
                prod = s_sign[j]
                for k in neighbors:
                    if k != i:
                        val = np.clip(v2c[j, k] / 2.0, -15.0, 15.0)
                        prod *= np.tanh(val)
                prod = np.clip(prod, -1.0 + 1e-15, 1.0 - 1e-15)
                c2v[j, i] = 2.0 * np.arctanh(prod)

        # Variable-to-check with per-node damping.
        prev_v2c = v2c.copy()
        for i in range(n):
            check_neighbors = [j for j in range(m) if H_f[j, i] > 0.5]
            total = llr[i] + sum(c2v[j, i] for j in check_neighbors)
            for j in check_neighbors:
                new_msg = total - c2v[j, i]
                d = per_node_damping[i]
                v2c[j, i] = d * prev_v2c[j, i] + (1.0 - d) * new_msg

        # Total LLR and hard decision.
        L_total = llr.copy()
        for i in range(n):
            for j in range(m):
                if H_f[j, i] > 0.5:
                    L_total[i] += c2v[j, i]

        correction = (L_total < 0.0).astype(np.uint8)
        s_residual = _compute_syndrome(H, correction)
        r_norm = float(np.linalg.norm(
            s_residual.astype(np.float64) - syndrome_vec.astype(np.float64),
        ))
        residual_norms.append(round(r_norm, 12))

        if np.array_equal(s_residual, syndrome_vec.astype(np.uint8)):
            return correction, iteration, residual_norms

    return correction, max_iters, residual_norms


# ── Main experiment function ──────────────────────────────────────────

def run_spectral_decoder_control_experiment(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    risk_result: dict[str, Any],
    prediction: dict[str, Any],
    *,
    base_damping: float = _DEFAULT_BASE_DAMPING,
    alpha: float = _DEFAULT_ALPHA,
    damping_min: float = _DEFAULT_DAMPING_MIN,
    damping_max: float = _DEFAULT_DAMPING_MAX,
    risk_threshold: float = _DEFAULT_RISK_THRESHOLD,
    schedule_threshold_fraction: float = _DEFAULT_SCHEDULE_THRESHOLD_FRACTION,
    max_iters: int = _DEFAULT_MAX_ITERS,
) -> dict[str, Any]:
    """Run a spectral-guided decoder control experiment.

    Wraps the decoding process using spectral diagnostics and BP
    stability predictor outputs to steer decoder behavior.

    Pipeline:
      1. Select control mode based on predictor output.
      2. Compute adaptive per-node damping if risk-guided.
      3. Run baseline decode (standard scalar damping).
      4. Run controlled decode (with adaptive damping/scheduling).
      5. Compare and record outcomes.

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
        Must contain ``node_risk_scores``, ``cluster_risk_scores``,
        and ``top_risk_clusters``.
    prediction : dict[str, Any]
        Output of ``compute_bp_stability_prediction()`` (v6.8).
        Must contain ``bp_failure_risk``, ``predicted_instability``,
        and ``spectral_instability_ratio``.
    base_damping : float
        Base damping for adaptive controller.  Default 0.5.
    alpha : float
        Risk-to-damping scaling factor.  Default 0.25.
    damping_min : float
        Minimum per-node damping.  Default 0.5.
    damping_max : float
        Maximum per-node damping.  Default 0.9.
    risk_threshold : float
        BP failure risk threshold for mode switching.  Default 0.6.
    schedule_threshold_fraction : float
        Fraction of max risk for identifying high-risk nodes used
        in scheduling.  Default 0.5.
    max_iters : int
        Maximum BP iterations.  Default 100.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``control_mode``: str — selected control mode.
        - ``predicted_instability``: bool — from predictor.
        - ``bp_failure_risk``: float — from predictor.
        - ``adaptive_damping_enabled``: bool — whether adaptive
          damping was applied.
        - ``scheduled_high_risk_nodes``: list[int] — high-risk
          nodes used for scheduling.
        - ``baseline_metrics``: dict — baseline decode metrics.
        - ``controlled_metrics``: dict — controlled decode metrics.
        - ``delta_iterations``: int — iteration difference
          (controlled - baseline).
        - ``delta_success``: int — success difference.
        - ``controller_metadata``: dict — additional control info.
        - ``node_risk_scores``: pass-through from risk_result.
        - ``cluster_risk_scores``: pass-through from risk_result.
        - ``top_risk_clusters``: pass-through from risk_result.
    """
    n = H.shape[1]
    llr = np.asarray(llr, dtype=np.float64)
    syndrome_vec = np.asarray(syndrome_vec, dtype=np.uint8)

    node_risk_scores = risk_result.get("node_risk_scores", [])

    # ── 1. Select control mode ────────────────────────────────────
    control_mode = _select_control_mode(
        prediction, risk_threshold=risk_threshold,
    )

    # ── 2. Determine high-risk nodes ──────────────────────────────
    high_risk_nodes = _identify_high_risk_nodes(
        node_risk_scores,
        threshold_fraction=schedule_threshold_fraction,
    )

    # ── 3. Compute adaptive damping ───────────────────────────────
    adaptive_damping_enabled = control_mode != CONTROL_MODE_STANDARD
    if adaptive_damping_enabled:
        controlled_damping = _compute_adaptive_damping(
            n, node_risk_scores,
            base_damping=base_damping,
            alpha=alpha,
            damping_min=damping_min,
            damping_max=damping_max,
        )
    else:
        controlled_damping = np.full(n, base_damping, dtype=np.float64)
        controlled_damping = np.clip(controlled_damping, damping_min, damping_max)

    # ── 4. Baseline decode ────────────────────────────────────────
    baseline_damping = np.full(n, base_damping, dtype=np.float64)
    baseline_damping = np.clip(baseline_damping, damping_min, damping_max)
    baseline_correction, baseline_iters, baseline_residuals = (
        _experimental_bp_flooding(
            H, llr, syndrome_vec, max_iters, baseline_damping,
        )
    )
    baseline_success = bool(np.array_equal(
        _compute_syndrome(H, baseline_correction),
        syndrome_vec,
    ))

    # ── 5. Controlled decode ──────────────────────────────────────
    controlled_correction, controlled_iters, controlled_residuals = (
        _experimental_bp_flooding(
            H, llr, syndrome_vec, max_iters, controlled_damping,
        )
    )
    controlled_success = bool(np.array_equal(
        _compute_syndrome(H, controlled_correction),
        syndrome_vec,
    ))

    # ── 6. Comparison ─────────────────────────────────────────────
    delta_iterations = controlled_iters - baseline_iters
    delta_success = int(controlled_success) - int(baseline_success)

    baseline_metrics = {
        "iterations": baseline_iters,
        "success": baseline_success,
        "residual_norms": baseline_residuals,
        "final_residual_norm": (
            baseline_residuals[-1] if baseline_residuals else 0.0
        ),
    }

    controlled_metrics = {
        "iterations": controlled_iters,
        "success": controlled_success,
        "residual_norms": controlled_residuals,
        "final_residual_norm": (
            controlled_residuals[-1] if controlled_residuals else 0.0
        ),
    }

    # ── 7. Controller metadata ────────────────────────────────────
    bp_failure_risk = round(
        float(prediction.get("bp_failure_risk", 0.0)), 12,
    )
    predicted_instability = bool(
        prediction.get("predicted_instability", False),
    )
    spectral_instability_ratio = round(
        float(prediction.get("spectral_instability_ratio", 0.0)), 12,
    )

    controller_metadata = {
        "base_damping": base_damping,
        "alpha": alpha,
        "damping_min": damping_min,
        "damping_max": damping_max,
        "risk_threshold": risk_threshold,
        "schedule_threshold_fraction": schedule_threshold_fraction,
        "spectral_instability_ratio": spectral_instability_ratio,
        "num_high_risk_nodes": len(high_risk_nodes),
    }

    return {
        "control_mode": control_mode,
        "predicted_instability": predicted_instability,
        "bp_failure_risk": bp_failure_risk,
        "adaptive_damping_enabled": adaptive_damping_enabled,
        "scheduled_high_risk_nodes": high_risk_nodes,
        "baseline_metrics": baseline_metrics,
        "controlled_metrics": controlled_metrics,
        "delta_iterations": delta_iterations,
        "delta_success": delta_success,
        "controller_metadata": controller_metadata,
        "node_risk_scores": node_risk_scores,
        "cluster_risk_scores": risk_result.get("cluster_risk_scores", []),
        "top_risk_clusters": risk_result.get("top_risk_clusters", []),
    }
