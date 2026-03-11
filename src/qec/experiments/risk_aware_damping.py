"""
v6.5.0 — Risk-Aware Damping Experiment.

Tests whether structural risk signals (v6.4 node_risk_scores) can guide
per-node damping to improve BP convergence.

Algorithm:
  1. Identify high-risk nodes: nodes where node_risk_score >= 0.5 * max.
  2. Run baseline BP decode (standard scalar damping).
  3. Run experimental BP decode with elevated damping for high-risk nodes.
  4. Compare iterations, success, and residual norms.

The experimental decode uses a minimal flooding BP loop that supports
per-node damping.  This is a self-contained research implementation —
the existing bp_decode function is not modified.

Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _identify_high_risk_nodes(
    node_risk_scores: list[list[int | float]],
    *,
    threshold_fraction: float = 0.5,
) -> list[int]:
    """Return node indices with risk >= threshold_fraction * max_risk.

    Parameters
    ----------
    node_risk_scores : list[list[int | float]]
        Output of ``compute_spectral_failure_risk()["node_risk_scores"]``.
        Each element is ``[node_index, score]``.
    threshold_fraction : float
        Fraction of max risk used as threshold.  Default 0.5.

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


def _compute_syndrome(H: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute binary syndrome s = H @ x (mod 2)."""
    return (H.astype(np.int32) @ x.astype(np.int32)) % 2


def _residual_norm(H: np.ndarray, correction: np.ndarray,
                   syndrome_vec: np.ndarray) -> float:
    """Compute L2 norm of syndrome residual."""
    s = _compute_syndrome(H, correction)
    return float(np.linalg.norm(s.astype(np.float64) - syndrome_vec.astype(np.float64)))


def _experimental_bp_flooding(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    max_iters: int,
    per_node_damping: np.ndarray,
) -> tuple[np.ndarray, int, list[float]]:
    """Minimal flooding BP with per-node damping for experiment.

    This is a self-contained experimental implementation used only
    for research comparison.  It does not modify the existing decoder.

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
        Per-variable damping factors, length n.  Values in [0, 1).

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

    # Sign adjustment for syndrome: s_j = 1 → flip sign convention.
    s_sign = np.where(syndrome_vec.astype(np.float64) > 0.5, -1.0, 1.0)

    # Initialize messages: variable-to-check = llr.
    v2c = np.zeros((m, n), dtype=np.float64)
    for j in range(m):
        for i in range(n):
            if H_f[j, i] > 0.5:
                v2c[j, i] = llr[i]

    c2v = np.zeros((m, n), dtype=np.float64)
    residual_norms: list[float] = []

    for iteration in range(1, max_iters + 1):
        # Check-to-variable: sum-product (tanh rule).
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

        # Variable-to-check update with per-node damping.
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

        # Convergence check.
        if np.array_equal(s_residual, syndrome_vec.astype(np.uint8)):
            return correction, iteration, residual_norms

    return correction, max_iters, residual_norms


def run_risk_aware_damping_experiment(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    risk_result: dict[str, Any],
    *,
    base_damping: float = 0.0,
    risk_damping_multiplier: float = 1.5,
    threshold_fraction: float = 0.5,
    max_iters: int = 100,
    bp_decode_fn: Any = None,
) -> dict[str, Any]:
    """Run a risk-aware damping experiment.

    Performs two deterministic decodes:
      1. Baseline: standard scalar damping.
      2. Risk-aware: elevated damping for high-risk nodes.

    Compares iterations to convergence, decoder success, and residual
    norms.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    llr : np.ndarray
        Per-variable log-likelihood ratios, length n.
    syndrome_vec : np.ndarray
        Binary syndrome vector, length m.
    risk_result : dict[str, Any]
        Output of ``compute_spectral_failure_risk()``.  Must contain
        ``node_risk_scores``, ``cluster_risk_scores``, and
        ``top_risk_clusters``.
    base_damping : float
        Scalar damping for baseline decode.  Default 0.0.
    risk_damping_multiplier : float
        Multiplier for high-risk node damping.  Default 1.5.
    threshold_fraction : float
        Fraction of max risk for high-risk threshold.  Default 0.5.
    max_iters : int
        Maximum BP iterations.  Default 100.
    bp_decode_fn : callable, optional
        Decoder function for baseline decode.  If None, uses the
        internal experimental BP implementation for both decodes
        to ensure a fair comparison.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``high_risk_nodes``: list of high-risk node indices.
        - ``base_damping``: float, baseline damping factor.
        - ``risk_damping``: float, damping applied to high-risk nodes.
        - ``baseline_metrics``: dict with baseline decode results.
        - ``experiment_metrics``: dict with risk-aware decode results.
        - ``delta_iterations``: int, iteration difference.
        - ``delta_success``: int, success difference (1, 0, or -1).
        - ``node_risk_scores``: pass-through from risk_result.
        - ``cluster_risk_scores``: pass-through from risk_result.
        - ``top_risk_clusters``: pass-through from risk_result.
    """
    n = H.shape[1]
    llr = np.asarray(llr, dtype=np.float64)
    syndrome_vec = np.asarray(syndrome_vec, dtype=np.uint8)

    node_risk_scores = risk_result.get("node_risk_scores", [])
    high_risk_nodes = _identify_high_risk_nodes(
        node_risk_scores,
        threshold_fraction=threshold_fraction,
    )

    risk_damping = min(base_damping * risk_damping_multiplier, 0.99)

    # ── Baseline decode ──────────────────────────────────────────
    baseline_damping_arr = np.full(n, base_damping, dtype=np.float64)
    baseline_correction, baseline_iters, baseline_residuals = (
        _experimental_bp_flooding(
            H, llr, syndrome_vec, max_iters, baseline_damping_arr,
        )
    )
    baseline_success = bool(np.array_equal(
        _compute_syndrome(H, baseline_correction),
        syndrome_vec,
    ))

    # ── Risk-aware decode ────────────────────────────────────────
    risk_damping_arr = np.full(n, base_damping, dtype=np.float64)
    for node_idx in high_risk_nodes:
        if node_idx < n:
            risk_damping_arr[node_idx] = risk_damping

    risk_correction, risk_iters, risk_residuals = (
        _experimental_bp_flooding(
            H, llr, syndrome_vec, max_iters, risk_damping_arr,
        )
    )
    risk_success = bool(np.array_equal(
        _compute_syndrome(H, risk_correction),
        syndrome_vec,
    ))

    # ── Comparison ───────────────────────────────────────────────
    delta_iterations = risk_iters - baseline_iters
    delta_success = int(risk_success) - int(baseline_success)

    baseline_metrics = {
        "iterations": baseline_iters,
        "success": baseline_success,
        "residual_norms": baseline_residuals,
        "final_residual_norm": baseline_residuals[-1] if baseline_residuals else 0.0,
    }

    experiment_metrics = {
        "iterations": risk_iters,
        "success": risk_success,
        "residual_norms": risk_residuals,
        "final_residual_norm": risk_residuals[-1] if risk_residuals else 0.0,
    }

    return {
        "high_risk_nodes": high_risk_nodes,
        "base_damping": base_damping,
        "risk_damping": round(risk_damping, 12),
        "baseline_metrics": baseline_metrics,
        "experiment_metrics": experiment_metrics,
        "delta_iterations": delta_iterations,
        "delta_success": delta_success,
        "node_risk_scores": node_risk_scores,
        "cluster_risk_scores": risk_result.get("cluster_risk_scores", []),
        "top_risk_clusters": risk_result.get("top_risk_clusters", []),
    }
