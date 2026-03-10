"""
v6.5.0 — Risk-Guided Perturbation Experiment.

Tests whether applying a deterministic perturbation to high-risk nodes
when BP stalls can help escape local minima.

Algorithm:
  1. Run BP until convergence or stall detection.
  2. Stall = residual norm change < epsilon for N consecutive iterations.
  3. If stall detected, apply deterministic perturbation to high-risk
     node LLRs:  llr[node] += perturbation_strength * sign(llr[node])
  4. Continue decoding after perturbation.

The experiment uses a minimal flooding BP loop that supports stall
detection and mid-decode perturbation.  The existing bp_decode function
is not modified.

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


def _experimental_bp_with_perturbation(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    max_iters: int,
    damping: float,
    high_risk_nodes: list[int],
    perturbation_strength: float,
    stall_window: int,
    stall_epsilon: float,
) -> dict[str, Any]:
    """Minimal flooding BP with stall-triggered perturbation.

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
    damping : float
        Scalar damping factor in [0, 1).
    high_risk_nodes : list[int]
        Node indices for perturbation targeting.
    perturbation_strength : float
        Perturbation magnitude.  If <= 0, auto-computed as
        0.1 * mean(|llr|).
    stall_window : int
        Number of consecutive iterations with small residual change
        to trigger stall detection.
    stall_epsilon : float
        Threshold for residual norm change to count as "stalled".

    Returns
    -------
    dict[str, Any]
        Result dictionary with decode outcome and perturbation info.
    """
    H_f = H.astype(np.float64)
    m, n = H_f.shape
    llr_current = np.asarray(llr, dtype=np.float64).copy()

    # Sign adjustment for syndrome.
    s_sign = np.where(syndrome_vec.astype(np.float64) > 0.5, -1.0, 1.0)

    # Auto-compute perturbation strength.
    if perturbation_strength <= 0.0:
        mean_abs_llr = float(np.mean(np.abs(llr_current)))
        perturbation_strength = 0.1 * mean_abs_llr if mean_abs_llr > 0.0 else 0.1

    # Initialize v2c messages.
    v2c = np.zeros((m, n), dtype=np.float64)
    for j in range(m):
        for i in range(n):
            if H_f[j, i] > 0.5:
                v2c[j, i] = llr_current[i]

    c2v = np.zeros((m, n), dtype=np.float64)
    residual_norms: list[float] = []
    stall_detected = False
    perturbation_applied = False
    perturbation_iteration = -1
    stall_counter = 0

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

        # Variable-to-check update with scalar damping.
        prev_v2c = v2c.copy()
        for i in range(n):
            check_neighbors = [j for j in range(m) if H_f[j, i] > 0.5]
            total = llr_current[i] + sum(c2v[j, i] for j in check_neighbors)
            for j in check_neighbors:
                new_msg = total - c2v[j, i]
                v2c[j, i] = damping * prev_v2c[j, i] + (1.0 - damping) * new_msg

        # Total LLR and hard decision.
        L_total = llr_current.copy()
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
            return {
                "correction": correction,
                "iterations": iteration,
                "success": True,
                "residual_norms": residual_norms,
                "stall_detected": stall_detected,
                "perturbation_applied": perturbation_applied,
                "perturbation_iteration": perturbation_iteration,
                "iterations_after_perturbation": (
                    iteration - perturbation_iteration
                    if perturbation_applied else 0
                ),
            }

        # Stall detection (only if perturbation not yet applied).
        if not perturbation_applied and len(residual_norms) >= 2:
            change = abs(residual_norms[-1] - residual_norms[-2])
            if change < stall_epsilon:
                stall_counter += 1
            else:
                stall_counter = 0

            if stall_counter >= stall_window:
                stall_detected = True
                # Apply deterministic perturbation to high-risk nodes.
                if high_risk_nodes:
                    perturbation_applied = True
                    perturbation_iteration = iteration
                    for node_idx in high_risk_nodes:
                        if node_idx < n:
                            sign = 1.0 if llr_current[node_idx] >= 0.0 else -1.0
                            llr_current[node_idx] += (
                                perturbation_strength * sign
                            )
                    # Re-initialize v2c from perturbed LLRs.
                    for j in range(m):
                        for i in range(n):
                            if H_f[j, i] > 0.5:
                                v2c[j, i] = llr_current[i]

    return {
        "correction": correction,
        "iterations": max_iters,
        "success": False,
        "residual_norms": residual_norms,
        "stall_detected": stall_detected,
        "perturbation_applied": perturbation_applied,
        "perturbation_iteration": perturbation_iteration,
        "iterations_after_perturbation": (
            max_iters - perturbation_iteration
            if perturbation_applied else 0
        ),
    }


def run_risk_guided_perturbation(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    risk_result: dict[str, Any],
    *,
    base_damping: float = 0.0,
    perturbation_strength: float = 0.0,
    threshold_fraction: float = 0.5,
    stall_window: int = 3,
    stall_epsilon: float = 1e-6,
    max_iters: int = 100,
) -> dict[str, Any]:
    """Run a risk-guided perturbation experiment.

    Performs two deterministic decodes:
      1. Baseline: standard BP without perturbation.
      2. Experimental: BP with stall-triggered perturbation on
         high-risk nodes.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    llr : np.ndarray
        Per-variable log-likelihood ratios, length n.
    syndrome_vec : np.ndarray
        Binary syndrome vector, length m.
    risk_result : dict[str, Any]
        Output of ``compute_spectral_failure_risk()``.
    base_damping : float
        Scalar damping factor.  Default 0.0.
    perturbation_strength : float
        Perturbation magnitude.  If <= 0 (default), auto-computed as
        0.1 * mean(|llr|).
    threshold_fraction : float
        Fraction of max risk for high-risk threshold.  Default 0.5.
    stall_window : int
        Consecutive stalled iterations before perturbation.  Default 3.
    stall_epsilon : float
        Residual change threshold for stall detection.  Default 1e-6.
    max_iters : int
        Maximum BP iterations.  Default 100.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``high_risk_nodes``: list of high-risk node indices.
        - ``baseline_metrics``: dict with baseline decode results.
        - ``experiment_metrics``: dict with perturbation decode results.
        - ``stall_detected``: bool, whether stall was detected.
        - ``perturbation_applied``: bool, whether perturbation fired.
        - ``delta_iterations``: int, iteration difference.
        - ``delta_success``: int, success difference.
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

    # ── Baseline decode (no perturbation) ────────────────────────
    baseline_result = _experimental_bp_with_perturbation(
        H, llr, syndrome_vec, max_iters, base_damping,
        high_risk_nodes=[],  # No perturbation targets.
        perturbation_strength=0.0,
        stall_window=stall_window,
        stall_epsilon=stall_epsilon,
    )

    # ── Experimental decode (with perturbation) ──────────────────
    experiment_result = _experimental_bp_with_perturbation(
        H, llr, syndrome_vec, max_iters, base_damping,
        high_risk_nodes=high_risk_nodes,
        perturbation_strength=perturbation_strength,
        stall_window=stall_window,
        stall_epsilon=stall_epsilon,
    )

    # ── Build output ─────────────────────────────────────────────
    baseline_metrics = {
        "iterations": baseline_result["iterations"],
        "success": baseline_result["success"],
        "residual_norms": baseline_result["residual_norms"],
        "final_residual_norm": (
            baseline_result["residual_norms"][-1]
            if baseline_result["residual_norms"] else 0.0
        ),
    }

    experiment_metrics = {
        "iterations": experiment_result["iterations"],
        "success": experiment_result["success"],
        "residual_norms": experiment_result["residual_norms"],
        "final_residual_norm": (
            experiment_result["residual_norms"][-1]
            if experiment_result["residual_norms"] else 0.0
        ),
        "stall_detected": experiment_result["stall_detected"],
        "perturbation_applied": experiment_result["perturbation_applied"],
        "perturbation_iteration": experiment_result["perturbation_iteration"],
        "iterations_after_perturbation": (
            experiment_result["iterations_after_perturbation"]
        ),
    }

    delta_iterations = (
        experiment_result["iterations"] - baseline_result["iterations"]
    )
    delta_success = (
        int(experiment_result["success"]) - int(baseline_result["success"])
    )

    return {
        "high_risk_nodes": high_risk_nodes,
        "baseline_metrics": baseline_metrics,
        "experiment_metrics": experiment_metrics,
        "stall_detected": experiment_result["stall_detected"],
        "perturbation_applied": experiment_result["perturbation_applied"],
        "delta_iterations": delta_iterations,
        "delta_success": delta_success,
        "node_risk_scores": node_risk_scores,
        "cluster_risk_scores": risk_result.get("cluster_risk_scores", []),
        "top_risk_clusters": risk_result.get("top_risk_clusters", []),
    }
