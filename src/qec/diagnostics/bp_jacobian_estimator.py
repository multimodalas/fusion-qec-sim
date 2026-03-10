"""
v6.0.0 — BP Jacobian Spectral Radius Estimator.

Estimates the dominant eigenvalue (spectral radius) of the BP Jacobian
without constructing the Jacobian explicitly.  Uses the power iteration
heuristic on consecutive LLR differences:

    delta_t = x_{t+1} - x_t
    lambda_est ≈ ||delta_{t+1}|| / ||delta_t||

The estimate is averaged over the final iterations of the LLR history
for robustness.

This is a lightweight observational diagnostic.  Does not run BP
decoding.  Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def estimate_bp_jacobian_spectral_radius(
    llr_history: np.ndarray | list,
    tail_fraction: float = 0.5,
    min_tail_iterations: int = 2,
) -> dict[str, Any]:
    """Estimate the BP Jacobian spectral radius from LLR history.

    Uses consecutive LLR differences to approximate the dominant
    eigenvalue via a power-iteration-like ratio:

        lambda_est = mean(||delta_{t+1}|| / ||delta_t||)

    averaged over the tail (final) iterations of the history.

    Parameters
    ----------
    llr_history : np.ndarray or list
        LLR values at each BP iteration.  Shape (T, N) where T is
        the number of iterations and N is the number of variable nodes.
    tail_fraction : float
        Fraction of iterations to use for averaging (from the end).
        Default 0.5.
    min_tail_iterations : int
        Minimum number of ratio samples required.  If fewer are
        available, all available ratios are used.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:
        - ``jacobian_spectral_radius_est``: float, the estimated
          spectral radius
        - ``tail_iterations_used``: int, number of ratio samples used
    """
    hist = np.asarray(llr_history, dtype=np.float64)

    if hist.ndim != 2 or hist.shape[0] < 3:
        # Need at least 3 iterations to compute 2 deltas and 1 ratio.
        return {
            "jacobian_spectral_radius_est": 0.0,
            "tail_iterations_used": 0,
        }

    T = hist.shape[0]

    # ── Compute consecutive differences ───────────────────────────
    deltas = np.diff(hist, axis=0)  # shape (T-1, N)

    # ── Compute L2 norms of each delta ────────────────────────────
    norms = np.linalg.norm(deltas, axis=1)  # shape (T-1,)

    # ── Compute ratios between consecutive norms ──────────────────
    ratios: list[float] = []
    for i in range(len(norms) - 1):
        if norms[i] > 0.0:
            ratios.append(float(norms[i + 1] / norms[i]))

    if not ratios:
        return {
            "jacobian_spectral_radius_est": 0.0,
            "tail_iterations_used": 0,
        }

    # ── Average over tail iterations ──────────────────────────────
    num_ratios = len(ratios)
    tail_count = max(min_tail_iterations, int(num_ratios * tail_fraction))
    tail_count = min(tail_count, num_ratios)

    tail_ratios = ratios[-tail_count:]
    jacobian_spectral_radius_est = sum(tail_ratios) / len(tail_ratios)

    return {
        "jacobian_spectral_radius_est": float(jacobian_spectral_radius_est),
        "tail_iterations_used": len(tail_ratios),
    }
