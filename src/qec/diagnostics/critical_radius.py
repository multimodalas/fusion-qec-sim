"""
v8.2.0 — Critical Spectral Radius Estimator.

Estimates the BP critical spectral radius from a dataset of
(spectral_radius, bp_converged) observations by locating the
transition region.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any


_ROUND = 12


def estimate_critical_spectral_radius(
    dataset: list[dict[str, Any]],
) -> dict[str, Any]:
    """Estimate the critical spectral radius from convergence data.

    Sorts observations by spectral_radius and finds the midpoint
    between the largest radius with convergence and the smallest
    radius with failure.

    Parameters
    ----------
    dataset : list[dict]
        Observations with ``spectral_radius`` and ``bp_converged``.

    Returns
    -------
    dict[str, Any]
        Dictionary with ``critical_radius`` and ``transition_width``.
    """
    if not dataset:
        return {"critical_radius": 0.0, "transition_width": 0.0}

    # Sort by spectral radius
    sorted_obs = sorted(dataset, key=lambda d: d["spectral_radius"])

    # Find largest radius with convergence
    max_converged_sr = None
    for obs in sorted_obs:
        if obs["bp_converged"] == 1:
            max_converged_sr = obs["spectral_radius"]

    # Find smallest radius with failure
    min_failed_sr = None
    for obs in sorted_obs:
        if obs["bp_converged"] == 0:
            if min_failed_sr is None or obs["spectral_radius"] < min_failed_sr:
                min_failed_sr = obs["spectral_radius"]

    if max_converged_sr is None and min_failed_sr is None:
        return {"critical_radius": 0.0, "transition_width": 0.0}

    if max_converged_sr is None:
        return {
            "critical_radius": round(min_failed_sr, _ROUND),
            "transition_width": 0.0,
        }

    if min_failed_sr is None:
        return {
            "critical_radius": round(max_converged_sr, _ROUND),
            "transition_width": 0.0,
        }

    critical_radius = (max_converged_sr + min_failed_sr) / 2.0
    transition_width = abs(max_converged_sr - min_failed_sr)

    return {
        "critical_radius": round(critical_radius, _ROUND),
        "transition_width": round(transition_width, _ROUND),
    }
