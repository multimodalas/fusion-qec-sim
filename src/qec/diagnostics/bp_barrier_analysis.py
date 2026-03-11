"""
Deterministic BP free-energy barrier estimation (v5.1.0).

Estimates the minimum deterministic perturbation required to escape the
current BP attractor basin.  This measures the free-energy barrier height
around BP fixed points.

Operates post-decode only.  Does not modify BP decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
No use of Python ``hash()`` (salted per process; forbidden).

The decoder is treated as a pure function — diagnostics must not alter
decoder behaviour, state, or execution paths.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .bp_fixed_point_analysis import compute_bp_fixed_point_analysis


# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_EPS_VALUES: list[float] = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]

DEFAULT_PERTURBATION_PATTERNS: list[float] = [1.0, -1.0, 2.0, -2.0]


# ── Internal helpers ─────────────────────────────────────────────────


def _classify_attractor(
    decode_fn: Callable,
    llr: np.ndarray,
) -> str:
    """Run a decode and classify the attractor via fixed-point analysis.

    Parameters
    ----------
    decode_fn : Callable
        A function ``decode_fn(llr) -> dict`` that decodes the given LLR
        vector and returns a dict containing at least:
        ``llr_trace``, ``energy_trace``, ``syndrome_trace``,
        ``final_syndrome_weight``.
    llr : np.ndarray
        LLR vector to decode.

    Returns
    -------
    str
        The fixed-point type classification.
    """
    decode_result = decode_fn(llr)
    fp_result = compute_bp_fixed_point_analysis(
        llr_trace=decode_result["llr_trace"],
        energy_trace=decode_result["energy_trace"],
        syndrome_trace=decode_result["syndrome_trace"],
        final_syndrome_weight=decode_result["final_syndrome_weight"],
    )
    return fp_result["fixed_point_type"]


# ── Public API ───────────────────────────────────────────────────────


def compute_bp_barrier_analysis(
    decode_fn: Callable,
    llr_init: np.ndarray,
    *,
    eps_values: Optional[list[float]] = None,
) -> dict:
    """Estimate the free-energy barrier around the current BP attractor.

    Measures the minimum deterministic perturbation required to escape
    the current attractor basin.  The barrier height is estimated as the
    smallest epsilon value at which the attractor classification changes.

    Parameters
    ----------
    decode_fn : Callable
        A function ``decode_fn(llr) -> dict`` that decodes the given LLR
        vector and returns a dict containing at least:
        ``llr_trace``, ``energy_trace``, ``syndrome_trace``,
        ``final_syndrome_weight``.
    llr_init : np.ndarray
        Initial LLR vector (baseline, unperturbed).
    eps_values : list[float], optional
        Perturbation magnitudes to probe.  Defaults to
        ``[1e-4, 5e-4, 1e-3, 2e-3, 5e-3]``.

    Returns
    -------
    dict with sorted keys:
        ``barrier_eps`` (float or None),
        ``baseline_attractor`` (str),
        ``escaped`` (bool),
        ``num_trials`` (int).
    All values are JSON-serializable.
    """
    if eps_values is None:
        eps_values = list(DEFAULT_EPS_VALUES)

    llr_arr = np.array(llr_init, dtype=np.float64)

    # Classify baseline attractor (unperturbed).
    baseline_attractor = _classify_attractor(decode_fn, llr_arr)

    # Deterministic perturbation patterns.
    patterns = DEFAULT_PERTURBATION_PATTERNS

    # Probe perturbations in deterministic order: eps (ascending) × patterns.
    barrier_eps: Optional[float] = None
    num_trials = 0

    for eps in eps_values:
        for pattern in patterns:
            # Explicit copy — baseline inputs never modified in-place.
            llr_perturbed = np.array(llr_arr, dtype=np.float64) + eps * pattern
            num_trials += 1

            perturbed_attractor = _classify_attractor(decode_fn, llr_perturbed)

            if perturbed_attractor != baseline_attractor:
                barrier_eps = float(eps)
                # Return with sorted keys for canonical ordering.
                return {
                    "barrier_eps": float(barrier_eps),
                    "baseline_attractor": str(baseline_attractor),
                    "escaped": True,
                    "num_trials": int(num_trials),
                }

    # No escape found across all perturbations.
    return {
        "barrier_eps": None,
        "baseline_attractor": str(baseline_attractor),
        "escaped": False,
        "num_trials": int(num_trials),
    }
