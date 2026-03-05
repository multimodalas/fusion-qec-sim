"""
BP free-energy landscape diagnostics.

Operates exclusively on energy traces produced by bp_decode.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from src.qec_qldpc_codes import bp_decode, syndrome


# ── Energy Landscape Metrics ────────────────────────────────────────


def compute_energy_gradient(trace: List[float]) -> List[float]:
    """Compute first differences: ΔE_i = E_{i+1} − E_i."""
    return [trace[i + 1] - trace[i] for i in range(len(trace) - 1)]


def compute_energy_curvature(trace: List[float]) -> List[float]:
    """Compute second differences: Δ²E_i = E_{i+2} − 2E_{i+1} + E_i."""
    return [
        trace[i + 2] - 2.0 * trace[i + 1] + trace[i]
        for i in range(len(trace) - 2)
    ]


def detect_plateau(trace: List[float], tolerance: float = 1e-6) -> List[tuple]:
    """Detect plateaus where ≥ 3 consecutive data points are within tolerance.

    A run of k flat gradient steps corresponds to (k+1) plateau points.
    Threshold: ≥ 2 consecutive gradient steps (i.e. ≥ 3 data points).

    Returns list of (start_index, length) tuples where length counts
    the number of flat gradient steps.
    """
    grad = compute_energy_gradient(trace)
    plateaus = []
    run_start = None
    run_len = 0

    for i, g in enumerate(grad):
        if abs(g) < tolerance:
            if run_start is None:
                run_start = i
                run_len = 1
            else:
                run_len += 1
        else:
            if run_start is not None and run_len >= 2:
                plateaus.append((run_start, run_len))
            run_start = None
            run_len = 0

    if run_start is not None and run_len >= 2:
        plateaus.append((run_start, run_len))

    return plateaus


def detect_local_minima(trace: List[float]) -> List[int]:
    """Detect local minima: E_i < E_{i-1} and E_i < E_{i+1}.

    Returns list of indices.
    """
    minima = []
    for i in range(1, len(trace) - 1):
        if trace[i] < trace[i - 1] and trace[i] < trace[i + 1]:
            minima.append(i)
    return minima


def detect_barrier_crossings(trace: List[float]) -> List[int]:
    """Detect barrier crossings: gradient sign change (+ → −).

    Returns list of gradient indices where crossing occurs.
    """
    grad = compute_energy_gradient(trace)
    crossings = []
    for i in range(len(grad) - 1):
        if grad[i] > 0 and grad[i + 1] < 0:
            crossings.append(i + 1)
    return crossings


# ── Energy Landscape Classification ─────────────────────────────────


def classify_energy_landscape(trace: List[float]) -> dict[str, Any]:
    """Classify a BP energy trace.

    Returns
    -------
    dict with keys:
        monotonic_descent : bool
        plateau_detected : bool
        local_minima : int
        barrier_crossings : int
        final_energy : float
        iterations : int
    """
    grad = compute_energy_gradient(trace)
    plateaus = detect_plateau(trace)
    minima = detect_local_minima(trace)
    crossings = detect_barrier_crossings(trace)

    return {
        "monotonic_descent": all(g <= 0 for g in grad),
        "plateau_detected": len(plateaus) > 0,
        "local_minima": len(minima),
        "barrier_crossings": len(crossings),
        "final_energy": trace[-1] if trace else 0.0,
        "iterations": len(trace),
    }


# ── Geometry-Induced Basin Switching Detection ──────────────────────

_BASIN_EPSILON = 1e-3


def detect_basin_switch(
    H: np.ndarray,
    llr_base: np.ndarray,
    base_correction: np.ndarray,
    base_trace: list,
    max_iters: int,
    bp_mode: str,
    schedule: str,
    syndrome_vec: np.ndarray,
) -> dict[str, Any]:
    """Detect geometry-induced basin switching.

    Compares the existing baseline decode result against a small
    deterministic perturbation to detect basin switches.
    """
    llr_base = np.asarray(llr_base, dtype=np.float64)
    corr1 = base_correction
    trace1 = base_trace

    # Small deterministic perturbation
    eps = 1e-3
    sign = np.sign(llr_base)
    sign[sign == 0] = 1.0
    llr_perturbed = llr_base + eps * sign

    # Run perturbed decode
    r2 = bp_decode(
        H,
        llr_perturbed,
        max_iters=max_iters,
        mode=bp_mode,
        schedule=schedule,
        syndrome_vec=syndrome_vec,
        energy_trace=True,
    )
    corr2, _iters2 = r2[0], r2[1]
    trace2 = r2[-1]

    energy1 = trace1[-1]
    energy2 = trace2[-1]
    switch = (
        not np.array_equal(corr1, corr2)
        or abs(energy1 - energy2) > 1e-9
    )

    return {
        "switch": bool(switch),
        "energy_base": float(energy1),
        "energy_perturbed": float(energy2),
    }
