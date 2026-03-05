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


# ── Improved Basin Switch Classification (v4.1.0) ──────────────────

_ENERGY_DELTA_THRESHOLD = 1e-6
_GRADIENT_FLIP_THRESHOLD = 3


def _count_gradient_sign_flips(trace: List[float]) -> int:
    """Count sign changes in the energy gradient.

    A sign flip occurs when consecutive gradient steps change sign.
    Deterministic: identical trace → identical count.
    """
    grad = compute_energy_gradient(trace)
    if len(grad) < 2:
        return 0
    flips = 0
    for i in range(len(grad) - 1):
        if grad[i] * grad[i + 1] < 0:
            flips += 1
    return flips


def _check_convergence(trace: List[float], tolerance: float = 1e-6) -> bool:
    """Check whether the energy trace converged (final gradient near zero).

    Deterministic: identical trace → identical result.
    """
    if len(trace) < 2:
        return True
    last_grad = trace[-1] - trace[-2]
    return abs(last_grad) < tolerance


def classify_basin_switch(
    H: np.ndarray,
    llr_base: np.ndarray,
    base_correction: np.ndarray,
    base_trace: List[float],
    max_iters: int,
    bp_mode: str,
    schedule: str,
    syndrome_vec: np.ndarray,
) -> dict[str, Any]:
    """Improved basin switch classifier (v4.1.0).

    Performs three deterministic decodes (baseline, +epsilon, -epsilon)
    and classifies the result into one of four regimes:

      - ``"metastable_oscillation"``
            Baseline gradient oscillates without settling.
      - ``"shallow_sensitivity"``
            Perturbations alter trajectory but not final correction.
      - ``"true_basin_switch"``
            Perturbations move decoding into a different attractor basin.
      - ``"none"``
            All outcomes match; no basin switching detected.

    All perturbations are deterministic.  Baseline inputs are never
    modified in-place.  The decoder is treated as a pure function.

    Returns
    -------
    dict with keys:
        basin_switch_class : str
        basin_switch_evidence : dict
    """
    llr_base = np.asarray(llr_base, dtype=np.float64)
    corr_baseline = base_correction
    trace_baseline = list(base_trace)

    # Deterministic perturbation vectors.
    eps = _BASIN_EPSILON
    s = np.sign(llr_base.copy())
    s[s == 0] = 1.0

    llr_plus = llr_base + eps * s
    llr_minus = llr_base - eps * s

    # Run perturbed decodes on explicit copies.
    r_plus = bp_decode(
        H, llr_plus,
        max_iters=max_iters,
        mode=bp_mode,
        schedule=schedule,
        syndrome_vec=syndrome_vec,
        energy_trace=True,
    )
    corr_plus = r_plus[0]
    trace_plus = list(r_plus[-1])

    r_minus = bp_decode(
        H, llr_minus,
        max_iters=max_iters,
        mode=bp_mode,
        schedule=schedule,
        syndrome_vec=syndrome_vec,
        energy_trace=True,
    )
    corr_minus = r_minus[0]
    trace_minus = list(r_minus[-1])

    # ── Collect evidence ───────────────────────────────────────────
    energy_baseline = trace_baseline[-1] if trace_baseline else 0.0
    energy_plus = trace_plus[-1] if trace_plus else 0.0
    energy_minus = trace_minus[-1] if trace_minus else 0.0

    energy_delta_plus = abs(energy_plus - energy_baseline)
    energy_delta_minus = abs(energy_minus - energy_baseline)

    gradient_flip_count = _count_gradient_sign_flips(trace_baseline)

    corrections_differ_plus = not np.array_equal(corr_baseline, corr_plus)
    corrections_differ_minus = not np.array_equal(corr_baseline, corr_minus)

    converged = _check_convergence(trace_baseline)

    evidence = {
        "energy_delta_plus": float(energy_delta_plus),
        "energy_delta_minus": float(energy_delta_minus),
        "gradient_flip_count": int(gradient_flip_count),
        "corrections_differ_plus": bool(corrections_differ_plus),
        "corrections_differ_minus": bool(corrections_differ_minus),
        "converged": bool(converged),
        "energy_baseline": float(energy_baseline),
        "energy_plus": float(energy_plus),
        "energy_minus": float(energy_minus),
    }

    # ── Classification logic ───────────────────────────────────────

    # 1. Metastable Oscillation:
    #    Repeated gradient sign flips and failure to converge.
    if (gradient_flip_count >= _GRADIENT_FLIP_THRESHOLD
            and not converged):
        return {
            "basin_switch_class": "metastable_oscillation",
            "basin_switch_evidence": evidence,
        }

    # 2. True Basin Switch:
    #    Perturbations produce different final corrections AND
    #    energy difference exceeds deterministic threshold.
    if ((corrections_differ_plus or corrections_differ_minus)
            and (energy_delta_plus > _ENERGY_DELTA_THRESHOLD
                 or energy_delta_minus > _ENERGY_DELTA_THRESHOLD)):
        return {
            "basin_switch_class": "true_basin_switch",
            "basin_switch_evidence": evidence,
        }

    # 3. Shallow Sensitivity:
    #    Perturbations alter energy trajectory but final corrections
    #    are identical (or nearly identical energy).
    if (energy_delta_plus > _ENERGY_DELTA_THRESHOLD
            or energy_delta_minus > _ENERGY_DELTA_THRESHOLD):
        return {
            "basin_switch_class": "shallow_sensitivity",
            "basin_switch_evidence": evidence,
        }

    # 4. None: all outcomes match.
    return {
        "basin_switch_class": "none",
        "basin_switch_evidence": evidence,
    }
