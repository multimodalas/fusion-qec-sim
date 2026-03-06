"""
Deterministic BP fixed-point trap analysis (v4.8.0).

Classifies BP decoding outcomes as:
  - correct_fixed_point:   syndrome weight zero, BP converged
  - incorrect_fixed_point: syndrome weight nonzero, BP converged
  - degenerate_fixed_point: symmetric attractor detected (low LLR entropy
    or nearly uniform LLR magnitudes)
  - no_convergence:        energy trace not stabilized

Operates post-decode only.  Does not modify BP decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
No use of Python ``hash()`` (salted per process; forbidden).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_ENERGY_STABILITY_WINDOW: int = 5
DEFAULT_ENERGY_STABILITY_RTOL: float = 1e-6
DEFAULT_ENTROPY_THRESHOLD: float = 0.1
DEFAULT_VARIANCE_THRESHOLD: float = 1e-6


# ── Internal helpers ─────────────────────────────────────────────────


def _check_energy_converged(
    energy_trace: list,
    window: int = DEFAULT_ENERGY_STABILITY_WINDOW,
    rtol: float = DEFAULT_ENERGY_STABILITY_RTOL,
) -> tuple:
    """Check whether energy has stabilized in the tail of the trace.

    Returns (converged: bool, iterations_to_fixed_point: int).
    """
    T = len(energy_trace)
    if T < 2:
        return False, T

    # Check tail window for stability.
    tail_size = min(window, T)
    tail = energy_trace[-tail_size:]

    # Energy is stabilized if all values in the tail are within rtol
    # of the final value.
    final = tail[-1]
    ref = abs(final) if final != 0.0 else 1.0
    stabilized = all(
        abs(float(v) - float(final)) <= rtol * ref for v in tail
    )

    if not stabilized:
        return False, T

    # Find the first iteration where energy stabilized.
    # Walk backwards from the end to find where stability begins.
    stable_start = T - 1
    for i in range(T - 2, -1, -1):
        if abs(float(energy_trace[i]) - float(final)) <= rtol * ref:
            stable_start = i
        else:
            break

    return True, stable_start


def _compute_llr_entropy(llr_vec: np.ndarray) -> float:
    """Compute normalized entropy of LLR magnitude distribution.

    Uses histogram-based probability estimation with deterministic bins.
    Returns a value in [0, 1] where 0 means all magnitudes identical
    and 1 means maximally spread.
    """
    magnitudes = np.abs(np.asarray(llr_vec, dtype=np.float64).ravel())
    n = len(magnitudes)
    if n == 0:
        return 0.0

    # Use fixed bin count for determinism.
    n_bins = min(max(10, n // 5), 50)
    mag_min = float(np.min(magnitudes))
    mag_max = float(np.max(magnitudes))

    if mag_max - mag_min < 1e-15:
        # All magnitudes identical → zero entropy.
        return 0.0

    # Deterministic histogram with fixed edges.
    edges = np.linspace(mag_min, mag_max, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n):
        # Find bin index deterministically.
        idx = int((magnitudes[i] - mag_min) / (mag_max - mag_min) * n_bins)
        idx = min(idx, n_bins - 1)
        counts[idx] += 1

    # Compute Shannon entropy.
    probs = counts[counts > 0].astype(np.float64) / float(n)
    entropy = -float(np.sum(probs * np.log2(probs)))

    # Normalize by max possible entropy.
    max_entropy = np.log2(float(n_bins))
    if max_entropy < 1e-15:
        return 0.0

    return float(entropy / max_entropy)


def _compute_llr_variance(llr_vec: np.ndarray) -> float:
    """Compute variance of LLR magnitudes."""
    magnitudes = np.abs(np.asarray(llr_vec, dtype=np.float64).ravel())
    if len(magnitudes) == 0:
        return 0.0
    return float(np.var(magnitudes))


# ── Public API ───────────────────────────────────────────────────────


def compute_bp_fixed_point_analysis(
    llr_trace: list,
    energy_trace: list,
    syndrome_trace: list,
    final_syndrome_weight: int,
    *,
    energy_stability_window: int = DEFAULT_ENERGY_STABILITY_WINDOW,
    energy_stability_rtol: float = DEFAULT_ENERGY_STABILITY_RTOL,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> dict:
    """Classify BP decoding outcome as a fixed-point type.

    Parameters
    ----------
    llr_trace : list
        Per-iteration LLR vectors (list of arrays).
    energy_trace : list
        Per-iteration energy values.
    syndrome_trace : list
        Per-iteration syndrome weight values.
    final_syndrome_weight : int
        Final syndrome weight after decoding.
    energy_stability_window : int
        Number of tail iterations to check for energy stability.
    energy_stability_rtol : float
        Relative tolerance for energy stability.
    entropy_threshold : float
        LLR entropy below this value indicates degenerate symmetry.
    variance_threshold : float
        LLR magnitude variance below this value indicates degenerate symmetry.

    Returns
    -------
    dict with keys:
        ``fixed_point_type`` (str),
        ``converged`` (bool),
        ``iterations_to_fixed_point`` (int),
        ``final_syndrome_weight`` (int),
        ``llr_entropy`` (float),
        ``llr_variance`` (float).
    All values are JSON-serializable.  Keys are sorted.
    """
    # Validate inputs.
    n_llr = len(llr_trace)
    n_energy = len(energy_trace)
    n_syndrome = len(syndrome_trace)
    if n_llr != n_energy:
        raise ValueError(
            "llr_trace and energy_trace must have equal length "
            f"(got {n_llr} and {n_energy})"
        )
    if n_llr != n_syndrome:
        raise ValueError(
            "llr_trace and syndrome_trace must have equal length "
            f"(got {n_llr} and {n_syndrome})"
        )

    # Edge case: insufficient data.
    if n_llr < 1:
        return {
            "converged": False,
            "final_syndrome_weight": int(final_syndrome_weight),
            "fixed_point_type": "no_convergence",
            "iterations_to_fixed_point": 0,
            "llr_entropy": 0.0,
            "llr_variance": 0.0,
        }

    # Check energy convergence.
    converged, iters_to_fp = _check_energy_converged(
        energy_trace,
        window=energy_stability_window,
        rtol=energy_stability_rtol,
    )

    # Compute LLR statistics from the final iteration.
    final_llr = np.asarray(llr_trace[-1], dtype=np.float64).ravel()
    llr_entropy = _compute_llr_entropy(final_llr)
    llr_variance = _compute_llr_variance(final_llr)

    # Classification logic (deterministic, first-match).
    if not converged:
        fixed_point_type = "no_convergence"
    elif llr_entropy < entropy_threshold or llr_variance < variance_threshold:
        fixed_point_type = "degenerate_fixed_point"
    elif int(final_syndrome_weight) == 0:
        fixed_point_type = "correct_fixed_point"
    else:
        fixed_point_type = "incorrect_fixed_point"

    # Return with sorted keys for canonical ordering.
    return {
        "converged": bool(converged),
        "final_syndrome_weight": int(final_syndrome_weight),
        "fixed_point_type": str(fixed_point_type),
        "iterations_to_fixed_point": int(iters_to_fp),
        "llr_entropy": float(llr_entropy),
        "llr_variance": float(llr_variance),
    }
