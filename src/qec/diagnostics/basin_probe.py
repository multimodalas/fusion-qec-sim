"""
v5.8.0 — Local Basin Probe (Deterministic).

Deterministically probes the neighborhood of a decoding point by
perturbing the LLR vector along fixed directions and classifying
the resulting decode outcomes into ternary states.

Produces a deterministic local basin map.

Does not modify decoder internals.  Treats the decoder as a pure
function.  All outputs are JSON-serializable.
No random perturbations.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def probe_local_ternary_basin(
    llr_vector: np.ndarray,
    decode_function: Callable[[np.ndarray], np.ndarray],
    perturbation_scale: float,
    directions: int | list[np.ndarray] | None = None,
    *,
    syndrome_function: Callable[[np.ndarray], np.ndarray] | None = None,
    syndrome_target: np.ndarray | None = None,
    parity_check_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """Probe the local ternary basin around a decoding point.

    For each direction, perturbs the LLR vector, runs decode, and
    classifies the result into a ternary state (+1, 0, -1).

    Parameters
    ----------
    llr_vector : np.ndarray
        The LLR vector to probe around.  Not modified in-place.
    decode_function : callable
        Decoder entrypoint: ``decode_function(llr) -> correction``.
    perturbation_scale : float
        Magnitude of the perturbation (epsilon).
    directions : int | list[np.ndarray] | None
        If int, use the first ``directions`` standard basis vectors.
        If list, use the provided direction vectors.
        If None, defaults to ``min(10, len(llr_vector))`` standard
        basis vectors.
    syndrome_function : callable | None
        Optional: ``syndrome_function(correction) -> syndrome``.
        If provided with *syndrome_target*, enables ternary
        classification based on syndrome satisfaction.
    syndrome_target : np.ndarray | None
        Target syndrome for success classification.
    parity_check_matrix : np.ndarray | None
        Optional parity-check matrix.  If provided (along with
        *syndrome_target*) and *syndrome_function* is None, syndrome
        is computed as ``parity_check_matrix @ correction mod 2``.

    Returns
    -------
    dict[str, Any]
        JSON-serializable basin probe results.

    Raises
    ------
    ValueError
        If llr_vector is empty or perturbation_scale is non-positive.
    """
    llr = np.asarray(llr_vector, dtype=np.float64)
    n = len(llr)

    if n == 0:
        raise ValueError("llr_vector must not be empty")
    if perturbation_scale <= 0.0:
        raise ValueError("perturbation_scale must be positive")

    # ── Build direction set ───────────────────────────────────────
    if directions is None:
        n_dirs = min(10, n)
        dir_vectors = [_standard_basis_vector(i, n) for i in range(n_dirs)]
    elif isinstance(directions, int):
        n_dirs = min(directions, n)
        dir_vectors = [_standard_basis_vector(i, n) for i in range(n_dirs)]
    else:
        dir_vectors = [np.asarray(d, dtype=np.float64) for d in directions]
        n_dirs = len(dir_vectors)

    # ── Baseline decode ───────────────────────────────────────────
    baseline_correction = decode_function(llr.copy())

    # ── Probe each direction ──────────────────────────────────────
    probe_results: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    boundary_count = 0

    for idx, d in enumerate(dir_vectors):
        # Perturb: never mutate the original llr.
        llr_perturbed = llr.copy() + perturbation_scale * d
        perturbed_correction = decode_function(llr_perturbed)

        state = _classify_probe_result(
            correction=perturbed_correction,
            baseline_correction=baseline_correction,
            syndrome_function=syndrome_function,
            syndrome_target=syndrome_target,
            parity_check_matrix=parity_check_matrix,
        )

        probe_results.append({
            "direction": idx,
            "state": state,
        })

        if state == 1:
            success_count += 1
        elif state == -1:
            failure_count += 1
        else:
            boundary_count += 1

    total = len(probe_results) if probe_results else 1

    return {
        "probe_results": probe_results,
        "success_fraction": float(success_count) / float(total),
        "failure_fraction": float(failure_count) / float(total),
        "boundary_fraction": float(boundary_count) / float(total),
    }


def _standard_basis_vector(i: int, n: int) -> np.ndarray:
    """Return the i-th standard basis vector of dimension n."""
    e = np.zeros(n, dtype=np.float64)
    e[i] = 1.0
    return e


def _classify_probe_result(
    correction: np.ndarray,
    baseline_correction: np.ndarray,
    syndrome_function: Callable[[np.ndarray], np.ndarray] | None,
    syndrome_target: np.ndarray | None,
    parity_check_matrix: np.ndarray | None,
) -> int:
    """Classify a single probe decode result into ternary state.

    +1  syndrome satisfied (success)
     0  syndrome unknown or ambiguous (boundary)
    -1  syndrome not satisfied (failure)

    When no syndrome information is available, falls back to comparing
    the correction to the baseline correction.
    """
    # If syndrome_function is provided, use it.
    if syndrome_function is not None and syndrome_target is not None:
        s = syndrome_function(np.asarray(correction))
        if np.array_equal(np.asarray(s), np.asarray(syndrome_target)):
            return 1
        return -1

    # If parity_check_matrix is provided, compute syndrome directly.
    if parity_check_matrix is not None and syndrome_target is not None:
        H = np.asarray(parity_check_matrix)
        c = np.asarray(correction, dtype=int)
        s = H @ c % 2
        if np.array_equal(s, np.asarray(syndrome_target, dtype=int)):
            return 1
        return -1

    # Fallback: compare to baseline correction.
    if np.array_equal(
        np.asarray(correction), np.asarray(baseline_correction)
    ):
        return 1  # Same as baseline — assume same basin.
    return 0  # Different from baseline — boundary/ambiguous.
