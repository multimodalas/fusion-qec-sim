"""
Deterministic BP boundary analysis (v5.3.0).

Estimates the distance in LLR space to the nearest competing BP attractor
basin.  This complements the v5.1 barrier analysis: barrier analysis measures
the difficulty of escaping the current basin, while boundary analysis measures
the distance to the nearest competing basin.

Operates post-decode only.  Does not modify BP decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
No use of Python ``hash()`` (salted per process; forbidden).

The decoder is treated as a pure function — diagnostics must not alter
decoder behaviour, state, or execution paths.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_PARAMS: dict[str, Any] = {
    "epsilon_max": 5.0,
    "delta": 1e-6,
    "least_reliable_k": 8,
}


# ── Internal helpers ─────────────────────────────────────────────────


def _attractors_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Compare two attractor (correction) vectors for equality."""
    return np.array_equal(a, b)


def generate_deterministic_directions(
    H: np.ndarray,
    llr_vector: np.ndarray,
    params: dict[str, Any],
) -> list[np.ndarray]:
    """Generate a deterministic, ordered list of perturbation directions.

    Direction families are appended in strict order:
    1. Parity-check hyperplane directions (normalized rows of H, ±).
    2. Least-reliable bit directions (unit vectors along weakest bits, ±).
    3. Global sign direction (sign structure of LLR, ±).

    Returns a Python list (never a set or dict) to guarantee ordering.
    """
    n = llr_vector.shape[0]
    directions: list[np.ndarray] = []

    # Family 1 — Parity-check hyperplane directions.
    for row_idx in range(H.shape[0]):
        row = H[row_idx].astype(np.float64)
        norm = np.linalg.norm(row)
        if norm > 0:
            d = row / norm
            directions.append(d)
            directions.append(-d)

    # Family 2 — Least-reliable bit directions.
    abs_llr = np.abs(llr_vector)
    sorted_indices = np.argsort(abs_llr)
    k = min(params["least_reliable_k"], n)
    for idx in range(k):
        i = int(sorted_indices[idx])
        e = np.zeros(n, dtype=np.float64)
        e[i] = 1.0
        directions.append(e)
        directions.append(-e)

    # Family 3 — Global sign direction.
    sign_vec = np.sign(llr_vector).astype(np.float64)
    norm = np.linalg.norm(sign_vec)
    if norm > 0:
        d = sign_vec / norm
        directions.append(d)
        directions.append(-d)

    return directions


# ── Public API ───────────────────────────────────────────────────────


def compute_bp_boundary_analysis(
    llr_vector: np.ndarray,
    decoder_fn: Callable[[np.ndarray], np.ndarray],
    parity_check_matrix: np.ndarray,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Estimate the distance to the nearest competing BP attractor basin.

    Performs a deterministic binary search along each direction to find the
    minimal perturbation epsilon that causes the decoder to converge to a
    different attractor.

    Parameters
    ----------
    llr_vector : np.ndarray
        Original LLR vector (baseline, unperturbed).
    decoder_fn : Callable[[np.ndarray], np.ndarray]
        A function ``decoder_fn(llr) -> correction`` that decodes the given
        LLR vector and returns the correction (attractor) vector.
        Must be deterministic.
    parity_check_matrix : np.ndarray
        The parity-check matrix H.
    params : dict, optional
        Override default parameters.  Keys:
        ``epsilon_max`` (float), ``delta`` (float),
        ``least_reliable_k`` (int).

    Returns
    -------
    dict with keys:
        ``baseline_attractor`` (list),
        ``boundary_eps`` (float or None),
        ``boundary_direction`` (list or None),
        ``boundary_crossed`` (bool),
        ``num_directions`` (int).
    All values are JSON-serializable.
    """
    merged_params: dict[str, Any] = {**DEFAULT_PARAMS, **(params or {})}

    llr_arr = np.array(llr_vector, dtype=np.float64)
    H = np.array(parity_check_matrix, dtype=np.float64)

    # Decode baseline.
    baseline_attractor = decoder_fn(np.copy(llr_arr))

    # Generate deterministic directions.
    directions = generate_deterministic_directions(H, llr_arr, merged_params)

    # Edge case: no valid directions.
    if not directions:
        return {
            "baseline_attractor": baseline_attractor.tolist(),
            "boundary_eps": None,
            "boundary_direction": None,
            "boundary_crossed": False,
            "num_directions": 0,
        }

    eps_max_param = float(merged_params["epsilon_max"])
    delta = float(merged_params["delta"])

    best_boundary_eps: Optional[float] = None
    best_boundary_direction: Optional[np.ndarray] = None
    num_directions = 0

    for d in directions:
        num_directions += 1

        # Binary search for boundary along direction d.
        eps_min = 0.0
        eps_max = eps_max_param

        # Quick check: does max perturbation cross at all?
        perturbed_llr = np.copy(llr_arr) + eps_max * d
        attractor_at_max = decoder_fn(perturbed_llr)

        if _attractors_equal(attractor_at_max, baseline_attractor):
            # No crossing along this direction within epsilon_max.
            continue

        # Binary search to find the boundary.
        while (eps_max - eps_min) > delta:
            eps_mid = (eps_min + eps_max) / 2.0
            perturbed_llr = np.copy(llr_arr) + eps_mid * d
            attractor = decoder_fn(perturbed_llr)

            if not _attractors_equal(attractor, baseline_attractor):
                eps_max = eps_mid
            else:
                eps_min = eps_mid

        boundary_eps = eps_max

        if best_boundary_eps is None or boundary_eps < best_boundary_eps:
            best_boundary_eps = boundary_eps
            best_boundary_direction = d

        # Early exit: boundary_eps <= delta means no smaller can exist.
        if best_boundary_eps <= delta:
            break

    boundary_crossed = best_boundary_eps is not None

    return {
        "baseline_attractor": baseline_attractor.tolist(),
        "boundary_eps": float(best_boundary_eps) if best_boundary_eps is not None else None,
        "boundary_direction": best_boundary_direction.tolist() if best_boundary_direction is not None else None,
        "boundary_crossed": boundary_crossed,
        "num_directions": num_directions,
    }
