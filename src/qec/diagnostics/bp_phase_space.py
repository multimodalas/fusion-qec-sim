"""
v5.8.0 — BP Phase-Space Explorer.

Deterministic, observational diagnostic that treats BP decoding as a
trajectory through an observable phase space.  Records per-iteration
observable decoder states and projects them into a reduced coordinate
system for analysis.

v5.8.0 additions:
    - compute_metastability_score(): mean absolute difference of
      residual norms over last N iterations, normalized by mean residual.

Does not modify decoder internals.  Treats the decoder as a pure
function.  All outputs are JSON-serializable.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_bp_phase_space(
    trajectory_states: list[np.ndarray],
    spectral_basis: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute phase-space diagnostics for an observable BP trajectory.

    Parameters
    ----------
    trajectory_states : list[np.ndarray]
        Ordered per-iteration observable decoder states.  Each element
        is a 1-D belief vector in the decoder's observable space.
    spectral_basis : np.ndarray | None
        Optional projection basis.  If provided, columns are
        deterministic basis vectors used for reduced-coordinate
        projection (e.g. Tanner spectral modes from v5.4).

    Returns
    -------
    dict[str, Any]
        JSON-serializable phase-space diagnostics.

    Raises
    ------
    ValueError
        If trajectory is empty or dimensionally inconsistent.
    """
    # ── Input validation ─────────────────────────────────────────────
    if not trajectory_states:
        raise ValueError("trajectory_states must not be empty")

    state_dim = len(trajectory_states[0])
    if state_dim == 0:
        raise ValueError("trajectory states must have non-zero dimension")

    for i, state in enumerate(trajectory_states):
        if len(state) != state_dim:
            raise ValueError(
                f"trajectory state {i} has dimension {len(state)}, "
                f"expected {state_dim}"
            )

    trajectory_length = len(trajectory_states)

    # ── Residual norms ───────────────────────────────────────────────
    residual_norms: list[float] = []
    for t in range(trajectory_length - 1):
        diff = np.asarray(trajectory_states[t + 1], dtype=np.float64) - \
               np.asarray(trajectory_states[t], dtype=np.float64)
        residual_norms.append(float(np.linalg.norm(diff, ord=2)))

    # ── Phase-space projection ───────────────────────────────────────
    if spectral_basis is not None:
        basis = np.asarray(spectral_basis, dtype=np.float64)
        if basis.ndim != 2:
            raise ValueError("spectral_basis must be a 2-D array")
        if basis.shape[0] < state_dim:
            raise ValueError(
                f"spectral_basis has {basis.shape[0]} rows, "
                f"expected at least {state_dim}"
            )
        # Restrict to state_dim rows (variable-node components).
        basis_restricted = basis[:state_dim, :]
        n_coords = basis_restricted.shape[1]
        phase_coordinates: list[list[float]] = []
        for state in trajectory_states:
            x = np.asarray(state, dtype=np.float64)
            coords = [
                float(np.dot(x, basis_restricted[:, k]))
                for k in range(n_coords)
            ]
            phase_coordinates.append(coords)
    else:
        # Deterministic fallback: first min(3, state_dim) standard
        # basis coordinates.
        n_coords = min(3, state_dim)
        phase_coordinates = []
        for state in trajectory_states:
            x = np.asarray(state, dtype=np.float64)
            coords = [float(x[k]) for k in range(n_coords)]
            phase_coordinates.append(coords)

    final_phase_coordinate = phase_coordinates[-1] if phase_coordinates else []

    # ── Oscillation score ────────────────────────────────────────────
    # Average sign-change rate of first differences in residual norms.
    oscillation_score = 0.0
    if len(residual_norms) >= 2:
        first_diffs = [
            residual_norms[t + 1] - residual_norms[t]
            for t in range(len(residual_norms) - 1)
        ]
        sign_changes = 0
        for t in range(len(first_diffs) - 1):
            if first_diffs[t] * first_diffs[t + 1] < 0.0:
                sign_changes += 1
        n_possible = len(first_diffs) - 1
        oscillation_score = float(sign_changes) / float(n_possible) if n_possible > 0 else 0.0

    return {
        "trajectory_length": trajectory_length,
        "state_dimension": state_dim,
        "residual_norms": residual_norms,
        "phase_coordinates": phase_coordinates,
        "final_phase_coordinate": final_phase_coordinate,
        "oscillation_score": oscillation_score,
    }


def compute_metastability_score(
    residual_norms: list[float],
    tail_length: int = 5,
) -> float:
    """Compute metastability score from residual norms.

    The metastability score is the mean absolute difference between
    consecutive residual norms over the last *tail_length* iterations,
    normalized by the mean residual over the same window.

    Parameters
    ----------
    residual_norms : list[float]
        Per-iteration residual norms (from ``compute_bp_phase_space``).
    tail_length : int
        Number of tail iterations to consider.  If the residual norms
        list is shorter than *tail_length*, the full list is used.

    Returns
    -------
    float
        Metastability score.  Low → convergence, medium → plateau
        metastability, high → oscillation.  Returns 0.0 if fewer
        than 2 residual norms are available.
    """
    if len(residual_norms) < 2:
        return 0.0

    # Restrict to tail window.
    tail = residual_norms[-tail_length:] if tail_length < len(residual_norms) else list(residual_norms)

    # Mean absolute difference of consecutive values.
    abs_diffs = [abs(tail[i + 1] - tail[i]) for i in range(len(tail) - 1)]
    mean_abs_diff = sum(abs_diffs) / float(len(abs_diffs))

    # Normalize by mean residual in the window.
    mean_residual = sum(abs(r) for r in tail) / float(len(tail))
    if mean_residual < 1e-15:
        return 0.0

    return float(mean_abs_diff / mean_residual)
