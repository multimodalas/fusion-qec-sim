"""
v5.7.0 — Ternary Decoder Topology Classifier.

Classifies BP decoding trajectories into a ternary topological state
in observable decoder space:

    +1  stable success basin
     0  boundary / metastable region
    -1  failure basin

Operates on phase-space diagnostics from ``compute_bp_phase_space``
and optionally integrates evidence from existing v5 diagnostics
(alignment, boundary, barrier).

Does not modify decoder internals.  Treats the decoder as a pure
function.  All outputs are JSON-serializable.
"""

from __future__ import annotations

from typing import Any


def compute_ternary_decoder_topology(
    phase_space_result: dict[str, Any],
    syndrome_residuals: list[int] | list[float] | None = None,
    alignment_result: dict[str, Any] | None = None,
    boundary_result: dict[str, Any] | None = None,
    barrier_result: dict[str, Any] | None = None,
    residual_epsilon: float = 1e-6,
    oscillation_epsilon: float = 1e-3,
    boundary_epsilon: float = 1e-3,
    alignment_threshold: float = 0.7,
) -> dict[str, Any]:
    """Classify a BP trajectory into ternary topological states.

    Parameters
    ----------
    phase_space_result : dict
        Output from ``compute_bp_phase_space``.
    syndrome_residuals : list[int] | list[float] | None
        Per-iteration syndrome residual weights (0 = satisfied).
    alignment_result : dict | None
        v5.5 spectral-boundary alignment result.
    boundary_result : dict | None
        v5.3 boundary analysis result.
    barrier_result : dict | None
        v5.1 barrier analysis result.
    residual_epsilon : float
        Threshold for residual norm convergence.
    oscillation_epsilon : float
        Threshold for oscillation significance.
    boundary_epsilon : float
        Threshold for boundary proximity.
    alignment_threshold : float
        Threshold for strong spectral alignment.

    Returns
    -------
    dict[str, Any]
        JSON-serializable ternary topology classification.

    Raises
    ------
    ValueError
        If phase_space_result is missing required keys.
    """
    # ── Extract phase-space evidence ─────────────────────────────────
    residual_norms = phase_space_result.get("residual_norms", [])
    oscillation_score = phase_space_result.get("oscillation_score", 0.0)
    trajectory_length = phase_space_result.get("trajectory_length", 0)

    if trajectory_length == 0:
        raise ValueError("phase_space_result has zero trajectory_length")

    residual_norm_final = float(residual_norms[-1]) if residual_norms else 0.0

    # ── Extract optional evidence ────────────────────────────────────
    alignment_max_final: float | None = None
    if alignment_result is not None:
        alignment_max_final = float(alignment_result.get("max_alignment", 0.0))

    boundary_eps_final: float | None = None
    if boundary_result is not None:
        boundary_eps_final = float(boundary_result.get("boundary_eps", 0.0)) \
            if boundary_result.get("boundary_eps") is not None else None

    barrier_eps_final: float | None = None
    if barrier_result is not None:
        barrier_eps_final = float(barrier_result.get("barrier_eps", 0.0)) \
            if barrier_result.get("barrier_eps") is not None else None

    syndrome_residual_final: float | int | None = None
    if syndrome_residuals is not None and len(syndrome_residuals) > 0:
        syndrome_residual_final = syndrome_residuals[-1]
        if isinstance(syndrome_residual_final, (int, float)):
            pass
        else:
            syndrome_residual_final = int(syndrome_residual_final)

    # ── Per-iteration ternary trace ──────────────────────────────────
    ternary_trace: list[int] = []
    for t in range(trajectory_length):
        state = _classify_iteration(
            t=t,
            residual_norms=residual_norms,
            syndrome_residuals=syndrome_residuals,
            residual_epsilon=residual_epsilon,
            oscillation_epsilon=oscillation_epsilon,
        )
        ternary_trace.append(state)

    # ── Final ternary state ──────────────────────────────────────────
    final_state, reason = _classify_final(
        residual_norm_final=residual_norm_final,
        oscillation_score=oscillation_score,
        syndrome_residual_final=syndrome_residual_final,
        alignment_max_final=alignment_max_final,
        boundary_eps_final=boundary_eps_final,
        barrier_eps_final=barrier_eps_final,
        residual_epsilon=residual_epsilon,
        oscillation_epsilon=oscillation_epsilon,
        boundary_epsilon=boundary_epsilon,
        alignment_threshold=alignment_threshold,
    )

    # ── State durations ──────────────────────────────────────────────
    state_durations = {
        "positive": sum(1 for s in ternary_trace if s == 1),
        "zero": sum(1 for s in ternary_trace if s == 0),
        "negative": sum(1 for s in ternary_trace if s == -1),
    }

    # ── Transition iteration ─────────────────────────────────────────
    # First iteration where ternary state matches the final state
    # and remains there for all subsequent iterations.
    transition_iteration: int | None = None
    if ternary_trace:
        for t in range(len(ternary_trace)):
            if all(s == final_state for s in ternary_trace[t:]):
                transition_iteration = t
                break

    return {
        "ternary_trace": ternary_trace,
        "final_ternary_state": final_state,
        "state_durations": state_durations,
        "transition_iteration": transition_iteration,
        "classification_reason": reason,
        "evidence": {
            "residual_norm_final": residual_norm_final,
            "oscillation_score": oscillation_score,
            "alignment_max_final": alignment_max_final,
            "boundary_eps_final": boundary_eps_final,
            "barrier_eps_final": barrier_eps_final,
            "syndrome_residual_final": syndrome_residual_final,
        },
    }


def _classify_iteration(
    t: int,
    residual_norms: list[float],
    syndrome_residuals: list[int] | list[float] | None,
    residual_epsilon: float,
    oscillation_epsilon: float,
) -> int:
    """Classify a single iteration into ternary state.

    Uses local residual behavior and syndrome information.
    """
    # If we have a syndrome residual for this iteration, use it.
    if syndrome_residuals is not None and t < len(syndrome_residuals):
        sr = syndrome_residuals[t]
        if sr == 0:
            # Syndrome satisfied — check residual convergence.
            if t < len(residual_norms) and residual_norms[t] < residual_epsilon:
                return 1  # converged to success
            # Syndrome zero but still moving — boundary.
            return 0
        else:
            # Syndrome not satisfied.
            if t < len(residual_norms) and residual_norms[t] < residual_epsilon:
                return -1  # converged to failure
            return 0  # still evolving

    # No syndrome information — use residual norms only.
    if t < len(residual_norms):
        if residual_norms[t] < residual_epsilon:
            return 0  # converged but unknown basin without syndrome
        return 0  # still evolving
    # Last iteration (no residual beyond).
    return 0


def _classify_final(
    residual_norm_final: float,
    oscillation_score: float,
    syndrome_residual_final: float | int | None,
    alignment_max_final: float | None,
    boundary_eps_final: float | None,
    barrier_eps_final: float | None,
    residual_epsilon: float,
    oscillation_epsilon: float,
    boundary_epsilon: float,
    alignment_threshold: float,
) -> tuple[int, str]:
    """Classify the final ternary state with a reason string."""
    converged = residual_norm_final < residual_epsilon
    low_oscillation = oscillation_score < oscillation_epsilon
    syndrome_zero = (
        syndrome_residual_final is not None and syndrome_residual_final == 0
    )
    syndrome_nonzero = (
        syndrome_residual_final is not None and syndrome_residual_final != 0
    )
    near_boundary = (
        boundary_eps_final is not None and boundary_eps_final < boundary_epsilon
    )
    strong_alignment = (
        alignment_max_final is not None
        and alignment_max_final >= alignment_threshold
    )

    # ── +1: Stable success basin ─────────────────────────────────────
    if syndrome_zero and converged and low_oscillation and not near_boundary:
        return 1, (
            "stable success: syndrome satisfied, residual converged, "
            "low oscillation, not near boundary"
        )
    if syndrome_zero and converged and low_oscillation:
        return 1, (
            "stable success: syndrome satisfied, residual converged, "
            "low oscillation"
        )
    if syndrome_zero and (converged or low_oscillation):
        return 1, "stable success: syndrome satisfied, convergence indicators positive"

    # ── -1: Failure basin ────────────────────────────────────────────
    if syndrome_nonzero and converged and low_oscillation:
        return -1, (
            "failure basin: syndrome unsatisfied, residual converged, "
            "low oscillation"
        )
    if syndrome_nonzero and converged and strong_alignment:
        return -1, (
            "failure basin: syndrome unsatisfied, residual converged, "
            "strong spectral alignment"
        )
    if syndrome_nonzero and converged:
        return -1, "failure basin: syndrome unsatisfied, residual converged"

    # ── 0: Boundary / metastable ─────────────────────────────────────
    if not converged and not low_oscillation:
        return 0, (
            "boundary/metastable: residual not converged, "
            "significant oscillation"
        )
    if near_boundary:
        return 0, "boundary/metastable: near decision boundary"
    if not converged:
        return 0, "boundary/metastable: residual not converged"
    if not low_oscillation:
        return 0, "boundary/metastable: significant oscillation"

    # ── Fallback with syndrome unknown ───────────────────────────────
    if syndrome_residual_final is None:
        if converged and low_oscillation:
            return 0, (
                "boundary/metastable: converged but syndrome information "
                "unavailable for basin assignment"
            )
        return 0, "boundary/metastable: insufficient evidence for classification"

    # Syndrome available but doesn't match clear patterns.
    return 0, "boundary/metastable: ambiguous convergence signature"
