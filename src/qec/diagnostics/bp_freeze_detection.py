"""
Deterministic BP freeze detection (v4.7.0).

Detects early metastability / frozen BP dynamics by computing a composite
freeze score from BP dynamics metrics over a sliding window.

Operates post-decode only.  Does not modify BP decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
No use of Python ``hash()`` (salted per process; forbidden).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .bp_dynamics import compute_bp_dynamics_metrics, classify_bp_regime


# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_WINDOW: int = 12
DEFAULT_FREEZE_THRESHOLD: float = 0.85


# ── Public API ───────────────────────────────────────────────────────


def compute_bp_freeze_detection(
    llr_trace: list,
    energy_trace: list,
    *,
    window: int = DEFAULT_WINDOW,
    freeze_threshold: float = DEFAULT_FREEZE_THRESHOLD,
) -> dict:
    """Detect early metastability / frozen BP dynamics.

    Computes a composite freeze score from BP dynamics metrics over a
    sliding window.  Freeze is declared when the score exceeds the
    threshold AND the regime is ``"metastable_state"``.

    Parameters
    ----------
    llr_trace : list
        Per-iteration LLR vectors.
    energy_trace : list
        Per-iteration energy values.
    window : int
        Sliding window size (default 12).
    freeze_threshold : float
        Score threshold for freeze declaration (default 0.85).

    Returns
    -------
    dict with keys:
        ``freeze_detected`` (bool),
        ``freeze_iteration`` (int or None),
        ``freeze_score`` (float),
        ``freeze_regime`` (str or None).
    All values are JSON-serializable.
    """
    # Validate trace lengths explicitly to avoid silent misalignment.
    n_energy = len(energy_trace)
    n_llr = len(llr_trace)
    if n_energy != n_llr:
        raise ValueError(
            "llr_trace and energy_trace must have equal length "
            f"(got {n_llr} and {n_energy})"
        )
    T = n_energy

    # Edge case: insufficient data.
    if T < 2:
        return {
            "freeze_detected": False,
            "freeze_iteration": None,
            "freeze_score": 0.0,
            "freeze_regime": None,
        }

    best_score: float = 0.0
    freeze_iteration: Optional[int] = None
    freeze_regime: Optional[str] = None

    # Slide window across iterations: evaluate at each endpoint.
    for end in range(2, T + 1):
        start = max(0, end - window)
        w_llr = llr_trace[start:end]
        w_energy = energy_trace[start:end]

        if len(w_energy) < 2 or len(w_llr) < 2:
            continue

        # Compute metrics on the current window.
        dynamics = compute_bp_dynamics_metrics(
            llr_trace=w_llr,
            energy_trace=w_energy,
            correction_vectors=None,
        )

        metrics = dynamics["metrics"]
        regime = dynamics["regime"]

        # Extract component scores.
        msi = float(metrics.get("msi", 0.0) or 0.0)
        eds_desc = float(metrics.get("eds_descent_fraction", 1.0) or 1.0)
        gos = float(metrics.get("gos", 0.0) or 0.0)
        cpi_strength = float(metrics.get("cpi_strength", 0.0) or 0.0)

        # Composite freeze score.
        score = (
            0.4 * msi
            + 0.3 * (1.0 - eds_desc)
            + 0.2 * (1.0 - gos)
            + 0.1 * cpi_strength
        )
        score = float(max(0.0, min(1.0, score)))

        # Track best score across all windows.
        if score > best_score:
            best_score = score

        # Freeze condition: score exceeds threshold AND metastable regime.
        if (
            score > freeze_threshold
            and regime == "metastable_state"
            and freeze_iteration is None
        ):
            freeze_iteration = end - 1  # 0-indexed iteration
            freeze_regime = regime

    return {
        "freeze_detected": freeze_iteration is not None,
        "freeze_iteration": freeze_iteration,
        "freeze_score": float(best_score),
        "freeze_regime": freeze_regime,
    }
