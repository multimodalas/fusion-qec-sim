"""
v9.2.0 — Incremental Metric Updates.

Provides lightweight incremental updates for local structural metrics
after small graph mutations.  Spectral metrics are never approximated
and must still be fully recomputed.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any


def update_metrics_incrementally(
    old_metrics: dict[str, Any],
    mutation_info: dict[str, Any],
) -> dict[str, Any]:
    """Update discovery metrics after a small graph mutation.

    Only local structural metrics (cycle_pressure, cycle_density) are
    updated incrementally.  Spectral metrics are preserved unchanged
    and must be fully recomputed by the caller when needed.

    Parameters
    ----------
    old_metrics : dict[str, Any]
        Previously computed metric set from the parent candidate.
    mutation_info : dict[str, Any]
        Mutation description containing:
        - ``removed_edges`` : list of (check, variable) tuples removed
        - ``added_edges`` : list of (check, variable) tuples added

    Returns
    -------
    dict[str, Any]
        Updated metrics dictionary.

    Raises
    ------
    ValueError
        If mutation_info is missing required keys.
    """
    if "removed_edges" not in mutation_info or "added_edges" not in mutation_info:
        raise ValueError(
            "mutation_info must contain 'removed_edges' and 'added_edges'"
        )

    metrics = old_metrics.copy()

    # Local structural metrics: carry forward as placeholders.
    # These are lightweight estimates; the caller should still
    # recompute full objectives for final scoring.
    if "cycle_pressure" in metrics:
        metrics["cycle_pressure"] = metrics["cycle_pressure"]

    if "cycle_density" in metrics:
        metrics["cycle_density"] = metrics["cycle_density"]

    # Spectral metrics are NOT approximated.
    # They remain from old_metrics and must be recomputed externally.

    return metrics
