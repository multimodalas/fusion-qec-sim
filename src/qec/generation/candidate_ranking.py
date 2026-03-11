"""
v8.4.0 — Candidate Ranking.

Ranks evaluated Tanner graph candidates using a deterministic
multi-objective scoring rule.

Layer 3 — Generation.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any


def rank_tanner_graph_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank candidates by stability, lower instability_score is better.

    Deterministic ranking uses a composite key:

    1. Lower ``instability_score`` (primary).
    2. Lower ``spectral_radius`` (secondary).
    3. Higher ``entropy`` (tertiary, negated for ascending sort).
    4. Higher ``bethe_margin`` (quaternary, negated for ascending sort).
    5. ``candidate_id`` (tie-breaker for total ordering).

    Parameters
    ----------
    candidates : list[dict[str, Any]]
        List of candidates, each containing at minimum:

        - ``candidate_id`` : str
        - ``instability_score`` : float
        - ``spectral_radius`` : float
        - ``entropy`` : float
        - ``bethe_margin`` : float

    Returns
    -------
    list[dict[str, Any]]
        Sorted copy of candidates (best first).
    """
    return sorted(
        candidates,
        key=lambda c: (
            c["instability_score"],
            c["spectral_radius"],
            -c["entropy"],
            -c["bethe_margin"],
            c["candidate_id"],
        ),
    )
