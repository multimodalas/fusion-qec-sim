"""
v9.0.0 — Discovery Search State.

Defines the canonical structure for a discovery candidate, tracking
provenance, evaluation results, and archive metadata.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_ROUND = 12


def make_search_state(
    *,
    candidate_id: str,
    generation: int,
    parent_id: str | None,
    operator: str | None,
    H: np.ndarray,
    metrics: dict[str, Any] | None = None,
    objectives: dict[str, Any] | None = None,
    novelty: float = 0.0,
    dominance_rank: int = 0,
    is_feasible: bool = True,
) -> dict[str, Any]:
    """Create a canonical search state dictionary.

    Parameters
    ----------
    candidate_id : str
        Unique identifier for this candidate.
    generation : int
        Generation number when this candidate was created.
    parent_id : str or None
        Identifier of the parent candidate, or None for initial population.
    operator : str or None
        Name of the mutation operator that produced this candidate.
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    metrics : dict or None
        Spectral metrics dictionary.
    objectives : dict or None
        Discovery objectives dictionary.
    novelty : float
        Novelty score relative to archive.
    dominance_rank : int
        Pareto dominance rank (0 = non-dominated).
    is_feasible : bool
        Whether the candidate satisfies structural constraints.

    Returns
    -------
    dict[str, Any]
        Canonical search state dictionary.
    """
    return {
        "candidate_id": candidate_id,
        "generation": generation,
        "parent_id": parent_id,
        "operator": operator,
        "H": np.asarray(H, dtype=np.float64),
        "metrics": metrics if metrics is not None else {},
        "objectives": objectives if objectives is not None else {},
        "novelty": round(float(novelty), _ROUND),
        "dominance_rank": dominance_rank,
        "is_feasible": is_feasible,
    }


def serialize_search_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert a search state to a JSON-serializable dictionary.

    Converts numpy arrays to nested lists and ensures all values are
    plain Python types.

    Parameters
    ----------
    state : dict[str, Any]
        Search state from ``make_search_state``.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary.
    """
    result = {}
    for key, val in sorted(state.items()):
        if key == "H":
            result[key] = val.tolist() if isinstance(val, np.ndarray) else val
        elif isinstance(val, np.ndarray):
            result[key] = val.tolist()
        elif isinstance(val, (np.integer,)):
            result[key] = int(val)
        elif isinstance(val, (np.floating,)):
            result[key] = float(val)
        elif isinstance(val, dict):
            result[key] = {
                str(k): float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in sorted(val.items())
            }
        else:
            result[key] = val
    return result
