"""
v9.0.0 — Novelty System.

Computes novelty scores for discovery candidates based on deterministic
distance from archive elites.  Prevents convergence to a single
structural basin and encourages discovery of multiple Tanner graph
families.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_ROUND = 12

_FEATURE_KEYS = [
    "instability_score",
    "spectral_radius",
    "bethe_margin",
    "cycle_density",
    "entropy",
    "curvature",
    "ipr_localization",
]


def extract_feature_vector(objectives: dict[str, Any]) -> np.ndarray:
    """Extract the novelty feature vector from objectives.

    Parameters
    ----------
    objectives : dict[str, Any]
        Discovery objectives dictionary.

    Returns
    -------
    np.ndarray
        Feature vector of shape (7,).
    """
    return np.array(
        [float(objectives.get(k, 0.0)) for k in _FEATURE_KEYS],
        dtype=np.float64,
    )


def compute_novelty_score(
    feature_vector: np.ndarray,
    archive_features: list[np.ndarray],
    *,
    k_nearest: int = 5,
) -> float:
    """Compute novelty score as mean distance to k-nearest archive elites.

    Parameters
    ----------
    feature_vector : np.ndarray
        Feature vector of the candidate, shape (d,).
    archive_features : list[np.ndarray]
        Feature vectors of archive elites.
    k_nearest : int
        Number of nearest neighbours to average.

    Returns
    -------
    float
        Novelty score.  Higher = more novel.
    """
    if not archive_features:
        return 1.0

    distances = []
    for af in archive_features:
        d = float(np.linalg.norm(feature_vector - af))
        distances.append(d)

    distances.sort()
    k = min(k_nearest, len(distances))
    mean_dist = sum(distances[:k]) / k if k > 0 else 0.0

    return round(mean_dist, _ROUND)


def compute_population_novelty(
    population_objectives: list[dict[str, Any]],
    archive_features: list[np.ndarray],
    *,
    k_nearest: int = 5,
) -> list[float]:
    """Compute novelty scores for an entire population.

    Parameters
    ----------
    population_objectives : list[dict[str, Any]]
        Objectives for each candidate in the population.
    archive_features : list[np.ndarray]
        Feature vectors of archive elites.
    k_nearest : int
        Number of nearest neighbours for novelty computation.

    Returns
    -------
    list[float]
        Novelty scores for each candidate.
    """
    scores = []
    for obj in population_objectives:
        fv = extract_feature_vector(obj)
        score = compute_novelty_score(fv, archive_features, k_nearest=k_nearest)
        scores.append(score)
    return scores
