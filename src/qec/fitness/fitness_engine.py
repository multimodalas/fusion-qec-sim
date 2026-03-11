"""
v10.0.0 — Fitness Engine.

Computes weighted composite fitness scores from spectral metrics for
LDPC/QLDPC parity-check matrices.  Supports caching by matrix hash.

Layer 3 — Fitness.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from src.qec.fitness.spectral_metrics import (
    compute_nbt_spectral_radius,
    compute_girth_spectrum,
    compute_ace_spectrum,
    estimate_eigenvector_ipr,
)


_ROUND = 12


def _matrix_hash(H: np.ndarray) -> str:
    """Compute a deterministic content hash for a parity-check matrix."""
    data = np.asarray(H, dtype=np.float64).tobytes()
    return hashlib.sha256(data).hexdigest()


class FitnessEngine:
    """Computes composite fitness scores for parity-check matrices.

    The composite fitness is a weighted combination of:
    - girth (maximize)
    - NBT spectral radius (minimize)
    - ACE spectrum variance (minimize)
    - expansion coefficient (maximize)
    - cycle density (minimize)
    - sparsity (maintain)

    Parameters
    ----------
    weights : dict[str, float] or None
        Optional custom weights for fitness components.
        Default weights are used if not provided.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or {
            "girth": 3.0,
            "nbt_spectral_radius": -2.0,
            "ace_variance": -1.5,
            "expansion": 2.0,
            "cycle_density": -1.0,
            "sparsity": 0.5,
        }
        self._cache: dict[str, dict[str, Any]] = {}

    def evaluate(self, H: np.ndarray) -> dict[str, Any]:
        """Compute composite fitness for a parity-check matrix.

        Parameters
        ----------
        H : np.ndarray
            Binary parity-check matrix, shape (m, n).

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - ``composite`` : float — weighted composite score
            - ``components`` : dict — individual weighted components
            - ``metrics`` : dict — raw metric values
        """
        H_arr = np.asarray(H, dtype=np.float64)
        h = _matrix_hash(H_arr)

        if h in self._cache:
            return self._cache[h]

        metrics = self._compute_metrics(H_arr)
        components = self._compute_components(metrics)
        composite = sum(components.values())

        result = {
            "composite": round(composite, _ROUND),
            "components": {k: round(v, _ROUND) for k, v in sorted(components.items())},
            "metrics": {k: round(v, _ROUND) if isinstance(v, float) else v
                        for k, v in sorted(metrics.items())},
        }

        self._cache[h] = result
        return result

    def _compute_metrics(self, H: np.ndarray) -> dict[str, Any]:
        """Compute all raw metrics."""
        m, n = H.shape

        nbt_radius = compute_nbt_spectral_radius(H)
        girth_result = compute_girth_spectrum(H)
        ace = compute_ace_spectrum(H)
        ipr_result = estimate_eigenvector_ipr(H)

        # Expansion coefficient: ratio of unique 2-hop neighbours to
        # expected for a tree-like graph
        total_edges = float(H.sum())
        if total_edges > 0 and n > 0:
            avg_var_deg = total_edges / n
            avg_check_deg = total_edges / m if m > 0 else 0.0
            # Expected 2-hop reach for tree-like
            expected = avg_var_deg * (avg_check_deg - 1)
            # Actual average 2-hop reach
            HtH = H.T @ H
            np.fill_diagonal(HtH, 0)
            actual_reach = np.mean(np.sum(HtH > 0, axis=1))
            expansion = actual_reach / max(expected, 1.0)
        else:
            expansion = 0.0

        # Cycle density: total short cycles normalised by edges
        total_cycles = sum(girth_result["cycle_counts"].values())
        cycle_density = total_cycles / max(total_edges, 1.0)

        # Sparsity: fraction of non-zero entries
        sparsity = total_edges / max(m * n, 1)

        # ACE variance
        ace_var = float(np.var(ace)) if len(ace) > 0 else 0.0

        return {
            "nbt_spectral_radius": nbt_radius,
            "girth": girth_result["girth"],
            "cycle_counts": girth_result["cycle_counts"],
            "ace_variance": ace_var,
            "ace_min": float(np.min(ace)) if len(ace) > 0 else 0.0,
            "ace_mean": float(np.mean(ace)) if len(ace) > 0 else 0.0,
            "expansion": expansion,
            "cycle_density": cycle_density,
            "sparsity": sparsity,
            "mean_ipr": ipr_result["mean_ipr"],
            "max_ipr": ipr_result["max_ipr"],
        }

    def _compute_components(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Compute weighted component scores from metrics."""
        w = self._weights

        components = {
            "girth": w.get("girth", 0.0) * float(metrics["girth"]),
            "nbt_spectral_radius": w.get("nbt_spectral_radius", 0.0) * metrics["nbt_spectral_radius"],
            "ace_variance": w.get("ace_variance", 0.0) * metrics["ace_variance"],
            "expansion": w.get("expansion", 0.0) * metrics["expansion"],
            "cycle_density": w.get("cycle_density", 0.0) * metrics["cycle_density"],
            "sparsity": w.get("sparsity", 0.0) * metrics["sparsity"],
        }

        return components

    def clear_cache(self) -> None:
        """Clear the fitness evaluation cache."""
        self._cache.clear()
