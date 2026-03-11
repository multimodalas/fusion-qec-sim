"""
v11.0.0 — Fitness Engine.

Computes weighted composite fitness scores from spectral metrics for
LDPC/QLDPC parity-check matrices.  Supports caching by matrix hash.

v11.0.0 adds decoder-aware fitness components:
- trapping_set_penalty (from TrappingSetDetector)
- bp_stability_score (from BPStabilityProbe)
- jacobian_instability_penalty (from estimate_bp_instability)

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
from src.qec.analysis.trapping_sets import TrappingSetDetector
from src.qec.decoder.stability_probe import BPStabilityProbe, estimate_bp_instability


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

    v11.0.0 decoder-aware mode adds:
    - bp_stability_score (maximize)
    - trapping_set_penalty (minimize)
    - jacobian_stability (maximize, derived from 1 - instability)

    Parameters
    ----------
    weights : dict[str, float] or None
        Optional custom weights for fitness components.
        Default weights are used if not provided.
    decoder_aware : bool
        Enable decoder-aware fitness evaluation (default False).
    bp_trials : int
        Number of BP probe trials when decoder_aware is True (default 50).
    bp_iterations : int
        Max BP iterations per probe trial (default 10).
    bp_seed : int
        Base seed for BP stability probe (default 0).
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        decoder_aware: bool = False,
        bp_trials: int = 50,
        bp_iterations: int = 10,
        bp_seed: int = 0,
    ) -> None:
        self._decoder_aware = decoder_aware
        if weights is not None:
            self._weights = weights
        elif decoder_aware:
            self._weights = {
                "girth": 1.65,
                "nbt_spectral_radius": -1.1,
                "ace_variance": -0.825,
                "expansion": 1.1,
                "cycle_density": -0.55,
                "sparsity": 0.275,
                "bp_stability_score": 2.5,
                "trapping_set_penalty": -1.0,
                "jacobian_stability": 1.0,
            }
        else:
            self._weights = {
                "girth": 3.0,
                "nbt_spectral_radius": -2.0,
                "ace_variance": -1.5,
                "expansion": 2.0,
                "cycle_density": -1.0,
                "sparsity": 0.5,
            }
        self._cache: dict[str, dict[str, Any]] = {}
        self._trapping_detector = TrappingSetDetector() if decoder_aware else None
        self._bp_probe = (
            BPStabilityProbe(trials=bp_trials, iterations=bp_iterations, seed=bp_seed)
            if decoder_aware else None
        )

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

        metrics = {
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

        # v11.0.0 decoder-aware metrics
        if self._decoder_aware:
            # Trapping set detection
            ts_result = self._trapping_detector.detect(H)
            total_edges = float(H.sum())
            ts_density = ts_result["total"] / max(total_edges, 1.0)
            metrics["trapping_set_total"] = ts_result["total"]
            metrics["trapping_set_min_size"] = ts_result["min_size"]
            metrics["trapping_set_penalty"] = min(ts_density, 1.0)

            # BP stability probe
            bp_result = self._bp_probe.probe(H)
            metrics["bp_stability_score"] = bp_result["bp_stability_score"]
            metrics["bp_divergence_rate"] = bp_result["divergence_rate"]
            metrics["bp_stagnation_rate"] = bp_result["stagnation_rate"]
            metrics["bp_oscillation_score"] = bp_result["oscillation_score"]

            # Jacobian instability estimate
            jac_result = estimate_bp_instability(H)
            rho = jac_result["jacobian_spectral_radius_est"]
            metrics["jacobian_spectral_radius"] = rho
            metrics["jacobian_stability"] = max(0.0, 1.0 - rho)

        return metrics

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

        # v11.0.0 decoder-aware components
        if self._decoder_aware:
            components["bp_stability_score"] = (
                w.get("bp_stability_score", 0.0) * metrics.get("bp_stability_score", 0.0)
            )
            components["trapping_set_penalty"] = (
                w.get("trapping_set_penalty", 0.0) * metrics.get("trapping_set_penalty", 0.0)
            )
            components["jacobian_stability"] = (
                w.get("jacobian_stability", 0.0) * metrics.get("jacobian_stability", 0.0)
            )

        return components

    def clear_cache(self) -> None:
        """Clear the fitness evaluation cache."""
        self._cache.clear()
