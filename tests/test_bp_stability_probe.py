"""
Tests for v11.0.0 — BPStabilityProbe and estimate_bp_instability.

Verifies:
- deterministic BP probe results
- stability score formula correctness
- Jacobian spectral radius estimation
- edge cases
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.qec.decoder.stability_probe import BPStabilityProbe, estimate_bp_instability


def _simple_H() -> np.ndarray:
    """A small (4,8) regular parity-check matrix."""
    return np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 1],
    ], dtype=np.float64)


def _high_density_H() -> np.ndarray:
    """A dense matrix that may cause BP instability."""
    return np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
    ], dtype=np.float64)


class TestBPStabilityProbe:
    """Test suite for BPStabilityProbe."""

    def test_deterministic_probe(self):
        """Same matrix and seed produce identical results."""
        H = _simple_H()
        probe = BPStabilityProbe(trials=20, iterations=5, seed=42)
        result1 = probe.probe(H)
        result2 = probe.probe(H)
        assert result1 == result2

    def test_different_seeds_may_differ(self):
        """Different seeds may produce different results."""
        H = _simple_H()
        probe1 = BPStabilityProbe(trials=20, iterations=5, seed=42)
        probe2 = BPStabilityProbe(trials=20, iterations=5, seed=99)
        result1 = probe1.probe(H)
        result2 = probe2.probe(H)
        # Results may or may not differ; just check they're valid
        assert 0.0 <= result1["bp_stability_score"] <= 1.0
        assert 0.0 <= result2["bp_stability_score"] <= 1.0

    def test_empty_matrix(self):
        """Empty matrix returns default stability."""
        H = np.zeros((0, 0), dtype=np.float64)
        probe = BPStabilityProbe()
        result = probe.probe(H)
        assert result["bp_stability_score"] == 1.0
        assert result["divergence_rate"] == 0.0

    def test_result_structure(self):
        """Result has all required keys."""
        H = _simple_H()
        probe = BPStabilityProbe(trials=10, iterations=5, seed=0)
        result = probe.probe(H)
        assert "bp_stability_score" in result
        assert "divergence_rate" in result
        assert "stagnation_rate" in result
        assert "oscillation_score" in result
        assert "average_iterations" in result

    def test_stability_score_range(self):
        """Stability score is in [0, 1]."""
        H = _simple_H()
        probe = BPStabilityProbe(trials=30, iterations=8, seed=0)
        result = probe.probe(H)
        assert 0.0 <= result["bp_stability_score"] <= 1.0

    def test_divergence_rate_range(self):
        """Divergence rate is in [0, 1]."""
        H = _simple_H()
        probe = BPStabilityProbe(trials=20, iterations=5, seed=0)
        result = probe.probe(H)
        assert 0.0 <= result["divergence_rate"] <= 1.0

    def test_stagnation_rate_range(self):
        """Stagnation rate is in [0, 1]."""
        H = _simple_H()
        probe = BPStabilityProbe(trials=20, iterations=5, seed=0)
        result = probe.probe(H)
        assert 0.0 <= result["stagnation_rate"] <= 1.0

    def test_stability_formula(self):
        """Stability score follows the specified formula."""
        H = _simple_H()
        probe = BPStabilityProbe(trials=30, iterations=8, seed=0)
        result = probe.probe(H)
        expected = (
            (1.0 - result["divergence_rate"])
            * (1.0 - result["stagnation_rate"])
            * math.exp(-result["oscillation_score"])
        )
        assert abs(result["bp_stability_score"] - round(expected, 12)) < 1e-10

    def test_single_trial(self):
        """Works with a single trial."""
        H = _simple_H()
        probe = BPStabilityProbe(trials=1, iterations=3, seed=0)
        result = probe.probe(H)
        assert 0.0 <= result["bp_stability_score"] <= 1.0


class TestEstimateBPInstability:
    """Test suite for estimate_bp_instability."""

    def test_deterministic(self):
        """Same matrix produces identical results."""
        H = _simple_H()
        result1 = estimate_bp_instability(H)
        result2 = estimate_bp_instability(H)
        assert result1 == result2

    def test_empty_matrix(self):
        """Empty matrix returns zero radius."""
        H = np.zeros((0, 0), dtype=np.float64)
        result = estimate_bp_instability(H)
        assert result["jacobian_spectral_radius_est"] == 0.0

    def test_result_structure(self):
        """Result has required key."""
        H = _simple_H()
        result = estimate_bp_instability(H)
        assert "jacobian_spectral_radius_est" in result

    def test_nonnegative_radius(self):
        """Spectral radius estimate is non-negative."""
        H = _simple_H()
        result = estimate_bp_instability(H)
        assert result["jacobian_spectral_radius_est"] >= 0.0

    def test_identity_stable(self):
        """Identity matrix should have low spectral radius."""
        H = np.eye(4, dtype=np.float64)
        result = estimate_bp_instability(H)
        # Identity has d_v=1, d_c=1, so inv_deg=0 -> radius ~0
        assert result["jacobian_spectral_radius_est"] < 1.0

    def test_dense_matrix_higher_radius(self):
        """Dense matrix tends to have higher instability."""
        H_sparse = np.eye(4, dtype=np.float64)
        H_dense = _high_density_H()
        r_sparse = estimate_bp_instability(H_sparse)["jacobian_spectral_radius_est"]
        r_dense = estimate_bp_instability(H_dense)["jacobian_spectral_radius_est"]
        # Dense matrix should generally have higher instability
        assert r_dense >= r_sparse
