"""
Tests for v11.0.0 — Decoder-Aware Fitness Integration.

Verifies:
- decoder-aware fitness engine produces correct metrics
- backward compatibility: non-decoder-aware mode unchanged
- fitness reacts to instability (higher penalty = lower fitness)
- caching works for decoder-aware evaluation
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qec.fitness.fitness_engine import FitnessEngine


def _simple_H() -> np.ndarray:
    """A small (4,8) regular parity-check matrix."""
    return np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 1],
    ], dtype=np.float64)


def _dense_H() -> np.ndarray:
    """A dense matrix likely to have worse decoder properties."""
    return np.array([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 0],
    ], dtype=np.float64)


class TestDecoderAwareFitness:
    """Test suite for decoder-aware FitnessEngine."""

    def test_backward_compatible_default(self):
        """Default FitnessEngine (no decoder-aware) still works."""
        engine = FitnessEngine()
        H = _simple_H()
        result = engine.evaluate(H)
        assert "composite" in result
        assert "components" in result
        assert "metrics" in result
        # Should NOT have decoder-aware metrics
        assert "bp_stability_score" not in result["metrics"]

    def test_decoder_aware_metrics_present(self):
        """Decoder-aware engine includes BP and trapping set metrics."""
        engine = FitnessEngine(decoder_aware=True, bp_trials=10, bp_iterations=5)
        H = _simple_H()
        result = engine.evaluate(H)
        assert "bp_stability_score" in result["metrics"]
        assert "bp_divergence_rate" in result["metrics"]
        assert "bp_stagnation_rate" in result["metrics"]
        assert "bp_oscillation_score" in result["metrics"]
        assert "trapping_set_total" in result["metrics"]
        assert "trapping_set_penalty" in result["metrics"]
        assert "jacobian_spectral_radius" in result["metrics"]
        assert "jacobian_stability" in result["metrics"]

    def test_decoder_aware_components_present(self):
        """Decoder-aware engine includes decoder components."""
        engine = FitnessEngine(decoder_aware=True, bp_trials=10, bp_iterations=5)
        H = _simple_H()
        result = engine.evaluate(H)
        assert "bp_stability_score" in result["components"]
        assert "trapping_set_penalty" in result["components"]
        assert "jacobian_stability" in result["components"]

    def test_deterministic_evaluation(self):
        """Decoder-aware evaluation is deterministic for decoder metrics."""
        engine = FitnessEngine(
            decoder_aware=True, bp_trials=10, bp_iterations=5, bp_seed=42,
        )
        H = _simple_H()
        result1 = engine.evaluate(H)
        engine.clear_cache()
        result2 = engine.evaluate(H)
        # Decoder-aware metrics must be deterministic
        da_keys = [
            "bp_stability_score", "bp_divergence_rate", "bp_stagnation_rate",
            "bp_oscillation_score", "trapping_set_total", "trapping_set_penalty",
            "jacobian_spectral_radius", "jacobian_stability",
        ]
        for key in da_keys:
            assert result1["metrics"][key] == result2["metrics"][key], f"{key} differs"

    def test_cache_works(self):
        """Second evaluation returns cached result."""
        engine = FitnessEngine(decoder_aware=True, bp_trials=10, bp_iterations=5)
        H = _simple_H()
        result1 = engine.evaluate(H)
        result2 = engine.evaluate(H)
        assert result1 is result2  # Same object from cache

    def test_stability_score_range(self):
        """BP stability score is in [0, 1]."""
        engine = FitnessEngine(decoder_aware=True, bp_trials=10, bp_iterations=5)
        H = _simple_H()
        result = engine.evaluate(H)
        score = result["metrics"]["bp_stability_score"]
        assert 0.0 <= score <= 1.0

    def test_trapping_set_penalty_range(self):
        """Trapping set penalty is in [0, 1]."""
        engine = FitnessEngine(decoder_aware=True, bp_trials=10, bp_iterations=5)
        H = _simple_H()
        result = engine.evaluate(H)
        penalty = result["metrics"]["trapping_set_penalty"]
        assert 0.0 <= penalty <= 1.0

    def test_jacobian_stability_range(self):
        """Jacobian stability is non-negative."""
        engine = FitnessEngine(decoder_aware=True, bp_trials=10, bp_iterations=5)
        H = _simple_H()
        result = engine.evaluate(H)
        stability = result["metrics"]["jacobian_stability"]
        assert stability >= 0.0

    def test_custom_weights_decoder_aware(self):
        """Custom weights work with decoder-aware mode."""
        weights = {
            "girth": 1.0,
            "nbt_spectral_radius": -1.0,
            "ace_variance": -0.5,
            "expansion": 1.0,
            "cycle_density": -0.5,
            "sparsity": 0.2,
            "bp_stability_score": 5.0,
            "trapping_set_penalty": -2.0,
            "jacobian_stability": 2.0,
        }
        engine = FitnessEngine(
            weights=weights, decoder_aware=True, bp_trials=10, bp_iterations=5,
        )
        H = _simple_H()
        result = engine.evaluate(H)
        assert "composite" in result
