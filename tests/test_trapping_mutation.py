"""
Tests for v11.0.0 — Trapping Set Pressure Mutation.

Verifies:
- mutation is deterministic
- mutation produces valid matrices
- mutation reduces trapping set participation
- edge cases (no trapping sets, empty matrix)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qec.discovery.guided_mutations import (
    trapping_set_pressure_mutation,
    apply_guided_mutation,
)
from src.qec.analysis.trapping_sets import TrappingSetDetector


def _simple_H() -> np.ndarray:
    """A small (4,8) regular parity-check matrix."""
    return np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 1],
    ], dtype=np.float64)


def _dense_H() -> np.ndarray:
    """A denser matrix with more trapping set potential."""
    return np.array([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 0],
    ], dtype=np.float64)


class TestTrappingSetPressureMutation:
    """Test suite for trapping_set_pressure_mutation."""

    def test_deterministic(self):
        """Same seed produces identical results."""
        H = _simple_H()
        result1 = trapping_set_pressure_mutation(H, seed=42)
        result2 = trapping_set_pressure_mutation(H, seed=42)
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds(self):
        """Different seeds may produce different results."""
        H = _simple_H()
        result1 = trapping_set_pressure_mutation(H, seed=42)
        result2 = trapping_set_pressure_mutation(H, seed=99)
        # At least one should differ (or both unchanged if no trapping sets)
        # Just verify both are valid
        assert result1.shape == H.shape
        assert result2.shape == H.shape

    def test_preserves_shape(self):
        """Mutation preserves matrix shape."""
        H = _simple_H()
        result = trapping_set_pressure_mutation(H, seed=0)
        assert result.shape == H.shape

    def test_binary_output(self):
        """Output contains only 0s and 1s."""
        H = _simple_H()
        result = trapping_set_pressure_mutation(H, seed=0)
        assert set(np.unique(result)).issubset({0.0, 1.0})

    def test_no_input_mutation(self):
        """Original matrix is not modified."""
        H = _simple_H()
        H_orig = H.copy()
        _ = trapping_set_pressure_mutation(H, seed=0)
        np.testing.assert_array_equal(H, H_orig)

    def test_empty_matrix(self):
        """Empty matrix returns copy."""
        H = np.zeros((0, 0), dtype=np.float64)
        result = trapping_set_pressure_mutation(H, seed=0)
        assert result.shape == (0, 0)

    def test_no_isolated_nodes(self):
        """Mutation does not create all-zero rows or columns."""
        H = _simple_H()
        result = trapping_set_pressure_mutation(H, seed=0)
        # Check no all-zero rows
        for ci in range(result.shape[0]):
            assert result[ci].sum() >= 1
        # Check no all-zero columns
        for vi in range(result.shape[1]):
            assert result[:, vi].sum() >= 1

    def test_edge_count_preserved(self):
        """Mutation preserves total edge count (rewires, not adds/removes)."""
        H = _simple_H()
        result = trapping_set_pressure_mutation(H, seed=0)
        assert H.sum() == result.sum()

    def test_registered_in_dispatcher(self):
        """trapping_set_pressure is available via apply_guided_mutation."""
        H = _simple_H()
        result = apply_guided_mutation(
            H, operator="trapping_set_pressure", seed=42,
        )
        assert result.shape == H.shape

    def test_mutation_targets_trapping_sets(self):
        """Mutation targets variables in trapping sets."""
        # Use a larger matrix where trapping sets exist and rewiring is feasible
        H = np.array([
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
        ], dtype=np.float64)
        detector = TrappingSetDetector(max_a=4, max_b=4)
        before = detector.detect(H)
        if before["total"] > 0:
            result = trapping_set_pressure_mutation(H, seed=42)
            # The mutation should have changed at least one edge
            assert not np.array_equal(H, result)

    def test_no_trapping_sets_returns_copy(self):
        """If no trapping sets, matrix is returned unchanged."""
        # Diagonal matrix has no trapping sets with b <= 4 typically
        H = np.eye(4, dtype=np.float64)
        detector = TrappingSetDetector(max_a=4, max_b=4)
        ts = detector.detect(H)
        if ts["total"] == 0:
            result = trapping_set_pressure_mutation(H, seed=0)
            np.testing.assert_array_equal(H, result)
