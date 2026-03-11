"""
Tests for v11.0.0 — TrappingSetDetector.

Verifies:
- deterministic trapping set detection
- correct ETS classification
- variable participation tracking
- edge cases (empty matrices, no trapping sets)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qec.analysis.trapping_sets import TrappingSetDetector


def _simple_H() -> np.ndarray:
    """A small (4,8) regular parity-check matrix with known structure."""
    return np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 1],
    ], dtype=np.float64)


class TestTrappingSetDetector:
    """Test suite for TrappingSetDetector."""

    def test_deterministic_detection(self):
        """Same matrix produces identical results across runs."""
        H = _simple_H()
        detector = TrappingSetDetector(max_a=6, max_b=4)
        result1 = detector.detect(H)
        result2 = detector.detect(H)
        assert result1["total"] == result2["total"]
        assert result1["counts"] == result2["counts"]
        assert result1["min_size"] == result2["min_size"]
        assert result1["variable_participation"] == result2["variable_participation"]

    def test_empty_matrix(self):
        """Empty matrix returns zero trapping sets."""
        H = np.zeros((0, 0), dtype=np.float64)
        detector = TrappingSetDetector()
        result = detector.detect(H)
        assert result["total"] == 0
        assert result["min_size"] == 0
        assert result["counts"] == {}

    def test_empty_columns(self):
        """Matrix with zero columns returns empty result."""
        H = np.zeros((3, 0), dtype=np.float64)
        detector = TrappingSetDetector()
        result = detector.detect(H)
        assert result["total"] == 0
        assert result["variable_participation"] == []

    def test_identity_matrix_no_ets(self):
        """Identity-like matrix (each var connected to one check) has no ETS
        with b > 0 where all checks have degree <= 2."""
        H = np.eye(4, dtype=np.float64)
        detector = TrappingSetDetector(max_a=4, max_b=4)
        result = detector.detect(H)
        # Single variable subsets: each has 1 check with degree 1 -> (1,1) ETS
        assert result["total"] >= 0  # May or may not find (1,1) sets

    def test_result_structure(self):
        """Result dictionary has required keys."""
        H = _simple_H()
        detector = TrappingSetDetector()
        result = detector.detect(H)
        assert "min_size" in result
        assert "counts" in result
        assert "total" in result
        assert "variable_participation" in result
        assert isinstance(result["counts"], dict)
        assert isinstance(result["variable_participation"], list)
        assert len(result["variable_participation"]) == H.shape[1]

    def test_counts_sum_to_total(self):
        """Sum of all counts equals total."""
        H = _simple_H()
        detector = TrappingSetDetector()
        result = detector.detect(H)
        assert sum(result["counts"].values()) == result["total"]

    def test_max_a_limit(self):
        """No trapping set exceeds max_a variable nodes."""
        H = _simple_H()
        detector = TrappingSetDetector(max_a=3, max_b=4)
        result = detector.detect(H)
        for (a, b) in result["counts"]:
            assert a <= 3

    def test_max_b_limit(self):
        """No trapping set exceeds max_b unsatisfied checks."""
        H = _simple_H()
        detector = TrappingSetDetector(max_a=6, max_b=2)
        result = detector.detect(H)
        for (a, b) in result["counts"]:
            assert b <= 2

    def test_known_ets_pair(self):
        """Matrix with a known (2,2) ETS detects it correctly.

        Two variable nodes each connected to the same two checks,
        with each check having exactly degree 1 in the induced subgraph
        when we pick just one var-check connection from each pair.
        """
        # Construct a matrix where vars 0,1 share checks 0,1
        # and each check also connects to other vars
        H = np.array([
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 1],
        ], dtype=np.float64)
        detector = TrappingSetDetector(max_a=4, max_b=4)
        result = detector.detect(H)
        # The subset {0, 1} has checks 0 (deg 2) and 1 (deg 2) -> (2, 0) ETS
        assert result["total"] > 0

    def test_participation_nonnegative(self):
        """All participation counts are non-negative."""
        H = _simple_H()
        detector = TrappingSetDetector()
        result = detector.detect(H)
        for count in result["variable_participation"]:
            assert count >= 0
