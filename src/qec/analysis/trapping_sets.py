"""
v11.0.0 — Trapping Set Detector.

Detects small elementary trapping sets (ETS) in Tanner graphs derived
from LDPC/QLDPC parity-check matrices.

An (a, b)-elementary trapping set is a subgraph on *a* variable nodes
where exactly *b* neighbouring check nodes have odd degree (degree 1)
within the induced subgraph.  All checks in the induced subgraph have
degree 1 or 2.

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_ROUND = 12


class TrappingSetDetector:
    """Detect small elementary trapping sets in a Tanner graph.

    Parameters
    ----------
    max_a : int
        Maximum number of variable nodes in a trapping set (default 6).
    max_b : int
        Maximum number of odd-degree check nodes (default 4).
    """

    def __init__(self, max_a: int = 6, max_b: int = 4) -> None:
        self.max_a = max_a
        self.max_b = max_b

    def detect(self, H: np.ndarray) -> dict[str, Any]:
        """Detect small elementary trapping sets in *H*.

        Parameters
        ----------
        H : np.ndarray
            Binary parity-check matrix, shape (m, n).

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - ``min_size`` : int — smallest *a* found (0 if none)
            - ``counts`` : dict — mapping ``(a, b)`` → count
            - ``total`` : int — total number of ETS found
            - ``variable_participation`` : list[int] — per-variable ETS count
        """
        H_arr = np.asarray(H, dtype=np.float64)
        m, n = H_arr.shape

        if m == 0 or n == 0:
            return {
                "min_size": 0,
                "counts": {},
                "total": 0,
                "variable_participation": [0] * n,
            }

        # Build adjacency: variable -> list of checks
        var_to_checks: list[list[int]] = [[] for _ in range(n)]
        for vi in range(n):
            for ci in range(m):
                if H_arr[ci, vi] != 0:
                    var_to_checks[vi].append(ci)

        counts: dict[tuple[int, int], int] = {}
        variable_participation = [0] * n
        found_sets: set[tuple[int, ...]] = set()

        # Enumerate variable-node subsets of size a = 1..max_a
        # Use iterative DFS to enumerate subsets in sorted order
        for a in range(1, self.max_a + 1):
            self._enumerate_subsets(
                a, n, var_to_checks, m, counts, variable_participation,
                found_sets,
            )

        total = sum(counts.values())
        min_size = min((ab[0] for ab in counts), default=0)

        return {
            "min_size": min_size,
            "counts": counts,
            "total": total,
            "variable_participation": variable_participation,
        }

    def _enumerate_subsets(
        self,
        a: int,
        n: int,
        var_to_checks: list[list[int]],
        m: int,
        counts: dict[tuple[int, int], int],
        variable_participation: list[int],
        found_sets: set[tuple[int, ...]],
    ) -> None:
        """Enumerate all variable-node subsets of size *a* and check ETS."""
        if a > n:
            return

        # Generate combinations iteratively using sorted indices
        indices = list(range(a))
        while True:
            subset = tuple(indices)
            if subset not in found_sets:
                b = self._check_ets(subset, var_to_checks, m)
                if b is not None and b <= self.max_b:
                    key = (a, b)
                    counts[key] = counts.get(key, 0) + 1
                    found_sets.add(subset)
                    for vi in subset:
                        variable_participation[vi] += 1

            # Generate next combination
            i = a - 1
            while i >= 0 and indices[i] == n - a + i:
                i -= 1
            if i < 0:
                break
            indices[i] += 1
            for j in range(i + 1, a):
                indices[j] = indices[j - 1] + 1

    @staticmethod
    def _check_ets(
        subset: tuple[int, ...],
        var_to_checks: list[list[int]],
        m: int,
    ) -> int | None:
        """Check if a variable-node subset forms an ETS.

        Returns the *b* value (number of odd-degree checks) if the subset
        is an elementary trapping set, or None if it is not.
        """
        var_set = set(subset)

        # Count degree of each check in the induced subgraph
        check_degree: dict[int, int] = {}
        for vi in subset:
            for ci in var_to_checks[vi]:
                check_degree[ci] = check_degree.get(ci, 0) + 1

        # ETS requires all checks have degree 1 or 2
        b = 0
        for ci, deg in check_degree.items():
            if deg > 2:
                return None
            if deg == 1:
                b += 1

        return b
